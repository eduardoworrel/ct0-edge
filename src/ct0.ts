/**
 * ct0.ts — lib tipada que orquestra o ONNX padrão (grafo único prefill+decode, scan) do
 * CT0/alias3-micro, via onnxruntime-web (WebGPU). NÃO depende do transformers.js nem de PR:
 * dirige a interface padrão do grafo diretamente.
 *
 * Um único grafo dinâmico roda os dois modos:
 *  - System 2 (generate): prefill do prompt (batched) + decode autoregressivo.
 *  - System 1 (decide):   decisão sim/não em 1 forward, lendo os logits (zero tokens gerados).
 *
 * Formatos (dtype): "q4" (int4, menor; ótimo p/ System 1 e geração), "int8", "fp16", "fp32".
 * int4 cobre System 1 (curto) e geração; p/ prefill de ENTRADA muito longa use int8/fp16.
 *
 * Uso:
 *   import { CT0 } from "./ct0.js";
 *   const ct0 = await CT0.load({ dtype: "q4", onStatus: console.log });
 *   for await (const piece of ct0.generate("Olá", { temperature: 0.7 })) process(piece);
 *   const p = await ct0.decide("path:/etc/cron.d/x\n...", "Is this an IOC?"); // 0..1
 */
import * as ort from "onnxruntime-web/webgpu";
import { AutoTokenizer, env } from "@huggingface/transformers";

export type Dtype = "q4" | "q4f16" | "int8" | "fp16" | "fp32";

export interface LoadOptions {
  /** URL base onde estão a pasta onnx/ e o tokenizer (default: ao lado do módulo). */
  baseUrl?: string;
  /** Precisão do grafo a carregar. Default "q4". */
  dtype?: Dtype;
  /** Callback de progresso. */
  onStatus?: (msg: string) => void;
}

export interface GenerateOptions {
  maxNewTokens?: number;      // default 256
  temperature?: number;       // 0 = greedy; default 0.7
  topP?: number;              // default 0.9
  topK?: number;              // default 40 (0 = desligado)
  repetitionPenalty?: number; // default 1.1
  seed?: number;              // default 0
  system?: string;            // system prompt
  think?: boolean;            // habilita <think>; default true
  signal?: AbortSignal;       // cancelamento
}

const EOS = new Set<number>([248046, 248044]);
const FILE: Record<Dtype, string> = {
  fp32: "model.onnx", fp16: "model_fp16.onnx", int8: "model_int8.onnx",
  q4: "model_q4.onnx", q4f16: "model_q4f16.onnx",
};

const prod = (a: number[]) => a.reduce((x, y) => x * y, 1);
function mulberry32(seed: number) {
  let a = seed >>> 0 || 1;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** Descritor de uma entrada de cache (past_*): shape com eixos dinâmicos e dtype. */
interface CacheSpec { name: string; dims: (number | string)[]; f16: boolean; seqAxis: number; }

export class CT0 {
  readonly maxContext = 262144;
  readonly vocab = 248320;
  private tok: any;
  private S: any;                       // ort.InferenceSession
  private pastSpecs: CacheSpec[] = [];  // entradas past_*
  private p2p: Record<string, string> = {}; // present(output) -> past(input)
  private outNames: string[] = [];

  private constructor(tok: any, S: any) { this.tok = tok; this.S = S; }

  static async load(opts: LoadOptions = {}): Promise<CT0> {
    const status = opts.onStatus ?? (() => {});
    const base = opts.baseUrl ?? new URL(".", import.meta.url).href;
    const dtype = opts.dtype ?? "q4";
    env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = base;
    status("carregando tokenizer…");
    const tok = await AutoTokenizer.from_pretrained("tok");
    const file = "onnx/" + FILE[dtype];
    status(`carregando grafo ${dtype} (WebGPU)…`);
    const S = await ort.InferenceSession.create(new URL(file, base).href, {
      executionProviders: ["webgpu"],
      externalData: [{ path: FILE[dtype] + "_data", data: new URL(file + "_data", base).href }],
    });
    const ct0 = new CT0(tok, S);
    ct0.outNames = S.outputNames.slice();
    // past_* inputs (cache) e mapeamento present->past
    for (const name of S.inputNames) {
      if (!name.startsWith("past_")) continue;
      const meta = (S as any).inputMetadata?.[name];
      // fallback: deduz shape/dtype pelos nomes conhecidos do qwen3_5
      const { dims, f16, seqAxis } = ct0.specFor(name, meta);
      ct0.pastSpecs.push({ name, dims, f16, seqAxis });
    }
    for (const o of S.outputNames) {
      if (o === "logits") continue;
      const past = o.replace("present_conv", "past_conv")
                    .replace("present_recurrent", "past_recurrent")
                    .replace(/^present\./, "past_key_values.");
      ct0.p2p[o] = past;
    }
    status("pronto.");
    return ct0;
  }

  /** Shape/dtype de uma entrada past_*; usa metadata quando disponível, senão nomes do qwen3_5. */
  private specFor(name: string, meta: any): { dims: (number | string)[]; f16: boolean; seqAxis: number } {
    let dims: (number | string)[]; let f16 = false; let seqAxis = -1;
    if (name.startsWith("past_conv")) { dims = [1, 8192, 4]; }
    else if (name.startsWith("past_recurrent")) { dims = [1, 32, 128, 128]; }
    else { dims = [1, 4, 0, 256]; seqAxis = 2; } // past_key_values.N.key/value (seq cresce no eixo 2)
    if (meta) {
      const t = (meta.type || meta.dataType || "").toString();
      if (t.includes("16")) f16 = true;
    }
    return { dims, f16, seqAxis };
  }

  /** Cache vazio (seq=0 nas entradas de atenção full; conv/recurrent têm tamanho fixo). */
  private emptyCache(): Record<string, any> {
    const c: Record<string, any> = {};
    for (const s of this.pastSpecs) {
      const dims = s.dims.map((d, i) => (i === s.seqAxis ? 0 : (typeof d === "number" ? d : 1)));
      const n = prod(dims as number[]);
      c[s.name] = new ort.Tensor(s.f16 ? "float16" : "float32",
        s.f16 ? new Uint16Array(n) : new Float32Array(n), dims as number[]);
    }
    return c;
  }

  /** Um forward do grafo. ids: tokens deste passo; pos0: posição do 1º token; total: tokens vistos. */
  private async forward(ids: number[], pos0: number, total: number, cache: Record<string, any>) {
    const seq = ids.length;
    const feeds: Record<string, any> = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, seq]),
      attention_mask: new ort.Tensor("int64", BigInt64Array.from(Array(total).fill(1n)), [1, total]),
      position_ids: new ort.Tensor("int64", BigInt64Array.from(Array.from({ length: seq }, (_, i) => BigInt(pos0 + i))), [1, seq]),
      ...cache,
    };
    const out = await this.S.run(feeds);
    const next: Record<string, any> = {};
    for (const o of this.outNames) if (o !== "logits") next[this.p2p[o]] = out[o];
    return { logits: out.logits, cache: next };
  }

  private lastRow(L: any): Float32Array {
    const V = L.dims.at(-1), off = (L.dims.at(-2) - 1) * V, d = L.data, r = new Float32Array(V);
    for (let i = 0; i < V; i++) r[i] = Number(d[off + i]);
    return r;
  }
  private ids(text: string): number[] { return this.tok(text, { add_special_tokens: false }).input_ids.tolist()[0].map(Number); }
  private turn(msg: string, system: string | undefined, think: boolean) {
    let t = "";
    if (system) t += "<|im_start|>system\n" + system + "<|im_end|>\n";
    t += "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n";
    t += think ? "<think>\n" : "<think>\n\n</think>\n\n";
    return t;
  }

  /** System 2 — geração autoregressiva (streaming de texto). Prefill batched + decode. */
  async *generate(prompt: string, opts: GenerateOptions = {}): AsyncGenerator<string> {
    const o = { maxNewTokens: 256, temperature: 0.7, topP: 0.9, topK: 40, repetitionPenalty: 1.1, seed: 0, think: true, ...opts };
    const rand = mulberry32(o.seed);
    const ids = this.ids(this.turn(prompt, o.system, o.think));
    let cache = this.emptyCache();
    let { logits, cache: c } = await this.forward(ids, 0, ids.length, cache); cache = c;
    let pos = ids.length; const gen: number[] = [];
    for (let s = 0; s < o.maxNewTokens; s++) {
      if (opts.signal?.aborted) break;
      const tid = this.sample(this.lastRow(logits), gen, rand, o);
      if (EOS.has(tid)) break;
      gen.push(tid);
      yield this.tok.decode([tid], { skip_special_tokens: true });
      ({ logits, cache } = await this.forward([tid], pos, pos + 1, cache)); pos++;
    }
  }

  /** System 1 — decisão sim/não: retorna P(sim) em [0,1], 1 forward, zero tokens. */
  async decide(state: string, question: string): Promise<number> {
    const sys = "You are a security decision function. Given a STATE and a yes/no QUESTION, answer with exactly one word: Yes or No.";
    const text = "<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\nSTATE:\n" + state +
      "\n\nQUESTION: " + question + "\nAnswer Yes or No.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const ids = this.ids(text);
    const { logits } = await this.forward(ids, 0, ids.length, this.emptyCache());
    const L = this.lastRow(logits);
    const yes = this.firstIds(["Yes", " Yes", "yes", " yes", "YES"]);
    const no = this.firstIds(["No", " No", "no", " no", "NO"]);
    const all = [...yes, ...no], mx = Math.max(...all.map((i) => L[i]));
    const Z = all.reduce((a, i) => a + Math.exp(L[i] - mx), 0);
    return yes.reduce((a, i) => a + Math.exp(L[i] - mx), 0) / Z;
  }
  private firstIds(words: string[]): number[] {
    const s = new Set<number>();
    for (const w of words) { const a = this.ids(w); if (a.length) s.add(a[0]); }
    return [...s];
  }

  /** Sessão de chat multi-turno: mantém o cache/posição entre mensagens (System 2). */
  newChat(opts: { system?: string } = {}): Chat { return new Chat(this, opts.system); }
  /** @internal */ _empty() { return this.emptyCache(); }
  /** @internal */ _fwd(ids: number[], pos0: number, total: number, cache: any) { return this.forward(ids, pos0, total, cache); }
  /** @internal */ _last(L: any) { return this.lastRow(L); }
  /** @internal */ _ids(t: string) { return this.ids(t); }
  /** @internal */ _dec(ids: number[]) { return this.tok.decode(ids, { skip_special_tokens: true }); }
  /** @internal */ _sample(l: Float32Array, gen: number[], rand: () => number, o: any) { return this.sample(l, gen, rand, o); }

  private sample(logits: Float32Array, gen: number[], rand: () => number, o: any): number {
    const { temperature: temp, topP, topK, repetitionPenalty: rep } = o;
    if (rep !== 1) for (const t of new Set(gen)) logits[t] = logits[t] > 0 ? logits[t] / rep : logits[t] * rep;
    if (temp <= 0) { let bi = 0, bv = -1e30; for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; bi = i; } return bi; }
    let idx = Array.from(logits.keys()); idx.sort((a, b) => logits[b] - logits[a]);
    if (topK > 0) idx = idx.slice(0, topK);
    const mx = logits[idx[0]]; let probs = idx.map((i) => Math.exp((logits[i] - mx) / temp));
    let s = probs.reduce((a, b) => a + b, 0); probs = probs.map((p) => p / s);
    if (topP < 1) { let c = 0, cut = idx.length; for (let i = 0; i < idx.length; i++) { c += probs[i]; if (c >= topP) { cut = i + 1; break; } } idx = idx.slice(0, cut); probs = probs.slice(0, cut); const z = probs.reduce((a, b) => a + b, 0); probs = probs.map((p) => p / z); }
    let r = rand(), acc = 0; for (let i = 0; i < idx.length; i++) { acc += probs[i]; if (r <= acc) return idx[i]; } return idx[idx.length - 1];
  }
}

/** Sessão de chat multi-turno: mantém cache/posição entre mensagens (System 2). */
export class Chat {
  private cache: Record<string, any>;
  private pos = 0;
  private started = false;
  constructor(private ct0: CT0, private system?: string) { this.cache = ct0._empty(); }
  get contextTokens() { return this.pos; }
  get maxContext() { return this.ct0.maxContext; }
  reset() { this.cache = this.ct0._empty(); this.pos = 0; this.started = false; }

  async *send(msg: string, opts: GenerateOptions = {}): AsyncGenerator<string> {
    const o = { maxNewTokens: 512, temperature: 0.7, topP: 0.9, topK: 40, repetitionPenalty: 1.1, seed: 0, think: true, ...opts };
    const rand = mulberry32(o.seed + this.pos);
    let turn = "";
    if (!this.started && (o.system ?? this.system)) turn += "<|im_start|>system\n" + (o.system ?? this.system) + "<|im_end|>\n";
    turn += "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n";
    turn += o.think ? "<think>\n" : "<think>\n\n</think>\n\n";
    this.started = true;
    const ids = this.ct0._ids(turn);
    let { logits, cache } = await this.ct0._fwd(ids, this.pos, this.pos + ids.length, this.cache);
    this.cache = cache; this.pos += ids.length;
    const gen: number[] = [];
    for (let s = 0; s < o.maxNewTokens; s++) {
      if (opts.signal?.aborted) break;
      const tid = this.ct0._sample(this.ct0._last(logits), gen, rand, o);
      if (EOS.has(tid)) break;
      gen.push(tid);
      yield this.ct0._dec([tid]);
      ({ logits, cache } = await this.ct0._fwd([tid], this.pos, this.pos + 1, this.cache));
      this.cache = cache; this.pos++;
    }
  }
}
