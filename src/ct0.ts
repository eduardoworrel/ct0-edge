/**
 * ct0.ts — SDK de browser/edge do CT0 (alias3-micro-1026), dirigindo o grafo **decode-only int4**
 * via onnxruntime-web (WebGPU). É o caminho que roda de verdade no navegador: recorrência O(1)
 * por token (geração longa fluida), cabe no teto de 4 GB do WASM (~3.3 GB), sem depender de PR.
 *
 *  - System 1 (decide):   decisão sim/não lendo os logits após o prefill (zero tokens gerados).
 *  - System 2 (generate): geração autoregressiva (streaming), token a token.
 *
 * O grafo scan/standard (generate() do transformers.js, prefill ilimitado) é ótimo pra Node/servidor,
 * mas no browser o solve-por-chunk o torna lento demais — por isso o browser usa o decode-only.
 *
 * Uso:
 *   const ct0 = await CT0.load({ repo: "eduardoworrel/ct0-alias3-micro-onnx" });
 *   for await (const piece of ct0.generate("Olá")) process(piece);
 *   const p = await ct0.decide("STATE...", "Is this an IOC?");
 */
import * as ort from "onnxruntime-web/webgpu";
import { AutoTokenizer, env } from "@huggingface/transformers";

export interface LoadOptions {
  /** HF repo id (tokenizer + grafo decode-only). */
  repo?: string;
  /** URL base alternativa (modelo local); tokenizer ainda vem do repo se dado. */
  baseUrl?: string;
  /** Nome do arquivo do grafo decode-only (default onnx/model_q4f16_decode.onnx). */
  decodeModel?: string;
  /** Teto saudável de tokens de entrada no navegador (default 2048; ort-web/WebGPU
   *  estoura ~4000 tok com "unaligned accesses"). Medido: sem degradação até ~2k. */
  maxSafeTokens?: number;
  onStatus?: (msg: string) => void;
}
export interface GenerateOptions {
  maxNewTokens?: number; temperature?: number; topP?: number; topK?: number;
  repetitionPenalty?: number; seed?: number; system?: string; think?: boolean; signal?: AbortSignal;
}

const EOS = new Set<number>([248046, 248044]);
const F16: any = (globalThis as any).Float16Array;
interface Spec { d: number[]; f16: boolean; seq: number; }
/** 64 estados: 8 blocos × [conv,rec,conv,rec,conv,rec,kv,kv]. */
function buildSpecs(): Spec[] {
  const out: Spec[] = [];
  const conv: Spec = { d: [1, 8192, 4], f16: true, seq: -1 };
  const rec: Spec = { d: [1, 32, 128, 128], f16: false, seq: -1 };
  const kv: Spec = { d: [1, 4, 0, 256], f16: true, seq: 2 };
  for (let b = 0; b < 8; b++) { for (let i = 0; i < 3; i++) out.push(conv, rec); out.push(kv, kv); }
  return out;
}
const prod = (a: number[]) => a.reduce((x, y) => x * y, 1);
function mulberry32(seed: number) {
  let a = seed >>> 0 || 1;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

export class CT0 {
  /** Contexto nominal do modelo (recorrência linear O(1)). */
  readonly maxContext = 262144;
  /** Teto SAUDÁVEL de tokens de entrada no navegador (ort-web/WebGPU). */
  maxSafeTokens = 2048;
  private tok: any; private S: any; private NS: number; private SP: Spec[];
  private yesIds: number[] = []; private noIds: number[] = [];
  private constructor(tok: any, S: any, SP: Spec[]) { this.tok = tok; this.S = S; this.SP = SP; this.NS = SP.length; }

  static async load(opts: LoadOptions = {}): Promise<CT0> {
    const status = opts.onStatus ?? (() => {});
    status("carregando tokenizer…");
    let tok: any, base: string;
    if (opts.repo) {
      env.allowRemoteModels = true; env.allowLocalModels = true;
      tok = await AutoTokenizer.from_pretrained(opts.repo);
      base = opts.baseUrl ?? `https://huggingface.co/${opts.repo}/resolve/main/`;
    } else {
      base = opts.baseUrl ?? new URL(".", import.meta.url).href;
      env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = base;
      tok = await AutoTokenizer.from_pretrained("tok");
    }
    const file = (opts.decodeModel ?? "onnx/model_q4f16_decode.onnx");
    const dataPath = file + ".data"; // model_q4f16_decode.onnx.data (ao lado)
    status("carregando grafo decode-only (WebGPU)…");
    const S = await ort.InferenceSession.create(new URL(file, base).href, {
      executionProviders: ["webgpu"],
      externalData: [{ path: dataPath.split("/").pop()!, data: new URL(dataPath, base).href }],
    });
    const ct0 = new CT0(tok, S, buildSpecs());
    if (opts.maxSafeTokens) ct0.maxSafeTokens = opts.maxSafeTokens;
    const firstId = (s: string) => { const a = ct0.ids(s); return a.length ? a[0] : null; };
    ct0.yesIds = [...new Set(["Yes", " Yes", "yes", " yes", "YES"].map(firstId).filter((x): x is number => x != null))];
    ct0.noIds = [...new Set(["No", " No", "no", " no", "NO"].map(firstId).filter((x): x is number => x != null))];
    status("pronto.");
    return ct0;
  }

  private ids(text: string): number[] { return this.tok(text, { add_special_tokens: false }).input_ids.tolist()[0].map(Number); }

  /** Conta tokens de um texto. */
  countTokens(text: string): number { return this.ids(text).length; }

  /** Trunca `text` para caber em `maxSafeTokens - reserve` (mantém início + fim,
   *  onde a informação costuma estar). Chama `onTrim` se cortar. */
  fitContext(text: string, opts: { reserve?: number; onTrim?: (i: { original: number; kept: number }) => void } = {}): string {
    const reserve = opts.reserve ?? 300;
    const budget = Math.max(64, this.maxSafeTokens - reserve);
    const toks = this.ids(text);
    if (toks.length <= budget) return text;
    const head = Math.floor(budget * 0.6), tail = budget - head;
    const kept = [...toks.slice(0, head), ...toks.slice(toks.length - tail)];
    opts.onTrim?.({ original: toks.length, kept: budget });
    return this.tok.decode(kept, { skip_special_tokens: true });
  }

  /**
   * chunk-watch: varre um texto GRANDE (que estoura maxSafeTokens — ex. HTML gigante)
   * em janelas deslizantes e roda System 1 (`decide`) por chunk, acumulando as regiões
   * suspeitas. Cobre documentos que não cabem no contexto, sem truncar/perder sinal.
   */
  async scanLong(text: string, question: string,
    opts: { windowTokens?: number; overlapTokens?: number; threshold?: number; onChunk?: (i: number, p: number, total: number) => void } = {}
  ): Promise<{ chunks: number; max: number; hits: { i: number; start: number; p: number; preview: string }[]; flagged: { i: number; start: number; p: number; preview: string }[] }> {
    const win = opts.windowTokens ?? (this.maxSafeTokens - 400);
    const overlap = opts.overlapTokens ?? 128;
    const thr = opts.threshold ?? 0.5;
    const toks = this.ids(text);
    const hits: { i: number; start: number; p: number; preview: string }[] = [];
    if (toks.length <= win) { const p = await this.decide(text, question); opts.onChunk?.(0, p, 1); hits.push({ i: 0, start: 0, p, preview: text.slice(0, 100) }); }
    else {
      const total = Math.ceil((toks.length - overlap) / (win - overlap));
      let i = 0, idx = 0;
      while (i < toks.length) {
        const chunk = this.tok.decode(toks.slice(i, i + win), { skip_special_tokens: true });
        const p = await this.decide(chunk, question);
        hits.push({ i: idx, start: i, p, preview: chunk.slice(0, 100) });
        opts.onChunk?.(idx, p, total); idx++; i += (win - overlap);
      }
    }
    return { chunks: hits.length, max: Math.max(...hits.map(h => h.p)), hits, flagged: hits.filter(h => h.p >= thr) };
  }

  /** Visão LIMPA e condensada de uma página (para agentes de segurança): forms,
   *  inputs, botões, links, comentários HTML, inline scripts e localStorage —
   *  onde vulnerabilidades/segredos costumam se esconder, usando pouco contexto. */
  static inspectPage(doc: any): any {
    const q = (s: string, f: (e: any) => any) => [...doc.querySelectorAll(s)].map(f);
    const comments: string[] = []; const w = doc.createTreeWalker(doc.body || doc.documentElement, 128 /* SHOW_COMMENT */);
    while (w.nextNode()) comments.push((w.currentNode.nodeValue || "").trim().slice(0, 180));
    const store: Record<string, string> = {};
    try { const ls = (doc.defaultView || window).localStorage; for (let i = 0; i < ls.length; i++) { const k = ls.key(i)!; store[k] = (ls.getItem(k) || "").slice(0, 80); } } catch {}
    return {
      title: doc.title, url: (doc.defaultView || {}).location?.search || "/",
      forms: q("form", (f: any) => ({ action: f.getAttribute("action"), method: f.method, inputs: [...f.querySelectorAll("input")].map((i: any) => ({ name: i.name, type: i.type, value: i.type === "hidden" ? i.value : undefined, autocomplete: i.autocomplete })) })),
      inputs: q("input,textarea", (i: any) => ({ sel: i.id ? "#" + i.id : i.name ? `[name=${i.name}]` : i.tagName.toLowerCase(), type: i.type })),
      buttons: q("button", (b: any) => ({ sel: b.id ? "#" + b.id : null, text: b.textContent.trim() })),
      links: q("a[href]", (a: any) => a.getAttribute("href")),
      // texto visível notável (stack traces, banners de erro/debug) — onde credenciais costumam vazar
      texts: q("pre,code,[class*='debug'],[class*='error'],[class*='stack']", (e: any) => (e.textContent || "").trim().slice(0, 300)).filter(Boolean),
      comments, inlineScripts: q("script:not([src])", (s: any) => s.textContent.trim().slice(0, 220)).filter(Boolean), localStorage: store,
    };
  }

  /** HTML cru do body (fallback quando a segurança exige ver o markup completo). */
  static pageSource(doc: any, max = 2500): string { return (doc.body || doc.documentElement).outerHTML.slice(0, max); }
  private initState(): Record<string, any> {
    const st: Record<string, any> = {};
    for (let i = 0; i < this.NS; i++) {
      const sp = this.SP[i], d = sp.d.slice(); if (sp.seq >= 0) d[sp.seq] = 0;
      const n = prod(d);
      st["si" + i] = new ort.Tensor(sp.f16 ? "float16" : "float32",
        sp.f16 ? (F16 ? new F16(n) : new Uint16Array(n)) : new Float32Array(n), d);
    }
    return st;
  }
  private async step(tid: number, pos: number, st: Record<string, any>) {
    const feed: Record<string, any> = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(tid)]), [1, 1]),
      cache_position: new ort.Tensor("int64", BigInt64Array.from([BigInt(pos)]), [1]), ...st,
    };
    const r = await this.S.run(feed);
    const ns: Record<string, any> = {}; for (let i = 0; i < this.NS; i++) ns["si" + i] = r["so" + i];
    return { logits: r.logits, st: ns };
  }
  private lastRow(L: any): Float32Array {
    const V = L.dims.at(-1), off = (L.dims.at(-2) - 1) * V, d = L.data, r = new Float32Array(V);
    for (let i = 0; i < V; i++) r[i] = Number(d[off + i]); return r;
  }
  private turn(msg: string, system: string | undefined, think: boolean) {
    let t = ""; if (system) t += "<|im_start|>system\n" + system + "<|im_end|>\n";
    t += "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n";
    return t + (think ? "<think>\n" : "<think>\n\n</think>\n\n");
  }

  /** Single-turn generation: wraps `prompt` in a chat turn (System 2). */
  async *generate(prompt: string, opts: GenerateOptions = {}): AsyncGenerator<string> {
    yield* this.stream(this.turn(prompt, opts.system, opts.think ?? true), opts);
  }

  /**
   * Low-level streaming from a RAW prompt (no chat-turn wrapping). Use this to
   * drive multi-turn / agentic (ReAct) loops where you build the full prompt
   * yourself. `temperature <= 0` ⇒ greedy (deterministic tool calls).
   */
  async *stream(prompt: string, opts: GenerateOptions = {}): AsyncGenerator<string> {
    const o = { maxNewTokens: 512, temperature: 0.7, topP: 0.9, topK: 40, repetitionPenalty: 1.1, seed: 0, ...opts };
    const rand = mulberry32(o.seed);
    let st = this.initState(), pos = 0, last: any = null;
    for (const t of this.ids(prompt)) { ({ logits: last, st } = await this.step(t, pos, st)); pos++; }
    const gen: number[] = [];
    for (let s = 0; s < o.maxNewTokens; s++) {
      if (opts.signal?.aborted) break;
      const tid = this.sample(this.lastRow(last), gen, rand, o);
      if (EOS.has(tid)) break;
      gen.push(tid); yield this.tok.decode([tid], { skip_special_tokens: true });
      ({ logits: last, st } = await this.step(tid, pos, st)); pos++;
    }
  }

  async decide(state: string, question: string): Promise<number> {
    const sys = "You are a security decision function. Given a STATE and a yes/no QUESTION, answer with exactly one word: Yes or No.";
    const text = "<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\nSTATE:\n" + state +
      "\n\nQUESTION: " + question + "\nAnswer Yes or No.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let st = this.initState(), pos = 0, last: any = null;
    for (const t of this.ids(text)) { ({ logits: last, st } = await this.step(t, pos, st)); pos++; }
    const L = this.lastRow(last), all = [...this.yesIds, ...this.noIds];
    if (!all.length) return NaN;
    const mx = Math.max(...all.map((i) => L[i]));
    const Z = all.reduce((a, i) => a + Math.exp(L[i] - mx), 0);
    return this.yesIds.reduce((a, i) => a + Math.exp(L[i] - mx), 0) / Z;
  }

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
