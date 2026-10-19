import * as ort from "onnxruntime-web/webgpu";
import { AutoTokenizer, env } from "@huggingface/transformers";
const EOS = /* @__PURE__ */ new Set([248046, 248044]);
const FILE = {
  fp32: "model.onnx",
  fp16: "model_fp16.onnx",
  int8: "model_int8.onnx",
  q4: "model_q4.onnx",
  q4f16: "model_q4f16.onnx"
};
const prod = (a) => a.reduce((x, y) => x * y, 1);
function mulberry32(seed) {
  let a = seed >>> 0 || 1;
  return () => {
    a |= 0;
    a = a + 1831565813 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
class CT0 {
  maxContext = 262144;
  vocab = 248320;
  tok;
  S;
  // ort.InferenceSession
  pastSpecs = [];
  // entradas past_*
  p2p = {};
  // present(output) -> past(input)
  outNames = [];
  constructor(tok, S) {
    this.tok = tok;
    this.S = S;
  }
  static async load(opts = {}) {
    const status = opts.onStatus ?? (() => {
    });
    const base = opts.baseUrl ?? new URL(".", import.meta.url).href;
    const dtype = opts.dtype ?? "q4";
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = base;
    status("carregando tokenizer\u2026");
    const tok = await AutoTokenizer.from_pretrained("tok");
    const file = "onnx/" + FILE[dtype];
    status(`carregando grafo ${dtype} (WebGPU)\u2026`);
    const S = await ort.InferenceSession.create(new URL(file, base).href, {
      executionProviders: ["webgpu"],
      externalData: [{ path: FILE[dtype] + "_data", data: new URL(file + "_data", base).href }]
    });
    const ct0 = new CT0(tok, S);
    ct0.outNames = S.outputNames.slice();
    for (const name of S.inputNames) {
      if (!name.startsWith("past_")) continue;
      const meta = S.inputMetadata?.[name];
      const { dims, f16, seqAxis } = ct0.specFor(name, meta);
      ct0.pastSpecs.push({ name, dims, f16, seqAxis });
    }
    for (const o of S.outputNames) {
      if (o === "logits") continue;
      const past = o.replace("present_conv", "past_conv").replace("present_recurrent", "past_recurrent").replace(/^present\./, "past_key_values.");
      ct0.p2p[o] = past;
    }
    status("pronto.");
    return ct0;
  }
  /** Shape/dtype de uma entrada past_*; usa metadata quando disponível, senão nomes do qwen3_5. */
  specFor(name, meta) {
    let dims;
    let f16 = false;
    let seqAxis = -1;
    if (name.startsWith("past_conv")) {
      dims = [1, 8192, 4];
    } else if (name.startsWith("past_recurrent")) {
      dims = [1, 32, 128, 128];
    } else {
      dims = [1, 4, 0, 256];
      seqAxis = 2;
    }
    if (meta) {
      const t = (meta.type || meta.dataType || "").toString();
      if (t.includes("16")) f16 = true;
    }
    return { dims, f16, seqAxis };
  }
  /** Cache vazio (seq=0 nas entradas de atenção full; conv/recurrent têm tamanho fixo). */
  emptyCache() {
    const c = {};
    for (const s of this.pastSpecs) {
      const dims = s.dims.map((d, i) => i === s.seqAxis ? 0 : typeof d === "number" ? d : 1);
      const n = prod(dims);
      c[s.name] = new ort.Tensor(
        s.f16 ? "float16" : "float32",
        s.f16 ? new Uint16Array(n) : new Float32Array(n),
        dims
      );
    }
    return c;
  }
  /** Um forward do grafo. ids: tokens deste passo; pos0: posição do 1º token; total: tokens vistos. */
  async forward(ids, pos0, total, cache) {
    const seq = ids.length;
    const feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, seq]),
      attention_mask: new ort.Tensor("int64", BigInt64Array.from(Array(total).fill(1n)), [1, total]),
      position_ids: new ort.Tensor("int64", BigInt64Array.from(Array.from({ length: seq }, (_, i) => BigInt(pos0 + i))), [1, seq]),
      ...cache
    };
    const out = await this.S.run(feeds);
    const next = {};
    for (const o of this.outNames) if (o !== "logits") next[this.p2p[o]] = out[o];
    return { logits: out.logits, cache: next };
  }
  lastRow(L) {
    const V = L.dims.at(-1), off = (L.dims.at(-2) - 1) * V, d = L.data, r = new Float32Array(V);
    for (let i = 0; i < V; i++) r[i] = Number(d[off + i]);
    return r;
  }
  ids(text) {
    return this.tok(text, { add_special_tokens: false }).input_ids.tolist()[0].map(Number);
  }
  turn(msg, system, think) {
    let t = "";
    if (system) t += "<|im_start|>system\n" + system + "<|im_end|>\n";
    t += "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n";
    t += think ? "<think>\n" : "<think>\n\n</think>\n\n";
    return t;
  }
  /** System 2 — geração autoregressiva (streaming de texto). Prefill batched + decode. */
  async *generate(prompt, opts = {}) {
    const o = { maxNewTokens: 256, temperature: 0.7, topP: 0.9, topK: 40, repetitionPenalty: 1.1, seed: 0, think: true, ...opts };
    const rand = mulberry32(o.seed);
    const ids = this.ids(this.turn(prompt, o.system, o.think));
    let cache = this.emptyCache();
    let { logits, cache: c } = await this.forward(ids, 0, ids.length, cache);
    cache = c;
    let pos = ids.length;
    const gen = [];
    for (let s = 0; s < o.maxNewTokens; s++) {
      if (opts.signal?.aborted) break;
      const tid = this.sample(this.lastRow(logits), gen, rand, o);
      if (EOS.has(tid)) break;
      gen.push(tid);
      yield this.tok.decode([tid], { skip_special_tokens: true });
      ({ logits, cache } = await this.forward([tid], pos, pos + 1, cache));
      pos++;
    }
  }
  /** System 1 — decisão sim/não: retorna P(sim) em [0,1], 1 forward, zero tokens. */
  async decide(state, question) {
    const sys = "You are a security decision function. Given a STATE and a yes/no QUESTION, answer with exactly one word: Yes or No.";
    const text = "<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\nSTATE:\n" + state + "\n\nQUESTION: " + question + "\nAnswer Yes or No.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const ids = this.ids(text);
    const { logits } = await this.forward(ids, 0, ids.length, this.emptyCache());
    const L = this.lastRow(logits);
    const yes = this.firstIds(["Yes", " Yes", "yes", " yes", "YES"]);
    const no = this.firstIds(["No", " No", "no", " no", "NO"]);
    const all = [...yes, ...no], mx = Math.max(...all.map((i) => L[i]));
    const Z = all.reduce((a, i) => a + Math.exp(L[i] - mx), 0);
    return yes.reduce((a, i) => a + Math.exp(L[i] - mx), 0) / Z;
  }
  firstIds(words) {
    const s = /* @__PURE__ */ new Set();
    for (const w of words) {
      const a = this.ids(w);
      if (a.length) s.add(a[0]);
    }
    return [...s];
  }
  /** Sessão de chat multi-turno: mantém o cache/posição entre mensagens (System 2). */
  newChat(opts = {}) {
    return new Chat(this, opts.system);
  }
  /** @internal */
  _empty() {
    return this.emptyCache();
  }
  /** @internal */
  _fwd(ids, pos0, total, cache) {
    return this.forward(ids, pos0, total, cache);
  }
  /** @internal */
  _last(L) {
    return this.lastRow(L);
  }
  /** @internal */
  _ids(t) {
    return this.ids(t);
  }
  /** @internal */
  _dec(ids) {
    return this.tok.decode(ids, { skip_special_tokens: true });
  }
  /** @internal */
  _sample(l, gen, rand, o) {
    return this.sample(l, gen, rand, o);
  }
  sample(logits, gen, rand, o) {
    const { temperature: temp, topP, topK, repetitionPenalty: rep } = o;
    if (rep !== 1) for (const t of new Set(gen)) logits[t] = logits[t] > 0 ? logits[t] / rep : logits[t] * rep;
    if (temp <= 0) {
      let bi = 0, bv = -1e30;
      for (let i = 0; i < logits.length; i++) if (logits[i] > bv) {
        bv = logits[i];
        bi = i;
      }
      return bi;
    }
    let idx = Array.from(logits.keys());
    idx.sort((a, b) => logits[b] - logits[a]);
    if (topK > 0) idx = idx.slice(0, topK);
    const mx = logits[idx[0]];
    let probs = idx.map((i) => Math.exp((logits[i] - mx) / temp));
    let s = probs.reduce((a, b) => a + b, 0);
    probs = probs.map((p) => p / s);
    if (topP < 1) {
      let c = 0, cut = idx.length;
      for (let i = 0; i < idx.length; i++) {
        c += probs[i];
        if (c >= topP) {
          cut = i + 1;
          break;
        }
      }
      idx = idx.slice(0, cut);
      probs = probs.slice(0, cut);
      const z = probs.reduce((a, b) => a + b, 0);
      probs = probs.map((p) => p / z);
    }
    let r = rand(), acc = 0;
    for (let i = 0; i < idx.length; i++) {
      acc += probs[i];
      if (r <= acc) return idx[i];
    }
    return idx[idx.length - 1];
  }
}
class Chat {
  constructor(ct0, system) {
    this.ct0 = ct0;
    this.system = system;
    this.cache = ct0._empty();
  }
  ct0;
  system;
  cache;
  pos = 0;
  started = false;
  get contextTokens() {
    return this.pos;
  }
  get maxContext() {
    return this.ct0.maxContext;
  }
  reset() {
    this.cache = this.ct0._empty();
    this.pos = 0;
    this.started = false;
  }
  async *send(msg, opts = {}) {
    const o = { maxNewTokens: 512, temperature: 0.7, topP: 0.9, topK: 40, repetitionPenalty: 1.1, seed: 0, think: true, ...opts };
    const rand = mulberry32(o.seed + this.pos);
    let turn = "";
    if (!this.started && (o.system ?? this.system)) turn += "<|im_start|>system\n" + (o.system ?? this.system) + "<|im_end|>\n";
    turn += "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n";
    turn += o.think ? "<think>\n" : "<think>\n\n</think>\n\n";
    this.started = true;
    const ids = this.ct0._ids(turn);
    let { logits, cache } = await this.ct0._fwd(ids, this.pos, this.pos + ids.length, this.cache);
    this.cache = cache;
    this.pos += ids.length;
    const gen = [];
    for (let s = 0; s < o.maxNewTokens; s++) {
      if (opts.signal?.aborted) break;
      const tid = this.ct0._sample(this.ct0._last(logits), gen, rand, o);
      if (EOS.has(tid)) break;
      gen.push(tid);
      yield this.ct0._dec([tid]);
      ({ logits, cache } = await this.ct0._fwd([tid], this.pos, this.pos + 1, this.cache));
      this.cache = cache;
      this.pos++;
    }
  }
}
export {
  CT0,
  Chat
};
