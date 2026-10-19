import * as ort from "onnxruntime-web/webgpu";
import { AutoTokenizer, env } from "@huggingface/transformers";
const EOS = /* @__PURE__ */ new Set([248046, 248044]);
const F16 = globalThis.Float16Array;
function buildSpecs() {
  const out = [];
  const conv = { d: [1, 8192, 4], f16: true, seq: -1 };
  const rec = { d: [1, 32, 128, 128], f16: false, seq: -1 };
  const kv = { d: [1, 4, 0, 256], f16: true, seq: 2 };
  for (let b = 0; b < 8; b++) {
    for (let i = 0; i < 3; i++) out.push(conv, rec);
    out.push(kv, kv);
  }
  return out;
}
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
  tok;
  S;
  NS;
  SP;
  yesIds = [];
  noIds = [];
  constructor(tok, S, SP) {
    this.tok = tok;
    this.S = S;
    this.SP = SP;
    this.NS = SP.length;
  }
  static async load(opts = {}) {
    const status = opts.onStatus ?? (() => {
    });
    status("carregando tokenizer\u2026");
    let tok, base;
    if (opts.repo) {
      env.allowRemoteModels = true;
      env.allowLocalModels = true;
      tok = await AutoTokenizer.from_pretrained(opts.repo);
      base = opts.baseUrl ?? `https://huggingface.co/${opts.repo}/resolve/main/`;
    } else {
      base = opts.baseUrl ?? new URL(".", import.meta.url).href;
      env.allowRemoteModels = false;
      env.allowLocalModels = true;
      env.localModelPath = base;
      tok = await AutoTokenizer.from_pretrained("tok");
    }
    const file = opts.decodeModel ?? "onnx/real_decode_gbq_int4.onnx";
    const dataName = file.replace(/[^/]*$/, (m) => m) + ".data";
    const dataPath = file + ".data";
    status("carregando grafo decode-only (WebGPU)\u2026");
    const S = await ort.InferenceSession.create(new URL(file, base).href, {
      executionProviders: ["webgpu"],
      externalData: [{ path: dataPath.split("/").pop(), data: new URL(dataPath, base).href }]
    });
    const ct0 = new CT0(tok, S, buildSpecs());
    const firstId = (s) => {
      const a = ct0.ids(s);
      return a.length ? a[0] : null;
    };
    ct0.yesIds = [...new Set(["Yes", " Yes", "yes", " yes", "YES"].map(firstId).filter((x) => x != null))];
    ct0.noIds = [...new Set(["No", " No", "no", " no", "NO"].map(firstId).filter((x) => x != null))];
    status("pronto.");
    return ct0;
  }
  ids(text) {
    return this.tok(text, { add_special_tokens: false }).input_ids.tolist()[0].map(Number);
  }
  initState() {
    const st = {};
    for (let i = 0; i < this.NS; i++) {
      const sp = this.SP[i], d = sp.d.slice();
      if (sp.seq >= 0) d[sp.seq] = 0;
      const n = prod(d);
      st["si" + i] = new ort.Tensor(
        sp.f16 ? "float16" : "float32",
        sp.f16 ? F16 ? new F16(n) : new Uint16Array(n) : new Float32Array(n),
        d
      );
    }
    return st;
  }
  async step(tid, pos, st) {
    const feed = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(tid)]), [1, 1]),
      cache_position: new ort.Tensor("int64", BigInt64Array.from([BigInt(pos)]), [1]),
      ...st
    };
    const r = await this.S.run(feed);
    const ns = {};
    for (let i = 0; i < this.NS; i++) ns["si" + i] = r["so" + i];
    return { logits: r.logits, st: ns };
  }
  lastRow(L) {
    const V = L.dims.at(-1), off = (L.dims.at(-2) - 1) * V, d = L.data, r = new Float32Array(V);
    for (let i = 0; i < V; i++) r[i] = Number(d[off + i]);
    return r;
  }
  turn(msg, system, think) {
    let t = "";
    if (system) t += "<|im_start|>system\n" + system + "<|im_end|>\n";
    t += "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n";
    return t + (think ? "<think>\n" : "<think>\n\n</think>\n\n");
  }
  async *generate(prompt, opts = {}) {
    const o = { maxNewTokens: 512, temperature: 0.7, topP: 0.9, topK: 40, repetitionPenalty: 1.1, seed: 0, think: true, ...opts };
    const rand = mulberry32(o.seed);
    let st = this.initState(), pos = 0, last = null;
    for (const t of this.ids(this.turn(prompt, o.system, o.think))) {
      ({ logits: last, st } = await this.step(t, pos, st));
      pos++;
    }
    const gen = [];
    for (let s = 0; s < o.maxNewTokens; s++) {
      if (opts.signal?.aborted) break;
      const tid = this.sample(this.lastRow(last), gen, rand, o);
      if (EOS.has(tid)) break;
      gen.push(tid);
      yield this.tok.decode([tid], { skip_special_tokens: true });
      ({ logits: last, st } = await this.step(tid, pos, st));
      pos++;
    }
  }
  async decide(state, question) {
    const sys = "You are a security decision function. Given a STATE and a yes/no QUESTION, answer with exactly one word: Yes or No.";
    const text = "<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\nSTATE:\n" + state + "\n\nQUESTION: " + question + "\nAnswer Yes or No.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let st = this.initState(), pos = 0, last = null;
    for (const t of this.ids(text)) {
      ({ logits: last, st } = await this.step(t, pos, st));
      pos++;
    }
    const L = this.lastRow(last), all = [...this.yesIds, ...this.noIds];
    if (!all.length) return NaN;
    const mx = Math.max(...all.map((i) => L[i]));
    const Z = all.reduce((a, i) => a + Math.exp(L[i] - mx), 0);
    return this.yesIds.reduce((a, i) => a + Math.exp(L[i] - mx), 0) / Z;
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
export {
  CT0
};
