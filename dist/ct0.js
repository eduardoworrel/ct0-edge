// src/ct0.ts
import * as ort from "onnxruntime-web/webgpu";
import { AutoTokenizer, env } from "@huggingface/transformers";
var EOS = /* @__PURE__ */ new Set([248046, 248044]);
var F16 = globalThis.Float16Array;
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
var prod = (a) => a.reduce((x, y) => x * y, 1);
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
var CT0 = class _CT0 {
  /** Contexto nominal do modelo (recorrência linear O(1)). */
  maxContext = 262144;
  /** Teto SAUDÁVEL de tokens de entrada no navegador (ort-web/WebGPU). */
  maxSafeTokens = 2048;
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
      env.allowLocalModels = false;
      tok = await AutoTokenizer.from_pretrained(opts.repo, { local_files_only: false });
      base = opts.baseUrl ?? `https://huggingface.co/${opts.repo}/resolve/main/`;
    } else {
      base = opts.baseUrl ?? new URL(".", import.meta.url).href;
      env.allowRemoteModels = false;
      env.allowLocalModels = true;
      env.localModelPath = base;
      tok = await AutoTokenizer.from_pretrained("tok");
    }
    const file = opts.decodeModel ?? "onnx/model_q4f16_decode.onnx";
    const dataPath = opts.externalDataFile ?? file + ".data";
    status("carregando grafo decode-only (WebGPU)\u2026");
    const isLocal = /^(\.\/|\/)/.test(dataPath) || /^(\.\/|\/)/.test(file);
    const modelUrl = isLocal ? new URL(file, location.href).href : new URL(file, base).href;
    const dataUrl = isLocal ? new URL(dataPath, location.href).href : new URL(dataPath, base).href;
    const declared = (() => {
      try {
        const t = (S0) => S0;
        return dataPath.split("/").pop();
      } catch {
        return dataPath.split("/").pop();
      }
    })();
    const S = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["webgpu"],
      externalData: [{ path: declared, data: dataUrl }]
    });
    const ct0 = new _CT0(tok, S, buildSpecs());
    const outNames = S.outputNames;
    if (outNames) {
      const h = outNames.find((n) => n === "type_as_80" || /hidden/i.test(n));
      if (h) {
        ct0.hiddenOutput = h;
        status("hidden state disponivel: " + h);
      }
    }
    if (opts.maxSafeTokens) ct0.maxSafeTokens = opts.maxSafeTokens;
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
  /** Conta tokens de um texto. */
  countTokens(text) {
    return this.ids(text).length;
  }
  /** Trunca `text` para caber em `maxSafeTokens - reserve` (mantém início + fim,
   *  onde a informação costuma estar). Chama `onTrim` se cortar. */
  fitContext(text, opts = {}) {
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
  async scanLong(text, question, opts = {}) {
    const win = opts.windowTokens ?? this.maxSafeTokens - 400;
    const overlap = opts.overlapTokens ?? 128;
    const thr = opts.threshold ?? 0.5;
    const toks = this.ids(text);
    const hits = [];
    if (toks.length <= win) {
      const p = await this.decide(text, question);
      opts.onChunk?.(0, p, 1);
      hits.push({ i: 0, start: 0, p, preview: text.slice(0, 100) });
    } else {
      const total = Math.ceil((toks.length - overlap) / (win - overlap));
      let i = 0, idx = 0;
      while (i < toks.length) {
        const chunk = this.tok.decode(toks.slice(i, i + win), { skip_special_tokens: true });
        const p = await this.decide(chunk, question);
        hits.push({ i: idx, start: i, p, preview: chunk.slice(0, 100) });
        opts.onChunk?.(idx, p, total);
        idx++;
        i += win - overlap;
      }
    }
    return { chunks: hits.length, max: Math.max(...hits.map((h) => h.p)), hits, flagged: hits.filter((h) => h.p >= thr) };
  }
  /** Visão LIMPA e condensada de uma página (para agentes de segurança): forms,
   *  inputs, botões, links, comentários HTML, inline scripts e localStorage —
   *  onde vulnerabilidades/segredos costumam se esconder, usando pouco contexto. */
  static inspectPage(doc) {
    const q = (s, f) => [...doc.querySelectorAll(s)].map(f);
    const comments = [];
    const w = doc.createTreeWalker(
      doc.body || doc.documentElement,
      128
      /* SHOW_COMMENT */
    );
    while (w.nextNode()) comments.push((w.currentNode.nodeValue || "").trim().slice(0, 180));
    const store = {};
    try {
      const ls = (doc.defaultView || window).localStorage;
      for (let i = 0; i < ls.length; i++) {
        const k = ls.key(i);
        store[k] = (ls.getItem(k) || "").slice(0, 80);
      }
    } catch {
    }
    return {
      title: doc.title,
      url: (doc.defaultView || {}).location?.search || "/",
      forms: q("form", (f) => ({ action: f.getAttribute("action"), method: f.method, inputs: [...f.querySelectorAll("input")].map((i) => ({ name: i.name, type: i.type, value: i.type === "hidden" ? i.value : void 0, autocomplete: i.autocomplete })) })),
      inputs: q("input,textarea", (i) => ({ sel: i.id ? "#" + i.id : i.name ? `[name=${i.name}]` : i.tagName.toLowerCase(), type: i.type })),
      buttons: q("button", (b) => ({ sel: b.id ? "#" + b.id : null, text: b.textContent.trim() })),
      links: q("a[href]", (a) => a.getAttribute("href")),
      // texto visível notável (stack traces, banners de erro/debug) — onde credenciais costumam vazar
      texts: q("pre,code,[class*='debug'],[class*='error'],[class*='stack']", (e) => (e.textContent || "").trim().slice(0, 300)).filter(Boolean),
      comments,
      inlineScripts: q("script:not([src])", (s) => s.textContent.trim().slice(0, 220)).filter(Boolean),
      localStorage: store
    };
  }
  /** HTML cru do body (fallback quando a segurança exige ver o markup completo). */
  static pageSource(doc, max = 2500) {
    return (doc.body || doc.documentElement).outerHTML.slice(0, max);
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
  async step(tid, pos, st, hiddenOut) {
    const feed = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(tid)]), [1, 1]),
      cache_position: new ort.Tensor("int64", BigInt64Array.from([BigInt(pos)]), [1]),
      ...st
    };
    let r;
    if (hiddenOut && !this.allOutputs) {
      const names = ["logits", hiddenOut];
      for (let i = 0; i < this.NS; i++) names.push("so" + i);
      r = await this.S.run(feed, names);
    } else {
      r = await this.S.run(feed);
    }
    const ns = {};
    for (let i = 0; i < this.NS; i++) ns["si" + i] = r["so" + i];
    let hidden;
    if (hiddenOut && r[hiddenOut]) {
      const t = r[hiddenOut];
      this.__lastHiddenTensor = t;
      if (t.data instanceof Float32Array) hidden = t.data;
      else if (typeof globalThis.Float16Array !== "undefined" && t.data instanceof globalThis.Float16Array) {
        hidden = Float32Array.from(t.data);
      } else if (t.type === "float16" || t.data instanceof Uint16Array) {
        const u = t.data;
        hidden = new Float32Array(u.length);
        for (let i = 0; i < u.length; i++) {
          const h = u[i];
          const s = h & 32768 ? -1 : 1, e = (h & 31744) >> 10, f = h & 1023;
          hidden[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? f ? NaN : s * Infinity : s * Math.pow(2, e - 15) * (1 + f / 1024);
        }
      } else hidden = Float32Array.from(t.data);
    }
    return { logits: r.logits, st: ns, hidden };
  }
  lastRow(L) {
    const V = L.dims.at(-1), off = (L.dims.at(-2) - 1) * V, d = L.data, r = new Float32Array(V);
    const isF16Arr = typeof globalThis.Float16Array !== "undefined" && d instanceof globalThis.Float16Array;
    if (d instanceof Float32Array || isF16Arr) {
      for (let i = 0; i < V; i++) r[i] = Number(d[off + i]);
    } else if (L.type === "float16" || d instanceof Uint16Array) {
      for (let i = 0; i < V; i++) {
        const h = d[off + i];
        const s = h & 32768 ? -1 : 1, e = (h & 31744) >> 10, f = h & 1023;
        r[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? f ? NaN : s * Infinity : s * Math.pow(2, e - 15) * (1 + f / 1024);
      }
    } else {
      for (let i = 0; i < V; i++) r[i] = Number(d[off + i]);
    }
    return r;
  }
  turn(msg, system, think) {
    let t = "";
    if (system) t += "<|im_start|>system\n" + system + "<|im_end|>\n";
    t += "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n";
    return t + (think ? "<think>\n" : "<think>\n\n</think>\n\n");
  }
  /** Single-turn generation: wraps `prompt` in a chat turn (System 2). */
  async *generate(prompt, opts = {}) {
    yield* this.stream(this.turn(prompt, opts.system, opts.think ?? true), opts);
  }
  /**
   * Low-level streaming from a RAW prompt (no chat-turn wrapping). Use this to
   * drive multi-turn / agentic (ReAct) loops where you build the full prompt
   * yourself. `temperature <= 0` ⇒ greedy (deterministic tool calls).
   */
  async *stream(prompt, opts = {}) {
    const o = { maxNewTokens: 512, temperature: 0.7, topP: 0.9, topK: 40, repetitionPenalty: 1.1, seed: 0, ...opts };
    const rand = mulberry32(o.seed);
    let st = this.initState(), pos = 0, last = null;
    for (const t of this.ids(prompt)) {
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
  /**
   * Prefill completo devolvendo os hidden states que o HEAD bilinear precisa.
   *
   * Requer o grafo com a saida extra `type_as_80` (hidden pos-norm, [1,1,2560]).
   * Para gerar: `python3 add_hidden_output.py` (adiciona a output ao onnx existente,
   * logits ficam bit-identicos) e carregue com `decodeModel: "onnx/model_q4f16_decode_head.onnx"`.
   *
   * Posicao do `<decide>` e a ultima posicao seguem a especificacao medida (EVIDENCE §20c):
   *   q le no ultimo token de `<decide>`; k le na ultima posicao do prompt.
   */
  async prefillHiddens(prompt) {
    const hiddenName = this.hiddenOutput;
    if (!hiddenName) return null;
    const ids = this.ids(prompt);
    let st = this.initState(), pos = 0;
    let atDecide = null;
    let atLast = null;
    const atOpts = [];
    const mid = this.ids("<decide>");
    const tail2 = mid.slice(-2);
    const optEnd = this.ids("</opt>").slice(-2);
    for (let i = 0; i < ids.length; i++) {
      const r = await this.step(ids[i], pos, st, hiddenName);
      st = r.st;
      const h = r.hidden;
      if (h) {
        if (i >= 1 && ids[i] === tail2[1] && ids[i - 1] === tail2[0]) atDecide = h;
        if (i >= 1 && ids[i] === optEnd[1] && ids[i - 1] === optEnd[0]) atOpts.push(h);
        if (i === ids.length - 1) atLast = h;
      }
      pos++;
    }
    if (!atDecide || !atLast) return null;
    return { atDecide, atLast, atOpts };
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
};
export {
  CT0
};
