// src/backend.ts
var GENERATE_DEFAULTS = { maxNewTokens: 512, temperature: 0.7, topP: 0.9, topK: 40, repetitionPenalty: 1.1, seed: 0 };
function resolveGenerateOptions(o = {}) {
  const num = (v, d) => typeof v === "number" && Number.isFinite(v) ? v : d;
  const D = GENERATE_DEFAULTS;
  const temperature = num(o.temperature, D.temperature);
  return {
    maxNewTokens: Math.max(0, Math.floor(num(o.maxNewTokens, D.maxNewTokens))),
    temperature,
    topP: num(o.topP, D.topP),
    topK: num(o.topK, D.topK),
    repetitionPenalty: num(o.repetitionPenalty, temperature <= 0 ? 1 : D.repetitionPenalty),
    seed: num(o.seed, D.seed),
    signal: o.signal
  };
}
function createGate(name) {
  let tail = Promise.resolve();
  let disposed = false;
  let disposing = null;
  const err = () => new Error(`${name}: backend already disposed`);
  const lock = async () => {
    const prev = tail;
    let rel;
    tail = new Promise((r) => {
      rel = r;
    });
    await prev;
    return rel;
  };
  return {
    get disposed() {
      return disposed;
    },
    assertLive() {
      if (disposed) throw err();
    },
    async acquire() {
      if (disposed) throw err();
      return lock();
    },
    dispose(release) {
      if (!disposing) {
        disposed = true;
        disposing = (async () => {
          const rel = await lock();
          try {
            await release();
          } finally {
            rel();
          }
        })();
      }
      return disposing;
    }
  };
}
var gates = /* @__PURE__ */ new WeakMap();
function gateFor(shared, name) {
  let g = gates.get(shared);
  if (!g) {
    g = createGate(name);
    gates.set(shared, g);
  }
  return g;
}

// src/backends/decodeOnly.ts
var EOS = /* @__PURE__ */ new Set([248046, 248044]);
var F16 = globalThis.Float16Array;
function halfToNumber(h) {
  const s = h & 32768 ? -1 : 1, e = (h & 31744) >> 10, f = h & 1023;
  return e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? f ? NaN : s * Infinity : s * Math.pow(2, e - 15) * (1 + f / 1024);
}

// src/ask.ts
var YES_ID = 9175;
var NO_ID = 2665;

// src/webllm.ts
function logitCapture() {
  let armed = false, cap = null, sampled = [];
  return {
    processor: {
      processLogits(l) {
        if (armed) {
          cap = new Float32Array(l);
          armed = false;
        }
        return l;
      },
      processSampledToken(t) {
        sampled.push(t);
      },
      resetState() {
      }
    },
    arm() {
      armed = true;
      cap = null;
    },
    take() {
      const c = cap;
      cap = null;
      armed = false;
      return c;
    },
    drain() {
      const s = sampled;
      sampled = [];
      return s;
    },
    reset() {
      armed = false;
      cap = null;
      sampled = [];
    }
  };
}
function normalizeRecovery(r) {
  const N = r.rows.length;
  if (!N || r.P.length % N) throw new Error(`webllm: P (${r.P.length}) is not a multiple of rows (${N})`);
  let P;
  if (r.P instanceof Uint16Array) {
    P = new Float32Array(r.P.length);
    for (let i = 0; i < P.length; i++) P[i] = halfToNumber(r.P[i]);
  } else P = r.P;
  return { rows: r.rows, P, D: P.length / N };
}
function recoverHidden(rec, logits) {
  const r = "D" in rec ? rec : normalizeRecovery(rec);
  const N = r.rows.length, sub = new Float32Array(N), h = new Float32Array(r.D);
  for (let n = 0; n < N; n++) sub[n] = logits[r.rows[n]];
  for (let d = 0; d < r.D; d++) {
    let acc = 0;
    const off = d * N;
    for (let n = 0; n < N; n++) acc += r.P[off + n] * sub[n];
    h[d] = acc;
  }
  return h;
}
async function loadRecovery(rowsUrl, pUrl, format = pUrl.endsWith(".f16") ? "f16" : "f32", fetchFn = fetch) {
  const [rb, pb] = await Promise.all([
    fetchFn(rowsUrl).then((r) => {
      if (!r.ok) throw new Error("webllm: " + rowsUrl + " " + r.status);
      return r.arrayBuffer();
    }),
    fetchFn(pUrl).then((r) => {
      if (!r.ok) throw new Error("webllm: " + pUrl + " " + r.status);
      return r.arrayBuffer();
    })
  ]);
  return { rows: new Int32Array(rb), P: format === "f16" ? new Uint16Array(pb) : new Float32Array(pb) };
}
async function tryLoadRecovery(base, files, fetchFn = fetch) {
  try {
    const rec = await loadRecovery(new URL(files.rows, base).href, new URL(files.P, base).href, files.P.endsWith(".f16") ? "f16" : "f32", fetchFn);
    const N = rec.rows.length;
    if (!N || rec.P.length % N) throw new Error(`P (${rec.P.length}) is not a multiple of rows (${N})`);
    return { recovery: rec };
  } catch (e) {
    return { reason: e?.message ?? String(e) };
  }
}
function webllmBackend(engine, capture, opts) {
  if (!opts.tok?.decode) throw new Error("webllm: opts.tok (tokenizer with decode) is required; WebLLM only accepts text");
  const eos = opts.eos ?? EOS;
  let rec = opts.recovery ? normalizeRecovery(opts.recovery) : null;
  const gate = gateFor(engine, "webllm");
  const decode = (ids) => opts.tok.decode(ids, { skip_special_tokens: false, clean_up_tokenization_spaces: false });
  async function logitsAt(ids, k) {
    await engine.resetChat();
    capture.reset();
    capture.arm();
    const r = await engine.completions.create({ prompt: decode(ids.slice(0, k + 1)), max_tokens: 1, temperature: 0 });
    const L = capture.take();
    if (!L) throw new Error("webllm: the LogitProcessor captured no logits (engine created without this capture's logitProcessorRegistry, or a Web Worker?)");
    const n = r?.usage?.prompt_tokens;
    if (n !== void 0 && n !== k + 1) throw new Error(`webllm: the prompt truncated at position ${k} re-tokenized to ${n} tokens (expected ${k + 1}); the position read would be wrong`);
    return L;
  }
  return {
    kind: "webllm",
    hiddenName: rec ? "recovered-from-logits" : null,
    dispose() {
      return gate.dispose(async () => {
        rec = null;
        await engine.unload?.();
      });
    },
    async prefill(ids, want) {
      gate.assertLive();
      if (!ids.length) throw new Error("prefill: empty ids");
      for (const p of want.positions) if (!Number.isInteger(p) || p < 0 || p >= ids.length) throw new Error(`webllm: position ${p} out of [0,${ids.length})`);
      if (want.positions.length && !rec) throw new Error("webllm: without opts.recovery the backend cannot recover hidden states; use method 'logits'");
      const rel = await gate.acquire();
      try {
        const last = ids.length - 1;
        const lastLogits = await logitsAt(ids, last);
        const at = /* @__PURE__ */ new Map();
        for (const p of new Set(want.positions)) at.set(p, recoverHidden(rec, p === last ? lastLogits : await logitsAt(ids, p)));
        return { hiddens: want.positions.map((p) => at.get(p).slice()), lastLogits };
      } finally {
        rel();
      }
    },
    /** Ids come from `processSampledToken`; EOS is not emitted. Uses the engine's sampler (temperature/top_p/seed/repetition_penalty; `topK` is IGNORED). */
    async *generate(ids, o) {
      gate.assertLive();
      const g = resolveGenerateOptions(o);
      if (!ids.length) throw new Error("generate: empty ids");
      if (g.maxNewTokens === 0) return;
      const rel = await gate.acquire();
      let iter = null, finished = false;
      try {
        if (o.signal?.aborted) return;
        await engine.resetChat();
        capture.reset();
        const stream = await engine.completions.create({
          prompt: decode(ids),
          stream: true,
          max_tokens: g.maxNewTokens,
          temperature: g.temperature,
          top_p: g.topP,
          seed: g.seed,
          repetition_penalty: g.repetitionPenalty
        });
        iter = stream[Symbol.asyncIterator]();
        let n = 0, stop = false;
        while (!stop) {
          if (o.signal?.aborted) break;
          const nx = await iter.next();
          const toks = capture.drain();
          if (nx.done) finished = true;
          for (const t of toks) {
            if (eos.has(t)) {
              stop = true;
              break;
            }
            if (n >= g.maxNewTokens) {
              stop = true;
              break;
            }
            n++;
            yield t;
            if (o.signal?.aborted) {
              stop = true;
              break;
            }
          }
          if (nx.done) break;
        }
      } finally {
        if (iter) {
          try {
            if (!finished) {
              await engine.interruptGenerate();
              for (let i = 0; i < 1e5; i++) {
                const nx = await iter.next();
                if (nx.done) break;
              }
            }
          } catch {
          }
          try {
            await iter.return?.();
          } catch {
          }
        }
        capture.reset();
        rel();
      }
    }
  };
}
async function createWebllm(webllm, opts) {
  const capture = logitCapture();
  const appConfig = opts.appConfig ?? { model_list: [{ model: opts.modelUrl, model_id: opts.modelId, model_lib: opts.libUrl }] };
  const recovery = opts.recovery ?? (opts.recoveryUrls ? await loadRecovery(opts.recoveryUrls.rows, opts.recoveryUrls.P, opts.recoveryUrls.format) : void 0);
  const engine = await webllm.CreateMLCEngine(opts.modelId, {
    appConfig,
    logitProcessorRegistry: /* @__PURE__ */ new Map([[opts.modelId, capture.processor]]),
    initProgressCallback: opts.initProgressCallback
  });
  return Object.assign(webllmBackend(engine, capture, { tok: opts.tok, recovery, eos: opts.eos }), { engine, capture });
}
var sigmoid = (z) => 1 / (1 + Math.exp(-z));
var clamp01 = (p) => Math.min(1 - 1e-6, Math.max(1e-6, p));
function webllmModel(engine, opts = {}) {
  const stop = opts.stop ?? ["<|im_end|>"];
  const gate = gateFor(engine, "webllm");
  const thinkTail = opts.noThink ? "/no_think\n" : "";
  return {
    countTokens: (t) => t.length / 4 | 0,
    fitContext: (p) => p,
    maxSafeTokens: opts.maxSafeTokens ?? 2048,
    async *stream(prompt, o = {}) {
      const rel = await gate.acquire();
      try {
        const it = await engine.completions.create({ prompt, temperature: o.temperature ?? 0, max_tokens: o.maxNewTokens ?? 300, stream: true, stop });
        for await (const ch of it) {
          const d = ch.choices?.[0]?.text || "";
          if (d) yield d;
        }
      } finally {
        rel();
      }
    },
    async decide(state, question) {
      const cap = opts.capture;
      if (!cap) throw new Error("webllmModel.decide(): requires logit capture. Use an MLCEngine on the MAIN THREAD (not a Web Worker) created with a `logitProcessorRegistry` containing `logitCapture().processor` (or use createWebllm(), which does this) and pass `{ capture }` to webllmModel. At temperature 0 WebLLM's top_logprobs are one-hot, so there is no fallback.");
      const prompt = `<|im_start|>system
You are a security decision function. Answer with exactly one word: Yes or No.<|im_end|>
<|im_start|>user
STATE:
${String(state).slice(0, 2e3)}

QUESTION: ${question}
Answer Yes or No.<|im_end|>
<|im_start|>assistant
<think>

</think>

` + thinkTail;
      const rel = await gate.acquire();
      let L;
      try {
        try {
          await engine.resetChat?.();
          cap.reset();
          cap.arm();
          await engine.completions.create({ prompt, max_tokens: 1, temperature: 0 });
          L = cap.take();
        } catch {
          return 0.5;
        }
      } finally {
        rel();
      }
      if (!L) throw new Error("webllmModel.decide(): no logits were captured in this call (the engine does not use this `capture`'s logitProcessorRegistry, or it is a Web Worker engine)");
      const p = sigmoid(L[YES_ID] - L[NO_ID]);
      if (Number.isNaN(p)) throw new Error("webllmModel.decide(): logits NaN");
      return clamp01(p);
    }
  };
}
export {
  createWebllm,
  loadRecovery,
  logitCapture,
  recoverHidden,
  tryLoadRecovery,
  webllmBackend,
  webllmModel
};
