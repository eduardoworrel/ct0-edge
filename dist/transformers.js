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

// src/backends/decodeOnly.ts
var EOS = /* @__PURE__ */ new Set([248046, 248044]);
var HIDDEN_NAMES = ["/model/norm/Mul_1_output_0", "mul_13450", "hidden_states", "type_as_80"];
var F16 = globalThis.Float16Array;
function halfToNumber(h) {
  const s = h & 32768 ? -1 : 1, e = (h & 31744) >> 10, f = h & 1023;
  return e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? f ? NaN : s * Infinity : s * Math.pow(2, e - 15) * (1 + f / 1024);
}
function tensorToF32(data, type, off = 0, n = data.length - off) {
  const out = new Float32Array(n);
  const isF16Arr = typeof F16 !== "undefined" && data instanceof F16;
  if (data instanceof Float32Array || isF16Arr) {
    for (let i = 0; i < n; i++) out[i] = Number(data[off + i]);
  } else if (type === "float16" || data instanceof Uint16Array) {
    for (let i = 0; i < n; i++) out[i] = halfToNumber(data[off + i]);
  } else for (let i = 0; i < n; i++) out[i] = Number(data[off + i]);
  return out;
}

// src/backends/transformersjs.ts
function transformersJsFromModel(model, Tensor) {
  const sess = model.sessions?.model ?? Object.values(model.sessions ?? {})[0];
  const outs = sess?.outputNames ?? [];
  const hiddenName = HIDDEN_NAMES.find((n) => outs.includes(n)) ?? null;
  const toTensors = (ids) => ({
    input_ids: new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    attention_mask: new Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length])
  });
  const gate = createGate("scan");
  return {
    kind: "scan",
    hiddenName,
    async prefill(ids, want) {
      gate.assertLive();
      const n = ids.length;
      if (!n) throw new Error("prefill: empty ids");
      for (const p of want.positions) if (!Number.isInteger(p) || p < 0 || p >= n) throw new Error(`transformers.js: position ${p} out of range [0,${n})`);
      if (want.positions.length && !hiddenName) throw new Error("transformers.js: graph has no hidden output; use method 'logits'");
      const rel = await gate.acquire();
      try {
        const out = await model.forward(toTensors(ids));
        const L = out.logits, V = L.dims.at(-1);
        const lastLogits = tensorToF32(L.data, L.type, (L.dims.at(-2) - 1) * V, V);
        const hiddens = [];
        if (want.positions.length) {
          const H = out[hiddenName], D = H.dims.at(-1);
          if (H.dims.at(-2) !== n) throw new Error(`transformers.js: hidden seq ${H.dims.at(-2)} != ${n}`);
          for (const p of want.positions) hiddens.push(tensorToF32(H.data, H.type, p * D, D));
        }
        return { hiddens, lastLogits };
      } finally {
        rel();
      }
    },
    dispose() {
      return gate.dispose(() => model.dispose?.());
    },
    /** Generates ids from `ids` (raw prompt) via `model.generate()`. Emits at most `maxNewTokens` (default 512;
     *  0 => none, without calling the library) ids, without EOS. `seed` is IGNORED (unsupported by the library): with
     *  temperature > 0 sampling is not reproducible. `signal` abort or a consumer `break` stops the library run
     *  (stopping criterion) and the generator only ends once it settles, so a following prefill/dispose never overlaps the session. */
    async *generate(ids, o) {
      gate.assertLive();
      const g = resolveGenerateOptions(o);
      const max = g.maxNewTokens, temp = g.temperature;
      if (!ids.length) throw new Error("generate: empty ids");
      if (max === 0) return;
      const rel = await gate.acquire();
      const queue = [];
      let done = false, wake = null, err = null, first = true, stop = false;
      const ping = () => {
        const w = wake;
        wake = null;
        w?.();
      };
      const streamer = {
        put(v) {
          if (first) {
            first = false;
            return;
          }
          if (stop) return;
          for (const t of v.flat ? v.flat() : v) queue.push(Number(t));
          ping();
        },
        end() {
          done = true;
          ping();
        }
      };
      const criterion = (batch) => batch.map(() => stop || !!o.signal?.aborted);
      const args = { ...toTensors(ids), max_new_tokens: max, streamer, stopping_criteria: [criterion], do_sample: temp > 0 };
      if (temp > 0) {
        args.temperature = temp;
        args.top_p = g.topP;
        args.top_k = g.topK;
      }
      args.repetition_penalty = g.repetitionPenalty;
      const run = Promise.resolve().then(() => model.generate(args)).then(() => {
        done = true;
        ping();
      }, (e) => {
        err = e;
        done = true;
        ping();
      });
      let emitted = 0;
      try {
        while (true) {
          if (o.signal?.aborted) break;
          if (queue.length) {
            const t = queue.shift();
            if (EOS.has(t)) break;
            yield t;
            if (++emitted >= max) break;
          } else if (done) break;
          else await new Promise((r) => {
            wake = r;
            if (queue.length || done) ping();
          });
        }
      } finally {
        stop = true;
        try {
          await run;
        } finally {
          rel();
        }
      }
      if (err && !o.signal?.aborted) throw err;
    }
  };
}
async function createTransformersJs(modelId, opts) {
  const tjs = await import("@huggingface/transformers");
  if (opts.localModelPath) {
    tjs.env.allowRemoteModels = false;
    tjs.env.allowLocalModels = true;
    tjs.env.localModelPath = opts.localModelPath;
  }
  const model = await tjs.AutoModelForCausalLM.from_pretrained(modelId, { dtype: opts.dtype, device: opts.device ?? "cpu" });
  return transformersJsFromModel(model, tjs.Tensor);
}
export {
  createTransformersJs,
  transformersJsFromModel
};
