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
function checkContinuation(name, h, ids, positions) {
  const k = h.ids.length;
  if (ids.length <= k || h.ids.some((x, i) => ids[i] !== x)) throw new Error(`${name}: prefillFrom ids must extend the snapshot prefix (${k} tokens) by at least one token`);
  for (const p of positions) if (!Number.isInteger(p) || p < k || p >= ids.length) throw new Error(`${name}: position ${p} outside the continuation [${k},${ids.length})`);
  return k;
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

// src/ortVersion.ts
function checkOrtVersion(v, min = [1, 30]) {
  const m = /^(\d+)\.(\d+)/.exec(v ?? "");
  if (!m) return;
  if (Number(m[1]) < min[0] || Number(m[1]) === min[0] && Number(m[2]) < min[1])
    throw new Error(`ct0-edge: onnxruntime-web ${v} is too old; requires >= ${min.join(".")} (WebGPU returns NaN below that; the genai backend needs the LinearAttention/CausalConvWithState kernels). Pass skipVersionCheck to ignore.`);
}

// src/backends/decodeOnly.ts
var EOS = /* @__PURE__ */ new Set([248046, 248044]);
var HIDDEN_NAMES = ["/model/norm/Mul_1_output_0", "mul_13450", "hidden_states", "type_as_80"];
var F16 = globalThis.Float16Array;
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
function lastRow(L) {
  const V = L.dims.at(-1), off = (L.dims.at(-2) - 1) * V;
  return tensorToF32(L.data, L.type, off, V);
}
function sample(logits, gen, rand, o) {
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
function pastNameFor(out) {
  let m;
  if (m = out.match(/^present_conv\.(\d+)$/)) return "past_conv." + m[1];
  if (m = out.match(/^present_recurrent\.(\d+)$/)) return "past_recurrent." + m[1];
  if (m = out.match(/^present\.(\d+)\.(key|value)$/)) return `past_key_values.${m[1]}.${m[2]}`;
  if (m = out.match(/^present\.(\d+)\.(conv|recurrent)$/)) return `past.${m[1]}.${m[2]}`;
  if (m = out.match(/^so(\d+)$/)) return "si" + m[1];
  return null;
}
function cacheDims(name, shape, pastLen) {
  const isKV = /^past_key_values\./.test(name) || shape.length === 4 && !/conv|recurrent/.test(name) && typeof shape[2] !== "number";
  return shape.map((d, i) => typeof d === "number" && d >= 0 ? d : isKV && i === 2 ? pastLen : 1);
}
async function createDecodeOnly(session, tok, opts) {
  if (!opts.skipVersionCheck) checkOrtVersion(opts.ort.env?.versions?.web);
  const T = opts.ort.Tensor;
  const pastLen = opts.pastLen ?? 1;
  const inNames = session.inputNames, outNames = session.outputNames;
  if (inNames.includes("attention_mask")) throw new Error("decode-only: a graph with attention_mask is not decode-only");
  const meta = Array.isArray(session.inputMetadata) ? session.inputMetadata : Object.entries(session.inputMetadata ?? {}).map(([name, m]) => ({ name, ...m }));
  const metaBy = new Map(meta.map((m) => [m.name, m]));
  const specs = [];
  for (const n of inNames) {
    if (n === "input_ids" || n === "cache_position") continue;
    const m = metaBy.get(n);
    if (!m?.shape) throw new Error("decode-only: no shape metadata for input " + n);
    specs.push({ name: n, type: m.type, dims: cacheDims(n, m.shape, pastLen) });
  }
  const pastOf = /* @__PURE__ */ new Map();
  for (const o of outNames) {
    const p = pastNameFor(o);
    if (p && inNames.includes(p)) pastOf.set(o, p);
  }
  const missing = specs.filter((s) => ![...pastOf.values()].includes(s.name));
  if (missing.length) throw new Error("decode-only: cache inputs without a matching present output: " + missing.slice(0, 3).map((s) => s.name).join(", "));
  const hiddenName = HIDDEN_NAMES.find((n) => outNames.includes(n)) ?? null;
  const hasCachePos = inNames.includes("cache_position");
  const zeros = (s) => {
    const n = s.dims.reduce((a, b) => a * b, 1);
    if (s.type === "float32") return new T("float32", new Float32Array(n), s.dims);
    if (s.type === "float16") return new T("float16", F16 ? new F16(n) : new Uint16Array(n), s.dims);
    throw new Error(`decode-only: unsupported cache dtype: ${s.type} (${s.name})`);
  };
  const initState = () => {
    const st = {};
    for (const s of specs) st[s.name] = zeros(s);
    return st;
  };
  const fetchBase = ["logits", ...pastOf.keys()];
  async function step(tid, pos, st, wantHidden) {
    const feed2 = { input_ids: new T("int64", BigInt64Array.from([BigInt(tid)]), [1, 1]), ...st };
    if (hasCachePos) feed2.cache_position = new T("int64", BigInt64Array.from([BigInt(pos)]), [1]);
    const names = wantHidden && hiddenName ? [...fetchBase, hiddenName] : fetchBase;
    const r = await session.run(feed2, names);
    const ns = {};
    for (const [o, p] of pastOf) ns[p] = r[o];
    const hidden = wantHidden && hiddenName ? tensorToF32(r[hiddenName].data, r[hiddenName].type) : void 0;
    return { logits: r.logits, st: ns, hidden };
  }
  async function feed(ids, from, st, want) {
    const wanted = new Set(want.positions), at = /* @__PURE__ */ new Map();
    let last = null;
    for (let i = from; i < ids.length; i++) {
      const r = await step(ids[i], i, st, wanted.has(i));
      st = r.st;
      last = r.logits;
      if (r.hidden) at.set(i, r.hidden);
    }
    return { hiddens: want.positions.map((p) => at.get(p)), lastLogits: lastRow(last) };
  }
  const checkHidden = (want) => {
    if (want.positions.length && !hiddenName) throw new Error("decode-only: the graph does not expose a hidden state (outputs: " + outNames.slice(-3).join(", ") + ")");
  };
  const gate = createGate("decode-only");
  const snaps = /* @__PURE__ */ new WeakMap();
  return {
    kind: "decode-only",
    hiddenName,
    dispose() {
      return gate.dispose(() => session.release?.());
    },
    async prefill(ids, want) {
      gate.assertLive();
      checkHidden(want);
      for (const p of new Set(want.positions)) if (p < 0 || p >= ids.length) throw new Error(`prefill: position ${p} out of [0,${ids.length})`);
      if (!ids.length) throw new Error("prefill: empty ids");
      const rel = await gate.acquire();
      try {
        return await feed(ids, 0, initState(), want);
      } finally {
        rel();
      }
    },
    async snapshot(ids) {
      gate.assertLive();
      if (!ids.length) throw new Error("decode-only: snapshot of empty ids");
      const rel = await gate.acquire();
      try {
        let st = initState();
        for (let i = 0; i < ids.length; i++) st = (await step(ids[i], i, st, false)).st;
        const h = { ids: ids.slice(), dispose() {
          snaps.delete(h);
        } };
        snaps.set(h, st);
        return h;
      } finally {
        rel();
      }
    },
    async prefillFrom(h, ids, want) {
      gate.assertLive();
      const st = snaps.get(h);
      if (!st) throw new Error("decode-only: unknown or released snapshot handle");
      const k = checkContinuation("decode-only", h, ids, want.positions);
      checkHidden(want);
      const rel = await gate.acquire();
      try {
        return await feed(ids, k, st, want);
      } finally {
        rel();
      }
    },
    async *generate(ids, o) {
      gate.assertLive();
      const g = resolveGenerateOptions(o);
      const rand = mulberry32(g.seed);
      if (!ids.length) throw new Error("generate: empty ids");
      if (g.maxNewTokens === 0) return;
      const rel = await gate.acquire();
      try {
        let st = initState(), pos = 0, last = null;
        for (const t of ids) {
          if (g.signal?.aborted) return;
          ({ logits: last, st } = await step(t, pos, st, false));
          pos++;
        }
        const gen = [];
        for (let s = 0; s < g.maxNewTokens; s++) {
          if (g.signal?.aborted) break;
          const tid = sample(lastRow(last), gen, rand, g);
          if (EOS.has(tid)) break;
          gen.push(tid);
          yield tid;
          if (s + 1 >= g.maxNewTokens || g.signal?.aborted) break;
          ({ logits: last, st } = await step(tid, pos, st, false));
          pos++;
        }
      } finally {
        rel();
      }
    }
  };
}
export {
  EOS,
  GENERATE_DEFAULTS,
  HIDDEN_NAMES,
  cacheDims,
  checkOrtVersion,
  createDecodeOnly,
  createGate,
  halfToNumber,
  lastRow,
  mulberry32,
  pastNameFor,
  resolveGenerateOptions,
  sample,
  tensorToF32
};
