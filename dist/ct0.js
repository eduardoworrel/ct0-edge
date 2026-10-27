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
var gates = /* @__PURE__ */ new WeakMap();
function gateFor(shared, name) {
  let g = gates.get(shared);
  if (!g) {
    g = createGate(name);
    gates.set(shared, g);
  }
  return g;
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

// src/backends/genai.ts
function parsePlacement(lines) {
  const out = {};
  for (const l of lines) {
    const m = /Node\(s\) placed on \[(\w+)\]\.\s*Number of nodes:\s*(\d+)/.exec(l);
    if (m) out[m[1]] = (out[m[1]] ?? 0) + Number(m[2]);
  }
  return out;
}
function assertWebGpuPlacement(p, maxCpuNodes = 32) {
  const gpu = Object.entries(p).filter(([k]) => /webgpu/i.test(k)).reduce((a, [, n]) => a + n, 0);
  const cpu = p.CPUExecutionProvider ?? 0;
  if (!Object.keys(p).length) throw new Error("genai: could not verify the EP placement (no 'Node(s) placed on' logs); reject a bundle without WebGPU");
  if (gpu === 0) throw new Error(`genai: no node on the WebGPU EP (placement: ${JSON.stringify(p)}); LinearAttention does not run on GPU`);
  if (cpu > maxCpuNodes) throw new Error(`genai: ${cpu} nodes on CPU (> ${maxCpuNodes}); probably a bundle without the WebGPU kernels (use onnxruntime-web/webgpu). Placement: ${JSON.stringify(p)}`);
}
function genaiFromSession(session, ort, opts = {}) {
  const T = ort.Tensor;
  const eos = opts.eos ?? EOS;
  const inNames = session.inputNames, outNames = session.outputNames;
  for (const need of ["input_ids"]) if (!inNames.includes(need)) throw new Error(`genai: graph without input ${need}`);
  if (!outNames.includes("logits")) throw new Error("genai: graph without a logits output");
  const hasMask = inNames.includes("attention_mask"), hasPos = inNames.includes("position_ids");
  const meta = Array.isArray(session.inputMetadata) ? session.inputMetadata : Object.entries(session.inputMetadata ?? {}).map(([name, m]) => ({ name, ...m }));
  const metaBy = new Map(meta.map((m) => [m.name, m]));
  const specs = [];
  for (const n of inNames) {
    if (n === "input_ids" || n === "attention_mask" || n === "position_ids") continue;
    const m = metaBy.get(n);
    if (!m?.shape) throw new Error("genai: no shape metadata for input " + n);
    const dims = cacheDims(n, m.shape, 0);
    if (/^past_key_values\./.test(n) && typeof m.shape[3] !== "number") dims[3] = opts.kvHeadDim ?? 256;
    specs.push({ name: n, type: m.type, dims });
  }
  const pastOf = /* @__PURE__ */ new Map();
  for (const o of outNames) {
    const p = pastNameFor(o);
    if (p && inNames.includes(p)) pastOf.set(o, p);
  }
  const covered = new Set(pastOf.values());
  const missing = specs.filter((s) => !covered.has(s.name));
  if (missing.length) throw new Error("genai: cache inputs without a present output: " + missing.slice(0, 3).map((s) => s.name).join(", "));
  const hiddenName = HIDDEN_NAMES.find((n) => outNames.includes(n)) ?? null;
  const posRows = (() => {
    const sh = metaBy.get("position_ids")?.shape;
    return typeof sh?.[0] === "number" ? sh[0] : 3;
  })();
  const zeros = (s) => {
    const n = s.dims.reduce((a, b) => a * b, 1);
    if (s.type === "float16") return new T("float16", new Uint16Array(n), s.dims);
    if (s.type === "float32") return new T("float32", new Float32Array(n), s.dims);
    throw new Error(`genai: unsupported cache dtype: ${s.type}`);
  };
  const initState = () => {
    const st = {};
    for (const s of specs) st[s.name] = zeros(s);
    return st;
  };
  const free = (st) => {
    if (st) {
      for (const t of Object.values(st)) if (t?.location === "gpu-buffer") {
        try {
          t.dispose?.();
        } catch {
        }
      }
    }
  };
  const gate = createGate("genai");
  async function run(chunk, past, st, wantLogits, wantHidden, keep = false) {
    const S = chunk.length;
    const feed = { input_ids: new T("int64", BigInt64Array.from(chunk.map(BigInt)), [1, S]), ...st };
    if (hasMask) feed.attention_mask = new T("int64", new BigInt64Array(past + S).fill(1n), [1, past + S]);
    if (hasPos) {
      const pos = new BigInt64Array(posRows * S);
      for (let r2 = 0; r2 < posRows; r2++) for (let i = 0; i < S; i++) pos[r2 * S + i] = BigInt(past + i);
      feed.position_ids = new T("int64", pos, [posRows, 1, S]);
    }
    const fetches = [...pastOf.keys()];
    if (wantLogits) fetches.unshift("logits");
    if (wantHidden && hiddenName) fetches.push(hiddenName);
    const r = await session.run(feed, fetches, { logSeverityLevel: 3 });
    const ns = {};
    for (const [o, p] of pastOf) ns[p] = r[o];
    if (!keep) free(st);
    const out = {};
    if (wantLogits) out.logits = r.logits;
    if (wantHidden && hiddenName) out.hidden = r[hiddenName];
    return { st: ns, out };
  }
  const read = async (t) => typeof t.getData === "function" ? await t.getData(true) : t.data;
  async function feedAll(ids, wanted, wantLast, signal, from) {
    let st = from?.st ?? initState(), past = from?.past ?? 0;
    const kept = from?.st;
    const hid = /* @__PURE__ */ new Map();
    let lastLogits;
    try {
      while (past < ids.length) {
        if (signal?.aborted) return { st, past, hid, lastLogits, aborted: true };
        const chunk = ids.slice(past, past + self.chunk), last = past + chunk.length >= ids.length;
        const need = [...wanted].filter((p) => p >= past && p < past + chunk.length);
        const r = await run(chunk, past, st, last && wantLast, need.length > 0, st === kept);
        st = r.st;
        if (r.out.hidden) {
          const H = r.out.hidden, D = H.dims.at(-1), d = await read(H);
          for (const p of need) hid.set(p, tensorToF32(d, H.type, (p - past) * D, D));
        }
        if (r.out.logits) {
          const L = r.out.logits, V = L.dims.at(-1), d = await read(L);
          lastLogits = tensorToF32(d, L.type, (L.dims.at(-2) - 1) * V, V);
        }
        past += chunk.length;
      }
    } catch (e) {
      if (st !== kept) free(st);
      throw e;
    }
    return { st, past, hid, lastLogits, aborted: false };
  }
  const snaps = /* @__PURE__ */ new WeakMap();
  const self = {
    kind: "genai",
    chunk: Math.max(1, opts.chunk ?? 128),
    hiddenName,
    dispose() {
      return gate.dispose(() => session.release?.());
    },
    async prefill(ids, want) {
      gate.assertLive();
      for (const p of want.positions) if (!Number.isInteger(p) || p < 0 || p >= ids.length) throw new Error(`genai: position ${p} out of [0,${ids.length})`);
      if (!ids.length) throw new Error("prefill: empty ids");
      if (want.positions.length && !hiddenName) throw new Error("genai: the graph does not expose a hidden state; use method 'logits'");
      const rel = await gate.acquire();
      try {
        const f = await feedAll(ids, new Set(want.positions), true);
        free(f.st);
        return { hiddens: want.positions.map((p) => f.hid.get(p)), lastLogits: f.lastLogits };
      } finally {
        rel();
      }
    },
    async snapshot(ids) {
      gate.assertLive();
      if (!ids.length) throw new Error("genai: snapshot of empty ids");
      const rel = await gate.acquire();
      try {
        const f = await feedAll(ids, /* @__PURE__ */ new Set(), false);
        let st = f.st;
        const h = { ids: ids.slice(), dispose: async () => {
          if (!st) return;
          const s0 = st;
          st = null;
          snaps.delete(h);
          let r = null;
          try {
            r = await gate.acquire();
          } catch {
          }
          try {
            free(s0);
          } finally {
            r?.();
          }
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
      if (!st) throw new Error("genai: unknown or released snapshot handle");
      const k = checkContinuation("genai", h, ids, want.positions);
      if (want.positions.length && !hiddenName) throw new Error("genai: the graph does not expose a hidden state; use method 'logits'");
      const rel = await gate.acquire();
      try {
        const f = await feedAll(ids, new Set(want.positions), true, void 0, { st, past: k });
        free(f.st);
        return { hiddens: want.positions.map((p) => f.hid.get(p)), lastLogits: f.lastLogits };
      } finally {
        rel();
      }
    },
    /** Generates ids from `ids`; stops at EOS (not emitted), `maxNewTokens` (default 512; 0 => none), `signal` or `break`.
     *  Runs are awaited inline, so no run is in flight when the generator ends. Sampler = decode-only's. */
    async *generate(ids, o) {
      gate.assertLive();
      const g = resolveGenerateOptions(o);
      const rand = mulberry32(g.seed);
      if (!ids.length) throw new Error("generate: empty ids");
      if (g.maxNewTokens === 0) return;
      const rel = await gate.acquire();
      let st = null;
      try {
        if (o.signal?.aborted) return;
        const f = await feedAll(ids, /* @__PURE__ */ new Set(), true, o.signal);
        st = f.st;
        if (f.aborted) return;
        let past = f.past, logits = f.lastLogits;
        const gen = [];
        for (let s = 0; s < g.maxNewTokens; s++) {
          if (o.signal?.aborted) break;
          const tid = sample(logits, gen, rand, g);
          if (eos.has(tid)) break;
          gen.push(tid);
          yield tid;
          if (o.signal?.aborted || s + 1 >= g.maxNewTokens) break;
          const r = await run([tid], past, st, true, false);
          st = r.st;
          past++;
          const L = r.out.logits, V = L.dims.at(-1);
          logits = tensorToF32(await read(L), L.type, (L.dims.at(-2) - 1) * V, V);
        }
      } finally {
        free(st);
        rel();
      }
    }
  };
  return self;
}
function hashStr(x) {
  let h = 2166136261;
  for (let i = 0; i < x.length; i++) {
    h ^= x.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}
async function adapterTag(opts) {
  if (typeof opts.adapterTag === "string") return opts.adapterTag;
  if (typeof opts.adapterTag === "function") return opts.adapterTag();
  try {
    const gpu = globalThis.navigator?.gpu;
    const ad = await gpu?.requestAdapter?.();
    const i = ad?.info ?? await ad?.requestAdapterInfo?.();
    if (!i) return "?";
    return hashStr([i.vendor, i.architecture, i.device, i.description].join("|"));
  } catch {
    return "?";
  }
}
function epStore(opts) {
  if (opts.epCache !== void 0) return opts.epCache;
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}
async function probePlacement(ort, modelUrl, sessionOpts, maxCpuNodes) {
  const captured = [];
  const names = ["log", "info", "warn", "error", "debug"];
  const saved = {};
  const prevLevel = ort.env?.logLevel;
  if (ort.env) ort.env.logLevel = "verbose";
  for (const n of names) {
    saved[n] = console[n];
    console[n] = (...a) => {
      const s = a.join(" ");
      if (s.includes("Node(s) placed on")) captured.push(s);
      else if (!s.includes("[V:onnxruntime") && !/WebGPU Shader code|^\s*$/.test(s)) saved[n].apply(console, a);
    };
  }
  let probe;
  try {
    probe = await ort.InferenceSession.create(modelUrl, sessionOpts(0));
  } finally {
    for (const n of names) console[n] = saved[n];
    if (ort.env) ort.env.logLevel = prevLevel;
  }
  try {
    assertWebGpuPlacement(parsePlacement(captured), maxCpuNodes);
  } finally {
    await probe.release?.();
  }
}
async function createGenai(modelUrl, dataUrl, opts = {}) {
  const providers = opts.providers ?? ["webgpu"];
  if (providers[0] !== "webgpu") throw new Error("genai: the graph requires the webgpu EP as first provider (the LinearAttention/CausalConvWithState kernels only exist on WebGPU)");
  const ort = opts.ort ?? await import("onnxruntime-web/webgpu");
  const verify = opts.verifyEp ?? true;
  const sessionOpts = (level) => ({
    executionProviders: providers.map((n) => n === "webgpu" ? { name: "webgpu", validationMode: "disabled" } : n),
    externalData: [{ path: "model.onnx.data", data: dataUrl }],
    graphOptimizationLevel: "all",
    enableGraphCapture: false,
    preferredOutputLocation: "gpu-buffer",
    logSeverityLevel: level
  });
  if (!opts.skipVersionCheck) checkOrtVersion(ort.env?.versions?.web);
  if (verify) {
    const store = epStore(opts);
    const maxCpu = opts.maxCpuNodes ?? 32;
    const key = store ? `ct0-edge:genai-ep:${ort.env?.versions?.web ?? "?"}:${await adapterTag(opts)}:${modelUrl}:${dataUrl}:${maxCpu}` : "";
    let cached = false;
    if (verify !== "force" && store) {
      try {
        cached = store.getItem(key) === "ok";
      } catch {
      }
    }
    if (!cached) {
      await probePlacement(ort, modelUrl, sessionOpts, maxCpu);
      if (store) {
        try {
          store.setItem(key, "ok");
        } catch {
        }
      }
    }
  }
  const session = await ort.InferenceSession.create(modelUrl, sessionOpts(3));
  try {
    const be = genaiFromSession(session, ort, opts);
    if (opts.warmup ?? true) await be.prefill([248045], { positions: [] });
    return be;
  } catch (e) {
    try {
      await session.release?.();
    } catch {
    }
    throw e;
  }
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

// src/positions.ts
var CHOICE_Q = "Which vulnerability class does this code contain?";
var SEV_Q = "How severe is the security risk in this content? (1=none, 5=critical)";
function renderPrompt(state, question) {
  return "<|im_start|>system\nYou are a security decision function. Answer with exactly one word: Yes or No.<|im_end|>\n<|im_start|>user\n<state>" + state + "</state>\n<q>" + question + "</q><decide><|im_end|>\n<|im_start|>assistant\n";
}
var KINDS = ["noul", "choice", "score"];
function assertProbe(p) {
  const q = p;
  if (!q || typeof q !== "object") throw new Error("probe: expected an object {kind, state, ...}");
  if (!KINDS.includes(q.kind)) {
    const hint = q.kind === void 0 && "type" in q ? ` (the field is \`kind\`, not \`type\`: {kind: ${JSON.stringify(q.type)}, state, q})` : "";
    throw new Error(`probe: invalid kind ${JSON.stringify(q.kind)}; use "noul", "choice" or "score"${hint}`);
  }
  if (typeof q.state !== "string") throw new Error("probe: state must be a string");
  if (q.kind === "choice") {
    const o = q.options;
    if (!Array.isArray(o) || !o.length) throw new Error("probe: choice requires options (non-empty list)");
    const seen = /* @__PURE__ */ new Set();
    for (const x of o) {
      if (typeof x !== "string" || !x.trim()) throw new Error("probe: option label empty or not a string: " + JSON.stringify(x));
      if (/<\/?opt>/.test(x)) throw new Error("probe: option label cannot contain <opt> / </opt>: " + JSON.stringify(x));
      if (seen.has(x)) throw new Error("probe: duplicate option label: " + JSON.stringify(x));
      seen.add(x);
    }
  }
}
function promptFor(p) {
  assertProbe(p);
  if (p.kind === "noul") return renderPrompt(p.state, p.q ?? "");
  if (p.kind === "choice") {
    return renderPrompt(p.state, CHOICE_Q + " " + (p.options ?? []).map((o) => "<opt>" + o + "</opt>").join(""));
  }
  return renderPrompt(p.state, SEV_Q);
}
function decidePos(tokens) {
  const imEnd = [];
  tokens.forEach((s, i) => {
    if (s === "<|im_end|>") imEnd.push(i);
  });
  if (!imEnd.length) throw new Error("decidePos: no <|im_end|> in " + JSON.stringify(tokens.slice(-8)));
  let lastAsst = -1;
  tokens.forEach((s, i) => {
    if (s.trim() === "assistant") lastAsst = i;
  });
  const cand = lastAsst >= 0 ? imEnd.filter((i) => i < lastAsst) : [];
  const end = cand.length ? cand[cand.length - 1] : imEnd[imEnd.length - 1];
  if (end === 0) throw new Error("decidePos: <|im_end|> at position 0, no preceding token");
  if (tokens[end - 1] !== ">") {
    throw new Error("decidePos: token before <|im_end|> is " + JSON.stringify(tokens[end - 1]) + ", expected '>' (end of <decide>)");
  }
  return end - 1;
}
function optPositions(prompt, offsets) {
  const out = [];
  const from = Math.max(0, prompt.lastIndexOf("<q>"));
  for (const m of prompt.slice(from).matchAll(/<\/opt>/g)) {
    const c = from + m.index + m[0].length - 1;
    const hit = offsets.findIndex(([s, e]) => s <= c && c < e);
    if (hit < 0) throw new Error("optPositions: no token covers char " + c);
    out.push(hit);
  }
  return out;
}
function optPositionsFor(prompt, offsets, nOptions) {
  const pos = optPositions(prompt, offsets);
  if (nOptions === void 0) nOptions = (prompt.slice(Math.max(0, prompt.lastIndexOf("<q>"))).match(/<opt>/g) ?? []).length;
  if (pos.length !== nOptions) throw new Error(`positions: ${pos.length} </opt> markers after <q> for ${nOptions} options (label containing </opt>?)`);
  return pos;
}
function tokenOffsets(tok, prompt) {
  const ids = Array.from(tok.encode(prompt, { add_special_tokens: false }), (x) => Number(x));
  const dec = (a) => tok.decode(a, { skip_special_tokens: false, clean_up_tokenization_spaces: false });
  const tokens = ids.map((i) => dec([i]));
  const offsets = [];
  let prev = 0;
  let full = "";
  for (let i = 0; i < ids.length; i++) {
    full = dec(ids.slice(0, i + 1));
    let end = full.length;
    while (end > prev && full.charCodeAt(end - 1) === 65533) end--;
    offsets.push([prev, end]);
    prev = end;
  }
  if (full !== prompt) throw new Error("tokenOffsets: reconstruction != prompt (offsets not reliable)");
  return { ids, tokens, offsets };
}

// src/systemone.ts
function loadHead(buf) {
  const dv = new DataView(buf);
  const jsonLen = dv.getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jsonLen)));
  let off = 4 + jsonLen;
  const take = (n) => {
    const avail = Math.floor((buf.byteLength - off) / 2);
    if (n > avail) throw new Error(`head binary too short: needs ${n} fp16, ${avail} available`);
    const f32 = new Float32Array(n);
    for (let i = 0; i < n; i++) f32[i] = f16ToF32(dv.getUint16(off + i * 2, true));
    off += n * 2;
    return f32;
  };
  return {
    q: take(meta.headDim * meta.hiddenSize),
    qb: take(meta.headDim),
    k: take(meta.headDim * meta.hiddenSize),
    kb: take(meta.headDim),
    headDim: meta.headDim,
    hiddenSize: meta.hiddenSize,
    temperature: meta.temperature,
    scoreOffset: meta.score_offset ?? 0
  };
}
function f16ToF32(h) {
  const s = (h & 32768) >> 15, e = (h & 31744) >> 10, f = h & 1023;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s ? -Infinity : Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
function proj(W, b, x, dim, hidden) {
  const y = new Float32Array(dim);
  for (let d = 0; d < dim; d++) {
    let acc = b[d];
    const row = d * hidden;
    for (let i = 0; i < hidden; i++) acc += W[row + i] * x[i];
    y[d] = acc;
  }
  return y;
}
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
var sigmoid = (z) => 1 / (1 + Math.exp(-z));
function headScore(hd, hq, hk) {
  if (!hq || !hk || hq.length < hd.hiddenSize || hk.length < hd.hiddenSize) return NaN;
  const a = proj(hd.q, hd.qb, hq, hd.headDim, hd.hiddenSize);
  const b = proj(hd.k, hd.kb, hk, hd.headDim, hd.hiddenSize);
  const s = dot(a, b) / hd.temperature + (hd.scoreOffset || 0);
  return Number.isFinite(s) ? s : NaN;
}
function answer(hd, q, hiddens) {
  if (q.type === "noul") {
    const raw2 = headScore(hd, hiddens.atDecide, hiddens.atLast);
    return { type: "noul", noul: sigmoid(raw2), raw: raw2 };
  }
  if (q.type === "choice") {
    const opts = q.options || [];
    const per = hiddens.perOption || [];
    if (!per.length) return { type: "choice", choice: {} };
    const choice = {};
    const raws = [];
    for (let i = 0; i < opts.length; i++) {
      const h = per[i];
      if (!h) {
        choice[opts[i]] = 0;
        raws.push(-Infinity);
        continue;
      }
      const r = headScore(hd, hiddens.atDecide, h);
      raws.push(r);
      choice[opts[i]] = r;
    }
    const mx = Math.max(...raws.filter(Number.isFinite));
    let z = 0;
    const exps = raws.map((r) => Number.isFinite(r) ? Math.exp(r - mx) : 0);
    z = exps.reduce((a, b) => a + b, 0) || 1;
    for (let i = 0; i < opts.length; i++) choice[opts[i]] = exps[i] / z;
    return { type: "choice", choice };
  }
  const K = q.levels || 5;
  const raw = headScore(hd, hiddens.atDecide, hiddens.atLast);
  const p = sigmoid(raw);
  const score = Math.max(1, Math.min(K, Math.round(p * (K - 1) + 1)));
  return { type: "score", score, raw };
}

// src/ask.ts
var YES_ID = 9175;
var NO_ID = 2665;
var DIGIT_IDS = [16, 17, 18, 19, 20];
var DEFAULT_METHOD = { noul: "logits", choice: "head", score: "head" };
var sigmoid2 = (z) => 1 / (1 + Math.exp(-z));
function softmax(xs) {
  const mx = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - mx));
  const z = e.reduce((a, b) => a + b, 0) || 1;
  return e.map((v) => v / z);
}
function argmax(xs) {
  let b = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[b]) b = i;
  return b;
}
function toChoice(options, probs) {
  const choice = {};
  options.forEach((o, i) => {
    choice[o] = probs[i];
  });
  return { choice, top: options[argmax(probs)] };
}
function levelFromCuts(x, cuts) {
  let l = 1;
  for (const c of cuts) if (x >= c) l++;
  return l;
}
function checkCuts(cuts, n, read) {
  if (cuts === void 0) return void 0;
  if (!Array.isArray(cuts) || cuts.length !== n || !cuts.every(Number.isFinite) || cuts.some((c, i) => i > 0 && c <= cuts[i - 1]))
    throw new Error(`ask: scoreCuts for the ${read} read must be ${n} finite ascending numbers, got ${JSON.stringify(cuts)}`);
  return cuts;
}
function planAsk(be, tok, heads, q, opts) {
  const maxSafe = opts.maxSafeTokens ?? 2048;
  const prompt = promptFor(q);
  const type = q.kind;
  const method = opts.method ?? DEFAULT_METHOD[type];
  if (method !== "head" && method !== "logits" && method !== "both") throw new Error(`ask: unknown method ${JSON.stringify(method)}; use "head", "logits" or "both"`);
  const options = q.options ?? [];
  const t = tokenOffsets(tok, prompt);
  if (t.ids.length > maxSafe) throw new Error(`ask: prompt has ${t.ids.length} tokens, over maxSafeTokens (${maxSafe})`);
  let headCuts, logitCuts;
  if (type === "score" && opts.scoreCuts !== void 0) {
    const sc = opts.scoreCuts;
    if (Array.isArray(sc)) {
      if (method === "both") throw new Error('ask: method "both" reads two scales; pass scoreCuts as { head, logits }');
      if (method === "head") headCuts = sc;
      else logitCuts = sc;
    } else ({ head: headCuts, logits: logitCuts } = sc ?? {});
    headCuts = checkCuts(headCuts, (q.levels || 5) - 1, "head");
    logitCuts = checkCuts(logitCuts, DIGIT_IDS.length - 1, "logits");
  }
  const optFirst = type === "choice" ? options.map((o) => Number(tok.encode(o, { add_special_tokens: false })[0])) : [];
  const fromLogits = (L) => {
    if (type === "noul") {
      const raw = L[YES_ID] - L[NO_ID];
      return { type, method: "logits", raw, p: sigmoid2(raw) };
    }
    if (type === "choice") return { type, method: "logits", ...toChoice(options, softmax(optFirst.map((id) => L[id]))) };
    const d = DIGIT_IDS.map((i) => L[i]);
    const expected = softmax(d).reduce((a, p, i) => a + (i + 1) * p, 0);
    return { type, method: "logits", level: logitCuts ? levelFromCuts(expected, logitCuts) : argmax(d) + 1, expected };
  };
  if (method === "logits") return { ids: t.ids, positions: [], read: (r) => fromLogits(r.lastLogits) };
  if (!be.hiddenName) throw new Error(`ask: method '${method}' needs a graph with a hidden output (backend.hiddenName is null); use method 'logits'`);
  const hd = type === "score" ? heads.score : heads.noulChoice;
  if (!hd) throw new Error(type === "score" ? `ask: method '${method}' for score needs the score head (head_score.bin); load it via loadHeads({score})` : `ask: method '${method}' needs a loaded head for '${type}'`);
  const dp = decidePos(t.tokens);
  const withLogits = (r, L) => method === "both" ? { ...r, logits: fromLogits(L) } : r;
  if (type === "choice") {
    const op = optPositionsFor(prompt, t.offsets, options.length);
    return { ids: t.ids, positions: [dp, ...op], read: ({ hiddens, lastLogits }) => {
      const perOption = hiddens.slice(1);
      const a = answer(hd, { type: "choice", options }, { atDecide: hiddens[0], atLast: perOption[perOption.length - 1], perOption });
      return withLogits({ type, method: "head", ...toChoice(options, options.map((o) => a.choice[o])) }, lastLogits);
    } };
  }
  return { ids: t.ids, positions: [dp, t.ids.length - 1], read: ({ hiddens, lastLogits }) => {
    const a = answer(hd, { type, levels: q.levels }, { atDecide: hiddens[0], atLast: hiddens[1] });
    if (type === "noul") return withLogits({ type, method: "head", raw: a.raw, p: a.noul }, lastLogits);
    const level = headCuts && Number.isFinite(a.raw) ? levelFromCuts(a.raw, headCuts) : a.score;
    return withLogits({ type, method: "head", raw: a.raw, level }, lastLogits);
  } };
}
async function ask(be, tok, heads, q, opts = {}) {
  const p = planAsk(be, tok, heads, q, opts);
  return p.read(await be.prefill(p.ids, { positions: p.positions }));
}
function sharedPrefix(plans) {
  let k = Math.min(...plans.map((p) => p.ids.length - 1), ...plans.flatMap((p) => p.positions));
  const a = plans[0].ids;
  for (const p of plans.slice(1)) {
    let i = 0;
    while (i < k && p.ids[i] === a[i]) i++;
    k = i;
  }
  return k;
}
function askMany(be, tok, heads, state, qs, opts = {}) {
  return askManyWith(be, tok, heads, state, qs, () => opts);
}
async function askManyWith(be, tok, heads, state, qs, optsFor) {
  if (typeof state !== "string") throw new Error("askMany: state must be a string");
  const plans = qs.map((q0) => {
    if (q0?.state !== void 0 && q0.state !== state) throw new Error("askMany: a question carries a state different from the shared state");
    const q = { ...q0, state };
    return planAsk(be, tok, heads, q, optsFor(q));
  });
  const out = [];
  const k = plans.length > 1 && be.snapshot && be.prefillFrom ? sharedPrefix(plans) : 0;
  if (k < 1) {
    for (const p of plans) out.push(p.read(await be.prefill(p.ids, { positions: p.positions })));
    return out;
  }
  const h = await be.snapshot(plans[0].ids.slice(0, k));
  try {
    for (const p of plans) out.push(p.read(await be.prefillFrom(h, p.ids, { positions: p.positions })));
  } finally {
    await h.dispose();
  }
  return out;
}

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

// src/backends/litert.ts
var LITERTJS_VERSION = "2.6.0";
var LITERT_WARNING = `ct0-edge: runtime "litert" is EXPERIMENTAL (LiteRT.js @litertjs/core ${LITERTJS_VERSION}, WebGPU with streamed weights; WASM only for small inline graphs).`;
function h2f(h) {
  const s = h & 32768 ? -1 : 1, e = h >> 10 & 31, f = h & 1023;
  return e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? f ? NaN : s * Infinity : s * 2 ** (e - 15) * (1 + f / 1024);
}
function embedFromF16Reader(read, hidden) {
  return {
    async rows(ids) {
      const out = new Float32Array(ids.length * hidden), got = /* @__PURE__ */ new Map();
      await Promise.all([...new Set(ids)].map(async (id) => {
        const b = await read(id * hidden * 2, hidden * 2);
        const c = b.byteOffset % 2 ? b.slice() : b;
        got.set(id, new Uint16Array(c.buffer, c.byteOffset, hidden));
      }));
      ids.forEach((id, r) => {
        const q = got.get(id);
        for (let j = 0; j < hidden; j++) out[r * hidden + j] = h2f(q[j]);
      });
      return out;
    }
  };
}
function httpRangeReader(url, fetchFn = fetch) {
  return async (offset, length) => {
    const r = await fetchFn(url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
    if (!r.ok) throw new Error(`litert: ${url} -> HTTP ${r.status}`);
    const b = new Uint8Array(await r.arrayBuffer());
    return r.status === 206 ? b : b.subarray(offset, offset + length);
  };
}
var MASK_OFF = globalThis.__CT0_MASK_OFF ?? -1e30;
function buildMask(pos, cache, dtype) {
  const rows = pos.length;
  if (dtype === "bool") {
    const m2 = new Uint8Array(rows * cache);
    for (let r = 0; r < rows; r++) m2.fill(1, r * cache, r * cache + pos[r] + 1);
    return m2;
  }
  const m = new Float32Array(rows * cache).fill(MASK_OFF);
  for (let r = 0; r < rows; r++) m.fill(0, r * cache, r * cache + pos[r] + 1);
  return m;
}
function prefillChunks(n, P) {
  const out = [];
  for (let s = 0; s < n; s += P) out.push({ start: s, len: Math.min(P, n - s) });
  return out;
}
var isState = (n) => n.startsWith("kv_cache_");
var zerosOf = (io) => {
  const n = io.shape.reduce((a, b) => a * b, 1);
  return io.dtype === "int32" ? new Int32Array(n) : new Float32Array(n);
};
async function createLiteRt(lrt, graph, manifest, embed, opts = {}) {
  if (!opts.quiet) console.warn(LITERT_WARNING);
  const accelerator = opts.accelerator ?? "webgpu";
  const sigNames = Object.keys(manifest.signatures);
  const prefillName = sigNames.filter((s) => /^prefill_\d+$/.test(s)).sort((a, b) => +b.split("_")[1] - +a.split("_")[1])[0];
  if (!prefillName || !manifest.signatures.decode) throw new Error(`litert: the graph needs prefill_<N> and decode signatures (has ${sigNames.join(", ")})`);
  const P = +prefillName.split("_")[1];
  const dec = manifest.signatures.decode;
  const stateIO = dec.inputs.filter((i) => isState(i.name));
  const maskIO = dec.inputs.find((i) => i.name === "mask");
  if (!maskIO) throw new Error("litert: decode signature has no mask input");
  const cache = maskIO.shape[maskIO.shape.length - 1];
  const H = manifest.hidden;
  const paramIO = dec.inputs.find((i) => i.name === "param_tensor");
  const inplace = !!paramIO;
  const gpuOptions = { precision: opts.gpuPrecision ?? "fp32" };
  const copyState = opts.copyState ?? accelerator === "webgpu";
  const model = accelerator === "webgpu" && opts.weights ? await lrt.loadModelAndWeights(graph, { getReader: () => opts.weights().getReader() }, { accelerator: "webgpu", gpuOptions }) : await lrt.loadAndCompile(graph, accelerator === "webgpu" ? { accelerator, gpuOptions } : { accelerator });
  const T = (data, shape, dtype) => dtype ? new lrt.Tensor(data, shape, dtype) : new lrt.Tensor(data, shape);
  const del = (t) => {
    try {
      t?.delete?.();
    } catch {
    }
  };
  const zeroStateHost = () => {
    const st = {};
    for (const io of stateIO) st[io.name] = T(zerosOf(io), io.shape);
    return st;
  };
  async function freshState() {
    const st = zeroStateHost();
    if (inplace && accelerator === "webgpu") for (const k in st) {
      const g = await st[k].moveTo("webgpu");
      st[k] = g;
    }
    return st;
  }
  const freeState = (st) => {
    for (const k in st) del(st[k]);
  };
  async function run(sig, emb, rows, pos, st, valid = rows) {
    const feed = { ...st, embeddings: T(emb, [1, rows, H]), input_pos: T(pos, [rows]), mask: T(buildMask(pos, cache, maskIO.dtype), [1, 1, rows, cache], maskIO.dtype === "bool" ? "bool" : void 0) };
    if (paramIO) {
      const p = new Int32Array(paramIO.shape.reduce((a, b) => a * b, 1));
      p[0] = pos[0];
      p[1] = pos[0] + valid;
      p[2] = pos[0] + valid;
      feed.param_tensor = T(p, paramIO.shape);
    }
    const out = await model.run(sig, feed);
    del(feed.embeddings);
    del(feed.input_pos);
    del(feed.mask);
    del(feed.param_tensor);
    if (inplace) return { st, logits: out.logits };
    const ns = {};
    for (const io of stateIO) {
      const t = out[io.name];
      if (!t) throw new Error(`litert: ${sig} did not return ${io.name}`);
      if (copyState && t.copyTo) {
        ns[io.name] = await t.copyTo("wasm");
        del(t);
      } else ns[io.name] = t;
    }
    freeState(st);
    return { st: ns, logits: out.logits };
  }
  async function readLogits(L) {
    const c = L.moveTo ? await L.moveTo("wasm") : L;
    const a = Float32Array.from(c.toTypedArray());
    if (c !== L) del(c);
    del(L);
    return a.subarray(a.length - manifest.vocab);
  }
  async function prefillTokens(ids, from, end, st) {
    if (opts.decodeOnlyPrefill) {
      for (let i = from; i < end; i++) {
        const r = await run("decode", await embed.rows([ids[i]]), 1, Int32Array.from([i]), st);
        st = r.st;
        del(r.logits);
      }
      return st;
    }
    const remDec = opts.remainderViaDecode ?? accelerator === "webgpu";
    for (const c of prefillChunks(end - from, P)) {
      if (remDec && c.len < P) {
        for (let i = from + c.start; i < from + c.start + c.len; i++) {
          const r = await run("decode", await embed.rows([ids[i]]), 1, Int32Array.from([i]), st);
          st = r.st;
          del(r.logits);
        }
        continue;
      }
      const toks = ids.slice(from + c.start, from + c.start + c.len);
      const emb = new Float32Array(P * H);
      emb.set(await embed.rows(toks));
      const pos = new Int32Array(P);
      for (let i = 0; i < c.len; i++) pos[i] = from + c.start + i;
      ({ st } = await run(prefillName, emb, P, pos, st, c.len));
    }
    return st;
  }
  async function decodeOne(tid, p, st) {
    const r = await run("decode", await embed.rows([tid]), 1, Int32Array.from([p]), st);
    return { st: r.st, logits: await readLogits(r.logits) };
  }
  async function lastLogitsFrom(ids, from, st) {
    st = await prefillTokens(ids, from, ids.length - 1, st);
    const r = await decodeOne(ids[ids.length - 1], ids.length - 1, st);
    freeState(r.st);
    return r.logits;
  }
  const noHidden = (want) => {
    if (want.positions.length) throw new Error('litert: this graph has no hidden-state output; use ask() with method "logits"');
  };
  const gate = createGate("litert");
  const snaps = /* @__PURE__ */ new WeakMap();
  const snapshotState = async (st) => Promise.all(stateIO.map(async (io) => {
    const t = st[io.name];
    const c = t.moveTo ? await t.moveTo("wasm") : t;
    const a = Float32Array.from(c.toTypedArray());
    if (c !== t) del(c);
    return a;
  }));
  const fromCopy = async (arrs) => {
    const st = {};
    stateIO.forEach((io, i) => {
      st[io.name] = T(arrs[i].slice(), io.shape);
    });
    if (inplace && accelerator === "webgpu") for (const k in st) st[k] = await st[k].moveTo("webgpu");
    return st;
  };
  return {
    kind: "litert",
    hiddenName: null,
    dispose() {
      return gate.dispose(() => model.delete?.());
    },
    async prefill(ids, want) {
      gate.assertLive();
      noHidden(want);
      if (!ids.length) throw new Error("prefill: empty ids");
      const rel = await gate.acquire();
      try {
        return { hiddens: [], lastLogits: await lastLogitsFrom(ids, 0, await freshState()) };
      } finally {
        rel();
      }
    },
    async snapshot(ids) {
      gate.assertLive();
      if (!ids.length) throw new Error("litert: snapshot of empty ids");
      const rel = await gate.acquire();
      try {
        const st = await prefillTokens(ids, 0, ids.length, await freshState());
        const h = { ids: ids.slice(), dispose() {
          snaps.delete(h);
        } };
        snaps.set(h, await snapshotState(st));
        freeState(st);
        return h;
      } finally {
        rel();
      }
    },
    async prefillFrom(h, ids, want) {
      gate.assertLive();
      noHidden(want);
      const arrs = snaps.get(h);
      if (!arrs) throw new Error("litert: unknown or released snapshot handle");
      const k = checkContinuation("litert", h, ids, want.positions);
      const rel = await gate.acquire();
      try {
        return { hiddens: [], lastLogits: await lastLogitsFrom(ids, k, await fromCopy(arrs)) };
      } finally {
        rel();
      }
    },
    async *generate(ids, o) {
      gate.assertLive();
      const g = resolveGenerateOptions(o), rand = mulberry32(g.seed);
      if (!ids.length) throw new Error("generate: empty ids");
      if (g.maxNewTokens === 0) return;
      const rel = await gate.acquire();
      let st = null;
      try {
        st = await prefillTokens(ids, 0, ids.length - 1, await freshState());
        let r = await decodeOne(ids[ids.length - 1], ids.length - 1, st);
        st = r.st;
        const gen = [];
        for (let s = 0, pos = ids.length; s < g.maxNewTokens; s++, pos++) {
          if (g.signal?.aborted || pos >= cache) break;
          const tid = sample(r.logits, gen, rand, g);
          if (EOS.has(tid)) break;
          gen.push(tid);
          yield tid;
          if (s + 1 >= g.maxNewTokens || g.signal?.aborted) break;
          r = await decodeOne(tid, pos, st);
          st = r.st;
        }
      } finally {
        if (st) freeState(st);
        rel();
      }
    }
  };
}

// src/runtime.ts
var DEFAULT_REPOS = {
  genai: "eduardoworrel/alias3-micro-1026-ONNX-genai",
  decodeOnly: "eduardoworrel/alias3-micro-1026-ONNX-decode-only",
  /** optimum/Scan graph (transformers.js). */
  onnx: "eduardoworrel/alias3-micro-1026-ONNX",
  mlcQ4f16: "eduardoworrel/alias3-micro-1026-MLC-q4f16_1",
  mlcQ4f32: "eduardoworrel/alias3-micro-1026-MLC-q4f32_1",
  /** NOT published yet (the LiteRT bundle waits for the owner's OK); pass `litertBaseUrl` to self-host. */
  litert: "eduardoworrel/alias3-micro-1026-LiteRT"
};
var DEFAULT_FILES = {
  head: "heads/head.bin",
  headScore: "heads/head_score.bin"
};
async function detectWebGPU(nav = globalThis.navigator) {
  try {
    const gpu = nav?.gpu;
    if (!gpu?.requestAdapter) return false;
    return !!await gpu.requestAdapter();
  } catch {
    return false;
  }
}
var RUNTIMES = ["genai", "decode-only", "transformers.js", "webllm", "litert"];
var DEVICES = ["webgpu", "wasm", "cpu", "cuda"];
var RUNTIME_LIMITS = {
  genai: { maxSafeTokens: 2048, byDevice: { webgpu: 7200 } },
  "decode-only": { maxSafeTokens: 2048, byDevice: { webgpu: 3600, cpu: 7200 } },
  "transformers.js": { maxSafeTokens: 2048 },
  webllm: { maxSafeTokens: 2048 },
  litert: { maxSafeTokens: 2e3 }
};
function maxSafeTokensFor(runtime, device) {
  const l = RUNTIME_LIMITS[runtime];
  return l.byDevice?.[device] ?? l.maxSafeTokens;
}
var MLC_Q4F32_LIB = "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/Qwen3.5-4B-q4f32_1_cs1k-webgpu.wasm";
var RUNTIME_DEFAULTS = {
  genai: {
    repo: DEFAULT_REPOS.genai,
    /** Graph per device; its data file is `<graph>.data`. */
    files: { webgpu: "webgpu/int4/model.onnx", cuda: "cuda/int4/model.onnx" },
    heads: true,
    methods: DEFAULT_METHOD
  },
  "decode-only": {
    repo: DEFAULT_REPOS.decodeOnly,
    /** Data file: `<graph>_data`. */
    file: "onnx/model_q4.onnx",
    heads: true,
    methods: DEFAULT_METHOD
  },
  "transformers.js": {
    repo: DEFAULT_REPOS.onnx,
    /** transformers.js dtype: loads `onnx/model_q4.onnx`. */
    dtype: "q4",
    heads: true,
    methods: DEFAULT_METHOD
  },
  webllm: {
    builds: {
      q4f16_1: {
        repo: DEFAULT_REPOS.mlcQ4f16,
        modelId: "alias3-micro-1026-q4f16_1",
        lib: "model-lib.wasm",
        recovery: { rows: "recovery/P_q4f16_rows.i32", P: "recovery/P_q4f16_8192.f16" }
      },
      q4f32_1: {
        repo: DEFAULT_REPOS.mlcQ4f32,
        modelId: "alias3-micro-1026-q4f32_1",
        lib: MLC_Q4F32_LIB,
        recovery: { rows: "recovery/P_q4f32_rows.i32", P: "recovery/P_q4f32_8192.f16" }
      }
    },
    /** q4f32_1: the faithful build (hidden cos 0.98 vs 0.86, keeps `<think>`, yes/no 54/65 vs 52/65); q4f16_1 is faster. */
    build: "q4f32_1",
    /** heads/ and recovery/ come from the build's own repo; the hidden state is recovered from the logits (h = P·logits[rows]),
     *  so the reads are the ONNX ones. If the recovery files fail to load, every mode falls back to the logits. */
    heads: true,
    methods: DEFAULT_METHOD,
    logitsOnlyMethods: { noul: "logits", choice: "logits", score: "logits" }
  },
  litert: {
    repo: DEFAULT_REPOS.litert,
    /** Bundle produced by litert-exp/rb_build.py: manifest + graph + streamed weights + fp16 embedding rows. */
    files: { manifest: "litert/manifest.json", graph: "litert/model.graph.tflite", weights: "litert/model.weights.bin", embed: "litert/embed.f16.bin" },
    /** @litertjs/core wasm files (pinned; the npm package's wasm/ directory). */
    wasm: "https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/",
    heads: false,
    methods: { noul: "logits", choice: "logits", score: "logits" }
  }
};
function defaultMethods(runtime, opts = {}) {
  if (runtime === "webllm" && !opts.webllmRecovery) return RUNTIME_DEFAULTS.webllm.logitsOnlyMethods;
  if (runtime === "litert") return RUNTIME_DEFAULTS.litert.methods;
  return DEFAULT_METHOD;
}
function resolveRuntimeChoice(o, fallback = "auto") {
  if (o.runtime === void 0) return { choice: fallback };
  if (o.runtime !== "auto" && !RUNTIMES.includes(o.runtime))
    throw new Error(`ct0-edge: unknown runtime "${o.runtime}". Use "auto", ${RUNTIMES.map((r) => `"${r}"`).join(", ")}.`);
  return { choice: o.runtime };
}
function detectPlatform(g = globalThis) {
  const isNode = !!g.process?.versions?.node;
  const hasDom = typeof g.window !== "undefined" && typeof g.document !== "undefined";
  const isWorker = typeof g.WorkerGlobalScope !== "undefined" && typeof g.importScripts === "function";
  return isNode && !hasDom && !isWorker ? "node" : "browser";
}
var GENAI_NODE_CPU_ERROR = 'ct0-edge: runtime "genai" does not run on onnxruntime-node CPU (the int4 graph fails with "not enough space: expected 49152, got 0"). In Node use runtime "decode-only" (CPU) or "transformers.js", or run genai on a CUDA machine (device "cuda").';
function checkDevice(platform, device) {
  if (!DEVICES.includes(device)) throw new Error(`ct0-edge: unknown device "${device}". Use "auto", ${DEVICES.map((d) => `"${d}"`).join(", ")}.`);
  if (platform === "browser" && device === "cuda") throw new Error('ct0-edge: device "cuda" is Node-only. In the browser use "webgpu" or "wasm".');
  if (platform === "node" && (device === "webgpu" || device === "wasm")) throw new Error(`ct0-edge: device "${device}" is browser-only. In Node use "cpu" or "cuda".`);
}
function deviceFor(runtime, want, caps) {
  const { platform } = caps;
  const dev = platform === "browser" && want === "cpu" ? "wasm" : want;
  if (dev !== "auto") checkDevice(platform, dev);
  const needGpu = (d) => {
    if (d === "webgpu" && !caps.webgpu) throw new Error(`ct0-edge: runtime "${runtime}" on WebGPU, but WebGPU is not available in this browser.`);
    return d;
  };
  const needCuda = (d) => {
    if (d === "cuda" && !caps.cuda) throw new Error(`ct0-edge: runtime "${runtime}" on CUDA, but onnxruntime-node lists no CUDA execution provider here.`);
    return d;
  };
  switch (runtime) {
    case "genai":
      if (platform === "browser") {
        if (dev === "wasm") throw new Error('ct0-edge: runtime "genai" needs WebGPU in the browser (its LinearAttention kernels have no WASM build). Use runtime "decode-only" on "wasm".');
        return needGpu("webgpu");
      }
      if (dev === "cpu") throw new Error(GENAI_NODE_CPU_ERROR);
      if (!caps.cuda) throw new Error(GENAI_NODE_CPU_ERROR + " No CUDA execution provider was found.");
      return "cuda";
    case "webllm":
      if (platform === "node") throw new Error('ct0-edge: runtime "webllm" runs only in the browser (WebGPU). In Node use "decode-only" or "transformers.js".');
      if (dev === "wasm") throw new Error('ct0-edge: runtime "webllm" needs WebGPU.');
      return needGpu("webgpu");
    case "litert":
      if (platform === "node") throw new Error('ct0-edge: runtime "litert" runs in the browser (LiteRT.js streams the weights into WebGPU buffers). In Node use "decode-only" or "transformers.js".');
      if (dev === "wasm") throw new Error('ct0-edge: runtime "litert" needs WebGPU (the CT0 weights exceed the 2 GiB LiteRT.js Wasm heap).');
      return needGpu("webgpu");
    case "decode-only":
    case "transformers.js":
      if (dev !== "auto") return platform === "browser" ? needGpu(dev) : needCuda(dev);
      return platform === "browser" ? caps.webgpu ? "webgpu" : "wasm" : "cpu";
  }
}
function planRuntime(choice, device = "auto", caps) {
  if (choice !== "auto") {
    if (!RUNTIMES.includes(choice)) throw new Error(`ct0-edge: unknown runtime "${String(choice)}".`);
    return [{ runtime: choice, device: deviceFor(choice, device, caps) }];
  }
  const { platform } = caps;
  const dev = platform === "browser" && device === "cpu" ? "wasm" : device;
  if (dev !== "auto") checkDevice(platform, dev);
  if (platform === "browser") {
    if (dev === "wasm") return [{ runtime: "decode-only", device: "wasm" }];
    if (dev === "webgpu" && !caps.webgpu) throw new Error('ct0-edge: device "webgpu" requested, but WebGPU is not available in this browser.');
    if (caps.webgpu) return [{ runtime: "genai", device: "webgpu" }, { runtime: "decode-only", device: "webgpu" }];
    return [{ runtime: "decode-only", device: "wasm" }];
  }
  const nodeCpu = [{ runtime: "transformers.js", device: "cpu" }, { runtime: "decode-only", device: "cpu" }];
  if (dev === "cpu") return nodeCpu;
  if (dev === "cuda") {
    if (!caps.cuda) throw new Error('ct0-edge: device "cuda" requested, but onnxruntime-node lists no CUDA execution provider here.');
    return [{ runtime: "genai", device: "cuda" }, { runtime: "transformers.js", device: "cuda" }];
  }
  if (caps.cuda) return [{ runtime: "genai", device: "cuda" }, { runtime: "transformers.js", device: "cuda" }, ...nodeCpu];
  return nodeCpu;
}
async function selectRuntime(choice, device, deps) {
  const status = deps.onStatus ?? (() => {
  });
  const plans = planRuntime(choice, device, deps.caps);
  if (choice !== "auto") return { ...plans[0], backend: await deps.load(plans[0]) };
  const reasons = [];
  if (plans[0].runtime !== "genai") {
    const { platform, webgpu, cuda } = deps.caps;
    reasons.push(platform === "browser" ? webgpu ? `device "${plans[0].device}" requested` : "WebGPU not available" : cuda ? `device "${plans[0].device}" requested` : "no CUDA execution provider (genai does not run on onnxruntime-node CPU)");
  }
  for (const [i, plan] of plans.entries()) {
    if (i > 0) status(`${reasons.at(-1)} -> ${plan.runtime} (${plan.device})`);
    try {
      const backend = await deps.load(plan);
      return { ...plan, backend, fallbackReason: reasons.length ? reasons.join("; ") : void 0 };
    } catch (e) {
      reasons.push(`${plan.runtime} (${plan.device}) unavailable: ${e?.message ?? String(e)}`);
      if (i === plans.length - 1) {
        const err = new Error(`ct0-edge: no runtime loaded. ${reasons.join("; ")}`);
        err.cause = e;
        throw err;
      }
    }
  }
  throw new Error("ct0-edge: no runtime candidate");
}
function packagesFor(runtime, platform) {
  const ort = platform === "node" ? "onnxruntime-node" : "onnxruntime-web";
  if (runtime === "genai" || runtime === "decode-only") return ["@huggingface/transformers", ort];
  if (runtime === "webllm") return ["@huggingface/transformers", "@mlc-ai/web-llm"];
  if (runtime === "litert") return ["@huggingface/transformers", "@litertjs/core"];
  return ["@huggingface/transformers"];
}
function isNotFound(e, pkg) {
  const msg = String(e?.message ?? e);
  if (e?.code === "ERR_MODULE_NOT_FOUND" || e?.code === "MODULE_NOT_FOUND" || /Cannot find (package|module)/.test(msg)) return msg.includes(pkg);
  return /Failed to resolve module specifier|Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed/i.test(msg);
}
async function importOptional(pkg, importer, why) {
  try {
    return await importer();
  } catch (e) {
    if (!isNotFound(e, pkg)) throw e;
    const err = new Error(`ct0-edge: ${why} needs the package "${pkg}", which is not installed. Install it: npm i ${pkg}`);
    err.cause = e;
    throw err;
  }
}
function detectCuda(ortNode) {
  try {
    return !!ortNode?.listSupportedBackends?.().some((b) => b?.name === "cuda");
  } catch {
    return false;
  }
}

// src/nodeFiles.ts
var nodeImport = (spec) => import(
  /* webpackIgnore: true */
  /* @vite-ignore */
  spec
);
async function defaultCacheDir() {
  const env = globalThis.process?.env ?? {};
  if (env.CT0_CACHE) return env.CT0_CACHE;
  const os = await nodeImport("node:os"), path = await nodeImport("node:path");
  return path.join(env.XDG_CACHE_HOME ?? path.join(os.homedir(), ".cache"), "ct0-edge");
}
async function localFile(url, cacheDir, onStatus = () => {
}) {
  const { fileURLToPath } = await nodeImport("node:url");
  if (url.startsWith("file:")) return fileURLToPath(url);
  if (!/^https?:/i.test(url)) return url;
  const fs = await nodeImport("node:fs"), path = await nodeImport("node:path");
  const u = new URL(url);
  const dest = path.join(cacheDir, u.hostname, ...decodeURIComponent(u.pathname).split("/").filter((s) => s && s !== ".."));
  if (fs.existsSync(dest)) return dest;
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const part = `${dest}.part-${globalThis.process?.pid ?? 0}`;
  const name = path.basename(dest);
  try {
    const res = await get(url);
    const total = Number(res.headers["content-length"]) || 0;
    let got = 0, shown = -1;
    res.on("data", (c) => {
      got += c.length;
      const pct = total ? Math.floor(100 * got / total / 10) * 10 : -1;
      if (pct !== shown && pct >= 0) {
        shown = pct;
        onStatus(`downloading ${name}: ${pct}% of ${(total / 1e9).toFixed(2)} GB`);
      }
    });
    const { pipeline } = await nodeImport("node:stream/promises");
    await pipeline(res, fs.createWriteStream(part));
    if (total && got !== total) throw new Error(`ct0-edge: download truncated: ${url} (${got} of ${total} bytes)`);
    fs.renameSync(part, dest);
  } catch (e) {
    try {
      fs.rmSync(part, { force: true });
    } catch {
    }
    throw e;
  }
  return dest;
}
async function get(url, hops = 0) {
  const mod = await nodeImport(url.startsWith("https:") ? "node:https" : "node:http");
  const res = await new Promise((resolve, reject) => mod.get(url, { headers: { "user-agent": "ct0-edge" } }, resolve).on("error", reject));
  if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location && hops < 10) {
    res.resume();
    return get(new URL(res.headers.location, url).href, hops + 1);
  }
  if (res.statusCode !== 200) {
    res.resume();
    throw new Error(`ct0-edge: download failed: ${url} -> HTTP ${res.statusCode}`);
  }
  return res;
}
async function readLocalBytes(url) {
  const { fileURLToPath } = await nodeImport("node:url");
  const fs = await nodeImport("node:fs");
  const b = fs.readFileSync(url.startsWith("file:") ? fileURLToPath(url) : url);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

// src/ct0.ts
var CT0 = class _CT0 {
  /** Nominal model context (linear recurrence). */
  maxContext = 262144;
  /** Longest prompt `ask()` accepts (default per runtime). */
  maxSafeTokens = 2048;
  tok;
  backend;
  yesIds = [];
  noIds = [];
  constructor(tok, backend) {
    this.tok = tok;
    this.backend = backend;
  }
  heads = {};
  methods = defaultMethods("decode-only");
  /** Runtime in use. */
  runtime;
  /** Device in use: "webgpu" | "wasm" | "cpu" | "cuda". */
  device;
  /** Why `"auto"` did not use genai (undefined when it did, or when the runtime was forced). */
  fallbackReason;
  /** Releases the graph / session. */
  dispose() {
    return this.backend.dispose();
  }
  /** Hidden-state output name of the graph (null when absent). */
  get hiddenOutput() {
    return this.backend.hiddenName;
  }
  /** `ask()` default read per mode for this runtime. */
  get defaultMethod() {
    return this.methods;
  }
  static async load(opts = {}) {
    const status = opts.onStatus ?? (() => {
    });
    const platform = detectPlatform();
    const { choice } = resolveRuntimeChoice(opts, opts.baseUrl && !opts.genaiBaseUrl && !opts.genaiRepo ? "decode-only" : "auto");
    const device = opts.device ?? "auto";
    const webgpu = platform === "browser" ? await detectWebGPU() : false;
    planRuntime(choice, device, { platform, webgpu, cuda: true });
    const mods = opts.modules ?? {};
    const needsOrt = choice === "auto" || choice === "genai" || choice === "decode-only";
    const loadOrt = async () => {
      if (mods.ort) return mods.ort;
      if (platform === "node") return importOptional("onnxruntime-node", () => import(
        /* webpackIgnore: true */
        /* @vite-ignore */
        NODE_ORT
      ), `runtime "${choice}" in Node`);
      return importOptional("onnxruntime-web", () => import("onnxruntime-web/webgpu"), `runtime "${choice}" in the browser`);
    };
    let ort = null;
    let cuda = false;
    if (platform === "node" && (needsOrt || device === "cuda")) {
      try {
        ort = await loadOrt();
        cuda = detectCuda(ort);
      } catch (e) {
        if (needsOrt) throw e;
      }
    }
    const caps = { platform, webgpu, cuda };
    status("loading tokenizer\u2026");
    const tjs = mods.transformers ?? await importOptional("@huggingface/transformers", () => import("@huggingface/transformers"), "ct0-edge (tokenizer)");
    const env = tjs.env;
    const pageBase = () => {
      if (typeof location !== "undefined") return location.href;
      const cwd = globalThis.process?.cwd?.();
      return cwd ? "file://" + encodeURI(cwd.replace(/\\/g, "/").replace(/^(?!\/)/, "/")) + "/" : void 0;
    };
    const absBase = (b) => new URL(b, pageBase()).href;
    const at = (p, b) => new URL(p, absBase(b)).href;
    const mlc = opts.mlc ?? {};
    const mlcBuild = RUNTIME_DEFAULTS.webllm.builds[mlc.build ?? RUNTIME_DEFAULTS.webllm.build];
    if (choice === "webllm" && !mlcBuild) throw new Error(`ct0-edge: unknown MLC build "${mlc.build}" (use "q4f32_1" or "q4f16_1")`);
    const repo = opts.repo ?? (choice === "webllm" ? mlcBuild.repo : opts.baseUrl ? void 0 : DEFAULT_REPOS.decodeOnly);
    let tok, base;
    if (repo) {
      env.allowRemoteModels = true;
      env.allowLocalModels = false;
      tok = await tjs.AutoTokenizer.from_pretrained(repo, { local_files_only: false });
      base = opts.baseUrl ?? `https://huggingface.co/${repo}/resolve/main/`;
    } else {
      base = absBase(opts.baseUrl);
      env.allowRemoteModels = false;
      env.allowLocalModels = true;
      const bu = new URL(base);
      if (bu.protocol === "file:") {
        const { fileURLToPath } = await import(
          /* webpackIgnore: true */
          /* @vite-ignore */
          NODE_URL
        );
        env.localModelPath = fileURLToPath(bu);
      } else env.localModelPath = typeof location !== "undefined" && bu.origin === location.origin ? bu.pathname : base;
      tok = await tjs.AutoTokenizer.from_pretrained("tok");
    }
    const genaiBase = opts.genaiBaseUrl ?? `https://huggingface.co/${opts.genaiRepo ?? DEFAULT_REPOS.genai}/resolve/main/`;
    const tjsRepo = opts.transformersRepo ?? RUNTIME_DEFAULTS["transformers.js"].repo;
    const tjsBase = `https://huggingface.co/${tjsRepo}/resolve/main/`;
    let cacheDir;
    const fileFor = async (url) => {
      if (platform !== "node") return url;
      cacheDir ??= opts.cacheDir ?? await defaultCacheDir();
      return localFile(url, cacheDir, status);
    };
    const loadDecodeOnly = async (dev) => {
      ort ??= await loadOrt();
      if (platform === "browser" && !opts.skipVersionCheck) checkOrtVersion(ort.env?.versions?.web);
      const file = opts.decodeModel ?? RUNTIME_DEFAULTS["decode-only"].file;
      const dataPath = opts.externalDataFile ?? file + "_data";
      const providers = opts.providers ?? [dev];
      const isLocal = /^(\.\/|\/)/.test(dataPath) || /^(\.\/|\/)/.test(file);
      const modelUrl = isLocal ? at(file, pageBase()) : at(file, base);
      const dataUrl = isLocal ? at(dataPath, pageBase()) : at(dataPath, base);
      status(`loading decode-only graph (${providers.join(",")})\u2026`);
      let S;
      if (platform === "node") {
        await fileFor(dataUrl);
        S = await ort.InferenceSession.create(await fileFor(modelUrl), { executionProviders: providers, logSeverityLevel: 3 });
      } else {
        S = await ort.InferenceSession.create(modelUrl, {
          executionProviders: providers,
          externalData: [{ path: dataPath.split("/").pop(), data: dataUrl }]
        });
      }
      try {
        return await createDecodeOnly(S, tok, { ort, pastLen: opts.pastLen, skipVersionCheck: opts.skipVersionCheck });
      } catch (e) {
        try {
          await S.release?.();
        } catch {
        }
        throw e;
      }
    };
    const loadGenai = async (dev) => {
      ort ??= await loadOrt();
      const file = opts.genaiModel ?? (dev === "cuda" ? RUNTIME_DEFAULTS.genai.files.cuda : RUNTIME_DEFAULTS.genai.files.webgpu);
      status(`loading genai int4 graph (${dev})\u2026`);
      if (platform === "browser")
        return createGenai(
          at(file, genaiBase),
          at(file + ".data", genaiBase),
          { ...opts.genai, verifyEp: opts.verifyEp ?? opts.genai?.verifyEp ?? false, ort, skipVersionCheck: opts.skipVersionCheck ?? opts.genai?.skipVersionCheck }
        );
      await fileFor(at(file + ".data", genaiBase));
      const S = await ort.InferenceSession.create(await fileFor(at(file, genaiBase)), { executionProviders: [dev], graphOptimizationLevel: "all", logSeverityLevel: 3 });
      try {
        const be = genaiFromSession(S, ort, opts.genai);
        if (opts.genai?.warmup ?? true) await be.prefill([248045], { positions: [] });
        return be;
      } catch (e) {
        try {
          await S.release?.();
        } catch {
        }
        throw e;
      }
    };
    const loadTransformers = async (dev) => {
      const dtype = opts.dtype ?? RUNTIME_DEFAULTS["transformers.js"].dtype;
      status(`loading transformers.js graph (${dtype}, ${dev})\u2026`);
      env.allowRemoteModels = true;
      env.allowLocalModels = false;
      if (platform === "node" && opts.cacheDir) env.cacheDir = opts.cacheDir;
      const model = await tjs.AutoModelForCausalLM.from_pretrained(tjsRepo, { dtype, device: dev });
      return transformersJsFromModel(model, tjs.Tensor);
    };
    let mlcBase = null, mlcNote, webllmRecovery = false;
    const loadWebllm = async () => {
      const webllm = mods.webllm ?? await importOptional("@mlc-ai/web-llm", () => import(
        /* webpackIgnore: true */
        /* @vite-ignore */
        WEBLLM_PKG
      ), 'runtime "webllm"');
      const build = mlcBuild;
      const modelUrl = mlc.modelUrl ?? `https://huggingface.co/${build.repo}/`;
      const modelLib = mlc.modelLib ?? (/^https?:/.test(build.lib) ? build.lib : new URL("resolve/main/" + build.lib, modelUrl).href);
      mlcBase = new URL("resolve/main/", modelUrl).href;
      let recovery = mlc.recovery || void 0;
      if (!recovery && mlc.recovery !== false) {
        status("loading WebLLM hidden-state recovery\u2026");
        const files = mlc.recoveryUrls ?? build.recovery;
        const r = await tryLoadRecovery(mlcBase, files);
        recovery = r.recovery;
        if (!recovery) {
          mlcNote = `webllm recovery unavailable (${r.reason}): every question type reads the logits`;
          status(mlcNote);
        }
      }
      webllmRecovery = !!recovery;
      status(`loading WebLLM build ${mlc.build ?? RUNTIME_DEFAULTS.webllm.build}\u2026`);
      return createWebllm(webllm, {
        modelId: mlc.modelId ?? build.modelId,
        modelUrl,
        libUrl: modelLib,
        appConfig: mlc.appConfig,
        tok,
        recovery,
        initProgressCallback: mlc.initProgressCallback
      });
    };
    const loadLitert = async (dev) => {
      const lrt = mods.litert ?? await importOptional("@litertjs/core", () => import(
        /* webpackIgnore: true */
        /* @vite-ignore */
        LITERT_PKG
      ), 'runtime "litert"');
      const D = RUNTIME_DEFAULTS.litert;
      const lbase = absBase(opts.litertBaseUrl ?? `https://huggingface.co/${opts.litertRepo ?? D.repo}/resolve/main/litert/`);
      const name = (f) => f.replace(/^litert\//, "");
      status(`loading LiteRT.js ${LITERTJS_VERSION} runtime\u2026`);
      try {
        await lrt.loadLiteRt(opts.litertWasmUrl ?? D.wasm, { jspi: !!globalThis.WebAssembly?.Suspending });
      } catch (e) {
        if (!/already loading|already loaded/i.test(String(e?.message ?? e))) throw e;
      }
      const manifest = await (await fetch(at(name(D.files.manifest), lbase))).json();
      const graph = new Uint8Array(await (await fetch(at(name(D.files.graph), lbase))).arrayBuffer());
      const weightsUrl = at(name(D.files.weights), lbase);
      status("streaming LiteRT weights into WebGPU\u2026");
      return createLiteRt(lrt, graph, manifest, embedFromF16Reader(httpRangeReader(at(name(D.files.embed), lbase)), manifest.hidden), {
        accelerator: dev === "webgpu" ? "webgpu" : "wasm",
        gpuPrecision: opts.litertPrecision,
        weights: manifest.externalized === false ? void 0 : () => {
          let body = null;
          return new ReadableStream({
            async pull(c) {
              body ??= (await fetch(weightsUrl)).body.getReader();
              const { done, value } = await body.read();
              if (done) c.close();
              else c.enqueue(value);
            },
            cancel() {
              body?.cancel();
            }
          });
        }
      });
    };
    const load = (p) => {
      switch (p.runtime) {
        case "genai":
          return loadGenai(p.device);
        case "decode-only":
          return loadDecodeOnly(p.device);
        case "transformers.js":
          return loadTransformers(p.device);
        case "webllm":
          return loadWebllm();
        case "litert":
          return loadLitert(p.device);
      }
    };
    const sel = await selectRuntime(choice, device, { caps, load, onStatus: status });
    const ct0 = new _CT0(tok, sel.backend);
    ct0.runtime = sel.runtime;
    ct0.device = sel.device;
    ct0.fallbackReason = [sel.fallbackReason, sel.runtime === "webllm" ? mlcNote : void 0].filter(Boolean).join("; ") || void 0;
    ct0.methods = defaultMethods(sel.runtime, { webllmRecovery });
    ct0.maxSafeTokens = opts.maxSafeTokens ?? maxSafeTokensFor(sel.runtime, sel.device);
    status(`runtime: ${sel.runtime} (${sel.device})`);
    const headsBase = sel.runtime === "genai" ? genaiBase : sel.runtime === "decode-only" ? base : sel.runtime === "transformers.js" ? tjsBase : sel.runtime === "webllm" && webllmRecovery ? mlcBase : null;
    if ((opts.heads ?? true) && headsBase) {
      try {
        await ct0.loadHeads({ noul: at(DEFAULT_FILES.head, headsBase), score: at(DEFAULT_FILES.headScore, headsBase) });
      } catch (e) {
        status("heads not loaded (" + (e?.message ?? e) + "); ask() with method head will throw");
      }
    }
    const firstId = (s) => {
      const a = ct0.ids(s);
      return a.length ? a[0] : null;
    };
    ct0.yesIds = [...new Set(["Yes", " Yes", "yes", " yes", "YES"].map(firstId).filter((x) => x != null))];
    ct0.noIds = [...new Set(["No", " No", "no", " no", "NO"].map(firstId).filter((x) => x != null))];
    status("ready.");
    return ct0;
  }
  ids(text) {
    return this.tok(text, { add_special_tokens: false }).input_ids.tolist()[0].map(Number);
  }
  /** Counts the tokens of a text. */
  countTokens(text) {
    return this.ids(text).length;
  }
  /** Truncates `text` to fit `maxSafeTokens - reserve` (keeps the start + end,
   *  where the information usually is). Calls `onTrim` if it cuts. */
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
   * chunk-watch: scans a LARGE text (over maxSafeTokens, e.g. huge HTML) in sliding windows,
   * running System 1 (`decide`) per chunk and accumulating the suspicious regions.
   * Covers documents that do not fit the context without truncating or losing signal.
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
  /** Clean, condensed view of a page (for security agents): forms, inputs, buttons, links,
   *  HTML comments, inline scripts and localStorage, where vulnerabilities/secrets tend to hide, in little context. */
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
      // notable visible text (stack traces, error/debug banners), where credentials tend to leak
      texts: q("pre,code,[class*='debug'],[class*='error'],[class*='stack']", (e) => (e.textContent || "").trim().slice(0, 300)).filter(Boolean),
      comments,
      inlineScripts: q("script:not([src])", (s) => s.textContent.trim().slice(0, 220)).filter(Boolean),
      localStorage: store
    };
  }
  /** Raw body HTML (fallback when a security review needs the full markup). */
  static pageSource(doc, max = 2500) {
    return (doc.body || doc.documentElement).outerHTML.slice(0, max);
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
    for await (const tid of this.backend.generate(this.ids(prompt), opts)) yield this.tok.decode([tid], { skip_special_tokens: true });
  }
  async decide(state, question) {
    const sys = "You are a security decision function. Given a STATE and a yes/no QUESTION, answer with exactly one word: Yes or No.";
    const text = "<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\nSTATE:\n" + state + "\n\nQUESTION: " + question + "\nAnswer Yes or No.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const { lastLogits: L } = await this.backend.prefill(this.ids(text), { positions: [] });
    const all = [...this.yesIds, ...this.noIds];
    if (!all.length) return NaN;
    const mx = Math.max(...all.map((i) => L[i]));
    const Z = all.reduce((a, i) => a + Math.exp(L[i] - mx), 0);
    return this.yesIds.reduce((a, i) => a + Math.exp(L[i] - mx), 0) / Z;
  }
  /**
   * Loads the bilinear heads used by `ask(..., {method:"head"})`. Each source is an ArrayBuffer
   * (the .bin contents) or a URL. `noul` and `choice` come from the same original head.bin (if both
   * are given, `noul` wins); `score` is head_score.bin (required for score with method head; no fallback).
   */
  async loadHeads(src) {
    const get2 = async (x) => {
      if (typeof x !== "string") return loadHead(x);
      if (x.startsWith("file:")) return loadHead(await readLocalBytes(x));
      const r = await fetch(x);
      if (!r.ok) throw new Error(`loadHeads: ${x} -> HTTP ${r.status}`);
      return loadHead(await r.arrayBuffer());
    };
    const nc = src.noul ?? src.choice;
    if (nc) this.heads.noulChoice = await get2(nc);
    if (src.score) this.heads.score = await get2(src.score);
  }
  /** Typed System-1 (noul / choice / score). Default read per mode: `ct0.defaultMethod` (ONNX runtimes: noul logits,
   *  choice head, score head_score.bin; webllm too, through recovery, else logits). `opts.method` overrides it. */
  ask(q, opts = {}) {
    return ask(this.backend, this.tok, this.heads, q, this.askOpts(q, opts));
  }
  /** Several questions about one `state` (each question may omit `state`): same results as `ask()` for each, in order.
   *  genai and decode-only prefill the shared prefix (system turn + state) once and run only each question's suffix;
   *  transformers.js and webllm make independent `ask()` calls. */
  askMany(state, qs, opts = {}) {
    return askManyWith(this.backend, this.tok, this.heads, state, qs, (q) => this.askOpts(q, opts));
  }
  askOpts(q, opts) {
    return { ...opts, method: opts.method ?? this.methods[q.kind], maxSafeTokens: opts.maxSafeTokens ?? this.maxSafeTokens };
  }
  /**
   * Full prefill returning the hidden states the bilinear HEAD needs.
   * Requires a graph with a hidden-state output (`backend.hiddenName`). Positions come from
   * `positions.ts`: q at the '>' of `<decide>`, k at the last token and, for
   * choice, at the token covering the end of each `</opt>`.
   */
  async prefillHiddens(prompt) {
    if (!this.backend.hiddenName) return null;
    const t = tokenOffsets(this.tok, prompt);
    const dp = decidePos(t.tokens), lp = t.ids.length - 1, op = optPositionsFor(prompt, t.offsets);
    const { hiddens } = await this.backend.prefill(t.ids, { positions: [dp, lp, ...op] });
    return { atDecide: hiddens[0], atLast: hiddens[1], atOpts: hiddens.slice(2) };
  }
};
var NODE_ORT = "onnxruntime-node";
var NODE_URL = "node:url";
var WEBLLM_PKG = "@mlc-ai/web-llm";
var LITERT_PKG = "@litertjs/core";
export {
  CT0,
  DEFAULT_FILES,
  DEFAULT_METHOD,
  DEFAULT_REPOS,
  GENAI_NODE_CPU_ERROR,
  GENERATE_DEFAULTS,
  MLC_Q4F32_LIB,
  RUNTIMES,
  RUNTIME_DEFAULTS,
  RUNTIME_LIMITS,
  askMany,
  defaultMethods,
  detectCuda,
  detectPlatform,
  detectWebGPU,
  importOptional,
  levelFromCuts,
  maxSafeTokensFor,
  packagesFor,
  planRuntime,
  resolveGenerateOptions,
  resolveRuntimeChoice,
  selectRuntime
};
