// src/ortVersion.ts
function checkOrtVersion(v, min = [1, 30]) {
  const m = /^(\d+)\.(\d+)/.exec(v ?? "");
  if (!m) return;
  if (Number(m[1]) < min[0] || Number(m[1]) === min[0] && Number(m[2]) < min[1])
    throw new Error(`ct0-edge: onnxruntime-web ${v} is too old; requires >= ${min.join(".")} (WebGPU returns NaN below that; the genai backend needs the LinearAttention/CausalConvWithState kernels). Pass skipVersionCheck to ignore.`);
}

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
export {
  assertWebGpuPlacement,
  checkOrtVersion,
  createGenai,
  genaiFromSession,
  hashStr,
  parsePlacement
};
