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

// src/backends/litert.ts
var LITERTJS_VERSION = "2.6.0";
var LITERT_WARNING = `ct0-edge: runtime "litert" is EXPERIMENTAL (LiteRT.js @litertjs/core ${LITERTJS_VERSION}, WebGPU with streamed weights; WASM only for small inline graphs).`;
function embedFromQ8(q8, scale, hidden) {
  const q = q8 instanceof Int8Array ? q8 : new Int8Array(q8.buffer, q8.byteOffset, q8.byteLength);
  return {
    async rows(ids) {
      const out = new Float32Array(ids.length * hidden);
      ids.forEach((id, r) => {
        const s = scale[id], o = id * hidden;
        for (let j = 0; j < hidden; j++) out[r * hidden + j] = q[o + j] * s;
      });
      return out;
    }
  };
}
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
function embedFromReader(read, scale, hidden) {
  return {
    async rows(ids) {
      const out = new Float32Array(ids.length * hidden);
      const uniq = [...new Set(ids)], got = /* @__PURE__ */ new Map();
      await Promise.all(uniq.map(async (id) => {
        const b = await read(id * hidden, hidden);
        got.set(id, new Int8Array(b.buffer, b.byteOffset, hidden));
      }));
      ids.forEach((id, r) => {
        const q = got.get(id), s = scale[id];
        for (let j = 0; j < hidden; j++) out[r * hidden + j] = q[j] * s;
      });
      return out;
    }
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
  const hiddenOut = (dec.outputs.find((o) => o.name === "hidden") ?? dec.outputs.find((o) => /(^|[_/])hidden(_states)?$/.test(o.name)))?.name ?? null;
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
    if (inplace) return { st, logits: out.logits, hidden: hiddenOut ? out[hiddenOut] : void 0 };
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
    return { st: ns, logits: out.logits, hidden: hiddenOut ? out[hiddenOut] : void 0 };
  }
  async function readLogits(L) {
    const c = L.moveTo ? await L.moveTo("wasm") : L;
    const a = Float32Array.from(c.toTypedArray());
    if (c !== L) del(c);
    del(L);
    return a.subarray(a.length - manifest.vocab);
  }
  async function collectHidden(hid, firstPos, rows, want, at) {
    if (!hid || !want.size) {
      del(hid);
      return;
    }
    const c = hid.moveTo ? await hid.moveTo("wasm") : hid;
    const a = c.toTypedArray();
    for (let r = 0; r < rows; r++) {
      const p = firstPos + r;
      if (want.has(p)) at.set(p, Float32Array.from(a.subarray(r * H, r * H + H)));
    }
    if (c !== hid) del(c);
    del(hid);
  }
  async function prefillTokens(ids, from, end, st, want, at) {
    const grab = async (r, firstPos, rows) => {
      if (want && at) await collectHidden(r.hidden, firstPos, rows, want, at);
      else del(r.hidden);
    };
    if (opts.decodeOnlyPrefill) {
      for (let i = from; i < end; i++) {
        const r = await run("decode", await embed.rows([ids[i]]), 1, Int32Array.from([i]), st);
        st = r.st;
        del(r.logits);
        await grab(r, i, 1);
      }
      return st;
    }
    const remDec = opts.remainderViaDecode ?? accelerator === "webgpu";
    for (const c of prefillChunks(end - from, P)) {
      if (remDec && c.len < P) {
        for (let i = from + c.start; i < from + c.start + c.len; i++) {
          const r2 = await run("decode", await embed.rows([ids[i]]), 1, Int32Array.from([i]), st);
          st = r2.st;
          del(r2.logits);
          await grab(r2, i, 1);
        }
        continue;
      }
      const toks = ids.slice(from + c.start, from + c.start + c.len);
      const emb = new Float32Array(P * H);
      emb.set(await embed.rows(toks));
      const pos = new Int32Array(P);
      for (let i = 0; i < c.len; i++) pos[i] = from + c.start + i;
      const r = await run(prefillName, emb, P, pos, st, c.len);
      st = r.st;
      await grab(r, from + c.start, c.len);
    }
    return st;
  }
  async function decodeOne(tid, p, st) {
    const r = await run("decode", await embed.rows([tid]), 1, Int32Array.from([p]), st);
    return { st: r.st, logits: await readLogits(r.logits), hidden: r.hidden };
  }
  async function feedFrom(ids, from, st, positions) {
    const want = new Set(positions);
    const at = /* @__PURE__ */ new Map();
    const last = ids.length - 1;
    st = await prefillTokens(ids, from, last, st, want, at);
    const r = await decodeOne(ids[last], last, st);
    await collectHidden(r.hidden, last, 1, want, at);
    freeState(r.st);
    return { lastLogits: r.logits, hiddens: positions.map((p) => at.get(p)) };
  }
  const checkHidden = (want) => {
    if (want.positions.length && !hiddenOut) throw new Error('litert: this graph has no hidden-state output; use ask() with method "logits"');
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
    hiddenName: hiddenOut,
    dispose() {
      return gate.dispose(() => model.delete?.());
    },
    async prefill(ids, want) {
      gate.assertLive();
      checkHidden(want);
      if (!ids.length) throw new Error("prefill: empty ids");
      const rel = await gate.acquire();
      try {
        return await feedFrom(ids, 0, await freshState(), want.positions);
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
      checkHidden(want);
      const arrs = snaps.get(h);
      if (!arrs) throw new Error("litert: unknown or released snapshot handle");
      const k = checkContinuation("litert", h, ids, want.positions);
      const rel = await gate.acquire();
      try {
        return await feedFrom(ids, k, await fromCopy(arrs), want.positions);
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
export {
  LITERTJS_VERSION,
  LITERT_WARNING,
  buildMask,
  createLiteRt,
  embedFromF16Reader,
  embedFromQ8,
  embedFromReader,
  httpRangeReader,
  prefillChunks
};
