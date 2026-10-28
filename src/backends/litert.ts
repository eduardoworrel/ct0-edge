/**
 * litert.ts — EXPERIMENTAL backend over LiteRT.js (`@litertjs/core`, pinned 2.6.0) for a CT0 `.tflite` exported with
 * litert-torch (Gated-DeltaNet decomposed, `gdn_mode=1`), packaged by `rb_build.py`:
 *
 *   manifest.json        signatures (prefill_<N>, decode), inputs/outputs, state tensors, file sizes + sha256
 *   model.graph.tflite   FlatBuffer; FullyConnected weights moved to LiteRT external buffers (group "tflite_weights")
 *   model.weights.bin    the external weight blob, streamed straight into WebGPU buffers (`loadModelAndWeights`)
 *   embed.q8.bin/.f32    token embedding rows (int8 per row + fp32 scales) looked up here, outside the graph
 *
 * State is owned by this backend (conv / recurrent / KV tensors in, updated tensors out), so every `prefill`
 * starts from zeros: there is no cross-call state leak (unlike LiteRT-LM sessions, issue #3165).
 * The graph has no hidden-state output: `hiddenName` is null, `ask()` reads logits only.
 * Prompts are fed like LiteRT-LM does: all tokens but the last through `prefill_<N>` (right-padded; pads get
 * zero embeddings and position 0, which the graph masks out), then the last token through `decode` -> logits.
 */
import { createGate, resolveGenerateOptions, checkContinuation, type Backend, type GenerateOptions, type StateHandle } from "../backend.js";
import { EOS, mulberry32, sample } from "./decodeOnly.js";

export const LITERTJS_VERSION = "2.6.0";
export const LITERT_WARNING = `ct0-edge: runtime "litert" is EXPERIMENTAL (LiteRT.js @litertjs/core ${LITERTJS_VERSION}, WebGPU with streamed weights; WASM only for small inline graphs).`;

export interface TensorIO { name: string; shape: number[]; dtype: string }
export interface LiteRtManifest {
  format: string; vocab: number; hidden: number;
  signatures: Record<string, { inputs: TensorIO[]; outputs: TensorIO[] }>;
  files?: Record<string, { bytes: number; sha256: string }>;
}
/** Token-embedding rows: `rows(ids)` returns ids.length x hidden floats. */
export interface EmbedSource { rows(ids: number[]): Promise<Float32Array> }
/** The subset of `@litertjs/core` this backend uses (injected so tests can fake it). */
export interface LiteRtJs {
  Tensor: new (data: any, shape: number[], dtype?: string) => any;
  loadAndCompile(model: Uint8Array | string, opts: { accelerator: string; gpuOptions?: { precision?: string } }): Promise<any>;
  loadModelAndWeights?(model: Uint8Array, weights: ReadableStream<Uint8Array>, opts: { accelerator: string; gpuOptions?: { precision?: string } }): Promise<any>;
}
export interface LiteRtOptions {
  accelerator?: "webgpu" | "wasm";
  /** Stream of model.weights.bin (required with WebGPU when the graph is externalized). */
  weights?: () => ReadableStream<Uint8Array>;
  /** WebGPU compute precision (LiteRT.js `gpuOptions.precision`). Default "fp32". */
  gpuPrecision?: "fp16" | "fp32";
  /** Copy state outputs before feeding them back (LiteRT.js may reuse output buffers on the next run). Default: true on WebGPU. */
  copyState?: boolean;
  /** Tokens that do not fill a whole prefill chunk go through `decode` one by one (no padded chunks).
   *  Default: true on WebGPU, where padded chunks on top of a non-zero state are wrong in LiteRT.js 2.6.0. */
  remainderViaDecode?: boolean;
  /** Debug: feed prompt tokens one by one through `decode` instead of `prefill_<N>`. */
  decodeOnlyPrefill?: boolean;
  /** Silence the experimental warning. */
  quiet?: boolean;
}

/** Embedding source over in-memory int8 rows + per-row fp32 scales. */
export function embedFromQ8(q8: Int8Array | Uint8Array, scale: Float32Array, hidden: number): EmbedSource {
  const q = q8 instanceof Int8Array ? q8 : new Int8Array(q8.buffer, q8.byteOffset, q8.byteLength);
  return {
    async rows(ids) {
      const out = new Float32Array(ids.length * hidden);
      ids.forEach((id, r) => { const s = scale[id], o = id * hidden; for (let j = 0; j < hidden; j++) out[r * hidden + j] = q[o + j] * s; });
      return out;
    },
  };
}
/** fp16 bits -> number. */
function h2f(h: number): number {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  return e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
}
/** Embedding source over fp16 rows read on demand (`read(byteOffset, byteLength)`: HTTP Range, fs, or a buffer). */
export function embedFromF16Reader(read: (offset: number, length: number) => Promise<Uint8Array>, hidden: number): EmbedSource {
  return {
    async rows(ids) {
      const out = new Float32Array(ids.length * hidden), got = new Map<number, Uint16Array>();
      await Promise.all([...new Set(ids)].map(async (id) => {
        const b = await read(id * hidden * 2, hidden * 2);
        const c = b.byteOffset % 2 ? b.slice() : b;
        got.set(id, new Uint16Array(c.buffer, c.byteOffset, hidden));
      }));
      ids.forEach((id, r) => { const q = got.get(id)!; for (let j = 0; j < hidden; j++) out[r * hidden + j] = h2f(q[j]); });
      return out;
    },
  };
}
/** `read(offset, length)` over HTTP Range requests (falls back to slicing a full response if the server ignores Range). */
export function httpRangeReader(url: string, fetchFn: typeof fetch = fetch): (offset: number, length: number) => Promise<Uint8Array> {
  return async (offset, length) => {
    const r = await fetchFn(url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
    if (!r.ok) throw new Error(`litert: ${url} -> HTTP ${r.status}`);
    const b = new Uint8Array(await r.arrayBuffer());
    return r.status === 206 ? b : b.subarray(offset, offset + length);
  };
}
/** Embedding source that reads only the needed rows (`read(offset, length)`, e.g. HTTP Range or fs). */
export function embedFromReader(read: (offset: number, length: number) => Promise<Uint8Array>, scale: Float32Array, hidden: number): EmbedSource {
  return {
    async rows(ids) {
      const out = new Float32Array(ids.length * hidden);
      const uniq = [...new Set(ids)], got = new Map<number, Int8Array>();
      await Promise.all(uniq.map(async (id) => { const b = await read(id * hidden, hidden); got.set(id, new Int8Array(b.buffer, b.byteOffset, hidden)); }));
      ids.forEach((id, r) => { const q = got.get(id)!, s = scale[id]; for (let j = 0; j < hidden; j++) out[r * hidden + j] = q[j] * s; });
      return out;
    },
  };
}

const MASK_OFF = (globalThis as any).__CT0_MASK_OFF ?? -1e30;
/** Additive (or boolean) attention mask [1,1,rows,cache]: row r sees cache columns <= pos[r]. */
export function buildMask(pos: Int32Array, cache: number, dtype: string): Float32Array | Uint8Array {
  const rows = pos.length;
  if (dtype === "bool") { const m = new Uint8Array(rows * cache); for (let r = 0; r < rows; r++) m.fill(1, r * cache, r * cache + pos[r] + 1); return m; }
  const m = new Float32Array(rows * cache).fill(MASK_OFF);
  for (let r = 0; r < rows; r++) m.fill(0, r * cache, r * cache + pos[r] + 1);
  return m;
}

/** Splits `n` prompt tokens into prefill chunks of `P` (the last one right-padded). */
export function prefillChunks(n: number, P: number): Array<{ start: number; len: number }> {
  const out = [];
  for (let s = 0; s < n; s += P) out.push({ start: s, len: Math.min(P, n - s) });
  return out;
}

const isState = (n: string) => n.startsWith("kv_cache_");
const zerosOf = (io: TensorIO) => { const n = io.shape.reduce((a, b) => a * b, 1); return io.dtype === "int32" ? new Int32Array(n) : new Float32Array(n); };

export async function createLiteRt(lrt: LiteRtJs, graph: Uint8Array, manifest: LiteRtManifest, embed: EmbedSource, opts: LiteRtOptions = {}): Promise<Backend> {
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
  // GPU-composite exports (litert-torch apply_gpu_composites=True) update conv/recurrent/KV state IN PLACE and take an
  // int32 param tensor [start, end, end, 0...] (LiteRT-LM FillSingleBufferCacheParamTensor). Their signatures have no
  // state outputs, so the state tensors are created on the GPU once per sequence and passed to every run.
  const paramIO = dec.inputs.find((i) => i.name === "param_tensor");
  const inplace = !!paramIO;
  // Post-norm hidden-state output (patch: litert-torch `output_hidden`). When present, ask() can read the
  // trained System-1 heads (head.bin / head_score.bin) from it, like the ONNX/genai runtimes. Absent -> logits only.
  const hiddenOut = (dec.outputs.find((o) => o.name === "hidden") ?? dec.outputs.find((o) => /(^|[_/])hidden(_states)?$/.test(o.name)))?.name ?? null;
  const gpuOptions = { precision: opts.gpuPrecision ?? "fp32" };
  const copyState = opts.copyState ?? accelerator === "webgpu";
  const model = accelerator === "webgpu" && opts.weights
    // LiteRT.js 2.6.0 runs one weight-loading session per WebGPU partition and each session reads the stream from
    // offset 0 (`streamOffset = 0` per callback), but it only ever calls `getReader()` on it. Hand it an object whose
    // getReader() opens a FRESH stream every time, so partially-delegated graphs load (one stream would end early:
    // "Stream ended before all weights were loaded").
    ? await lrt.loadModelAndWeights!(graph, { getReader: () => opts.weights!().getReader() } as unknown as ReadableStream<Uint8Array>, { accelerator: "webgpu", gpuOptions })
    : await lrt.loadAndCompile(graph, accelerator === "webgpu" ? { accelerator, gpuOptions } : { accelerator });

  const T = (data: any, shape: number[], dtype?: string) => (dtype ? new lrt.Tensor(data, shape, dtype) : new lrt.Tensor(data, shape));
  const del = (t: any) => { try { t?.delete?.(); } catch { /* already freed */ } };
  const zeroStateHost = () => { const st: Record<string, any> = {}; for (const io of stateIO) st[io.name] = T(zerosOf(io), io.shape); return st; };
  async function freshState() {
    const st = zeroStateHost();
    if (inplace && accelerator === "webgpu") for (const k in st) { const g = await st[k].moveTo("webgpu"); st[k] = g; }
    return st;
  }
  const freeState = (st: Record<string, any>) => { for (const k in st) del(st[k]); };

  async function run(sig: string, emb: Float32Array, rows: number, pos: Int32Array, st: Record<string, any>, valid = rows) {
    const feed: Record<string, any> = { ...st, embeddings: T(emb, [1, rows, H]), input_pos: T(pos, [rows]), mask: T(buildMask(pos, cache, maskIO!.dtype), [1, 1, rows, cache], maskIO!.dtype === "bool" ? "bool" : undefined) };
    if (paramIO) { const p = new Int32Array(paramIO.shape.reduce((a, b) => a * b, 1)); p[0] = pos[0]; p[1] = pos[0] + valid; p[2] = pos[0] + valid; feed.param_tensor = T(p, paramIO.shape); }
    const out = await model.run(sig, feed);
    del(feed.embeddings); del(feed.input_pos); del(feed.mask); del(feed.param_tensor);
    if (inplace) return { st, logits: out.logits, hidden: hiddenOut ? out[hiddenOut] : undefined };
    const ns: Record<string, any> = {};
    for (const io of stateIO) {
      const t = out[io.name];
      if (!t) throw new Error(`litert: ${sig} did not return ${io.name}`);
      if (copyState && t.copyTo) { ns[io.name] = await t.copyTo("wasm"); del(t); } else ns[io.name] = t;
    }
    freeState(st);
    return { st: ns, logits: out.logits, hidden: hiddenOut ? out[hiddenOut] : undefined };
  }
  async function readLogits(L: any): Promise<Float32Array> {
    const c = L.moveTo ? await L.moveTo("wasm") : L;
    const a = Float32Array.from(c.toTypedArray());
    if (c !== L) del(c);
    del(L);
    return a.subarray(a.length - manifest.vocab);
  }
  /** Reads rows [rowStart..rowStart+rows) of a `hidden` output ([1, rows, H]) into `at` for the wanted global
   *  positions. `firstPos` is the global position of row 0. */
  async function collectHidden(hid: any, firstPos: number, rows: number, want: Set<number>, at: Map<number, Float32Array>) {
    if (!hid || !want.size) { del(hid); return; }
    const c = hid.moveTo ? await hid.moveTo("wasm") : hid;
    const a = c.toTypedArray();
    for (let r = 0; r < rows; r++) { const p = firstPos + r; if (want.has(p)) at.set(p, Float32Array.from(a.subarray(r * H, r * H + H))); }
    if (c !== hid) del(c);
    del(hid);
  }
  /** Feeds ids[from..end) through prefill chunks (or decode for a short remainder) on top of `st`.
   *  Collects the `hidden` row for every global position in `want` into `at`. */
  async function prefillTokens(ids: number[], from: number, end: number, st: Record<string, any>, want?: Set<number>, at?: Map<number, Float32Array>) {
    const grab = async (r: any, firstPos: number, rows: number) => { if (want && at) await collectHidden(r.hidden, firstPos, rows, want, at); else del(r.hidden); };
    if (opts.decodeOnlyPrefill) {
      for (let i = from; i < end; i++) { const r = await run("decode", await embed.rows([ids[i]]), 1, Int32Array.from([i]), st); st = r.st; del(r.logits); await grab(r, i, 1); }
      return st;
    }
    const remDec = opts.remainderViaDecode ?? accelerator === "webgpu";
    for (const c of prefillChunks(end - from, P)) {
      if (remDec && c.len < P) {
        for (let i = from + c.start; i < from + c.start + c.len; i++) { const r = await run("decode", await embed.rows([ids[i]]), 1, Int32Array.from([i]), st); st = r.st; del(r.logits); await grab(r, i, 1); }
        continue;
      }
      const toks = ids.slice(from + c.start, from + c.start + c.len);
      const emb = new Float32Array(P * H); emb.set(await embed.rows(toks));
      const pos = new Int32Array(P); for (let i = 0; i < c.len; i++) pos[i] = from + c.start + i;
      const r = await run(prefillName, emb, P, pos, st, c.len); st = r.st; await grab(r, from + c.start, c.len);
    }
    return st;
  }
  async function decodeOne(tid: number, p: number, st: Record<string, any>) {
    const r = await run("decode", await embed.rows([tid]), 1, Int32Array.from([p]), st);
    return { st: r.st, logits: await readLogits(r.logits), hidden: r.hidden };
  }
  /** Runs ids[from..] and returns the last-position logits plus the hidden rows at `want.positions`. */
  async function feedFrom(ids: number[], from: number, st: Record<string, any>, positions: number[]) {
    const want = new Set(positions); const at = new Map<number, Float32Array>();
    const last = ids.length - 1;
    st = await prefillTokens(ids, from, last, st, want, at);
    const r = await decodeOne(ids[last], last, st);
    await collectHidden(r.hidden, last, 1, want, at);
    freeState(r.st);
    return { lastLogits: r.logits, hiddens: positions.map((p) => at.get(p)!) };
  }
  const checkHidden = (want: { positions: number[] }) => { if (want.positions.length && !hiddenOut) throw new Error('litert: this graph has no hidden-state output; use ask() with method "logits"'); };

  const gate = createGate("litert");
  const snaps = new WeakMap<StateHandle, Float32Array[]>();
  const snapshotState = async (st: Record<string, any>) => Promise.all(stateIO.map(async (io) => { const t = st[io.name]; const c = t.moveTo ? await t.moveTo("wasm") : t; const a = Float32Array.from(c.toTypedArray()); if (c !== t) del(c); return a; }));
  const fromCopy = async (arrs: Float32Array[]) => { const st: Record<string, any> = {}; stateIO.forEach((io, i) => { st[io.name] = T(arrs[i].slice(), io.shape); });
    if (inplace && accelerator === "webgpu") for (const k in st) st[k] = await st[k].moveTo("webgpu");
    return st; };
  return {
    kind: "litert",
    hiddenName: hiddenOut,
    dispose() { return gate.dispose(() => model.delete?.()); },
    async prefill(ids, want) {
      gate.assertLive(); checkHidden(want);
      if (!ids.length) throw new Error("prefill: empty ids");
      const rel = await gate.acquire();
      try { return await feedFrom(ids, 0, await freshState(), want.positions); } finally { rel(); }
    },
    async snapshot(ids) {
      gate.assertLive();
      if (!ids.length) throw new Error("litert: snapshot of empty ids");
      const rel = await gate.acquire();
      try {
        // the prefix state must end at a real token: run every prefix token through prefill (padded) chunks
        const st = await prefillTokens(ids, 0, ids.length, await freshState());
        const h: StateHandle = { ids: ids.slice(), dispose() { snaps.delete(h); } };
        snaps.set(h, await snapshotState(st)); freeState(st);
        return h;
      } finally { rel(); }
    },
    async prefillFrom(h, ids, want) {
      gate.assertLive(); checkHidden(want);
      const arrs = snaps.get(h);
      if (!arrs) throw new Error("litert: unknown or released snapshot handle");
      const k = checkContinuation("litert", h, ids, want.positions);
      const rel = await gate.acquire();
      try { return await feedFrom(ids, k, await fromCopy(arrs), want.positions); } finally { rel(); }
    },
    async *generate(ids, o: GenerateOptions) {
      gate.assertLive();
      const g = resolveGenerateOptions(o), rand = mulberry32(g.seed);
      if (!ids.length) throw new Error("generate: empty ids");
      if (g.maxNewTokens === 0) return;
      const rel = await gate.acquire();
      let st: Record<string, any> | null = null;
      try {
        st = await prefillTokens(ids, 0, ids.length - 1, await freshState());
        let r = await decodeOne(ids[ids.length - 1], ids.length - 1, st); st = r.st;
        const gen: number[] = [];
        for (let s = 0, pos = ids.length; s < g.maxNewTokens; s++, pos++) {
          if (g.signal?.aborted || pos >= cache) break;
          const tid = sample(r.logits, gen, rand, g);
          if (EOS.has(tid)) break;
          gen.push(tid); yield tid;
          if (s + 1 >= g.maxNewTokens || g.signal?.aborted) break;
          r = await decodeOne(tid, pos, st); st = r.st;
        }
      } finally { if (st) freeState(st); rel(); }
    },
  };
}
