/**
/**
 * genai.ts — backend over the onnxruntime-genai graph (WebGPU prototype, int4) running in the browser with
 * onnxruntime-web (native WebGPU EP). Inputs per run: input_ids [1,S], attention_mask [1,past+S] (ones),
 * position_ids [3,1,S] (the 3 rows are absolute positions) and the caches (growing KV fp16 [1,H,past,D];
 * fixed-size conv/recurrent). Cache names, dtypes and shapes come from `session.inputMetadata`.
 *
 * - State (present -> past) stays on the GPU (`preferredOutputLocation: "gpu-buffer"`); the previous tensor is
 *   disposed at each step. logits/hidden are read with `getData(true)`.
 * - The graph emits logits for EVERY prompt position (~0.5 MB/token): prefill runs in chunks and only the last
 *   one fetches `logits` (the others do not read them back).
 * - Loading: imports `onnxruntime-web/webgpu` (the default bundle falls back to CPU on 591 nodes) and VERIFIES,
 *   from the session's verbose placement logs, that almost all nodes landed on the WebGPU EP (LinearAttention included).
 * - Warm-up: 1 token after load (hides ~0.5 s of shader compilation).
 * Measured: 24/24 tokens match PyTorch, ~24 tok/s on an M2 Pro.
 */
import type { Backend, GenerateOptions, StateHandle } from "../backend.js";
import { checkOrtVersion } from "../ortVersion.js";
import { createGate, resolveGenerateOptions, checkContinuation } from "../backend.js";
import { tensorToF32, EOS, HIDDEN_NAMES, mulberry32, sample, pastNameFor, cacheDims } from "./decodeOnly.js";

export interface GenaiOptions {
  /** EPs (default ["webgpu"]; webgpu e obrigatorio). */
  providers?: string[];
  /** Already-loaded onnxruntime-web module (default: `import("onnxruntime-web/webgpu")`). */
  ort?: any;
  /** Tokens per prefill run (default 128; the last chunk reads back S x 248320 fp16 logits). */
  chunk?: number;
  /** Verifies the EP placement at load (default true). Without it the backend may silently run on CPU.
   *  COST of the first verification: a throwaway session with verbose logging, so the graph (~2.4 GB) is
   *  downloaded and compiled one extra time (double the traffic if the server does not cache; load went
   *  from ~11-14 s to ~20 s on an M2 Pro).
   *  To avoid paying this in production, an OK verdict is stored in localStorage keyed by (onnxruntime-web
   *  version, GPU adapter hash [vendor/architecture/device/description], model URL, data URL, `maxCpuNodes`):
   *  later loads do not probe. The cache does NOT expire: after updating browser/driver/onnxruntime-web
   *  without changing that key, use `"force"` to re-verify (ignores the cache and rewrites it);
   *  `false` never probes. Outside a browser (no localStorage) the verdict is not stored, so it probes every time. */
  verifyEp?: boolean | "force";
  /** Verdict storage (default `globalThis.localStorage`, if present). Useful for tests/SSR. */
  epCache?: { getItem(k: string): string | null; setItem(k: string, v: string): void } | null;
  /** GPU adapter identifier for the cache key (default: hash of `navigator.gpu.requestAdapter().info`,
   *  or "?" if unavailable). Injectable for tests. */
  adapterTag?: string | (() => Promise<string>);
  /** Maximum nodes outside WebGPU tolerated by the check (default 32; the graph has 6 on CPU: the int64 mask). */
  maxCpuNodes?: number;
  /** KV head_dim when the metadata leaves it symbolic (genai_config.decoder.head_size; default 256). */
  kvHeadDim?: number;
  /** Skip the onnxruntime-web >= 1.30 requirement (experiments only). */
  skipVersionCheck?: boolean;
  /** 1-token warm-up after load (default true). */
  warmup?: boolean;
  /** EOS ids (default: those in genai_config). */
  eos?: Set<number>;
}

// ---- placement verification (ORT verbose logs) ----
/** "Node(s) placed on [XExecutionProvider]. Number of nodes: N" -> { XExecutionProvider: N }. */
export function parsePlacement(lines: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const l of lines) {
    const m = /Node\(s\) placed on \[(\w+)\]\.\s*Number of nodes:\s*(\d+)/.exec(l);
    if (m) out[m[1]] = (out[m[1]] ?? 0) + Number(m[2]);
  }
  return out;
}

/** Throws if the graph is not (almost entirely) on the WebGPU EP. */
export function assertWebGpuPlacement(p: Record<string, number>, maxCpuNodes = 32): void {
  const gpu = Object.entries(p).filter(([k]) => /webgpu/i.test(k)).reduce((a, [, n]) => a + n, 0);
  const cpu = p.CPUExecutionProvider ?? 0;
  if (!Object.keys(p).length) throw new Error("genai: could not verify the EP placement (no 'Node(s) placed on' logs); reject a bundle without WebGPU");
  if (gpu === 0) throw new Error(`genai: no node on the WebGPU EP (placement: ${JSON.stringify(p)}); LinearAttention does not run on GPU`);
  if (cpu > maxCpuNodes) throw new Error(`genai: ${cpu} nodes on CPU (> ${maxCpuNodes}); probably a bundle without the WebGPU kernels (use onnxruntime-web/webgpu). Placement: ${JSON.stringify(p)}`);
}

export { checkOrtVersion };

// ---- backend ----
export type GenaiBackend = Backend & { /** tokens per prefill run (mutable) */ chunk: number };

export function genaiFromSession(session: any, ort: { Tensor: new (type: any, data: any, dims: number[], ...r: any[]) => any }, opts: GenaiOptions = {}): GenaiBackend {
  const T = ort.Tensor;
  const eos = opts.eos ?? EOS;
  const inNames: string[] = session.inputNames, outNames: string[] = session.outputNames;
  for (const need of ["input_ids"]) if (!inNames.includes(need)) throw new Error(`genai: graph without input ${need}`);
  if (!outNames.includes("logits")) throw new Error("genai: graph without a logits output");
  const hasMask = inNames.includes("attention_mask"), hasPos = inNames.includes("position_ids");
  const meta: any[] = Array.isArray(session.inputMetadata) ? session.inputMetadata
    : Object.entries(session.inputMetadata ?? {}).map(([name, m]: any) => ({ name, ...m }));
  const metaBy = new Map(meta.map((m) => [m.name, m]));
  const specs: { name: string; type: string; dims: number[] }[] = [];
  for (const n of inNames) {
    if (n === "input_ids" || n === "attention_mask" || n === "position_ids") continue;
    const m = metaBy.get(n);
    if (!m?.shape) throw new Error("genai: no shape metadata for input " + n);
    const dims = cacheDims(n, m.shape, 0); // KV: past inicial 0
    // the KV metadata leaves head_dim symbolic ("kv_cache_dim"); it comes from genai_config.decoder.head_size (256)
    if (/^past_key_values\./.test(n) && typeof m.shape[3] !== "number") dims[3] = opts.kvHeadDim ?? 256;
    specs.push({ name: n, type: m.type, dims });
  }
  const pastOf = new Map<string, string>(); // present -> past
  for (const o of outNames) { const p = pastNameFor(o); if (p && inNames.includes(p)) pastOf.set(o, p); }
  const covered = new Set(pastOf.values());
  const missing = specs.filter((s) => !covered.has(s.name));
  if (missing.length) throw new Error("genai: cache inputs without a present output: " + missing.slice(0, 3).map((s) => s.name).join(", "));
  const hiddenName = HIDDEN_NAMES.find((n) => outNames.includes(n)) ?? null;
  const posRows = (() => { const sh = metaBy.get("position_ids")?.shape; return typeof sh?.[0] === "number" ? sh[0] : 3; })();

  const zeros = (s: { type: string; dims: number[] }) => {
    const n = s.dims.reduce((a, b) => a * b, 1);
    if (s.type === "float16") return new T("float16", new Uint16Array(n), s.dims);
    if (s.type === "float32") return new T("float32", new Float32Array(n), s.dims);
    throw new Error(`genai: unsupported cache dtype: ${s.type}`);
  };
  const initState = () => { const st: Record<string, any> = {}; for (const s of specs) st[s.name] = zeros(s); return st; };
  const free = (st: Record<string, any> | null) => { if (st) for (const t of Object.values(st)) if (t?.location === "gpu-buffer") { try { t.dispose?.(); } catch { /* */ } } };

  // queue + dispose (contract in backend.ts): ORT does not allow concurrent runs on one session
  const gate = createGate("genai");

  /** One run of `chunk` tokens over [past, past+S). Returns the new state (and frees the previous one, unless `keep`: a snapshot). */
  async function run(chunk: number[], past: number, st: Record<string, any>, wantLogits: boolean, wantHidden: boolean, keep = false) {
    const S = chunk.length;
    const feed: Record<string, any> = { input_ids: new T("int64", BigInt64Array.from(chunk.map(BigInt)), [1, S]), ...st };
    if (hasMask) feed.attention_mask = new T("int64", new BigInt64Array(past + S).fill(1n), [1, past + S]);
    if (hasPos) {
      const pos = new BigInt64Array(posRows * S);
      for (let r = 0; r < posRows; r++) for (let i = 0; i < S; i++) pos[r * S + i] = BigInt(past + i);
      feed.position_ids = new T("int64", pos, [posRows, 1, S]);
    }
    const fetches = [...pastOf.keys()];
    if (wantLogits) fetches.unshift("logits");
    if (wantHidden && hiddenName) fetches.push(hiddenName);
    const r = await session.run(feed, fetches, { logSeverityLevel: 3 });
    const ns: Record<string, any> = {};
    for (const [o, p] of pastOf) ns[p] = r[o];
    if (!keep) free(st);
    const out: { logits?: any; hidden?: any } = {};
    if (wantLogits) out.logits = r.logits;
    if (wantHidden && hiddenName) out.hidden = r[hiddenName];
    return { st: ns, out };
  }
  const read = async (t: any) => (typeof t.getData === "function" ? await t.getData(true) : t.data);

  /** Chunked prefill. Returns the final state (to be freed by the caller) and, if requested, hiddens and the last logits.
   *  `from` = a snapshot (state after ids[0..past)): the run starts there and never frees the snapshot's tensors. */
  async function feedAll(ids: number[], wanted: Set<number>, wantLast: boolean, signal?: AbortSignal, from?: { st: Record<string, any>; past: number }) {
    let st = from?.st ?? initState(), past = from?.past ?? 0;
    const kept = from?.st;
    const hid = new Map<number, Float32Array>();
    let lastLogits: Float32Array | undefined;
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
    } catch (e) { if (st !== kept) free(st); throw e; }
    return { st, past, hid, lastLogits, aborted: false };
  }

  // snapshot state (kept on the GPU) per live handle
  const snaps = new WeakMap<StateHandle, Record<string, any>>();

  const self: GenaiBackend = {
    kind: "genai",
    chunk: Math.max(1, opts.chunk ?? 128),
    hiddenName,
    dispose() { return gate.dispose(() => session.release?.()); },
    async prefill(ids, want) {
      gate.assertLive();
      for (const p of want.positions) if (!Number.isInteger(p) || p < 0 || p >= ids.length) throw new Error(`genai: position ${p} out of [0,${ids.length})`);
      if (!ids.length) throw new Error("prefill: empty ids");
      if (want.positions.length && !hiddenName) throw new Error("genai: the graph does not expose a hidden state; use method 'logits'");
      const rel = await gate.acquire();
      try {
        const f = await feedAll(ids, new Set(want.positions), true);
        free(f.st);
        return { hiddens: want.positions.map((p) => f.hid.get(p)!), lastLogits: f.lastLogits! };
      } finally { rel(); }
    },
    async snapshot(ids) {
      gate.assertLive();
      if (!ids.length) throw new Error("genai: snapshot of empty ids");
      const rel = await gate.acquire();
      try {
        const f = await feedAll(ids, new Set(), false);
        let st: Record<string, any> | null = f.st;
        const h: StateHandle = { ids: ids.slice(), dispose: async () => {
          if (!st) return;
          const s0 = st; st = null; snaps.delete(h);
          // through the queue: a prefillFrom already queued with this state runs first
          let r: (() => void) | null = null;
          try { r = await gate.acquire(); } catch { /* backend released */ }
          try { free(s0); } finally { r?.(); }
        } };
        snaps.set(h, st);
        return h;
      } finally { rel(); }
    },
    async prefillFrom(h, ids, want) {
      gate.assertLive();
      const st = snaps.get(h);
      if (!st) throw new Error("genai: unknown or released snapshot handle");
      const k = checkContinuation("genai", h, ids, want.positions);
      if (want.positions.length && !hiddenName) throw new Error("genai: the graph does not expose a hidden state; use method 'logits'");
      const rel = await gate.acquire();
      try {
        const f = await feedAll(ids, new Set(want.positions), true, undefined, { st, past: k });
        free(f.st);
        return { hiddens: want.positions.map((p) => f.hid.get(p)!), lastLogits: f.lastLogits! };
      } finally { rel(); }
    },
    /** Generates ids from `ids`; stops at EOS (not emitted), `maxNewTokens` (default 512; 0 => none), `signal` or `break`.
     *  Runs are awaited inline, so no run is in flight when the generator ends. Sampler = decode-only's. */
    async *generate(ids, o: GenerateOptions) {
      gate.assertLive();
      const g = resolveGenerateOptions(o);
      const rand = mulberry32(g.seed);
      if (!ids.length) throw new Error("generate: empty ids");
      if (g.maxNewTokens === 0) return;
      const rel = await gate.acquire();
      let st: Record<string, any> | null = null;
      try {
        if (o.signal?.aborted) return;
        const f = await feedAll(ids, new Set(), true, o.signal);
        st = f.st; if (f.aborted) return; let past = f.past, logits = f.lastLogits!;
        const gen: number[] = [];
        for (let s = 0; s < g.maxNewTokens; s++) {
          if (o.signal?.aborted) break;
          const tid = sample(logits, gen, rand, g);
          if (eos.has(tid)) break;
          gen.push(tid); yield tid;
          if (o.signal?.aborted || s + 1 >= g.maxNewTokens) break;
          const r = await run([tid], past, st, true, false); st = r.st; past++;
          const L = r.out.logits, V = L.dims.at(-1);
          logits = tensorToF32(await read(L), L.type, (L.dims.at(-2) - 1) * V, V);
        }
      } finally { free(st); rel(); }
    },
  };
  return self;
}

/** Hash FNV-1a (hex) curto. */
export function hashStr(x: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < x.length; i++) { h ^= x.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(16).padStart(8, "0");
}

async function adapterTag(opts: GenaiOptions): Promise<string> {
  if (typeof opts.adapterTag === "string") return opts.adapterTag;
  if (typeof opts.adapterTag === "function") return opts.adapterTag();
  try {
    const gpu = (globalThis as any).navigator?.gpu;
    const ad = await gpu?.requestAdapter?.();
    const i = ad?.info ?? (await ad?.requestAdapterInfo?.());
    if (!i) return "?";
    return hashStr([i.vendor, i.architecture, i.device, i.description].join("|"));
  } catch { return "?"; }
}

function epStore(opts: GenaiOptions) {
  if (opts.epCache !== undefined) return opts.epCache;
  try { return (globalThis as any).localStorage ?? null; } catch { return null; }
}

/** Placement only shows in verbose logs, and the log level belongs to the SESSION (it stays for every run, ~50% slower).
 *  So it is checked in a throwaway session, released before the production one is created (level 3). Throws on failure. */
async function probePlacement(ort: any, modelUrl: string, sessionOpts: (level: number) => any, maxCpuNodes: number) {
  const captured: string[] = [];
  const names = ["log", "info", "warn", "error", "debug"] as const;
  const saved: Record<string, any> = {};
  const prevLevel = ort.env?.logLevel;
  if (ort.env) ort.env.logLevel = "verbose";
  for (const n of names) { saved[n] = (console as any)[n]; (console as any)[n] = (...a: any[]) => { const s = a.join(" "); if (s.includes("Node(s) placed on")) captured.push(s); else if (!s.includes("[V:onnxruntime") && !/WebGPU Shader code|^\s*$/.test(s)) saved[n].apply(console, a); }; }
  let probe: any;
  try { probe = await ort.InferenceSession.create(modelUrl, sessionOpts(0)); }
  finally { for (const n of names) (console as any)[n] = saved[n]; if (ort.env) ort.env.logLevel = prevLevel; }
  try { assertWebGpuPlacement(parsePlacement(captured), maxCpuNodes); }
  finally { await probe.release?.(); }
}

/** Loads the WebGPU graph (`modelUrl` = model_webgpu.onnx, `dataUrl` = model.onnx.data) and returns the backend. */
export async function createGenai(modelUrl: string, dataUrl: string, opts: GenaiOptions = {}): Promise<GenaiBackend> {
  const providers = opts.providers ?? ["webgpu"];
  if (providers[0] !== "webgpu") throw new Error("genai: the graph requires the webgpu EP as first provider (the LinearAttention/CausalConvWithState kernels only exist on WebGPU)");
  const ort: any = opts.ort ?? await import("onnxruntime-web/webgpu");
  const verify = opts.verifyEp ?? true;

  const sessionOpts = (level: number) => ({
    executionProviders: providers.map((n) => (n === "webgpu" ? { name: "webgpu", validationMode: "disabled" } : n)),
    externalData: [{ path: "model.onnx.data", data: dataUrl }],
    graphOptimizationLevel: "all", enableGraphCapture: false,
    preferredOutputLocation: "gpu-buffer",
    logSeverityLevel: level,
  });
  if (!opts.skipVersionCheck) checkOrtVersion(ort.env?.versions?.web);

  if (verify) {
    const store = epStore(opts);
    const maxCpu = opts.maxCpuNodes ?? 32;
    const key = store ? `ct0-edge:genai-ep:${ort.env?.versions?.web ?? "?"}:${await adapterTag(opts)}:${modelUrl}:${dataUrl}:${maxCpu}` : "";
    let cached = false;
    if (verify !== "force" && store) { try { cached = store.getItem(key) === "ok"; } catch { /* no cache */ } }
    if (!cached) {
      await probePlacement(ort, modelUrl, sessionOpts, maxCpu);
      if (store) { try { store.setItem(key, "ok"); } catch { /* cota/privado */ } }
    }
  }
  const session = await ort.InferenceSession.create(modelUrl, sessionOpts(3));

  try {
    const be = genaiFromSession(session, ort, opts);
    if (opts.warmup ?? true) await be.prefill([248045], { positions: [] }); // 1 token: compila shaders S=1
    return be;
  } catch (e) {
    // the session already exists: release it before rethrowing (else the facade's fallback loads another graph with this one in memory)
    try { await session.release?.(); } catch { /* */ }
    throw e;
  }
}
