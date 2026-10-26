/**
/**
 * decodeOnly.ts — backend for the decode-only graph (1 token per run, no attention_mask).
 * Caches are NOT hard-coded: names, dtypes and shapes come from `session.inputMetadata`, and the
 * hidden output is detected in `session.outputNames`. Mirrors the reference `run_prompt`:
 * symbolic dims -> 1, zeros, and present_* -> past_* at each step.
 */
import { createGate, resolveGenerateOptions, checkContinuation, type Backend, type GenerateOptions, type StateHandle } from "../backend.js";
import { checkOrtVersion } from "../ortVersion.js";

export { resolveGenerateOptions, GENERATE_DEFAULTS, createGate } from "../backend.js";
export { checkOrtVersion } from "../ortVersion.js";
export const EOS = new Set<number>([248046, 248044]);
export const HIDDEN_NAMES = ["/model/norm/Mul_1_output_0", "mul_13450", "hidden_states", "type_as_80"];
const F16: any = (globalThis as any).Float16Array;

export function mulberry32(seed: number) {
  let a = seed >>> 0 || 1;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** Decodes an fp16 (bit pattern) to a number. `Float32Array.from(Uint16Array)` does NOT do this. */
export function halfToNumber(h: number): number {
  const s = (h & 0x8000) ? -1 : 1, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  return e === 0 ? s * Math.pow(2, -14) * (f / 1024)
       : e === 0x1f ? (f ? NaN : s * Infinity)
       : s * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** Converts an ORT tensor's data (Float32Array, native Float16Array or raw fp16 Uint16Array)
 *  to a Float32Array, starting at `off` for `n` elements. */
export function tensorToF32(data: any, type: string | undefined, off = 0, n = data.length - off): Float32Array {
  const out = new Float32Array(n);
  const isF16Arr = typeof F16 !== "undefined" && data instanceof F16;
  if (data instanceof Float32Array || isF16Arr) { for (let i = 0; i < n; i++) out[i] = Number(data[off + i]); }
  else if (type === "float16" || data instanceof Uint16Array) { for (let i = 0; i < n; i++) out[i] = halfToNumber(data[off + i]); }
  else for (let i = 0; i < n; i++) out[i] = Number(data[off + i]);
  return out;
}

/** Last row ([.., T, V] -> V) of a logits tensor. */
export function lastRow(L: any): Float32Array {
  const V = L.dims.at(-1), off = (L.dims.at(-2) - 1) * V;
  return tensorToF32(L.data, L.type, off, V);
}

export function sample(logits: Float32Array, gen: number[], rand: () => number, o: any): number {
  const { temperature: temp, topP, topK, repetitionPenalty: rep } = o;
  if (rep !== 1) for (const t of new Set(gen)) logits[t] = logits[t] > 0 ? logits[t] / rep : logits[t] * rep;
  if (temp <= 0) { let bi = 0, bv = -1e30; for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; bi = i; } return bi; }
  let idx = Array.from(logits.keys()); idx.sort((a, b) => logits[b] - logits[a]);
  if (topK > 0) idx = idx.slice(0, topK);
  const mx = logits[idx[0]]; let probs = idx.map((i) => Math.exp((logits[i] - mx) / temp));
  let s = probs.reduce((a, b) => a + b, 0); probs = probs.map((p) => p / s);
  if (topP < 1) { let c = 0, cut = idx.length; for (let i = 0; i < idx.length; i++) { c += probs[i]; if (c >= topP) { cut = i + 1; break; } } idx = idx.slice(0, cut); probs = probs.slice(0, cut); const z = probs.reduce((a, b) => a + b, 0); probs = probs.map((p) => p / z); }
  let r = rand(), acc = 0; for (let i = 0; i < idx.length; i++) { acc += probs[i]; if (r <= acc) return idx[i]; } return idx[idx.length - 1];
}

export interface DecodeOnlyOptions {
  /** onnxruntime module (needs `Tensor`): `ort-web` or `ort-node`. */
  ort: { Tensor: new (type: any, data: any, dims: number[]) => any; env?: { versions?: { web?: string } } };
  /** Skip the onnxruntime-web >= 1.30 requirement (experiments only). */
  skipVersionCheck?: boolean;
  /** Initial length of the KV (past) sequence axis. Default 1.
   *  The reference implementation maps every symbolic dim to 1, i.e. starts with a past of length 1
   *  (one zeroed K/V); parity with the published numbers depends on it. Batch and other symbolic dims are always 1. */
  pastLen?: number;
}

interface CacheSpec { name: string; type: string; dims: number[]; }

/** present -> past. Optimum convention (`present_conv.N`, `present.N.key`) and the old one (`so<i>` -> `si<i>`). */
export function pastNameFor(out: string): string | null {
  let m: RegExpMatchArray | null;
  if ((m = out.match(/^present_conv\.(\d+)$/))) return "past_conv." + m[1];
  if ((m = out.match(/^present_recurrent\.(\d+)$/))) return "past_recurrent." + m[1];
  if ((m = out.match(/^present\.(\d+)\.(key|value)$/))) return `past_key_values.${m[1]}.${m[2]}`;
  if ((m = out.match(/^present\.(\d+)\.(conv|recurrent)$/))) return `past.${m[1]}.${m[2]}`; // genai graph (qwen3_5)
  if ((m = out.match(/^so(\d+)$/))) return "si" + m[1];
  return null;
}

/** Dims of a zeroed cache: numeric ones stay; symbolic ones become 1, except the KV sequence axis (2) -> pastLen. */
export function cacheDims(name: string, shape: any[], pastLen: number): number[] {
  const isKV = /^past_key_values\./.test(name) || (shape.length === 4 && !/conv|recurrent/.test(name) && typeof shape[2] !== "number");
  return shape.map((d: any, i: number) => (typeof d === "number" && d >= 0 ? d : isKV && i === 2 ? pastLen : 1));
}

export async function createDecodeOnly(session: any, tok: any, opts: DecodeOnlyOptions): Promise<Backend> {
  if (!opts.skipVersionCheck) checkOrtVersion(opts.ort.env?.versions?.web);
  const T = opts.ort.Tensor;
  const pastLen = opts.pastLen ?? 1;
  const inNames: string[] = session.inputNames, outNames: string[] = session.outputNames;
  if (inNames.includes("attention_mask")) throw new Error("decode-only: a graph with attention_mask is not decode-only");
  const meta: any[] = Array.isArray(session.inputMetadata) ? session.inputMetadata
    : Object.entries(session.inputMetadata ?? {}).map(([name, m]: any) => ({ name, ...m }));
  const metaBy = new Map(meta.map((m) => [m.name, m]));
  const specs: CacheSpec[] = [];
  for (const n of inNames) {
    if (n === "input_ids" || n === "cache_position") continue;
    const m = metaBy.get(n);
    if (!m?.shape) throw new Error("decode-only: no shape metadata for input " + n);
    specs.push({ name: n, type: m.type, dims: cacheDims(n, m.shape, pastLen) });
  }
  const pastOf = new Map<string, string>();
  for (const o of outNames) { const p = pastNameFor(o); if (p && inNames.includes(p)) pastOf.set(o, p); }
  const missing = specs.filter((s) => ![...pastOf.values()].includes(s.name));
  if (missing.length) throw new Error("decode-only: cache inputs without a matching present output: " + missing.slice(0, 3).map((s) => s.name).join(", "));
  const hiddenName = HIDDEN_NAMES.find((n) => outNames.includes(n)) ?? null;
  const hasCachePos = inNames.includes("cache_position");

  const zeros = (s: CacheSpec) => {
    const n = s.dims.reduce((a, b) => a * b, 1);
    if (s.type === "float32") return new T("float32", new Float32Array(n), s.dims);
    if (s.type === "float16") return new T("float16", F16 ? new F16(n) : new Uint16Array(n), s.dims);
    throw new Error(`decode-only: unsupported cache dtype: ${s.type} (${s.name})`);
  };
  const initState = () => { const st: Record<string, any> = {}; for (const s of specs) st[s.name] = zeros(s); return st; };

  // Fetch only logits + present_* (+ hidden when requested), never everything; a partial fetch
  // returns finite hiddens on ort-web WebGPU.
  const fetchBase = ["logits", ...pastOf.keys()];
  async function step(tid: number, pos: number, st: Record<string, any>, wantHidden: boolean) {
    const feed: Record<string, any> = { input_ids: new T("int64", BigInt64Array.from([BigInt(tid)]), [1, 1]), ...st };
    if (hasCachePos) feed.cache_position = new T("int64", BigInt64Array.from([BigInt(pos)]), [1]);
    const names = wantHidden && hiddenName ? [...fetchBase, hiddenName] : fetchBase;
    const r = await session.run(feed, names);
    const ns: Record<string, any> = {};
    for (const [o, p] of pastOf) ns[p] = r[o];
    const hidden = wantHidden && hiddenName ? tensorToF32(r[hiddenName].data, r[hiddenName].type) : undefined;
    return { logits: r.logits, st: ns, hidden };
  }

  /** Runs ids[from..] on top of `st` (the state after ids[0..from)); hiddens at `want.positions`, logits of the last token. */
  async function feed(ids: number[], from: number, st: Record<string, any>, want: { positions: number[] }) {
    const wanted = new Set(want.positions), at = new Map<number, Float32Array>();
    let last: any = null;
    for (let i = from; i < ids.length; i++) {
      const r = await step(ids[i], i, st, wanted.has(i));
      st = r.st; last = r.logits;
      if (r.hidden) at.set(i, r.hidden);
    }
    return { hiddens: want.positions.map((p) => at.get(p)!), lastLogits: lastRow(last) };
  }
  const checkHidden = (want: { positions: number[] }) => {
    if (want.positions.length && !hiddenName) throw new Error("decode-only: the graph does not expose a hidden state (outputs: " + outNames.slice(-3).join(", ") + ")");
  };

  const gate = createGate("decode-only");
  // snapshot state (CPU tensors, never fed back as outputs) per live handle
  const snaps = new WeakMap<StateHandle, Record<string, any>>();
  return {
    kind: "decode-only",
    hiddenName,
    dispose() { return gate.dispose(() => session.release?.()); },
    async prefill(ids, want) {
      gate.assertLive();
      checkHidden(want);
      for (const p of new Set(want.positions)) if (p < 0 || p >= ids.length) throw new Error(`prefill: position ${p} out of [0,${ids.length})`);
      if (!ids.length) throw new Error("prefill: empty ids");
      const rel = await gate.acquire();
      try { return await feed(ids, 0, initState(), want); } finally { rel(); }
    },
    async snapshot(ids) {
      gate.assertLive();
      if (!ids.length) throw new Error("decode-only: snapshot of empty ids");
      const rel = await gate.acquire();
      try {
        let st = initState();
        for (let i = 0; i < ids.length; i++) st = (await step(ids[i], i, st, false)).st;
        const h: StateHandle = { ids: ids.slice(), dispose() { snaps.delete(h); } };
        snaps.set(h, st);
        return h;
      } finally { rel(); }
    },
    async prefillFrom(h, ids, want) {
      gate.assertLive();
      const st = snaps.get(h);
      if (!st) throw new Error("decode-only: unknown or released snapshot handle");
      const k = checkContinuation("decode-only", h, ids, want.positions);
      checkHidden(want);
      const rel = await gate.acquire();
      try { return await feed(ids, k, st, want); } finally { rel(); }
    },
    async *generate(ids, o: GenerateOptions) {
      gate.assertLive();
      const g = resolveGenerateOptions(o);
      const rand = mulberry32(g.seed);
      if (!ids.length) throw new Error("generate: empty ids");
      if (g.maxNewTokens === 0) return;
      const rel = await gate.acquire();
      try {
        let st = initState(), pos = 0, last: any = null;
        for (const t of ids) { if (g.signal?.aborted) return; ({ logits: last, st } = await step(t, pos, st, false)); pos++; }
        const gen: number[] = [];
        for (let s = 0; s < g.maxNewTokens; s++) {
          if (g.signal?.aborted) break;
          const tid = sample(lastRow(last), gen, rand, g);
          if (EOS.has(tid)) break;
          gen.push(tid); yield tid;
          if (s + 1 >= g.maxNewTokens || g.signal?.aborted) break;
          ({ logits: last, st } = await step(tid, pos, st, false)); pos++;
        }
      } finally { rel(); }
    },
  };
}
