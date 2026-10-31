

export interface GenerateOptions {
  maxNewTokens?: number; temperature?: number; topP?: number; topK?: number;
  repetitionPenalty?: number; seed?: number; system?: string; think?: boolean; signal?: AbortSignal;
}

export interface ResolvedGenerateOptions {
  maxNewTokens: number; temperature: number; topP: number; topK: number; repetitionPenalty: number; seed: number;
  signal?: AbortSignal;
}

export const GENERATE_DEFAULTS = { maxNewTokens: 512, temperature: 0.7, topP: 0.9, topK: 40, repetitionPenalty: 1.1, seed: 0 } as const;

export function resolveGenerateOptions(o: GenerateOptions = {}): ResolvedGenerateOptions {
  const num = (v: unknown, d: number) => (typeof v === "number" && Number.isFinite(v) ? v : d);
  const D = GENERATE_DEFAULTS;
  const temperature = num(o.temperature, D.temperature);
  return {
    maxNewTokens: Math.max(0, Math.floor(num(o.maxNewTokens, D.maxNewTokens))),
    temperature,
    topP: num(o.topP, D.topP),
    topK: num(o.topK, D.topK),
    repetitionPenalty: num(o.repetitionPenalty, temperature <= 0 ? 1 : D.repetitionPenalty),
    seed: num(o.seed, D.seed),
    signal: o.signal,
  };
}

export interface Backend {

  kind: "decode-only" | "scan" | "genai" | "webllm" | (string & {});

  hiddenName: string | null;

  prefill(ids: number[], want: { positions: number[] }): Promise<{ hiddens: Float32Array[]; lastLogits: Float32Array }>;

  dispose(): Promise<void>;

  generate(ids: number[], opts: GenerateOptions): AsyncGenerator<number>;

  snapshot?(ids: number[]): Promise<StateHandle>;

  prefillFrom?(h: StateHandle, ids: number[], want: { positions: number[] }): Promise<{ hiddens: Float32Array[]; lastLogits: Float32Array }>;
}

export interface StateHandle {

  readonly ids: readonly number[];

  dispose(): void | Promise<void>;
}

export function checkContinuation(name: string, h: StateHandle, ids: number[], positions: number[]): number {
  const k = h.ids.length;
  if (ids.length <= k || h.ids.some((x, i) => ids[i] !== x)) throw new Error(`${name}: prefillFrom ids must extend the snapshot prefix (${k} tokens) by at least one token`);
  for (const p of positions) if (!Number.isInteger(p) || p < k || p >= ids.length) throw new Error(`${name}: position ${p} outside the continuation [${k},${ids.length})`);
  return k;
}

export interface Gate {
  readonly disposed: boolean;

  acquire(): Promise<() => void>;

  assertLive(): void;

  dispose(release: () => unknown): Promise<void>;
}

export function createGate(name: string): Gate {
  let tail: Promise<void> = Promise.resolve();
  let disposed = false;
  let disposing: Promise<void> | null = null;
  const err = () => new Error(`${name}: backend already disposed`);
  const lock = async () => { const prev = tail; let rel!: () => void; tail = new Promise<void>((r) => { rel = r; }); await prev; return rel; };
  return {
    get disposed() { return disposed; },
    assertLive() { if (disposed) throw err(); },
    async acquire() {
      if (disposed) throw err();
      return lock(); 
    },
    dispose(release) {
      if (!disposing) {
        disposed = true;
        disposing = (async () => { const rel = await lock(); try { await release(); } finally { rel(); } })();
      }
      return disposing;
    },
  };
}

const gates = new WeakMap<object, Gate>();
export function gateFor(shared: object, name: string): Gate {
  let g = gates.get(shared);
  if (!g) { g = createGate(name); gates.set(shared, g); }
  return g;
}
