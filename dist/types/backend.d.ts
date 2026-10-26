/**
/**
 * backend.ts — common contract of the CT0 execution backends (decode-only, scan, genai, webllm),
 * plus the two helpers ALL backends share: `Gate` (queue + dispose flag) and
 * `resolveGenerateOptions` (single defaults). The `CT0` facade delegates to the backend; backends know
 * nothing about prompts or the head.
 */
/**
/**
 * Generation options. Defaults live in ONE place (`resolveGenerateOptions`) and apply to every backend:
 * `maxNewTokens` 512, `temperature` 0.7, `topP` 0.9, `topK` 40, `repetitionPenalty` 1.1, `seed` 0.
 * `undefined` fields do NOT override the default. `maxNewTokens: 0` generates 0 tokens (the prompt is not even run).
 * `temperature <= 0` = pure greedy: without an explicit `repetitionPenalty` it becomes 1 (argmax of the raw
 * logits, as in PyTorch); passing `repetitionPenalty` explicitly re-enables the penalty.
 *
 * Options a runtime does not support (per backend):
 *  - scan (transformers.js): `seed` is IGNORED (not exposed; with temperature > 0 sampling is not reproducible).
 *  - webllm: `topK` is IGNORED (MLC's sampler has none); `seed`, `topP`, `temperature`, `repetitionPenalty` apply.
 *  - decode-only, genai: all apply (JS sampler, reproducible `seed`).
 * `system`/`think` are `CT0.generate` facade fields; backends ignore them.
 */
export interface GenerateOptions {
    maxNewTokens?: number;
    temperature?: number;
    topP?: number;
    topK?: number;
    repetitionPenalty?: number;
    seed?: number;
    system?: string;
    think?: boolean;
    signal?: AbortSignal;
}
/** `GenerateOptions` with the defaults applied (sampling fields only). */
export interface ResolvedGenerateOptions {
    maxNewTokens: number;
    temperature: number;
    topP: number;
    topK: number;
    repetitionPenalty: number;
    seed: number;
    signal?: AbortSignal;
}
export declare const GENERATE_DEFAULTS: {
    readonly maxNewTokens: 512;
    readonly temperature: 0.7;
    readonly topP: 0.9;
    readonly topK: 40;
    readonly repetitionPenalty: 1.1;
    readonly seed: 0;
};
/** The single normalization of `GenerateOptions`: applies defaults, ignores `undefined`/non-finite fields,
 *  and makes `maxNewTokens` an integer >= 0 (0 => no tokens). */
export declare function resolveGenerateOptions(o?: GenerateOptions): ResolvedGenerateOptions;
/**
 * Backend contract (applies to all four: decode-only, scan, genai, webllm).
 *
 * CONCURRENCY. A backend wraps ONE session/engine that does not allow concurrent runs, so calls are
 * SERIALIZED by the backend itself, in arrival order: `prefill()` or `generate()` called while another
 * call is in flight wait their turn (they never interleave). A `generate()` holds the turn until the
 * generator finishes (end, `break`, abort or error). So do NOT call `prefill()`/`ask()` inside the
 * `for await` of a `generate()` on the same backend and await it there (deadlock); finish or abandon the stream first.
 *
 * DISPOSE. `dispose()` is idempotent (repeated calls wait for the same release) and WAITS for in-flight
 * work before freeing resources. Calls accepted BEFORE `dispose()` (in flight or queued) finish first
 * (dispose is a barrier in the queue; use `signal` to cut a generate short). After `dispose()` every new
 * call (`prefill`, `generate`) REJECTS with "backend already disposed".
 */
export interface Backend {
    /** "scan" = optimum/Scan graph (transformers.js). */
    kind: "decode-only" | "scan" | "genai" | "webllm" | (string & {});
    /** Name of the hidden-state output in the graph (null if the graph does not expose it). */
    hiddenName: string | null;
    /** Runs `ids` from scratch and returns the (post-norm) hidden at each position of `want.positions`
     *  and the logits of the LAST position. `hiddens[i]` matches `want.positions[i]`, in the given order;
     *  duplicate positions give duplicate entries. Serialized (see contract above). */
    prefill(ids: number[], want: {
        positions: number[];
    }): Promise<{
        hiddens: Float32Array[];
        lastLogits: Float32Array;
    }>;
    /** Frees the session/native resources: idempotent, waits for in-flight work; afterwards everything rejects. */
    dispose(): Promise<void>;
    /** Autoregressive generation from `ids` (raw prompt); emits token ids until EOS/maxNewTokens. Serialized;
     *  holds the turn until the generator finishes. Options and defaults: see `GenerateOptions`. */
    generate(ids: number[], opts: GenerateOptions): AsyncGenerator<number>;
    /** Optional (decode-only, genai): run `ids` once and keep the model state after them. The handle stays valid for any
     *  number of `prefillFrom` calls until `handle.dispose()`. Serialized like `prefill`. */
    snapshot?(ids: number[]): Promise<StateHandle>;
    /** Optional, with `snapshot`: same result as `prefill(ids, want)`, running only `ids` after the snapshot's prefix.
     *  `ids` must start with `h.ids` and be longer; `want.positions` are positions in the whole prompt, all >= `h.ids.length`.
     *  Does not consume the snapshot. */
    prefillFrom?(h: StateHandle, ids: number[], want: {
        positions: number[];
    }): Promise<{
        hiddens: Float32Array[];
        lastLogits: Float32Array;
    }>;
}
/** Model state after a token prefix (`Backend.snapshot`). Only the backend that made it accepts it. */
export interface StateHandle {
    /** The prefix the state was computed from. */
    readonly ids: readonly number[];
    /** Frees the state (idempotent); later `prefillFrom` calls with it reject. */
    dispose(): void | Promise<void>;
}
/** Shared `prefillFrom` argument checks: `ids` extends `h.ids` by at least one token and every position is in
 *  [h.ids.length, ids.length). Returns the prefix length. */
export declare function checkContinuation(name: string, h: StateHandle, ids: number[], positions: number[]): number;
export interface Gate {
    readonly disposed: boolean;
    /** Waits for its turn and returns `release`. Rejects if the gate was disposed; calls queued before dispose still run. */
    acquire(): Promise<() => void>;
    /** Throws if already disposed (synchronous check before any work). */
    assertLive(): void;
    /** Idempotent: marks as disposed, waits for in-flight work and runs `release` exactly once. */
    dispose(release: () => unknown): Promise<void>;
}
export declare function createGate(name: string): Gate;
export declare function gateFor(shared: object, name: string): Gate;
