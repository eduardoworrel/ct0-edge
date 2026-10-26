/** WebLLM (MLC) for ct0-edge.
 *
 *  1. `webllmBackend` / `createWebllm`: a `Backend` (src/backend.ts) over an `MLCEngine` ON THE MAIN THREAD
 *     (`LogitProcessor` only exists on the main-thread engine, not with a Web Worker). Enables ask()
 *     (noul/choice/score) through the `logits` and `head` methods:
 *       - logits: the `LogitProcessor` returns the FULL logit vector (248,320) of the last prompt token.
 *       - hidden: tied embeddings => logits = E·h => h = P·logits[rows], with P = (EₙᵀEₙ)⁻¹Eₙᵀ precomputed
 *         offline from the build's own DEQUANTIZED int4 embedding over a fixed 8192-row vocab subset
 *         (files P_<build>_8192.f32|f16 + P_<build>_rows.i32). One dense mat-vec in JS on the CPU (~20 ms;
 *         ~80 MiB resident: f16 is widened to f32 at runtime, only the download shrinks to 40 MiB).
 *         Each requested position is read by running the prompt TRUNCATED at that token (one prefill
 *         per position; the model is causal).
 *     Limits: WebLLM only accepts TEXT, so the backend needs the tokenizer (`opts.tok`, ids -> text) and
 *     checks `usage.prompt_tokens === position+1` (if re-tokenization diverges it rejects instead of
 *     reading the wrong position). Each build (q4f16_1, q4f32_1) has its own P.
 *     For head-based score prefer the q4f32_1 build + head_score.bin (q4f16's fp16 activations shift
 *     the hiddens); only q4f16 was tested here.
 *  2. `webllmModel`: adapter for the ct0-cai `Agent` (stream + decide). `decide()` reads L[Yes]-L[No] from the
 *     logit vector and REQUIRES `opts.capture` (no fallback: at temperature 0 WebLLM's top_logprobs are one-hot).
 *  The engine is injected (no WebLLM import here); `createWebllm` receives the `@mlc-ai/web-llm` module. */
import { type Backend } from "./backend.js";
export interface LogitCapture {
    /** Register in `CreateMLCEngine(id, { logitProcessorRegistry: new Map([[id, capture.processor]]) })`. */
    processor: {
        processLogits(l: Float32Array): Float32Array;
        processSampledToken(t: number): void;
        resetState(): void;
    };
    /** The next processLogits call copies the vector (only the first one after arm()). */
    arm(): void;
    /** Returns (and clears) the captured vector; null if nothing was captured. */
    take(): Float32Array | null;
    /** Sampled tokens since the last drain, in order. */
    drain(): number[];
    /** Clears pending sampled tokens and disarms. */
    reset(): void;
}
export declare function logitCapture(): LogitCapture;
export interface Recovery {
    /** Vocab indices (N) used; P has D*N elements, row-major (P[d*N + n]). */
    rows: Int32Array;
    /** plain f32 or f16 (Uint16Array, widened once to f32 in `normalizeRecovery`). */
    P: Float32Array | Uint16Array;
}
interface RecoveryF32 {
    rows: Int32Array;
    P: Float32Array;
    D: number;
}
/** h = P · logits[rows]  (D elements). Accepts a normalized or raw `Recovery`. */
export declare function recoverHidden(rec: Recovery | RecoveryF32, logits: Float32Array): Float32Array;
/** Downloads the build's recovery files: `<prefix>_rows.i32` and `<prefix>_8192.f32|f16`. */
export declare function loadRecovery(rowsUrl: string, pUrl: string, format?: "f32" | "f16", fetchFn?: typeof fetch): Promise<Recovery>;
/** Recovery of an MLC build from its repo base (`<base>/<rows>`, `<base>/<P>`). Never throws: on failure returns the reason,
 *  and the caller reads every mode from the logits. */
export declare function tryLoadRecovery(base: string, files: {
    rows: string;
    P: string;
}, fetchFn?: typeof fetch): Promise<{
    recovery?: Recovery;
    reason?: string;
}>;
export interface WebllmBackendOptions {
    /** Required tokenizer (transformers.js or similar): `decode(ids, {skip_special_tokens:false, ...})` -> text. */
    tok: any;
    /** Hidden recovery; without it only the `logits` method works (hiddenName null). */
    recovery?: Recovery;
    /** EOS ids (default: CT0's). */
    eos?: Set<number>;
}
export declare function webllmBackend(engine: any, capture: LogitCapture, opts: WebllmBackendOptions): Backend;
export interface CreateWebllmOptions extends WebllmBackendOptions {
    /** Model id in appConfig. */
    modelId: string;
    /** Base URL of the build (mlc-chat-config.json, shards) and the wasm; or pass a ready `appConfig`. */
    modelUrl?: string;
    libUrl?: string;
    appConfig?: any;
    initProgressCallback?: (p: any) => void;
    /** Alternative to `recovery`: URLs of the build's P files (f32 or f16). */
    recoveryUrls?: {
        rows: string;
        P: string;
        format?: "f32" | "f16";
    };
}
/** Creates the MLCEngine (main thread) with the LogitProcessor registered and the backend on top.
 *  `webllm` = the `@mlc-ai/web-llm` module. Also returns engine and capture (for `webllmModel`). */
export declare function createWebllm(webllm: any, opts: CreateWebllmOptions): Promise<Backend & {
    engine: any;
    capture: LogitCapture;
}>;
/** `Agent` adapter. Shares the per-engine queue with `webllmBackend`/`createWebllm`, so decide()/stream() and
 *  prefill()/generate() never run together. After the backend's `dispose()`, decide()/stream() reject.
 *  `decide()` with `opts.capture` (a logitCapture() registered on the engine) reads L[Yes]-L[No] from the logit vector. */
export declare function webllmModel(engine: any, opts?: {
    stop?: string[];
    noThink?: boolean;
    capture?: LogitCapture;
    maxSafeTokens?: number;
}): {
    countTokens: (t: string) => number;
    fitContext: (p: string) => string;
    maxSafeTokens: number;
    stream(prompt: string, o?: {
        temperature?: number;
        maxNewTokens?: number;
    }): AsyncGenerator<any, void, unknown>;
    decide(state: string, question: string): Promise<number>;
};
export {};
