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
import { type Backend } from "../backend.js";
export declare const LITERTJS_VERSION = "2.6.0";
export declare const LITERT_WARNING = "ct0-edge: runtime \"litert\" is EXPERIMENTAL (LiteRT.js @litertjs/core 2.6.0, WebGPU with streamed weights; WASM only for small inline graphs).";
export interface TensorIO {
    name: string;
    shape: number[];
    dtype: string;
}
export interface LiteRtManifest {
    format: string;
    vocab: number;
    hidden: number;
    signatures: Record<string, {
        inputs: TensorIO[];
        outputs: TensorIO[];
    }>;
    files?: Record<string, {
        bytes: number;
        sha256: string;
    }>;
}
/** Token-embedding rows: `rows(ids)` returns ids.length x hidden floats. */
export interface EmbedSource {
    rows(ids: number[]): Promise<Float32Array>;
}
/** The subset of `@litertjs/core` this backend uses (injected so tests can fake it). */
export interface LiteRtJs {
    Tensor: new (data: any, shape: number[], dtype?: string) => any;
    loadAndCompile(model: Uint8Array | string, opts: {
        accelerator: string;
        gpuOptions?: {
            precision?: string;
        };
    }): Promise<any>;
    loadModelAndWeights?(model: Uint8Array, weights: ReadableStream<Uint8Array>, opts: {
        accelerator: string;
        gpuOptions?: {
            precision?: string;
        };
    }): Promise<any>;
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
export declare function embedFromQ8(q8: Int8Array | Uint8Array, scale: Float32Array, hidden: number): EmbedSource;
/** Embedding source over fp16 rows read on demand (`read(byteOffset, byteLength)`: HTTP Range, fs, or a buffer). */
export declare function embedFromF16Reader(read: (offset: number, length: number) => Promise<Uint8Array>, hidden: number): EmbedSource;
/** `read(offset, length)` over HTTP Range requests (falls back to slicing a full response if the server ignores Range). */
export declare function httpRangeReader(url: string, fetchFn?: typeof fetch): (offset: number, length: number) => Promise<Uint8Array>;
/** Embedding source that reads only the needed rows (`read(offset, length)`, e.g. HTTP Range or fs). */
export declare function embedFromReader(read: (offset: number, length: number) => Promise<Uint8Array>, scale: Float32Array, hidden: number): EmbedSource;
/** Additive (or boolean) attention mask [1,1,rows,cache]: row r sees cache columns <= pos[r]. */
export declare function buildMask(pos: Int32Array, cache: number, dtype: string): Float32Array | Uint8Array;
/** Splits `n` prompt tokens into prefill chunks of `P` (the last one right-padded). */
export declare function prefillChunks(n: number, P: number): Array<{
    start: number;
    len: number;
}>;
export declare function createLiteRt(lrt: LiteRtJs, graph: Uint8Array, manifest: LiteRtManifest, embed: EmbedSource, opts?: LiteRtOptions): Promise<Backend>;
