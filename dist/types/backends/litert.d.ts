import { type Backend } from "../backend.js";
export declare const LITERTJS_VERSION = "2.6.0";
export declare const LITERT_WARNING = "ct0-edge: runtime \"litert\" is opt-in (LiteRT.js @litertjs/core 2.6.0, WebGPU with streamed weights; about 10x slower per decision than \"genai\"). Pass quiet: true to silence this note.";
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
export interface EmbedSource {
    rows(ids: number[]): Promise<Float32Array>;
}
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
    weights?: () => ReadableStream<Uint8Array>;
    gpuPrecision?: "fp16" | "fp32";
    copyState?: boolean;
    remainderViaDecode?: boolean;
    decodeOnlyPrefill?: boolean;
    quiet?: boolean;
}
export declare function embedFromQ8(q8: Int8Array | Uint8Array, scale: Float32Array, hidden: number): EmbedSource;
export declare function embedFromF16Reader(read: (offset: number, length: number) => Promise<Uint8Array>, hidden: number): EmbedSource;
export declare function httpRangeReader(url: string, fetchFn?: typeof fetch): (offset: number, length: number) => Promise<Uint8Array>;
export declare function embedFromReader(read: (offset: number, length: number) => Promise<Uint8Array>, scale: Float32Array, hidden: number): EmbedSource;
export declare function buildMask(pos: Int32Array, cache: number, dtype: string): Float32Array | Uint8Array;
export declare function prefillChunks(n: number, P: number): Array<{
    start: number;
    len: number;
}>;
export declare function createLiteRt(lrt: LiteRtJs, graph: Uint8Array, manifest: LiteRtManifest, embed: EmbedSource, opts?: LiteRtOptions): Promise<Backend>;
