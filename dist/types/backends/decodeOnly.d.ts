/**
/**
 * decodeOnly.ts — backend for the decode-only graph (1 token per run, no attention_mask).
 * Caches are NOT hard-coded: names, dtypes and shapes come from `session.inputMetadata`, and the
 * hidden output is detected in `session.outputNames`. Mirrors the reference `run_prompt`:
 * symbolic dims -> 1, zeros, and present_* -> past_* at each step.
 */
import { type Backend } from "../backend.js";
export { resolveGenerateOptions, GENERATE_DEFAULTS, createGate } from "../backend.js";
export { checkOrtVersion } from "../ortVersion.js";
export declare const EOS: Set<number>;
export declare const HIDDEN_NAMES: string[];
export declare function mulberry32(seed: number): () => number;
/** Decodes an fp16 (bit pattern) to a number. `Float32Array.from(Uint16Array)` does NOT do this. */
export declare function halfToNumber(h: number): number;
/** Converts an ORT tensor's data (Float32Array, native Float16Array or raw fp16 Uint16Array)
 *  to a Float32Array, starting at `off` for `n` elements. */
export declare function tensorToF32(data: any, type: string | undefined, off?: number, n?: number): Float32Array;
/** Last row ([.., T, V] -> V) of a logits tensor. */
export declare function lastRow(L: any): Float32Array;
export declare function sample(logits: Float32Array, gen: number[], rand: () => number, o: any): number;
export interface DecodeOnlyOptions {
    /** onnxruntime module (needs `Tensor`): `ort-web` or `ort-node`. */
    ort: {
        Tensor: new (type: any, data: any, dims: number[]) => any;
        env?: {
            versions?: {
                web?: string;
            };
        };
    };
    /** Skip the onnxruntime-web >= 1.30 requirement (experiments only). */
    skipVersionCheck?: boolean;
    /** Initial length of the KV (past) sequence axis. Default 1.
     *  The reference implementation maps every symbolic dim to 1, i.e. starts with a past of length 1
     *  (one zeroed K/V); parity with the published numbers depends on it. Batch and other symbolic dims are always 1. */
    pastLen?: number;
}
/** present -> past. Optimum convention (`present_conv.N`, `present.N.key`) and the old one (`so<i>` -> `si<i>`). */
export declare function pastNameFor(out: string): string | null;
/** Dims of a zeroed cache: numeric ones stay; symbolic ones become 1, except the KV sequence axis (2) -> pastLen. */
export declare function cacheDims(name: string, shape: any[], pastLen: number): number[];
export declare function createDecodeOnly(session: any, tok: any, opts: DecodeOnlyOptions): Promise<Backend>;
