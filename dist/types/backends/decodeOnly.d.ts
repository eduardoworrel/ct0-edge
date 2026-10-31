import { type Backend } from "../backend.js";
export { resolveGenerateOptions, GENERATE_DEFAULTS, createGate } from "../backend.js";
export { checkOrtVersion } from "../ortVersion.js";
export declare const EOS: Set<number>;
export declare const HIDDEN_NAMES: string[];
export declare function mulberry32(seed: number): () => number;
export declare function halfToNumber(h: number): number;
export declare function tensorToF32(data: any, type: string | undefined, off?: number, n?: number): Float32Array;
export declare function lastRow(L: any): Float32Array;
export declare function sample(logits: Float32Array, gen: number[], rand: () => number, o: any): number;
export interface DecodeOnlyOptions {
    ort: {
        Tensor: new (type: any, data: any, dims: number[]) => any;
        env?: {
            versions?: {
                web?: string;
            };
        };
    };
    skipVersionCheck?: boolean;
    pastLen?: number;
}
export declare function pastNameFor(out: string): string | null;
export declare function cacheDims(name: string, shape: any[], pastLen: number): number[];
export declare function createDecodeOnly(session: any, tok: any, opts: DecodeOnlyOptions): Promise<Backend>;
