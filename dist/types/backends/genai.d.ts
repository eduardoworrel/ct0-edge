import type { Backend } from "../backend.js";
import { checkOrtVersion } from "../ortVersion.js";
export interface GenaiOptions {
    providers?: string[];
    ort?: any;
    chunk?: number;
    verifyEp?: boolean | "force";
    epCache?: {
        getItem(k: string): string | null;
        setItem(k: string, v: string): void;
    } | null;
    adapterTag?: string | (() => Promise<string>);
    maxCpuNodes?: number;
    kvHeadDim?: number;
    skipVersionCheck?: boolean;
    warmup?: boolean;
    eos?: Set<number>;
}
export declare function parsePlacement(lines: string[]): Record<string, number>;
export declare function assertWebGpuPlacement(p: Record<string, number>, maxCpuNodes?: number): void;
export { checkOrtVersion };
export type GenaiBackend = Backend & {
    chunk: number;
};
export declare function genaiFromSession(session: any, ort: {
    Tensor: new (type: any, data: any, dims: number[], ...r: any[]) => any;
}, opts?: GenaiOptions): GenaiBackend;
export declare function hashStr(x: string): string;
export declare function createGenai(modelUrl: string, dataUrl: string, opts?: GenaiOptions): Promise<GenaiBackend>;
