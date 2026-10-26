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
import type { Backend } from "../backend.js";
import { checkOrtVersion } from "../ortVersion.js";
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
    epCache?: {
        getItem(k: string): string | null;
        setItem(k: string, v: string): void;
    } | null;
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
/** "Node(s) placed on [XExecutionProvider]. Number of nodes: N" -> { XExecutionProvider: N }. */
export declare function parsePlacement(lines: string[]): Record<string, number>;
/** Throws if the graph is not (almost entirely) on the WebGPU EP. */
export declare function assertWebGpuPlacement(p: Record<string, number>, maxCpuNodes?: number): void;
export { checkOrtVersion };
export type GenaiBackend = Backend & {
    chunk: number;
};
export declare function genaiFromSession(session: any, ort: {
    Tensor: new (type: any, data: any, dims: number[], ...r: any[]) => any;
}, opts?: GenaiOptions): GenaiBackend;
/** Hash FNV-1a (hex) curto. */
export declare function hashStr(x: string): string;
/** Loads the WebGPU graph (`modelUrl` = model_webgpu.onnx, `dataUrl` = model.onnx.data) and returns the backend. */
export declare function createGenai(modelUrl: string, dataUrl: string, opts?: GenaiOptions): Promise<GenaiBackend>;
