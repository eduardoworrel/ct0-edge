/**
 * runtime.ts — runtime and device selection for `CT0.load()`, kept free of runtime packages so it can be tested with fakes.
 *
 * Runtimes (all behind the same `Backend` contract):
 *  - "genai":           ONNX Runtime GenAI graph (int4) on onnxruntime-web WebGPU (browser) or onnxruntime-node CUDA (Node).
 *  - "decode-only":     decode-only graph (q4) on onnxruntime-web (WebGPU / WASM) or onnxruntime-node (CPU / CUDA).
 *  - "transformers.js": optimum-exported graph (q4) through transformers.js (WebGPU / WASM in the browser, CPU in Node).
 *  - "webllm":          MLC build (q4f32_1; q4f16_1 is faster, less faithful) through WebLLM; browser with WebGPU only.
 *  - "litert":          EXPERIMENTAL. LiteRT.js (@litertjs/core 2.6.0) over a litert-torch .tflite (Gated-DeltaNet
 *                       decomposed) with the int4 weights streamed into WebGPU buffers; browser WebGPU only (the weights
 *                       do not fit the 2 GiB Wasm heap). Logits reads only. Never picked by "auto".
 *
 * `"auto"` picks from measured results: browser + WebGPU -> genai (fallback decode-only WebGPU); browser without WebGPU ->
 * decode-only WASM; Node + CUDA -> genai (fallback transformers.js CUDA, then CPU); Node CPU -> transformers.js CPU
 * (fallback decode-only CPU). On Node, transformers.js is the fastest measured path after genai (CPU ~8–10 s vs ~58 s
 * per question for decode-only; CUDA ~0.4–1 s vs ~4–6 s).
 * genai on onnxruntime-node CPU fails at run time ("not enough space: expected 49152, got 0"), so it is never picked on CPU
 * and forcing it throws.
 */
import type { Backend } from "./backend.js";
/** Public Hugging Face repos used by default. */
export declare const DEFAULT_REPOS: {
    readonly genai: "eduardoworrel/alias3-micro-1026-ONNX-genai";
    readonly decodeOnly: "eduardoworrel/alias3-micro-1026-ONNX-decode-only";
    /** optimum/Scan graph (transformers.js). */
    readonly onnx: "eduardoworrel/alias3-micro-1026-ONNX";
    readonly mlcQ4f16: "eduardoworrel/alias3-micro-1026-MLC-q4f16_1";
    readonly mlcQ4f32: "eduardoworrel/alias3-micro-1026-MLC-q4f32_1";
    /** NOT published yet (the LiteRT bundle waits for the owner's OK); pass `litertBaseUrl` to self-host. */
    readonly litert: "eduardoworrel/alias3-micro-1026-LiteRT";
};
/** Head files inside each runtime's repo (graph files: `RUNTIME_DEFAULTS`). */
export declare const DEFAULT_FILES: {
    readonly head: "heads/head.bin";
    readonly headScore: "heads/head_score.bin";
};
/** true when `navigator.gpu` exists and returns an adapter. Never throws. */
export declare function detectWebGPU(nav?: any): Promise<boolean>;
export type RuntimeName = "genai" | "decode-only" | "transformers.js" | "webllm" | "litert";
export type RuntimeChoice = RuntimeName | "auto";
export type Device = "webgpu" | "wasm" | "cpu" | "cuda";
export type DeviceChoice = Device | "auto";
export type Platform = "browser" | "node";
export type AskMethod = "head" | "logits";
export declare const RUNTIMES: readonly RuntimeName[];
/**
 * Longest prompt `ask()` accepts, per runtime (and optionally per device). `maxSafeTokens` in `LoadOptions` overrides it.
 * Measured cells are the longest prompt that kept every decision equal to the reference, minus 10%
 * (genai WebGPU passed 8000 tokens; decode-only passed 4000 in Chrome WebGPU and 8000 on Node CPU).
 * Cells without a measurement keep 2048.
 */
export declare const RUNTIME_LIMITS: Record<RuntimeName, {
    maxSafeTokens: number;
    byDevice?: Partial<Record<Device, number>>;
}>;
export declare function maxSafeTokensFor(runtime: RuntimeName, device: Device): number;
/** MLC lib (wasm) for q4f32_1: the official WebLLM prebuilt (the q4f32_1 repo ships no wasm). */
export declare const MLC_Q4F32_LIB = "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/Qwen3.5-4B-q4f32_1_cs1k-webgpu.wasm";
/** Default model files, heads and `ask()` reads per runtime. */
export declare const RUNTIME_DEFAULTS: {
    readonly genai: {
        readonly repo: "eduardoworrel/alias3-micro-1026-ONNX-genai";
        /** Graph per device; its data file is `<graph>.data`. */
        readonly files: {
            readonly webgpu: "webgpu/int4/model.onnx";
            readonly cuda: "cuda/int4/model.onnx";
        };
        readonly heads: true;
        readonly methods: Readonly<Record<"noul" | "choice" | "score", "head" | "logits">>;
    };
    readonly "decode-only": {
        readonly repo: "eduardoworrel/alias3-micro-1026-ONNX-decode-only";
        /** Data file: `<graph>_data`. */
        readonly file: "onnx/model_q4.onnx";
        readonly heads: true;
        readonly methods: Readonly<Record<"noul" | "choice" | "score", "head" | "logits">>;
    };
    readonly "transformers.js": {
        readonly repo: "eduardoworrel/alias3-micro-1026-ONNX";
        /** transformers.js dtype: loads `onnx/model_q4.onnx`. */
        readonly dtype: "q4";
        readonly heads: true;
        readonly methods: Readonly<Record<"noul" | "choice" | "score", "head" | "logits">>;
    };
    readonly webllm: {
        readonly builds: {
            readonly q4f16_1: {
                readonly repo: "eduardoworrel/alias3-micro-1026-MLC-q4f16_1";
                readonly modelId: "alias3-micro-1026-q4f16_1";
                readonly lib: "model-lib.wasm";
                readonly recovery: {
                    readonly rows: "recovery/P_q4f16_rows.i32";
                    readonly P: "recovery/P_q4f16_8192.f16";
                };
            };
            readonly q4f32_1: {
                readonly repo: "eduardoworrel/alias3-micro-1026-MLC-q4f32_1";
                readonly modelId: "alias3-micro-1026-q4f32_1";
                readonly lib: "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/Qwen3.5-4B-q4f32_1_cs1k-webgpu.wasm";
                readonly recovery: {
                    readonly rows: "recovery/P_q4f32_rows.i32";
                    readonly P: "recovery/P_q4f32_8192.f16";
                };
            };
        };
        /** q4f32_1: the faithful build (hidden cos 0.98 vs 0.86, keeps `<think>`, yes/no 54/65 vs 52/65); q4f16_1 is faster. */
        readonly build: "q4f32_1";
        /** heads/ and recovery/ come from the build's own repo; the hidden state is recovered from the logits (h = P·logits[rows]),
         *  so the reads are the ONNX ones. If the recovery files fail to load, every mode falls back to the logits. */
        readonly heads: true;
        readonly methods: Readonly<Record<"noul" | "choice" | "score", "head" | "logits">>;
        readonly logitsOnlyMethods: {
            readonly noul: "logits";
            readonly choice: "logits";
            readonly score: "logits";
        };
    };
    readonly litert: {
        readonly repo: "eduardoworrel/alias3-micro-1026-LiteRT";
        /** Bundle produced by litert-exp/rb_build.py: manifest + graph + streamed weights + fp16 embedding rows. */
        readonly files: {
            readonly manifest: "litert/manifest.json";
            readonly graph: "litert/model.graph.tflite";
            readonly weights: "litert/model.weights.bin";
            readonly embed: "litert/embed.f16.bin";
        };
        /** @litertjs/core wasm files (pinned; the npm package's wasm/ directory). */
        readonly wasm: "https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/";
        readonly heads: false;
        readonly methods: {
            readonly noul: "logits";
            readonly choice: "logits";
            readonly score: "logits";
        };
    };
};
/** `ask()` default read per mode for a runtime. WebLLM with recovery matrices reads like the ONNX runtimes. */
export declare function defaultMethods(runtime: RuntimeName, opts?: {
    webllmRecovery?: boolean;
}): Readonly<Record<"noul" | "choice" | "score", AskMethod>>;
/** Resolves `runtime` to a runtime choice (`fallback` when unset). Throws on unknown values. */
export declare function resolveRuntimeChoice(o: {
    runtime?: string;
}, fallback?: RuntimeChoice): {
    choice: RuntimeChoice;
};
/** "node" when running under Node (or Bun/Deno without a DOM), else "browser" (web pages, workers, Electron renderers). */
export declare function detectPlatform(g?: any): Platform;
export interface Caps {
    platform: Platform;
    webgpu: boolean;
    cuda: boolean;
}
export interface Plan {
    runtime: RuntimeName;
    device: Device;
}
/** Error for genai on onnxruntime-node CPU. */
export declare const GENAI_NODE_CPU_ERROR: string;
/**
 * Ordered candidates for a choice. An explicit runtime gives one candidate (no fallback); `"auto"` gives the measured
 * best first and its fallback second. Throws on impossible pairs (genai on Node CPU, webllm in Node, cuda in a browser...).
 */
export declare function planRuntime(choice: RuntimeChoice, device: DeviceChoice | undefined, caps: Caps): Plan[];
export interface SelectRuntimeDeps {
    caps: Caps;
    load: (plan: Plan) => Promise<Backend>;
    onStatus?: (msg: string) => void;
}
export interface SelectedRuntime extends Plan {
    backend: Backend;
    /** Why `"auto"` did not use genai (absent when it did, or when the runtime was forced). */
    fallbackReason?: string;
}
/** Plans and loads: tries each candidate in order; `"auto"` falls back and records why. */
export declare function selectRuntime(choice: RuntimeChoice, device: DeviceChoice, deps: SelectRuntimeDeps): Promise<SelectedRuntime>;
/** npm package that provides each runtime on each platform (besides @huggingface/transformers for the tokenizer). */
export declare function packagesFor(runtime: RuntimeName, platform: Platform): string[];
/** Imports an optional peer package; when it is missing, throws an error that names the package to install. */
export declare function importOptional<T = any>(pkg: string, importer: () => Promise<T>, why: string): Promise<T>;
/** true when onnxruntime-node lists the CUDA execution provider. Never throws. */
export declare function detectCuda(ortNode: any): boolean;
