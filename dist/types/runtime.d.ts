import type { Backend } from "./backend.js";
export declare const DEFAULT_REPOS: {
    readonly genai: "eduardoworrel/alias3-micro-1026-ONNX-genai";
    readonly decodeOnly: "eduardoworrel/alias3-micro-1026-ONNX-decode-only";
    readonly onnx: "eduardoworrel/alias3-micro-1026-ONNX";
    readonly mlcQ4f16: "eduardoworrel/alias3-micro-1026-MLC-q4f16_1";
    readonly mlcQ4f32: "eduardoworrel/alias3-micro-1026-MLC-q4f32_1";
    readonly litert: "eduardoworrel/alias3-micro-1026-LiteRT";
};
export declare const LITERT_QUANTS: readonly ["int4", "int8", "int4-b64"];
export type LiteRtQuant = (typeof LITERT_QUANTS)[number];
export declare function litertBundleBase(o?: {
    litertBaseUrl?: string;
    litertRepo?: string;
    litertQuant?: string;
}): string;
export declare const DEFAULT_FILES: {
    readonly head: "heads/head.bin";
    readonly headScore: "heads/head_score.bin";
};
export declare function detectWebGPU(nav?: any): Promise<boolean>;
export type RuntimeName = "genai" | "decode-only" | "transformers.js" | "webllm" | "litert";
export type RuntimeChoice = RuntimeName | "auto";
export type Device = "webgpu" | "wasm" | "cpu" | "cuda";
export type DeviceChoice = Device | "auto";
export type Platform = "browser" | "node";
export type AskMethod = "head" | "logits";
export declare const RUNTIMES: readonly RuntimeName[];
export declare const RUNTIME_LIMITS: Record<RuntimeName, {
    maxSafeTokens: number;
    byDevice?: Partial<Record<Device, number>>;
}>;
export declare function maxSafeTokensFor(runtime: RuntimeName, device: Device): number;
export declare const MLC_Q4F32_LIB = "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/Qwen3.5-4B-q4f32_1_cs1k-webgpu.wasm";
export declare const RUNTIME_DEFAULTS: {
    readonly genai: {
        readonly repo: "eduardoworrel/alias3-micro-1026-ONNX-genai";
        readonly files: {
            readonly webgpu: "webgpu/int4/model.onnx";
            readonly cuda: "cuda/int4/model.onnx";
        };
        readonly heads: true;
        readonly methods: Readonly<Record<"noul" | "choice" | "score", "head" | "logits">>;
    };
    readonly "decode-only": {
        readonly repo: "eduardoworrel/alias3-micro-1026-ONNX-decode-only";
        readonly file: "onnx/model_q4.onnx";
        readonly heads: true;
        readonly methods: Readonly<Record<"noul" | "choice" | "score", "head" | "logits">>;
    };
    readonly "transformers.js": {
        readonly repo: "eduardoworrel/alias3-micro-1026-ONNX";
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
        readonly build: "q4f32_1";
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
        readonly files: {
            readonly manifest: "litert/manifest.json";
            readonly graph: "litert/model.graph.tflite";
            readonly weights: "litert/model.weights.bin";
            readonly embed: "litert/embed.f16.bin";
        };
        readonly wasm: "https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/";
        readonly heads: false;
        readonly methods: {
            readonly noul: "logits";
            readonly choice: "logits";
            readonly score: "logits";
        };
    };
};
export declare function defaultMethods(runtime: RuntimeName, opts?: {
    webllmRecovery?: boolean;
}): Readonly<Record<"noul" | "choice" | "score", AskMethod>>;
export declare function resolveRuntimeChoice(o: {
    runtime?: string;
}, fallback?: RuntimeChoice): {
    choice: RuntimeChoice;
};
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
export declare const GENAI_NODE_CPU_ERROR: string;
export declare function planRuntime(choice: RuntimeChoice, device: DeviceChoice | undefined, caps: Caps): Plan[];
export interface SelectRuntimeDeps {
    caps: Caps;
    load: (plan: Plan) => Promise<Backend>;
    onStatus?: (msg: string) => void;
}
export interface SelectedRuntime extends Plan {
    backend: Backend;
    fallbackReason?: string;
}
export declare function selectRuntime(choice: RuntimeChoice, device: DeviceChoice, deps: SelectRuntimeDeps): Promise<SelectedRuntime>;
export declare function packagesFor(runtime: RuntimeName, platform: Platform): string[];
export declare function importOptional<T = any>(pkg: string, importer: () => Promise<T>, why: string): Promise<T>;
export declare function detectCuda(ortNode: any): boolean;
