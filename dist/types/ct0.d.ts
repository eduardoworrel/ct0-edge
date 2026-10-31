import type { GenerateOptions } from "./backend.js";
import { type GenaiOptions } from "./backends/genai.js";
import { type Recovery } from "./webllm.js";
import { type Probe } from "./positions.js";
import { type AskOptions, type AskResult, type AskManyQuestion } from "./ask.js";
import { type RuntimeName, type RuntimeChoice, type DeviceChoice, type Device, type AskMethod } from "./runtime.js";
export type { AskHeads, AskOptions, AskResult, AskQuestion, AskManyQuestion } from "./ask.js";
export { DEFAULT_METHOD, askMany, levelFromCuts } from "./ask.js";
export type { Backend, GenerateOptions, StateHandle } from "./backend.js";
export { resolveGenerateOptions, GENERATE_DEFAULTS } from "./backend.js";
export { RUNTIMES, RUNTIME_LIMITS, RUNTIME_DEFAULTS, LITERT_QUANTS, litertBundleBase, MLC_Q4F32_LIB, GENAI_NODE_CPU_ERROR, maxSafeTokensFor, defaultMethods, resolveRuntimeChoice, detectPlatform, detectCuda, detectWebGPU, planRuntime, selectRuntime, packagesFor, importOptional, DEFAULT_REPOS, DEFAULT_FILES, } from "./runtime.js";
export type { RuntimeName, RuntimeChoice, Device, DeviceChoice, Platform, Caps, Plan, SelectRuntimeDeps, SelectedRuntime, LiteRtQuant } from "./runtime.js";
export interface MlcOptions {
    build?: "q4f16_1" | "q4f32_1";
    modelUrl?: string;
    modelId?: string;
    modelLib?: string;
    appConfig?: any;
    recovery?: Recovery | false;
    recoveryUrls?: {
        rows: string;
        P: string;
        format?: "f32" | "f16";
    };
    initProgressCallback?: (p: any) => void;
}
export interface LoadOptions {
    runtime?: RuntimeChoice;
    device?: DeviceChoice;
    repo?: string;
    baseUrl?: string;
    genaiRepo?: string;
    genaiBaseUrl?: string;
    genaiModel?: string;
    verifyEp?: boolean | "force";
    genai?: Omit<GenaiOptions, "ort">;
    decodeModel?: string;
    providers?: string[];
    externalDataFile?: string;
    pastLen?: number;
    transformersRepo?: string;
    dtype?: string;
    mlc?: MlcOptions;
    litertBaseUrl?: string;
    litertRepo?: string;
    litertQuant?: "int4" | "int8" | "int4-b64";
    litertWasmUrl?: string;
    litertPrecision?: "fp32" | "fp16";
    modules?: {
        ort?: any;
        transformers?: any;
        webllm?: any;
        litert?: any;
    };
    cacheDir?: string;
    heads?: boolean;
    maxSafeTokens?: number;
    skipVersionCheck?: boolean;
    onStatus?: (msg: string) => void;
}
export declare class CT0 {
    readonly maxContext = 262144;
    maxSafeTokens: number;
    private tok;
    private backend;
    private yesIds;
    private noIds;
    private constructor();
    private heads;
    private methods;
    runtime: RuntimeName;
    device: Device;
    fallbackReason?: string;
    dispose(): Promise<void>;
    get hiddenOutput(): string | null;
    get defaultMethod(): Readonly<Record<"noul" | "choice" | "score", AskMethod>>;
    static load(opts?: LoadOptions): Promise<CT0>;
    private ids;
    countTokens(text: string): number;
    fitContext(text: string, opts?: {
        reserve?: number;
        onTrim?: (i: {
            original: number;
            kept: number;
        }) => void;
    }): string;
    scanLong(text: string, question: string, opts?: {
        windowTokens?: number;
        overlapTokens?: number;
        threshold?: number;
        onChunk?: (i: number, p: number, total: number) => void;
    }): Promise<{
        chunks: number;
        max: number;
        hits: {
            i: number;
            start: number;
            p: number;
            preview: string;
        }[];
        flagged: {
            i: number;
            start: number;
            p: number;
            preview: string;
        }[];
    }>;
    static inspectPage(doc: any): any;
    static pageSource(doc: any, max?: number): string;
    private turn;
    generate(prompt: string, opts?: GenerateOptions): AsyncGenerator<string>;
    stream(prompt: string, opts?: GenerateOptions): AsyncGenerator<string>;
    decide(state: string, question: string): Promise<number>;
    loadHeads(src: {
        noul?: ArrayBuffer | string;
        choice?: ArrayBuffer | string;
        score?: ArrayBuffer | string;
    }): Promise<void>;
    ask(q: Probe & {
        levels?: number;
    }, opts?: AskOptions): Promise<AskResult>;
    askMany(state: string, qs: AskManyQuestion[], opts?: AskOptions): Promise<AskResult[]>;
    private askOpts;
    prefillHiddens(prompt: string): Promise<{
        atDecide: Float32Array;
        atLast: Float32Array;
        atOpts: Float32Array[];
    } | null>;
}
