/**
 * ct0.ts — CT0 (alias3-micro-1026) SDK for the browser and Node. One entry point, `CT0.load({ runtime, device })`, over
 * four runtimes that share the `Backend` contract (see runtime.ts):
 *
 *   const ct0 = await CT0.load();                               // auto: best measured runtime for this machine
 *   const ct0 = await CT0.load({ runtime: "genai" });           // ONNX Runtime GenAI graph
 *   const ct0 = await CT0.load({ runtime: "decode-only" });     // ONNX Runtime Web / Node
 *   const ct0 = await CT0.load({ runtime: "transformers.js" }); // optimum-exported graph via transformers.js
 *   const ct0 = await CT0.load({ runtime: "webllm" });          // MLC
 *
 * Same API everywhere: ask() · askMany() · generate() · stream() · dispose(); ct0.runtime / ct0.device / ct0.fallbackReason.
 * Runtime packages are optional peers, imported only when the chosen runtime needs them.
 */
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
export { RUNTIMES, RUNTIME_LIMITS, RUNTIME_DEFAULTS, MLC_Q4F32_LIB, GENAI_NODE_CPU_ERROR, maxSafeTokensFor, defaultMethods, resolveRuntimeChoice, detectPlatform, detectCuda, detectWebGPU, planRuntime, selectRuntime, packagesFor, importOptional, DEFAULT_REPOS, DEFAULT_FILES, } from "./runtime.js";
export type { RuntimeName, RuntimeChoice, Device, DeviceChoice, Platform, Caps, Plan, SelectRuntimeDeps, SelectedRuntime } from "./runtime.js";
/** WebLLM (MLC) options for `runtime: "webllm"`. */
export interface MlcOptions {
    /** MLC build: `"q4f32_1"` (default, faithful; its lib is the official WebLLM prebuilt wasm) or `"q4f16_1"` (faster, less faithful; wasm in its repo). */
    build?: "q4f16_1" | "q4f32_1";
    /** Model base URL (default `https://huggingface.co/<build repo>/`). */
    modelUrl?: string;
    /** Model id in the WebLLM app config (default `alias3-micro-1026-<build>`). */
    modelId?: string;
    /** Model lib (wasm) URL (default: the build repo's `model-lib.wasm`, or the official prebuilt for q4f32_1). */
    modelLib?: string;
    /** Full WebLLM app config (replaces modelUrl / modelId / modelLib). */
    appConfig?: any;
    /** Hidden-state recovery (h = P·logits[rows]) that enables the head reads. Default: `recovery/` of the build's repo.
     *  `false` skips it (every mode reads the logits). */
    recovery?: Recovery | false;
    recoveryUrls?: {
        rows: string;
        P: string;
        format?: "f32" | "f16";
    };
    initProgressCallback?: (p: any) => void;
}
export interface LoadOptions {
    /** `"auto"` (default): the best measured runtime for this machine. `"genai"`, `"decode-only"`, `"transformers.js"`, `"webllm"` force one. */
    runtime?: RuntimeChoice;
    /** `"auto"` (default), `"webgpu"` / `"wasm"` (browser), `"cpu"` / `"cuda"` (Node). */
    device?: DeviceChoice;
    /** Tokenizer repo and decode-only repo (default `eduardoworrel/alias3-micro-1026-ONNX-decode-only`). */
    repo?: string;
    /** Self-hosted decode-only files (URL, or a directory in Node). Without `runtime`, `"auto"` then uses only decode-only. */
    baseUrl?: string;
    /** genai repo (default `eduardoworrel/alias3-micro-1026-ONNX-genai`). */
    genaiRepo?: string;
    /** Self-hosted genai files (replaces `genaiRepo`). */
    genaiBaseUrl?: string;
    /** genai graph inside its base (default `webgpu/int4/model.onnx` in the browser, `cuda/int4/model.onnx` in Node; data `<graph>.data`). */
    genaiModel?: string;
    /** Browser genai: check once that the graph runs on the WebGPU EP (default false: the probe costs an extra session). */
    verifyEp?: boolean | "force";
    /** Extra genai backend options (chunk, ...). */
    genai?: Omit<GenaiOptions, "ort">;
    /** decode-only graph (default `onnx/model_q4.onnx`, data `<graph>_data`). */
    decodeModel?: string;
    /** decode-only execution providers (overrides the ones derived from `device`). */
    providers?: string[];
    /** decode-only external data file name, if not `decodeModel + "_data"`. */
    externalDataFile?: string;
    /** decode-only initial KV past length (default 1, as in the reference implementation). */
    pastLen?: number;
    /** transformers.js model repo (default `eduardoworrel/alias3-micro-1026-ONNX`). */
    transformersRepo?: string;
    /** transformers.js dtype (default `"q4"`). */
    dtype?: string;
    /** WebLLM options. */
    mlc?: MlcOptions;
    /** Use these modules instead of importing them (custom builds, or bundlers that cannot resolve an optional import). */
    modules?: {
        ort?: any;
        transformers?: any;
        webllm?: any;
    };
    /** Node: where downloaded graphs are cached (default `$CT0_CACHE` or `~/.cache/ct0-edge`). */
    cacheDir?: string;
    /** Load `heads/head.bin` and `heads/head_score.bin` from the runtime's repo (default true; webllm: when its recovery loaded). */
    heads?: boolean;
    /** Longest prompt `ask()` accepts (default per runtime: `RUNTIME_LIMITS`). */
    maxSafeTokens?: number;
    /** Do not require onnxruntime-web >= 1.30 (experiments only). */
    skipVersionCheck?: boolean;
    onStatus?: (msg: string) => void;
}
export declare class CT0 {
    /** Nominal model context (linear recurrence). */
    readonly maxContext = 262144;
    /** Longest prompt `ask()` accepts (default per runtime). */
    maxSafeTokens: number;
    private tok;
    private backend;
    private yesIds;
    private noIds;
    private constructor();
    private heads;
    private methods;
    /** Runtime in use. */
    runtime: RuntimeName;
    /** Device in use: "webgpu" | "wasm" | "cpu" | "cuda". */
    device: Device;
    /** Why `"auto"` did not use genai (undefined when it did, or when the runtime was forced). */
    fallbackReason?: string;
    /** Releases the graph / session. */
    dispose(): Promise<void>;
    /** Hidden-state output name of the graph (null when absent). */
    get hiddenOutput(): string | null;
    /** `ask()` default read per mode for this runtime. */
    get defaultMethod(): Readonly<Record<"noul" | "choice" | "score", AskMethod>>;
    static load(opts?: LoadOptions): Promise<CT0>;
    private ids;
    /** Counts the tokens of a text. */
    countTokens(text: string): number;
    /** Truncates `text` to fit `maxSafeTokens - reserve` (keeps the start + end,
     *  where the information usually is). Calls `onTrim` if it cuts. */
    fitContext(text: string, opts?: {
        reserve?: number;
        onTrim?: (i: {
            original: number;
            kept: number;
        }) => void;
    }): string;
    /**
     * chunk-watch: scans a LARGE text (over maxSafeTokens, e.g. huge HTML) in sliding windows,
     * running System 1 (`decide`) per chunk and accumulating the suspicious regions.
     * Covers documents that do not fit the context without truncating or losing signal.
     */
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
    /** Clean, condensed view of a page (for security agents): forms, inputs, buttons, links,
     *  HTML comments, inline scripts and localStorage, where vulnerabilities/secrets tend to hide, in little context. */
    static inspectPage(doc: any): any;
    /** Raw body HTML (fallback when a security review needs the full markup). */
    static pageSource(doc: any, max?: number): string;
    private turn;
    /** Single-turn generation: wraps `prompt` in a chat turn (System 2). */
    generate(prompt: string, opts?: GenerateOptions): AsyncGenerator<string>;
    /**
     * Low-level streaming from a RAW prompt (no chat-turn wrapping). Use this to
     * drive multi-turn / agentic (ReAct) loops where you build the full prompt
     * yourself. `temperature <= 0` ⇒ greedy (deterministic tool calls).
     */
    stream(prompt: string, opts?: GenerateOptions): AsyncGenerator<string>;
    decide(state: string, question: string): Promise<number>;
    /**
     * Loads the bilinear heads used by `ask(..., {method:"head"})`. Each source is an ArrayBuffer
     * (the .bin contents) or a URL. `noul` and `choice` come from the same original head.bin (if both
     * are given, `noul` wins); `score` is head_score.bin (required for score with method head; no fallback).
     */
    loadHeads(src: {
        noul?: ArrayBuffer | string;
        choice?: ArrayBuffer | string;
        score?: ArrayBuffer | string;
    }): Promise<void>;
    /** Typed System-1 (noul / choice / score). Default read per mode: `ct0.defaultMethod` (ONNX runtimes: noul logits,
     *  choice head, score head_score.bin; webllm too, through recovery, else logits). `opts.method` overrides it. */
    ask(q: Probe & {
        levels?: number;
    }, opts?: AskOptions): Promise<AskResult>;
    /** Several questions about one `state` (each question may omit `state`): same results as `ask()` for each, in order.
     *  genai and decode-only prefill the shared prefix (system turn + state) once and run only each question's suffix;
     *  transformers.js and webllm make independent `ask()` calls. */
    askMany(state: string, qs: AskManyQuestion[], opts?: AskOptions): Promise<AskResult[]>;
    private askOpts;
    /**
     * Full prefill returning the hidden states the bilinear HEAD needs.
     * Requires a graph with a hidden-state output (`backend.hiddenName`). Positions come from
     * `positions.ts`: q at the '>' of `<decide>`, k at the last token and, for
     * choice, at the token covering the end of each `</opt>`.
     */
    prefillHiddens(prompt: string): Promise<{
        atDecide: Float32Array;
        atLast: Float32Array;
        atOpts: Float32Array[];
    } | null>;
}
