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
import { DEFAULT_METHOD } from "./ask.js";

/** Public Hugging Face repos used by default. */
export const DEFAULT_REPOS = {
  genai: "eduardoworrel/alias3-micro-1026-ONNX-genai",
  decodeOnly: "eduardoworrel/alias3-micro-1026-ONNX-decode-only",
  /** optimum/Scan graph (transformers.js). */
  onnx: "eduardoworrel/alias3-micro-1026-ONNX",
  mlcQ4f16: "eduardoworrel/alias3-micro-1026-MLC-q4f16_1",
  mlcQ4f32: "eduardoworrel/alias3-micro-1026-MLC-q4f32_1",
  /** NOT published yet (the LiteRT bundle waits for the owner's OK); pass `litertBaseUrl` to self-host. */
  litert: "eduardoworrel/alias3-micro-1026-LiteRT",
} as const;

/** Head files inside each runtime's repo (graph files: `RUNTIME_DEFAULTS`). */
export const DEFAULT_FILES = {
  head: "heads/head.bin",
  headScore: "heads/head_score.bin",
} as const;

/** true when `navigator.gpu` exists and returns an adapter. Never throws. */
export async function detectWebGPU(nav: any = (globalThis as any).navigator): Promise<boolean> {
  try {
    const gpu = nav?.gpu;
    if (!gpu?.requestAdapter) return false;
    return !!(await gpu.requestAdapter());
  } catch { return false; }
}

export type RuntimeName = "genai" | "decode-only" | "transformers.js" | "webllm" | "litert";
export type RuntimeChoice = RuntimeName | "auto";
export type Device = "webgpu" | "wasm" | "cpu" | "cuda";
export type DeviceChoice = Device | "auto";
export type Platform = "browser" | "node";
export type AskMethod = "head" | "logits";

export const RUNTIMES: readonly RuntimeName[] = ["genai", "decode-only", "transformers.js", "webllm", "litert"];
const DEVICES: readonly Device[] = ["webgpu", "wasm", "cpu", "cuda"];

/**
 * Longest prompt `ask()` accepts, per runtime (and optionally per device). `maxSafeTokens` in `LoadOptions` overrides it.
 * Measured cells are the longest prompt that kept every decision equal to the reference, minus 10%
 * (genai WebGPU passed 8000 tokens; decode-only passed 4000 in Chrome WebGPU and 8000 on Node CPU).
 * Cells without a measurement keep 2048.
 */
export const RUNTIME_LIMITS: Record<RuntimeName, { maxSafeTokens: number; byDevice?: Partial<Record<Device, number>> }> = {
  genai: { maxSafeTokens: 2048, byDevice: { webgpu: 7200 } },
  "decode-only": { maxSafeTokens: 2048, byDevice: { webgpu: 3600, cpu: 7200 } },
  "transformers.js": { maxSafeTokens: 2048 },
  webllm: { maxSafeTokens: 2048 },
  litert: { maxSafeTokens: 2000 },
};

export function maxSafeTokensFor(runtime: RuntimeName, device: Device): number {
  const l = RUNTIME_LIMITS[runtime];
  return l.byDevice?.[device] ?? l.maxSafeTokens;
}

/** MLC lib (wasm) for q4f32_1: the official WebLLM prebuilt (the q4f32_1 repo ships no wasm). */
export const MLC_Q4F32_LIB = "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/Qwen3.5-4B-q4f32_1_cs1k-webgpu.wasm";

/** Default model files, heads and `ask()` reads per runtime. */
export const RUNTIME_DEFAULTS = {
  genai: {
    repo: DEFAULT_REPOS.genai,
    /** Graph per device; its data file is `<graph>.data`. */
    files: { webgpu: "webgpu/int4/model.onnx", cuda: "cuda/int4/model.onnx" },
    heads: true,
    methods: DEFAULT_METHOD,
  },
  "decode-only": {
    repo: DEFAULT_REPOS.decodeOnly,
    /** Data file: `<graph>_data`. */
    file: "onnx/model_q4.onnx",
    heads: true,
    methods: DEFAULT_METHOD,
  },
  "transformers.js": {
    repo: DEFAULT_REPOS.onnx,
    /** transformers.js dtype: loads `onnx/model_q4.onnx`. */
    dtype: "q4",
    heads: true,
    methods: DEFAULT_METHOD,
  },
  webllm: {
    builds: {
      q4f16_1: { repo: DEFAULT_REPOS.mlcQ4f16, modelId: "alias3-micro-1026-q4f16_1", lib: "model-lib.wasm",
        recovery: { rows: "recovery/P_q4f16_rows.i32", P: "recovery/P_q4f16_8192.f16" } },
      q4f32_1: { repo: DEFAULT_REPOS.mlcQ4f32, modelId: "alias3-micro-1026-q4f32_1", lib: MLC_Q4F32_LIB,
        recovery: { rows: "recovery/P_q4f32_rows.i32", P: "recovery/P_q4f32_8192.f16" } },
    },
    /** q4f32_1: the faithful build (hidden cos 0.98 vs 0.86, keeps `<think>`, yes/no 54/65 vs 52/65); q4f16_1 is faster. */
    build: "q4f32_1",
    /** heads/ and recovery/ come from the build's own repo; the hidden state is recovered from the logits (h = P·logits[rows]),
     *  so the reads are the ONNX ones. If the recovery files fail to load, every mode falls back to the logits. */
    heads: true,
    methods: DEFAULT_METHOD,
    logitsOnlyMethods: { noul: "logits", choice: "logits", score: "logits" },
  },
  litert: {
    repo: DEFAULT_REPOS.litert,
    /** Bundle produced by litert-exp/rb_build.py: manifest + graph + streamed weights + fp16 embedding rows. */
    files: { manifest: "litert/manifest.json", graph: "litert/model.graph.tflite", weights: "litert/model.weights.bin", embed: "litert/embed.f16.bin" },
    /** @litertjs/core wasm files (pinned; the npm package's wasm/ directory). */
    wasm: "https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/",
    heads: false,
    methods: { noul: "logits", choice: "logits", score: "logits" },
  },
} as const;

/** `ask()` default read per mode for a runtime. WebLLM with recovery matrices reads like the ONNX runtimes. */
export function defaultMethods(runtime: RuntimeName, opts: { webllmRecovery?: boolean } = {}): Readonly<Record<"noul" | "choice" | "score", AskMethod>> {
  if (runtime === "webllm" && !opts.webllmRecovery) return RUNTIME_DEFAULTS.webllm.logitsOnlyMethods;
  if (runtime === "litert") return RUNTIME_DEFAULTS.litert.methods;
  return DEFAULT_METHOD;
}

/** Resolves `runtime` to a runtime choice (`fallback` when unset). Throws on unknown values. */
export function resolveRuntimeChoice(o: { runtime?: string }, fallback: RuntimeChoice = "auto"): { choice: RuntimeChoice } {
  if (o.runtime === undefined) return { choice: fallback };
  if (o.runtime !== "auto" && !RUNTIMES.includes(o.runtime as RuntimeName))
    throw new Error(`ct0-edge: unknown runtime "${o.runtime}". Use "auto", ${RUNTIMES.map((r) => `"${r}"`).join(", ")}.`);
  return { choice: o.runtime as RuntimeChoice };
}

/** "node" when running under Node (or Bun/Deno without a DOM), else "browser" (web pages, workers, Electron renderers). */
export function detectPlatform(g: any = globalThis): Platform {
  const isNode = !!g.process?.versions?.node;
  const hasDom = typeof g.window !== "undefined" && typeof g.document !== "undefined";
  const isWorker = typeof g.WorkerGlobalScope !== "undefined" && typeof g.importScripts === "function";
  return isNode && !hasDom && !isWorker ? "node" : "browser";
}

export interface Caps { platform: Platform; webgpu: boolean; cuda: boolean }

export interface Plan { runtime: RuntimeName; device: Device }

/** Error for genai on onnxruntime-node CPU. */
export const GENAI_NODE_CPU_ERROR = 'ct0-edge: runtime "genai" does not run on onnxruntime-node CPU (the int4 graph fails with "not enough space: expected 49152, got 0"). '
  + 'In Node use runtime "decode-only" (CPU) or "transformers.js", or run genai on a CUDA machine (device "cuda").';

function checkDevice(platform: Platform, device: Device): void {
  if (!DEVICES.includes(device)) throw new Error(`ct0-edge: unknown device "${device}". Use "auto", ${DEVICES.map((d) => `"${d}"`).join(", ")}.`);
  if (platform === "browser" && device === "cuda") throw new Error('ct0-edge: device "cuda" is Node-only. In the browser use "webgpu" or "wasm".');
  if (platform === "node" && (device === "webgpu" || device === "wasm")) throw new Error(`ct0-edge: device "${device}" is browser-only. In Node use "cpu" or "cuda".`);
}

/** Device for an explicit runtime: validates the pair, or picks the default device for the platform. */
function deviceFor(runtime: RuntimeName, want: DeviceChoice, caps: Caps): Device {
  const { platform } = caps;
  // in the browser "cpu" means the WASM CPU backend
  const dev: DeviceChoice = platform === "browser" && want === "cpu" ? "wasm" : want;
  if (dev !== "auto") checkDevice(platform, dev);
  const needGpu = (d: Device) => { if (d === "webgpu" && !caps.webgpu) throw new Error(`ct0-edge: runtime "${runtime}" on WebGPU, but WebGPU is not available in this browser.`); return d; };
  const needCuda = (d: Device) => { if (d === "cuda" && !caps.cuda) throw new Error(`ct0-edge: runtime "${runtime}" on CUDA, but onnxruntime-node lists no CUDA execution provider here.`); return d; };
  switch (runtime) {
    case "genai":
      if (platform === "browser") {
        if (dev === "wasm") throw new Error('ct0-edge: runtime "genai" needs WebGPU in the browser (its LinearAttention kernels have no WASM build). Use runtime "decode-only" on "wasm".');
        return needGpu("webgpu");
      }
      if (dev === "cpu") throw new Error(GENAI_NODE_CPU_ERROR);
      if (!caps.cuda) throw new Error(GENAI_NODE_CPU_ERROR + " No CUDA execution provider was found.");
      return "cuda";
    case "webllm":
      if (platform === "node") throw new Error('ct0-edge: runtime "webllm" runs only in the browser (WebGPU). In Node use "decode-only" or "transformers.js".');
      if (dev === "wasm") throw new Error('ct0-edge: runtime "webllm" needs WebGPU.');
      return needGpu("webgpu");
    case "litert":
      if (platform === "node") throw new Error('ct0-edge: runtime "litert" runs in the browser (LiteRT.js streams the weights into WebGPU buffers). In Node use "decode-only" or "transformers.js".');
      if (dev === "wasm") throw new Error('ct0-edge: runtime "litert" needs WebGPU (the CT0 weights exceed the 2 GiB LiteRT.js Wasm heap).');
      return needGpu("webgpu");
    case "decode-only":
    case "transformers.js":
      if (dev !== "auto") return platform === "browser" ? needGpu(dev) : needCuda(dev);
      return platform === "browser" ? (caps.webgpu ? "webgpu" : "wasm") : "cpu";
  }
}

/**
 * Ordered candidates for a choice. An explicit runtime gives one candidate (no fallback); `"auto"` gives the measured
 * best first and its fallback second. Throws on impossible pairs (genai on Node CPU, webllm in Node, cuda in a browser...).
 */
export function planRuntime(choice: RuntimeChoice, device: DeviceChoice = "auto", caps: Caps): Plan[] {
  if (choice !== "auto") {
    if (!RUNTIMES.includes(choice)) throw new Error(`ct0-edge: unknown runtime "${String(choice)}".`);
    return [{ runtime: choice, device: deviceFor(choice, device, caps) }];
  }
  const { platform } = caps;
  const dev: DeviceChoice = platform === "browser" && device === "cpu" ? "wasm" : device;
  if (dev !== "auto") checkDevice(platform, dev);
  if (platform === "browser") {
    if (dev === "wasm") return [{ runtime: "decode-only", device: "wasm" }];
    if (dev === "webgpu" && !caps.webgpu) throw new Error('ct0-edge: device "webgpu" requested, but WebGPU is not available in this browser.');
    if (caps.webgpu) return [{ runtime: "genai", device: "webgpu" }, { runtime: "decode-only", device: "webgpu" }];
    return [{ runtime: "decode-only", device: "wasm" }];
  }
  const nodeCpu: Plan[] = [{ runtime: "transformers.js", device: "cpu" }, { runtime: "decode-only", device: "cpu" }];
  if (dev === "cpu") return nodeCpu;
  if (dev === "cuda") {
    if (!caps.cuda) throw new Error('ct0-edge: device "cuda" requested, but onnxruntime-node lists no CUDA execution provider here.');
    return [{ runtime: "genai", device: "cuda" }, { runtime: "transformers.js", device: "cuda" }];
  }
  if (caps.cuda) return [{ runtime: "genai", device: "cuda" }, { runtime: "transformers.js", device: "cuda" }, ...nodeCpu];
  return nodeCpu;
}

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
export async function selectRuntime(choice: RuntimeChoice, device: DeviceChoice, deps: SelectRuntimeDeps): Promise<SelectedRuntime> {
  const status = deps.onStatus ?? (() => {});
  const plans = planRuntime(choice, device, deps.caps);
  if (choice !== "auto") return { ...plans[0], backend: await deps.load(plans[0]) };
  const reasons: string[] = [];
  if (plans[0].runtime !== "genai") {
    const { platform, webgpu, cuda } = deps.caps;
    reasons.push(platform === "browser"
      ? (webgpu ? `device "${plans[0].device}" requested` : "WebGPU not available")
      : (cuda ? `device "${plans[0].device}" requested` : "no CUDA execution provider (genai does not run on onnxruntime-node CPU)"));
  }
  for (const [i, plan] of plans.entries()) {
    if (i > 0) status(`${reasons.at(-1)} -> ${plan.runtime} (${plan.device})`);
    try {
      const backend = await deps.load(plan);
      return { ...plan, backend, fallbackReason: reasons.length ? reasons.join("; ") : undefined };
    } catch (e: any) {
      reasons.push(`${plan.runtime} (${plan.device}) unavailable: ${e?.message ?? String(e)}`);
      if (i === plans.length - 1) {
        const err = new Error(`ct0-edge: no runtime loaded. ${reasons.join("; ")}`);
        (err as any).cause = e;
        throw err;
      }
    }
  }
  throw new Error("ct0-edge: no runtime candidate"); // unreachable: planRuntime always returns >= 1
}

// ---------------------------------------------------------------- optional packages
/** npm package that provides each runtime on each platform (besides @huggingface/transformers for the tokenizer). */
export function packagesFor(runtime: RuntimeName, platform: Platform): string[] {
  const ort = platform === "node" ? "onnxruntime-node" : "onnxruntime-web";
  if (runtime === "genai" || runtime === "decode-only") return ["@huggingface/transformers", ort];
  if (runtime === "webllm") return ["@huggingface/transformers", "@mlc-ai/web-llm"];
  if (runtime === "litert") return ["@huggingface/transformers", "@litertjs/core"];
  return ["@huggingface/transformers"];
}

function isNotFound(e: any, pkg: string): boolean {
  const msg = String(e?.message ?? e);
  // Node: "Cannot find package 'x' imported from ..." (a missing dependency OF the package names another one: rethrown)
  if (e?.code === "ERR_MODULE_NOT_FOUND" || e?.code === "MODULE_NOT_FOUND" || /Cannot find (package|module)/.test(msg)) return msg.includes(pkg);
  // browsers: bare specifier not mapped / module script not fetched
  return /Failed to resolve module specifier|Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed/i.test(msg);
}

/** Imports an optional peer package; when it is missing, throws an error that names the package to install. */
export async function importOptional<T = any>(pkg: string, importer: () => Promise<T>, why: string): Promise<T> {
  try { return await importer(); }
  catch (e: any) {
    if (!isNotFound(e, pkg)) throw e;
    const err = new Error(`ct0-edge: ${why} needs the package "${pkg}", which is not installed. Install it: npm i ${pkg}`);
    (err as any).cause = e;
    throw err;
  }
}

/** true when onnxruntime-node lists the CUDA execution provider. Never throws. */
export function detectCuda(ortNode: any): boolean {
  try { return !!ortNode?.listSupportedBackends?.().some((b: any) => b?.name === "cuda"); }
  catch { return false; }
}
