
import type { Backend } from "./backend.js";
import { DEFAULT_METHOD } from "./ask.js";

export const DEFAULT_REPOS = {
  genai: "eduardoworrel/alias3-micro-1026-ONNX-genai",
  decodeOnly: "eduardoworrel/alias3-micro-1026-ONNX-decode-only",

  onnx: "eduardoworrel/alias3-micro-1026-ONNX",
  mlcQ4f16: "eduardoworrel/alias3-micro-1026-MLC-q4f16_1",
  mlcQ4f32: "eduardoworrel/alias3-micro-1026-MLC-q4f32_1",

  litert: "eduardoworrel/alias3-micro-1026-LiteRT",
} as const;

export const LITERT_QUANTS = ["int4", "int8", "int4-b64"] as const;
export type LiteRtQuant = (typeof LITERT_QUANTS)[number];

export function litertBundleBase(o: { litertBaseUrl?: string; litertRepo?: string; litertQuant?: string } = {}): string {
  if (o.litertBaseUrl) return o.litertBaseUrl;
  const q = o.litertQuant ?? "int4";
  if (!(LITERT_QUANTS as readonly string[]).includes(q)) throw new Error(`ct0-edge: litertQuant must be one of ${LITERT_QUANTS.join(", ")} (got ${JSON.stringify(q)})`);
  return `https://huggingface.co/${o.litertRepo ?? DEFAULT_REPOS.litert}/resolve/main/litert/${q}/`;
}

export const DEFAULT_FILES = {
  head: "heads/head.bin",
  headScore: "heads/head_score.bin",
} as const;

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

export const MLC_Q4F32_LIB = "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/Qwen3.5-4B-q4f32_1_cs1k-webgpu.wasm";

export const RUNTIME_DEFAULTS = {
  genai: {
    repo: DEFAULT_REPOS.genai,

    files: { webgpu: "webgpu/int4/model.onnx", cuda: "cuda/int4/model.onnx" },
    heads: true,
    methods: DEFAULT_METHOD,
  },
  "decode-only": {
    repo: DEFAULT_REPOS.decodeOnly,

    file: "onnx/model_q4.onnx",
    heads: true,
    methods: DEFAULT_METHOD,
  },
  "transformers.js": {
    repo: DEFAULT_REPOS.onnx,

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

    build: "q4f32_1",

    heads: true,
    methods: DEFAULT_METHOD,
    logitsOnlyMethods: { noul: "logits", choice: "logits", score: "logits" },
  },
  litert: {
    repo: DEFAULT_REPOS.litert,

    files: { manifest: "litert/manifest.json", graph: "litert/model.graph.tflite", weights: "litert/model.weights.bin", embed: "litert/embed.f16.bin" },

    wasm: "https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/",
    heads: false,
    methods: { noul: "logits", choice: "logits", score: "logits" },
  },
} as const;

export function defaultMethods(runtime: RuntimeName, opts: { webllmRecovery?: boolean } = {}): Readonly<Record<"noul" | "choice" | "score", AskMethod>> {
  if (runtime === "webllm" && !opts.webllmRecovery) return RUNTIME_DEFAULTS.webllm.logitsOnlyMethods;
  if (runtime === "litert") return RUNTIME_DEFAULTS.litert.methods;
  return DEFAULT_METHOD;
}

export function resolveRuntimeChoice(o: { runtime?: string }, fallback: RuntimeChoice = "auto"): { choice: RuntimeChoice } {
  if (o.runtime === undefined) return { choice: fallback };
  if (o.runtime !== "auto" && !RUNTIMES.includes(o.runtime as RuntimeName))
    throw new Error(`ct0-edge: unknown runtime "${o.runtime}". Use "auto", ${RUNTIMES.map((r) => `"${r}"`).join(", ")}.`);
  return { choice: o.runtime as RuntimeChoice };
}

export function detectPlatform(g: any = globalThis): Platform {
  const isNode = !!g.process?.versions?.node;
  const hasDom = typeof g.window !== "undefined" && typeof g.document !== "undefined";
  const isWorker = typeof g.WorkerGlobalScope !== "undefined" && typeof g.importScripts === "function";
  return isNode && !hasDom && !isWorker ? "node" : "browser";
}

export interface Caps { platform: Platform; webgpu: boolean; cuda: boolean }

export interface Plan { runtime: RuntimeName; device: Device }

export const GENAI_NODE_CPU_ERROR = 'ct0-edge: runtime "genai" does not run on onnxruntime-node CPU (the int4 graph fails with "not enough space: expected 49152, got 0"). '
  + 'In Node use runtime "decode-only" (CPU) or "transformers.js", or run genai on a CUDA machine (device "cuda").';

function checkDevice(platform: Platform, device: Device): void {
  if (!DEVICES.includes(device)) throw new Error(`ct0-edge: unknown device "${device}". Use "auto", ${DEVICES.map((d) => `"${d}"`).join(", ")}.`);
  if (platform === "browser" && device === "cuda") throw new Error('ct0-edge: device "cuda" is Node-only. In the browser use "webgpu" or "wasm".');
  if (platform === "node" && (device === "webgpu" || device === "wasm")) throw new Error(`ct0-edge: device "${device}" is browser-only. In Node use "cpu" or "cuda".`);
}

function deviceFor(runtime: RuntimeName, want: DeviceChoice, caps: Caps): Device {
  const { platform } = caps;

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

  fallbackReason?: string;
}

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
  throw new Error("ct0-edge: no runtime candidate"); 
}

export function packagesFor(runtime: RuntimeName, platform: Platform): string[] {
  const ort = platform === "node" ? "onnxruntime-node" : "onnxruntime-web";
  if (runtime === "genai" || runtime === "decode-only") return ["@huggingface/transformers", ort];
  if (runtime === "webllm") return ["@huggingface/transformers", "@mlc-ai/web-llm"];
  if (runtime === "litert") return ["@huggingface/transformers", "@litertjs/core"];
  return ["@huggingface/transformers"];
}

function isNotFound(e: any, pkg: string): boolean {
  const msg = String(e?.message ?? e);

  if (e?.code === "ERR_MODULE_NOT_FOUND" || e?.code === "MODULE_NOT_FOUND" || /Cannot find (package|module)/.test(msg)) return msg.includes(pkg);

  return /Failed to resolve module specifier|Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed/i.test(msg);
}

export async function importOptional<T = any>(pkg: string, importer: () => Promise<T>, why: string): Promise<T> {
  try { return await importer(); }
  catch (e: any) {
    if (!isNotFound(e, pkg)) throw e;
    const err = new Error(`ct0-edge: ${why} needs the package "${pkg}", which is not installed. Install it: npm i ${pkg}`);
    (err as any).cause = e;
    throw err;
  }
}

export function detectCuda(ortNode: any): boolean {
  try { return !!ortNode?.listSupportedBackends?.().some((b: any) => b?.name === "cuda"); }
  catch { return false; }
}
