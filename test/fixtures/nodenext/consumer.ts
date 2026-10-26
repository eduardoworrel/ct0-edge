// Consumer with moduleResolution nodenext: the published .d.ts files must resolve (relative imports with .js).
import { CT0, type LoadOptions, type RuntimeName, type RuntimeChoice, type DeviceChoice, RUNTIME_LIMITS, RUNTIME_DEFAULTS, planRuntime } from "ct0-edge";
import { ask } from "ct0-edge/ask";
import { createTransformersJs } from "ct0-edge/transformers";
export type X = [typeof CT0, typeof ask, typeof createTransformersJs];

// LoadOptions: runtime / device / per-runtime options
const runtimes: RuntimeChoice[] = ["auto", "genai", "decode-only", "transformers.js", "webllm"];
const devices: DeviceChoice[] = ["auto", "webgpu", "wasm", "cpu", "cuda"];
export const opts: LoadOptions[] = [
  {},
  { runtime: "genai", device: "webgpu", verifyEp: true, genaiModel: "webgpu/int4/model.onnx" },
  { runtime: "decode-only", device: "cpu", cacheDir: "/tmp/ct0", decodeModel: "onnx/model_q4.onnx" },
  { runtime: "transformers.js", dtype: "q4f16", transformersRepo: "eduardoworrel/alias3-micro-1026-ONNX" },
  { runtime: "webllm", mlc: { build: "q4f32_1", initProgressCallback: () => {} }, modules: { webllm: {} } },
  { runtime: "transformers.js", maxSafeTokens: 1024, onStatus: (m: string) => void m },
];
// @ts-expect-error no `backend` option (use runtime)
export const badBackend: LoadOptions = { backend: "decode-only" };
// @ts-expect-error unknown runtime
export const bad: LoadOptions = { runtime: "onnx" };
// @ts-expect-error unknown device
export const badDevice: LoadOptions = { device: "tpu" };

export async function use(): Promise<void> {
  const ct0 = await CT0.load({ runtime: runtimes[0], device: devices[0] });
  const r: RuntimeName = ct0.runtime;
  const d: "webgpu" | "wasm" | "cpu" | "cuda" = ct0.device;
  const why: string | undefined = ct0.fallbackReason;
  const n: number = RUNTIME_LIMITS[r].maxSafeTokens + ct0.maxSafeTokens;
  const m: "head" | "logits" = ct0.defaultMethod.choice;
  const f: string = RUNTIME_DEFAULTS.genai.files.cuda;
  const p = planRuntime("auto", "auto", { platform: "node", webgpu: false, cuda: false });
  void [d, why, n, m, f, p];
  const a = await ct0.ask({ kind: "noul", state: "s", q: "q?" });
  void a.p;
  for await (const piece of ct0.generate("x", { maxNewTokens: 1 })) void piece;
  await ct0.dispose();
}
