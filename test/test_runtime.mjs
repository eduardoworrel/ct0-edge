// test_runtime.mjs — runtime/device selection of CT0.load() with fakes (no model, no network): platform detection,
// every auto branch, forced runtimes and devices, impossible pairs (genai on Node CPU, webllm in Node, cuda in a browser),
// missing-package errors, runtime resolution, per-runtime limits and reads, and a full CT0.load()
// in Node over fake onnxruntime-node / transformers.js modules.
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  CT0, detectPlatform, detectCuda, planRuntime, selectRuntime, resolveRuntimeChoice, importOptional, packagesFor,
  RUNTIMES, RUNTIME_LIMITS, RUNTIME_DEFAULTS, GENAI_NODE_CPU_ERROR, MLC_Q4F32_LIB, maxSafeTokensFor, defaultMethods, DEFAULT_METHOD,
  detectWebGPU, DEFAULT_REPOS, DEFAULT_FILES,
} from "../dist/ct0.js";
import { YES_ID, NO_ID } from "../dist/ask.js";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };
const throws = (f, re) => { try { f(); return false; } catch (e) { return re.test(String(e.message)); } };
const rejects = async (f, re) => { try { await f(); return false; } catch (e) { return re.test(String(e.message)); } };
const plan = (c, d, caps) => JSON.stringify(planRuntime(c, d, caps).map((p) => `${p.runtime}/${p.device}`));
const B = (webgpu) => ({ platform: "browser", webgpu, cuda: false });
const N = (cuda) => ({ platform: "node", webgpu: false, cuda });

console.log("detectPlatform:");
ok(detectPlatform(globalThis) === "node", "this process -> node");
ok(detectPlatform({ window: {}, document: {} }) === "browser", "window + document -> browser");
ok(detectPlatform({ process: { versions: { node: "22" } }, window: {}, document: {} }) === "browser", "Electron renderer (node + DOM) -> browser");
ok(detectPlatform({ WorkerGlobalScope: function () {}, importScripts() {} }) === "browser", "web worker -> browser");
ok(detectPlatform({ process: { versions: { node: "22" } } }) === "node", "process.versions.node without DOM -> node");
ok(detectPlatform({}) === "browser", "nothing known -> browser");

console.log("\ndetectCuda:");
ok(detectCuda({ listSupportedBackends: () => [{ name: "cpu" }, { name: "cuda", bundled: false }] }) === true, "cuda listed -> true");
ok(detectCuda({ listSupportedBackends: () => [{ name: "cpu" }, { name: "coreml" }] }) === false, "no cuda -> false");
ok(detectCuda({}) === false && detectCuda(null) === false, "old onnxruntime-node without the API -> false");
ok(detectCuda({ listSupportedBackends: () => { throw new Error("x"); } }) === false, "never throws");

console.log("\nauto:");
ok(plan("auto", "auto", B(true)) === '["genai/webgpu","decode-only/webgpu"]', "browser + WebGPU -> genai, fallback decode-only WebGPU");
ok(plan("auto", "auto", B(false)) === '["decode-only/wasm"]', "browser without WebGPU -> decode-only WASM");
ok(plan("auto", "auto", N(true)) === '["genai/cuda","transformers.js/cuda","transformers.js/cpu","decode-only/cpu"]', "Node + CUDA -> genai, then transformers.js CUDA, then CPU");
ok(plan("auto", "auto", N(false)) === '["transformers.js/cpu","decode-only/cpu"]', "Node CPU -> transformers.js CPU, fallback decode-only (genai never auto-picked on CPU)");
ok(plan("auto", "wasm", B(true)) === '["decode-only/wasm"]', "browser, device wasm -> decode-only WASM");
ok(plan("auto", "cpu", B(true)) === '["decode-only/wasm"]', "browser, device cpu = wasm");
ok(plan("auto", "webgpu", B(true)) === '["genai/webgpu","decode-only/webgpu"]', "browser, device webgpu");
ok(throws(() => planRuntime("auto", "webgpu", B(false)), /WebGPU is not available/), "device webgpu without WebGPU throws");
ok(plan("auto", "cpu", N(true)) === '["transformers.js/cpu","decode-only/cpu"]', "Node, device cpu -> transformers.js CPU even with CUDA");
ok(plan("auto", "cuda", N(true)) === '["genai/cuda","transformers.js/cuda"]', "Node, device cuda -> genai CUDA, fallback transformers.js CUDA");
ok(throws(() => planRuntime("auto", "cuda", N(false)), /no CUDA execution provider/), "device cuda without CUDA throws");
ok(throws(() => planRuntime("auto", "cuda", B(true)), /Node-only/), "cuda in a browser throws");
ok(throws(() => planRuntime("auto", "webgpu", N(true)), /browser-only/) && throws(() => planRuntime("auto", "wasm", N(false)), /browser-only/), "webgpu / wasm in Node throw");
ok(throws(() => planRuntime("auto", "tpu", N(false)), /unknown device/), "unknown device throws");

console.log("\nforced runtime:");
ok(plan("genai", "auto", B(true)) === '["genai/webgpu"]', "genai, browser -> WebGPU, no fallback");
ok(throws(() => planRuntime("genai", "auto", B(false)), /WebGPU is not available/), "genai without WebGPU throws");
ok(throws(() => planRuntime("genai", "wasm", B(true)), /needs WebGPU/), "genai on wasm throws");
ok(plan("genai", "auto", N(true)) === '["genai/cuda"]', "genai, Node + CUDA -> CUDA");
ok(throws(() => planRuntime("genai", "auto", N(false)), /not enough space: expected 49152, got 0/), "genai forced on Node CPU (no CUDA) throws the known failure");
ok(throws(() => planRuntime("genai", "cpu", N(true)), /does not run on onnxruntime-node CPU/), "genai forced on device cpu throws even with CUDA");
ok(plan("decode-only", "auto", B(true)) === '["decode-only/webgpu"]' && plan("decode-only", "auto", B(false)) === '["decode-only/wasm"]', "decode-only, browser: WebGPU, else WASM");
ok(plan("decode-only", "auto", N(true)) === '["decode-only/cpu"]' && plan("decode-only", "cuda", N(true)) === '["decode-only/cuda"]', "decode-only, Node: CPU by default, CUDA on request");
ok(throws(() => planRuntime("decode-only", "cuda", N(false)), /no CUDA/), "decode-only cuda without CUDA throws");
ok(plan("transformers.js", "auto", B(true)) === '["transformers.js/webgpu"]' && plan("transformers.js", "auto", B(false)) === '["transformers.js/wasm"]', "transformers.js, browser: WebGPU, else WASM");
ok(plan("transformers.js", "auto", N(false)) === '["transformers.js/cpu"]', "transformers.js, Node -> CPU");
ok(throws(() => planRuntime("transformers.js", "webgpu", B(false)), /WebGPU is not available/), "transformers.js webgpu without WebGPU throws");
ok(plan("webllm", "auto", B(true)) === '["webllm/webgpu"]', "webllm, browser + WebGPU");
ok(throws(() => planRuntime("webllm", "auto", N(true)), /only in the browser/), "webllm in Node throws");
ok(throws(() => planRuntime("webllm", "auto", B(false)), /WebGPU is not available/) && throws(() => planRuntime("webllm", "wasm", B(true)), /needs WebGPU/), "webllm without WebGPU / on wasm throws");
ok(throws(() => planRuntime("onnx", "auto", N(false)), /unknown runtime/), "unknown runtime throws");

console.log("\nselectRuntime:");
function deps(caps, fails = {}) {
  const calls = [], status = [];
  return { calls, status, caps, onStatus: (m) => status.push(m),
    load: async (p) => { calls.push(`${p.runtime}/${p.device}`); if (fails[p.runtime]) throw new Error(fails[p.runtime]); return { kind: p.runtime }; } };
}
{
  const d = deps(B(true));
  const s = await selectRuntime("auto", "auto", d);
  ok(s.runtime === "genai" && s.device === "webgpu" && s.fallbackReason === undefined && d.calls.length === 1, "auto, WebGPU: genai, decode-only never loaded, no fallback reason");
}
{
  const d = deps(B(true), { genai: "genai: nenhum no no EP WebGPU" });
  const s = await selectRuntime("auto", "auto", d);
  ok(s.runtime === "decode-only" && s.device === "webgpu" && JSON.stringify(d.calls) === '["genai/webgpu","decode-only/webgpu"]', "auto: genai fails -> decode-only on WebGPU");
  ok(/genai \(webgpu\) unavailable: genai: nenhum no/.test(s.fallbackReason) && d.status.some((m) => /decode-only \(webgpu\)/.test(m)), "fallback reason carries the genai error; status reported");
}
{
  const s = await selectRuntime("auto", "auto", deps(B(false)));
  ok(s.runtime === "decode-only" && s.device === "wasm" && /WebGPU not available/.test(s.fallbackReason), "auto, no WebGPU: reason says so");
}
{
  const s = await selectRuntime("auto", "auto", deps(N(false)));
  ok(s.runtime === "transformers.js" && s.device === "cpu" && /no CUDA execution provider/.test(s.fallbackReason), "auto, Node CPU: transformers.js CPU, reason names CUDA");
}
{
  const d = deps(N(true), { genai: "cuda EP missing libcudnn" });
  const s = await selectRuntime("auto", "auto", d);
  ok(s.runtime === "transformers.js" && s.device === "cuda" && /libcudnn/.test(s.fallbackReason), "auto, Node CUDA: genai fails -> transformers.js CUDA");
}
{
  const d = deps(B(true), { genai: "genai EP fail", "decode-only": "decode OOM" });
  let msg = ""; try { await selectRuntime("auto", "auto", d); } catch (e) { msg = e.message; }
  ok(/no runtime loaded/.test(msg) && /genai EP fail/.test(msg) && /decode OOM/.test(msg), "both fail -> one error with both reasons");
}
{
  const d = deps(B(true), { genai: "boom" });
  ok(await rejects(() => selectRuntime("genai", "auto", d), /^boom$/) && d.calls.length === 1, "forced genai: no fallback, original error");
  const s = await selectRuntime("transformers.js", "auto", deps(N(false)));
  ok(s.runtime === "transformers.js" && s.device === "cpu" && s.fallbackReason === undefined, "forced runtime: no fallback reason");
}

console.log("\ndetectWebGPU / default repos:");
ok(await detectWebGPU(undefined) === false && await detectWebGPU({}) === false, "no navigator / no gpu -> false");
ok(await detectWebGPU({ gpu: { requestAdapter: async () => null } }) === false && await detectWebGPU({ gpu: { requestAdapter: async () => { throw new Error("x"); } } }) === false, "no adapter / requestAdapter throws -> false");
ok(await detectWebGPU({ gpu: { requestAdapter: async () => ({}) } }) === true, "adapter -> true");
ok(DEFAULT_REPOS.genai === "eduardoworrel/alias3-micro-1026-ONNX-genai" && DEFAULT_REPOS.decodeOnly === "eduardoworrel/alias3-micro-1026-ONNX-decode-only", "default repos are the public ones");
ok(DEFAULT_FILES.head === "heads/head.bin" && DEFAULT_FILES.headScore === "heads/head_score.bin", "head files");

console.log("\nruntime resolution:");
ok(resolveRuntimeChoice({}).choice === "auto", "nothing -> auto");
ok(resolveRuntimeChoice({}, "decode-only").choice === "decode-only", "fallback choice (self-hosted baseUrl) applies");
for (const r of ["auto", ...RUNTIMES]) ok(resolveRuntimeChoice({ runtime: r }).choice === r, `runtime "${r}"`);
ok(throws(() => resolveRuntimeChoice({ runtime: "scan" }), /unknown runtime/) && throws(() => resolveRuntimeChoice({ runtime: "onnx" }), /unknown runtime/), "unknown values throw");

console.log("\nmissing packages:");
const notFound = (pkg) => async () => { const e = new Error(`Cannot find package '${pkg}' imported from /x/ct0.js`); e.code = "ERR_MODULE_NOT_FOUND"; throw e; };
ok(await rejects(() => importOptional("onnxruntime-node", notFound("onnxruntime-node"), 'runtime "decode-only" in Node'), /needs the package "onnxruntime-node".*npm i onnxruntime-node/), "Node: missing package -> names it and how to install");
ok(await rejects(() => importOptional("@mlc-ai/web-llm", async () => { throw new TypeError('Failed to resolve module specifier "@mlc-ai/web-llm"'); }, 'runtime "webllm"'), /npm i @mlc-ai\/web-llm/), "browser: unresolved specifier -> names the package");
ok(await rejects(() => importOptional("onnxruntime-web", async () => { throw new TypeError("Failed to fetch dynamically imported module: https://x/ort.mjs"); }, "x"), /npm i onnxruntime-web/), "browser: module fetch failure -> names the package");
ok(await rejects(() => importOptional("onnxruntime-node", notFound("some-dep"), "x"), /Cannot find package 'some-dep'/), "a missing dependency OF the package is rethrown as is");
ok(await rejects(() => importOptional("onnxruntime-node", async () => { throw new Error("dlopen failed"); }, "x"), /^dlopen failed$/), "other load errors are rethrown");
ok((await importOptional("x", async () => ({ ok: 1 }), "x")).ok === 1, "present package -> module");
ok(JSON.stringify(packagesFor("decode-only", "node")) === '["@huggingface/transformers","onnxruntime-node"]' && packagesFor("genai", "browser").includes("onnxruntime-web")
  && packagesFor("webllm", "browser").includes("@mlc-ai/web-llm") && packagesFor("transformers.js", "node").length === 1, "packagesFor per runtime and platform");

console.log("\nper-runtime tables:");
ok(RUNTIMES.every((r) => RUNTIME_LIMITS[r].maxSafeTokens === 2048), "RUNTIME_LIMITS: 2048 default for every runtime");
ok(maxSafeTokensFor("genai", "webgpu") === 7200 && maxSafeTokensFor("genai", "cuda") === 2048, "genai: 7200 on WebGPU (measured), 2048 on CUDA (not measured)");
ok(maxSafeTokensFor("decode-only", "webgpu") === 3600 && maxSafeTokensFor("decode-only", "cpu") === 7200 && maxSafeTokensFor("decode-only", "wasm") === 2048, "decode-only: 3600 WebGPU, 7200 CPU, 2048 unmeasured");
ok(maxSafeTokensFor("transformers.js", "webgpu") === 2048 && maxSafeTokensFor("webllm", "webgpu") === 2048, "transformers.js and webllm: 2048 (not measured)");
RUNTIME_LIMITS.webllm.byDevice = { webgpu: 1000 };
ok(maxSafeTokensFor("webllm", "webgpu") === 1000 && maxSafeTokensFor("webllm", "wasm") === 2048, "per-device override hook");
delete RUNTIME_LIMITS.webllm.byDevice;
ok(defaultMethods("webllm").choice === "logits" && defaultMethods("webllm").score === "logits" && defaultMethods("webllm").noul === "logits", "webllm without recovery: logits for every mode");
ok(RUNTIME_DEFAULTS.webllm.methods === DEFAULT_METHOD && RUNTIME_DEFAULTS.webllm.heads === true, "webllm default (with recovery): DEFAULT_METHOD and heads");
ok(defaultMethods("webllm", { webllmRecovery: true }) === DEFAULT_METHOD && defaultMethods("genai") === DEFAULT_METHOD, "ONNX runtimes (and webllm with recovery): DEFAULT_METHOD");
ok(RUNTIME_DEFAULTS.genai.files.webgpu === "webgpu/int4/model.onnx" && RUNTIME_DEFAULTS.genai.files.cuda === "cuda/int4/model.onnx", "genai files per device");
ok(RUNTIME_DEFAULTS["decode-only"].file === "onnx/model_q4.onnx" && RUNTIME_DEFAULTS["transformers.js"].dtype === "q4", "decode-only q4, transformers.js q4");
ok(RUNTIME_DEFAULTS.webllm.build === "q4f32_1", "webllm: q4f32_1 (faithful) is the default build");
ok(MLC_Q4F32_LIB === "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/Qwen3.5-4B-q4f32_1_cs1k-webgpu.wasm" && RUNTIME_DEFAULTS.webllm.builds.q4f32_1.lib === MLC_Q4F32_LIB, "webllm q4f32_1: lib = the official WebLLM prebuilt wasm");
ok(RUNTIME_DEFAULTS.webllm.builds.q4f32_1.repo === "eduardoworrel/alias3-micro-1026-MLC-q4f32_1" && RUNTIME_DEFAULTS.webllm.builds.q4f16_1.lib === "model-lib.wasm", "webllm: q4f32_1 repo; q4f16_1 option keeps its repo wasm");
ok(/eduardoworrel\/alias3-micro-1026-/.test(RUNTIME_DEFAULTS.genai.repo + RUNTIME_DEFAULTS["transformers.js"].repo + RUNTIME_DEFAULTS.webllm.builds.q4f16_1.repo), "public repos");

// ---------------------------------------------------------------- CT0.load() in Node with fake modules
console.log("\nCT0.load (Node, fake modules):");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
class Tensor { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }
const V = 9300;
function decodeSession() {
  return { inputNames: ["input_ids", "past_conv.0"], outputNames: ["logits", "present_conv.0"],
    inputMetadata: [{ name: "input_ids", type: "int64", shape: [1, 1] }, { name: "past_conv.0", type: "float32", shape: [1, 2] }],
    async run() { await sleep(0); const l = new Float32Array(V); l[YES_ID] = 3; l[NO_ID] = 1; l[65] = 9; return { logits: new Tensor("float32", l, [1, 1, V]), "present_conv.0": new Tensor("float32", new Float32Array(2), [1, 2]) }; },
    async release() {} };
}
function fakeOrt({ cuda = false, genaiFails = "genai session failed" } = {}) {
  const created = [];
  return { created, Tensor, env: { versions: { node: "1.30.0" } },
    listSupportedBackends: () => [{ name: "cpu" }, ...(cuda ? [{ name: "cuda" }] : [])],
    InferenceSession: { async create(path, o) { created.push({ path, eps: o?.executionProviders }); if (/genai/.test(path)) throw new Error(genaiFails); return decodeSession(); } } };
}
// char tokenizer: ids = UTF-16 codes (all < V)
function fakeTok() {
  const enc = (s) => Array.from(s, (c) => c.charCodeAt(0));
  const tok = (text) => ({ input_ids: { tolist: () => [enc(text)] } });
  tok.encode = enc;
  tok.decode = (ids) => String.fromCharCode(...ids.map(Number));
  return tok;
}
const tjs = { env: {}, Tensor, AutoTokenizer: { async from_pretrained() { return fakeTok(); } } };
const dir = mkdtempSync(join(tmpdir(), "ct0-runtime-"));
mkdirSync(join(dir, "onnx"), { recursive: true }); mkdirSync(join(dir, "genai", "cuda", "int4"), { recursive: true });
writeFileSync(join(dir, "onnx", "model_q4.onnx"), ""); writeFileSync(join(dir, "onnx", "model_q4.onnx_data"), "");
try {
  {
    const ort = fakeOrt();
    const st = [];
    const ct0 = await CT0.load({ runtime: "auto", baseUrl: dir + "/", heads: false, modules: { ort, transformers: tjs }, onStatus: (m) => st.push(m) });
    ok(ct0.runtime === "decode-only" && ct0.device === "cpu" && /no CUDA/.test(ct0.fallbackReason), "auto on a CPU-only Node: decode-only CPU, fallbackReason set");
    ok(ort.created.length === 1 && ort.created[0].path === join(dir, "onnx", "model_q4.onnx") && JSON.stringify(ort.created[0].eps) === '["cpu"]', "graph opened from disk on the CPU EP");
    ok(ct0.maxSafeTokens === 7200 && ct0.defaultMethod.choice === "head", "maxSafeTokens from RUNTIME_LIMITS; ONNX reads");
    const r = await ct0.ask({ kind: "noul", state: "s", q: "q?" });
    ok(r.method === "logits" && Math.abs(r.raw - 2) < 1e-6, "ask() noul answers through the selected runtime");
    const sc = await ct0.ask({ kind: "score", state: "s" }, { method: "logits", scoreCuts: [1.5, 2.5, 3.5, 4.5] });
    ok(sc.expected === 3 && sc.level === 3, "ask() passes scoreCuts / returns expected (flat digits: expected 3 -> level 3)");
    const many = await ct0.askMany("s", [{ kind: "noul", q: "q?" }, { kind: "score" }], { method: "logits" });
    ok(JSON.stringify(many) === JSON.stringify([r, await ct0.ask({ kind: "score", state: "s" }, { method: "logits" })]), "askMany() == ask() per question");
    ok(await rejects(() => ct0.askMany("s", [{ kind: "noul", q: "q?" }, { kind: "choice", options: ["a", "b"] }]), /hidden output/), "askMany() uses the runtime's per-kind default (choice -> head)");
    ok(await rejects(() => ct0.ask({ kind: "noul", state: "s", q: "q?" }, { method: "both" }), /hidden output/), "ask() accepts method both (needs a hidden output)");
    let out = ""; for await (const p of ct0.generate("hi", { maxNewTokens: 2, temperature: 0 })) out += p;
    ok(out === "AA", "generate() streams through the selected runtime");
    await ct0.dispose();
    ok(st.some((m) => /runtime: decode-only \(cpu\)/.test(m)), "status names runtime and device");
  }
  {
    const ort = fakeOrt({ cuda: true, genaiFails: "cuda EP: libcublas missing" });
    const ct0 = await CT0.load({ runtime: "auto", baseUrl: dir + "/", genaiBaseUrl: join(dir, "genai") + "/", heads: false, modules: { ort, transformers: tjs }, maxSafeTokens: 999 });
    ok(ort.created[0].path.includes(join("genai", "cuda", "int4", "model.onnx")) && JSON.stringify(ort.created[0].eps) === '["cuda"]', "Node + CUDA: tries genai cuda/int4 on the CUDA EP first");
    ok(ct0.runtime === "decode-only" && ct0.device === "cpu" && /libcublas/.test(ct0.fallbackReason), "genai CUDA failure -> decode-only CPU with the reason");
    ok(ct0.maxSafeTokens === 999, "maxSafeTokens option overrides the table");
  }
  {
    const ort = fakeOrt();
    ok(await rejects(() => CT0.load({ runtime: "genai", modules: { ort, transformers: tjs } }), /not enough space: expected 49152/), "forced genai on Node CPU throws");
    ok(await rejects(() => CT0.load({ runtime: "genai", device: "cpu", modules: { ort, transformers: tjs } }), /does not run on onnxruntime-node CPU/), "forced genai + device cpu throws");
    ok(ort.created.length === 0, "...before any graph is opened");
    ok(await rejects(() => CT0.load({ runtime: "webllm", modules: { ort, transformers: tjs } }), /only in the browser/), "webllm in Node throws");
    const ct0 = await CT0.load({ runtime: "decode-only", baseUrl: dir + "/", heads: false, modules: { ort, transformers: tjs } });
    ok(ct0.runtime === "decode-only" && ct0.fallbackReason === undefined, "runtime: \"decode-only\" forced: no fallback reason");
  }
  {
    const created = [];
    const tjs2 = { ...tjs, AutoModelForCausalLM: { async from_pretrained(id, o) { created.push({ id, ...o }); return { sessions: { model: { outputNames: ["logits"] } }, async forward() { return {}; } }; } } };
    const ct0 = await CT0.load({ runtime: "transformers.js", heads: false, modules: { transformers: tjs2 } });
    ok(ct0.runtime === "transformers.js" && ct0.device === "cpu", "transformers.js in Node: no onnxruntime-node needed by the facade");
    ok(created[0].id === "eduardoworrel/alias3-micro-1026-ONNX" && created[0].dtype === "q4" && created[0].device === "cpu", "transformers.js defaults: ONNX repo, q4, cpu");
  }
} finally { rmSync(dir, { recursive: true, force: true }); }

ok(GENAI_NODE_CPU_ERROR.includes("decode-only"), "the genai/CPU error points to the working runtimes");

// ---------------------------------------------------------------- webllm in a (fake) browser: heads + recovery from the MLC repo
console.log("\nCT0.load webllm (fake browser):");
{
  const saved = { window: globalThis.window, document: globalThis.document, fetch: globalThis.fetch, nav: Object.getOwnPropertyDescriptor(globalThis, "navigator") };
  globalThis.window = {}; globalThis.document = {};
  Object.defineProperty(globalThis, "navigator", { value: { gpu: { requestAdapter: async () => ({}) } }, configurable: true });
  const N = 4, D = 3;
  const run = async (recoveryOk, opts = {}) => {
    const fetched = [], tokRepos = [], engines = [];
    globalThis.fetch = async (url) => {
      fetched.push(String(url));
      if (/recovery\/P_q4f\d+_rows\.i32$/.test(url) && recoveryOk) return new Response(new Int32Array([1, 2, 3, 4]).buffer);
      if (/recovery\/P_q4f\d+_8192\.f16$/.test(url) && recoveryOk) return new Response(new Uint16Array(N * D).buffer);
      return new Response("missing", { status: 404 });
    };
    const tjs3 = { ...tjs, AutoTokenizer: { async from_pretrained(repo) { tokRepos.push(repo); return fakeTok(); } } };
    const webllm = { async CreateMLCEngine(id, cfg) { engines.push({ id, cfg }); return { async unload() {} }; } };
    const st = [];
    const ct0 = await CT0.load({ runtime: "webllm", modules: { transformers: tjs3, webllm }, onStatus: (m) => st.push(m), ...opts });
    return { ct0, fetched, tokRepos, engines, st };
  };
  try {
    {
      const { ct0, fetched, tokRepos, engines } = await run(true);
      const base = "https://huggingface.co/eduardoworrel/alias3-micro-1026-MLC-q4f32_1/resolve/main/";
      ok(ct0.runtime === "webllm" && ct0.device === "webgpu" && ct0.fallbackReason === undefined, "webllm loads on WebGPU, no fallback reason");
      ok(JSON.stringify(tokRepos) === '["eduardoworrel/alias3-micro-1026-MLC-q4f32_1"]', "tokenizer from the MLC q4f32_1 repo itself");
      ok(fetched.includes(base + "recovery/P_q4f32_rows.i32") && fetched.includes(base + "recovery/P_q4f32_8192.f16"), "recovery rows + P (f16) fetched from the selected MLC repo");
      ok(fetched.includes(base + "heads/head.bin"), "heads fetched from the same MLC repo");
      ok(ct0.hiddenOutput === "recovered-from-logits" && ct0.defaultMethod === DEFAULT_METHOD, "recovery loaded: head reads with DEFAULT_METHOD (yes/no logits, choice head, severity head)");
      const lib = engines[0].cfg.appConfig.model_list[0];
      ok(engines[0].id === "alias3-micro-1026-q4f32_1" && lib.model_lib === MLC_Q4F32_LIB && lib.model === "https://huggingface.co/eduardoworrel/alias3-micro-1026-MLC-q4f32_1/", "engine: q4f32_1 model id, repo URL and the official wasm");
    }
    {
      const { ct0, fetched, st } = await run(false);
      ok(ct0.runtime === "webllm" && /webllm recovery unavailable/.test(ct0.fallbackReason) && /404/.test(ct0.fallbackReason), "recovery fails -> fallbackReason says so");
      ok(ct0.hiddenOutput === null && ct0.defaultMethod.choice === "logits" && ct0.defaultMethod.score === "logits", "...and every question type reads the logits");
      ok(!fetched.some((u) => /heads\//.test(u)) && st.some((m) => /logits/.test(m)), "...heads not fetched; status reported");
    }
    {
      const { ct0, fetched, tokRepos } = await run(true, { mlc: { build: "q4f16_1" } });
      ok(tokRepos[0] === "eduardoworrel/alias3-micro-1026-MLC-q4f16_1" && fetched.some((u) => u.endsWith("MLC-q4f16_1/resolve/main/recovery/P_q4f16_8192.f16")) && ct0.defaultMethod === DEFAULT_METHOD, "build q4f16_1: its own tokenizer, recovery and heads");
    }
    {
      const { ct0, fetched } = await run(true, { mlc: { recovery: false } });
      ok(!fetched.some((u) => /recovery\//.test(u)) && ct0.defaultMethod.choice === "logits" && ct0.fallbackReason === undefined, "mlc.recovery: false -> logits only, nothing fetched");
    }
  } finally {
    globalThis.window = saved.window; globalThis.document = saved.document; globalThis.fetch = saved.fetch;
    if (saved.nav) Object.defineProperty(globalThis, "navigator", saved.nav); else delete globalThis.navigator;
    if (saved.window === undefined) delete globalThis.window;
    if (saved.document === undefined) delete globalThis.document;
  }
}

console.log(`\n${pass} ok, ${fail} failures`);
process.exit(fail ? 1 : 0);
