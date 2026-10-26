# Changelog

## 0.6.0

First public release.

- **One entry point for the browser and Node: `CT0.load({ runtime, device })`.** Runtimes: `"genai"` (ONNX Runtime GenAI
  graph), `"decode-only"` (ONNX Runtime Web / Node), `"transformers.js"` (optimum-exported graph), `"webllm"` (MLC).
  Devices: `"webgpu"`, `"wasm"`, `"cpu"`, `"cuda"`. Same API on every runtime: `ask`, `askMany`, `generate`, `stream`,
  `dispose`, plus `ct0.runtime`, `ct0.device`, `ct0.fallbackReason` and `ct0.defaultMethod`.
- **`"auto"` picks the fastest measured runtime:** browser with WebGPU → genai int4 (fallback decode-only q4 on WebGPU);
  browser without WebGPU → decode-only q4 on WASM; Node with CUDA → genai (fallback transformers.js q4 on CUDA, then CPU);
  Node on CPU → transformers.js q4 (fallback decode-only). On Linux the CUDA provider of `onnxruntime-node` needs the
  CUDA 13 runtime; without it `"auto"` falls back and `fallbackReason` says why.
- **System 1 in three modes:** yes/no, multiple choice and severity, read from the logits or from the heads
  (`head.bin`, `head_score.bin`). `method`: `"head"`, `"logits"` or `"both"` (both reads from one forward pass); any
  other value throws. Logits severity also returns `expected` (Σ level × softmax of the digits 1..5); `scoreCuts` maps
  `raw`/`expected` to levels (`levelFromCuts` is exported).
- **`askMany(state, questions)`:** several questions about one state. On genai and decode-only the shared start of the
  prompts runs once (backend `snapshot` / `prefillFrom`). Three questions on one state: decode-only q4 on Node CPU
  1.6–2.4× faster than three `ask()` calls with bit-identical results; genai int4 in Chrome 1.27 s → 0.91 s with the same
  decisions (numbers can differ by about as much as another prefill chunk size). transformers.js and webllm make one
  `ask()` per question.
- **Context limits** measured per runtime and device (`RUNTIME_LIMITS`, `maxSafeTokens`): genai WebGPU 7200 tokens,
  decode-only WebGPU 3600 and Node CPU 7200, 2048 where not measured.
- **WebLLM:** default build q4f32_1 (official WebLLM prebuilt wasm); `mlc: { build: "q4f16_1" }` is faster and less
  faithful. Heads read a hidden state recovered from the logits (`recovery/` in the MLC repo); if recovery fails it reads
  the logits and sets `fallbackReason`.
- **Node:** graphs are downloaded once to `~/.cache/ct0-edge` (`cacheDir`, `$CT0_CACHE`). Runtime packages are optional
  peers (`onnxruntime-web`, `onnxruntime-node`, `@mlc-ai/web-llm`), imported only when their runtime loads; a missing one
  fails with an error that names it.
- Lower-level entry points: `ct0-edge/ask`, `ct0-edge/transformers` (`createTransformersJs`), the genai, decode-only and
  webllm backends, `planRuntime`, `selectRuntime`, `RUNTIME_DEFAULTS`, `DEFAULT_REPOS`, `DEFAULT_FILES`.
