# Changelog

## 0.7.0

- **Experimental `runtime: "litert"`** — CT0 through [LiteRT.js](https://www.npmjs.com/package/@litertjs/core)
  (`@litertjs/core` 2.6.0, pinned optional peer). Browser **WebGPU only**: the int4 FullyConnected weights are streamed
  into WebGPU buffers (`loadModelAndWeights`) because they exceed LiteRT.js's 2 GiB WASM heap; the token-embedding table
  is fp16 and read one row at a time with HTTP Range. Logits reads only (the graph has no hidden-state output). The
  backend owns the recurrent/conv/KV state and zeroes it per question, so there is no cross-call leak. Never chosen by
  `"auto"`; loading prints a warning that it is experimental.
- The bundle (`manifest.json`, `model.graph.tflite`, `model.weights.bin`, `embed.f16.bin`) is self-hosted via
  `litertBaseUrl` / `litertRepo`. Measured on the full CT0 bundle (Apple M2 Pro, Chrome WebGPU): 19/20 answers equal to
  the fp32 reference over 20 decisions (genai int4: 17/20), 9/12 greedy tokens vs HF fp32, ~3.2 s/decision median (vs
  ~0.5 s for `genai`), 24.8 s load. New entry point `ct0-edge/litert` and a mock-backed test suite.

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
