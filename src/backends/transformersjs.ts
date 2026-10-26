/**
/**
 * transformersjs.ts — backend over the optimum/Scan graph via transformers.js (Node/CPU or browser). `kind: "scan"`.
 * Subpath `ct0-edge/transformers`: `createTransformersJs` / `transformersJsFromModel` / `TransformersJsOptions`.
 * prefill = ONE `model.forward({input_ids, attention_mask})` (no past: the library builds empty caches);
 * the result carries every graph output by name, including `logits` and the (post-norm) hidden.
 * generate = the library's `model.generate()` (greedy or sampled), with a streamer.
 * Verified: 172/172 rows identical to Python.
 */
import { createGate, resolveGenerateOptions, type Backend, type GenerateOptions } from "../backend.js";
import { tensorToF32, EOS, HIDDEN_NAMES } from "./decodeOnly.js";


export interface TransformersJsOptions {
  dtype: string; device?: string; localModelPath?: string;
}

/** Backend from an already-loaded model (`AutoModelForCausalLM`) and the `Tensor` constructor. */
export function transformersJsFromModel(model: any, Tensor: any): Backend {
  const sess = model.sessions?.model ?? Object.values(model.sessions ?? {})[0] as any;
  const outs: string[] = sess?.outputNames ?? [];
  const hiddenName = HIDDEN_NAMES.find((n) => outs.includes(n)) ?? null;

  const toTensors = (ids: number[]) => ({
    input_ids: new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    attention_mask: new Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
  });

  const gate = createGate("scan");
  return {
    kind: "scan",
    hiddenName,
    async prefill(ids, want) {
      gate.assertLive();
      const n = ids.length;
      if (!n) throw new Error("prefill: empty ids");
      for (const p of want.positions) if (!Number.isInteger(p) || p < 0 || p >= n) throw new Error(`transformers.js: position ${p} out of range [0,${n})`);
      if (want.positions.length && !hiddenName) throw new Error("transformers.js: graph has no hidden output; use method 'logits'");
      const rel = await gate.acquire();
      try {
        const out = await model.forward(toTensors(ids));
        const L = out.logits, V = L.dims.at(-1);
        const lastLogits = tensorToF32(L.data, L.type, (L.dims.at(-2) - 1) * V, V);
        const hiddens: Float32Array[] = [];
        if (want.positions.length) {
          const H = out[hiddenName!], D = H.dims.at(-1);
          if (H.dims.at(-2) !== n) throw new Error(`transformers.js: hidden seq ${H.dims.at(-2)} != ${n}`);
          for (const p of want.positions) hiddens.push(tensorToF32(H.data, H.type, p * D, D));
        }
        return { hiddens, lastLogits };
      } finally { rel(); }
    },
    dispose() { return gate.dispose(() => model.dispose?.()); },
    /** Generates ids from `ids` (raw prompt) via `model.generate()`. Emits at most `maxNewTokens` (default 512;
     *  0 => none, without calling the library) ids, without EOS. `seed` is IGNORED (unsupported by the library): with
     *  temperature > 0 sampling is not reproducible. `signal` abort or a consumer `break` stops the library run
     *  (stopping criterion) and the generator only ends once it settles, so a following prefill/dispose never overlaps the session. */
    async *generate(ids, o: GenerateOptions) {
      gate.assertLive();
      const g = resolveGenerateOptions(o);
      const max = g.maxNewTokens, temp = g.temperature;
      if (!ids.length) throw new Error("generate: empty ids");
      if (max === 0) return;
      const rel = await gate.acquire();
      const queue: number[] = []; let done = false, wake: (() => void) | null = null, err: any = null, first = true, stop = false;
      const ping = () => { const w = wake; wake = null; w?.(); };
      const streamer = {
        put(v: any) { if (first) { first = false; return; } // 1a chamada = prompt
          if (stop) return;
          for (const t of v.flat ? v.flat() : v) queue.push(Number(t)); ping(); },
        end() { done = true; ping(); },
      };
      const criterion = (batch: any[]) => batch.map(() => stop || !!o.signal?.aborted);
      const args: any = { ...toTensors(ids), max_new_tokens: max, streamer, stopping_criteria: [criterion], do_sample: temp > 0 };
      if (temp > 0) { args.temperature = temp; args.top_p = g.topP; args.top_k = g.topK; }
      args.repetition_penalty = g.repetitionPenalty;
      const run = Promise.resolve().then(() => model.generate(args)).then(() => { done = true; ping(); }, (e: any) => { err = e; done = true; ping(); });
      let emitted = 0;
      try {
        while (true) {
          if (o.signal?.aborted) break;
          if (queue.length) {
            const t = queue.shift()!;
            if (EOS.has(t)) break;
            yield t;
            if (++emitted >= max) break;
          } else if (done) break;
          else await new Promise<void>((r) => { wake = r; if (queue.length || done) ping(); });
        }
      } finally {
        stop = true; // interrompe a biblioteca no proximo passo
        try { await run; } finally { rel(); } // never leave a pending promise; errors only propagate on the normal path
      }
      if (err && !o.signal?.aborted) throw err;
    },
  };
}

export async function createTransformersJs(modelId: string, opts: TransformersJsOptions): Promise<Backend> {
  const tjs: any = await import("@huggingface/transformers");
  if (opts.localModelPath) { tjs.env.allowRemoteModels = false; tjs.env.allowLocalModels = true; tjs.env.localModelPath = opts.localModelPath; }
  const model = await tjs.AutoModelForCausalLM.from_pretrained(modelId, { dtype: opts.dtype, device: opts.device ?? "cpu" });
  return transformersJsFromModel(model, tjs.Tensor);
}
