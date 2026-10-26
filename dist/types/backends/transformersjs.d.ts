/**
/**
 * transformersjs.ts — backend over the optimum/Scan graph via transformers.js (Node/CPU or browser). `kind: "scan"`.
 * Subpath `ct0-edge/transformers`: `createTransformersJs` / `transformersJsFromModel` / `TransformersJsOptions`.
 * prefill = ONE `model.forward({input_ids, attention_mask})` (no past: the library builds empty caches);
 * the result carries every graph output by name, including `logits` and the (post-norm) hidden.
 * generate = the library's `model.generate()` (greedy or sampled), with a streamer.
 * Verified: 172/172 rows identical to Python.
 */
import { type Backend } from "../backend.js";
export interface TransformersJsOptions {
    dtype: string;
    device?: string;
    localModelPath?: string;
}
/** Backend from an already-loaded model (`AutoModelForCausalLM`) and the `Tensor` constructor. */
export declare function transformersJsFromModel(model: any, Tensor: any): Backend;
export declare function createTransformersJs(modelId: string, opts: TransformersJsOptions): Promise<Backend>;
