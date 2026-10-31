import { type Backend } from "../backend.js";
export interface TransformersJsOptions {
    dtype: string;
    device?: string;
    localModelPath?: string;
}
export declare function transformersJsFromModel(model: any, Tensor: any): Backend;
export declare function createTransformersJs(modelId: string, opts: TransformersJsOptions): Promise<Backend>;
