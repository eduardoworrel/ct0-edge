import type { Backend } from "./backend.js";
import { type HeadWeights, type Answer } from "./systemone.js";
export interface HiddenRuntime {
    hiddens(prompt: string): Promise<{
        atDecide: Float32Array;
        atLast: Float32Array;
    }>;
    optHiddens?(prompt: string, nOptions?: number): Promise<{
        atDecide: Float32Array;
        atLast: Float32Array;
        atOpts: Float32Array[];
    }>;
}
export declare function backendRuntime(be: Backend, tok: any): HiddenRuntime;
export interface HeadDecideOptions {
    render?: (state: string, question: string) => string;
}
export declare function headDecide(hd: HeadWeights, rt: HiddenRuntime, opts?: HeadDecideOptions): (state: string, question: string) => Promise<number>;
export declare function headAnswer(hd: HeadWeights, rt: HiddenRuntime, q: {
    type: "noul" | "choice" | "score";
    state: string;
    question: string;
    options?: string[];
    levels?: number;
}, opts?: HeadDecideOptions): Promise<Answer>;
