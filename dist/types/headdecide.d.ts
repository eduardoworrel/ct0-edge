import type { Backend } from "./backend.js";
import { type HeadWeights, type Answer } from "./systemone.js";
/** What the adapter needs from the runtime: run a prompt and return the two hidden states. */
export interface HiddenRuntime {
    /** Runs the prefill and returns the hidden at `<decide>` and at the last position. */
    hiddens(prompt: string): Promise<{
        atDecide: Float32Array;
        atLast: Float32Array;
    }>;
    /** For `choice`: hiddens of a prompt with several `<opt>…</opt>` (one hidden per `</opt>`, at the
     *  positions of positions.ts). Required by headAnswer with `choice`. `nOptions` = how many options the
     *  prompt must have; if the number of `</opt>` after `<q>` differs, it throws (never returns shifted hiddens). */
    optHiddens?(prompt: string, nOptions?: number): Promise<{
        atDecide: Float32Array;
        atLast: Float32Array;
        atOpts: Float32Array[];
    }>;
}
/** HiddenRuntime over a Backend: same positions as ask(). */
export declare function backendRuntime(be: Backend, tok: any): HiddenRuntime;
export interface HeadDecideOptions {
    /** Prompt prefix with the `<>` markers. Default: the measured format. */
    render?: (state: string, question: string) => string;
}
/** Creates a head-backed `decide(state, question)`. Drop-in for `recon({ decide })`. */
export declare function headDecide(hd: HeadWeights, rt: HiddenRuntime, opts?: HeadDecideOptions): (state: string, question: string) => Promise<number>;
/** Answers a typed question (`noul`, `choice`, `score`), which `decide()` cannot. */
export declare function headAnswer(hd: HeadWeights, rt: HiddenRuntime, q: {
    type: "noul" | "choice" | "score";
    state: string;
    question: string;
    options?: string[];
    levels?: number;
}, opts?: HeadDecideOptions): Promise<Answer>;
