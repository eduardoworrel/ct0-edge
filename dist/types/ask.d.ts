/**
 * ask.ts — typed System-1 (noul / choice / score) over any Backend, by two methods:
 *
 *  - `head`:   bilinear head (systemone.answer) over the prefill hiddens. Needs a hidden output in the graph.
 *  - `logits`: reads the logits of the LAST position directly:
 *      noul   -> L[Yes=9175] > L[No=2665]; p = sigmoid(L[Yes]-L[No])
 *      choice -> logit of the FIRST token of each label; argmax; tie -> first option
 *      score  -> argmax over the digits '1'..'5' (ids 16..20) + 1; `expected` = sum of level x softmax(digit logits)
 *  - `both`:   the head read plus the logits read (`result.logits`) from the SAME prefill.
 *
 * `askMany` asks several questions about one state: it prefills the shared token prefix once (`Backend.snapshot`) and
 * runs only each question's suffix (`Backend.prefillFrom`); backends without snapshots get independent `ask` calls.
 *
 * Prompt and positions come from positions.ts.
 */
import type { Backend } from "./backend.js";
import { type Probe } from "./positions.js";
import { type HeadWeights } from "./systemone.js";
export declare const YES_ID = 9175, NO_ID = 2665;
export declare const DIGIT_IDS: number[];
export interface AskHeads {
    /** Original head (head.bin): noul and choice. */
    noulChoice?: HeadWeights;
    /** Score head (head_score.bin); required for score with method 'head' (no fallback). */
    score?: HeadWeights;
}
export type AskQuestion = Probe & {
    levels?: number;
};
/** Default read per mode (measured on a 50-probe set): noul from the logits (the head ties them on accuracy and its
 *  p saturates), choice from the head (0.84 vs 0.68 on q4), score from head_score.bin (exact 0.52 vs 0.40). */
export declare const DEFAULT_METHOD: Readonly<Record<"noul" | "choice" | "score", "head" | "logits">>;
export type AskManyQuestion = Omit<AskQuestion, "state"> & {
    state?: string;
};
export interface AskOptions {
    /** Read to use. Default per mode: see `DEFAULT_METHOD` (noul: logits; choice and score: head).
     *  `"both"`: the head result, with the logits read of the same prefill in `result.logits`. */
    method?: "head" | "logits" | "both";
    /** Prompt token ceiling (default 2048). */
    maxSafeTokens?: number;
    /** score: map a value to a level with K-1 ascending cuts (K = `levels`, default 5; the logits read always has K = 5):
     *  level = 1 + number of cuts <= value. The value is `raw` for the head and `expected` for the logits.
     *  Default: head round(sigmoid(raw)·(K-1)+1), logits argmax of the digits. With `"both"` the two reads have
     *  different scales, so pass `{ head, logits }`. Ignored by noul and choice. */
    scoreCuts?: number[] | {
        head?: number[];
        logits?: number[];
    };
}
export interface AskResult {
    type: "noul" | "choice" | "score";
    method: "head" | "logits";
    /** noul: raw score (head) or L[Yes]-L[No] (logits). score/head: raw score. */
    raw?: number;
    /** noul: P(yes). */
    p?: number;
    /** choice: probability (softmax) per option. */
    choice?: Record<string, number>;
    /** choice: winning option (tie -> the first). */
    top?: string;
    /** score: level 1..K. */
    level?: number;
    /** score, logits read: expected level, sum over l=1..5 of l x softmax(logits of the digits '1'..'5'). */
    expected?: number;
    /** method "both": the logits read of the same prefill. */
    logits?: AskResult;
}
/** Level from ascending cuts: 1 + number of cuts <= x. */
export declare function levelFromCuts(x: number, cuts: readonly number[]): number;
export declare function ask(be: Backend, tok: any, heads: AskHeads, q: AskQuestion, opts?: AskOptions): Promise<AskResult>;
/**
 * Several questions about ONE state: same results as `ask` for each question, in order. When the backend implements
 * `snapshot` / `prefillFrom` (decode-only, genai), the shared prompt prefix (system turn + state) runs once and each
 * question runs only its own suffix; otherwise it falls back to independent `ask` calls. A question may omit `state`
 * (the shared one is used); a different `state` rejects. Every question is validated before anything runs.
 */
export declare function askMany(be: Backend, tok: any, heads: AskHeads, state: string, qs: AskManyQuestion[], opts?: AskOptions): Promise<AskResult[]>;
/** `askMany` with per-question options (the facade uses it for its per-kind default method). */
export declare function askManyWith(be: Backend, tok: any, heads: AskHeads, state: string, qs: AskManyQuestion[], optsFor: (q: AskQuestion) => AskOptions): Promise<AskResult[]>;
