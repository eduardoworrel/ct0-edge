import type { Backend } from "./backend.js";
import { type Probe } from "./positions.js";
import { type HeadWeights } from "./systemone.js";
export declare const YES_ID = 9175, NO_ID = 2665;
export declare const DIGIT_IDS: number[];
export interface AskHeads {
    noulChoice?: HeadWeights;
    score?: HeadWeights;
}
export type AskQuestion = Probe & {
    levels?: number;
};
export declare const DEFAULT_METHOD: Readonly<Record<"noul" | "choice" | "score", "head" | "logits">>;
export type AskManyQuestion = Omit<AskQuestion, "state"> & {
    state?: string;
};
export interface AskOptions {
    method?: "head" | "logits" | "both";
    maxSafeTokens?: number;
    scoreCuts?: number[] | {
        head?: number[];
        logits?: number[];
    };
}
export interface AskResult {
    type: "noul" | "choice" | "score";
    method: "head" | "logits";
    raw?: number;
    p?: number;
    choice?: Record<string, number>;
    top?: string;
    level?: number;
    expected?: number;
    logits?: AskResult;
}
export declare function levelFromCuts(x: number, cuts: readonly number[]): number;
export declare function ask(be: Backend, tok: any, heads: AskHeads, q: AskQuestion, opts?: AskOptions): Promise<AskResult>;
export declare function askMany(be: Backend, tok: any, heads: AskHeads, state: string, qs: AskManyQuestion[], opts?: AskOptions): Promise<AskResult[]>;
export declare function askManyWith(be: Backend, tok: any, heads: AskHeads, state: string, qs: AskManyQuestion[], optsFor: (q: AskQuestion) => AskOptions): Promise<AskResult[]>;
