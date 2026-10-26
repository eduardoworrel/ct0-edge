/**
/**
 * positions.ts — prompts and hidden positions matching the reference Python implementation.
 *
 * Reproduces `render`, `prompt_for`, `decide_pos` and `opt_positions` exactly, so the library's
 * numbers are the published numbers.
 *
 * Pitfalls:
 *  - `<decide>` is NOT a single token: it becomes 4 tokens ('><' 'dec' 'ide' '>'). Never find it by
 *    subsequence; the position is the '>' before the `<|im_end|>` that precedes `assistant`.
 *  - `</opt><opt>` fuses into ['</','opt','><','opt']: option positions come from character OFFSETS
 *    (the token covering the last char of each `</opt>`).
 */
export type Probe = {
    kind: "noul" | "choice" | "score";
    state: string;
    q?: string;
    options?: string[];
    sev?: number;
    want?: string;
};
export declare const CHOICE_Q = "Which vulnerability class does this code contain?";
export declare const SEV_Q = "How severe is the security risk in this content? (1=none, 5=critical)";
export declare function renderPrompt(state: string, question: string): string;
/** Validates a Probe and throws a clear error (unknown kind, invalid choice options). Called by `promptFor`,
 *  so ask() / headAnswer() / CT0.ask() / prefillHiddens inherit the check. */
export declare function assertProbe(p: Probe): void;
export declare function promptFor(p: Probe): string;
/** Index of the query hidden: the '<|im_end|>' that precedes `assistant`, minus 1. `tokens` = decoded strings. */
export declare function decidePos(tokens: string[]): number;
/** Indices of the tokens covering the LAST character of each `</opt>` (via offsets, not subsequence).
 *  SECURITY: `state` is untrusted and comes BEFORE `<q>`, while the rendered options always come AFTER the
 *  last `<q>`. Search only from there, or a `</opt>` inside the state would shift the positions. */
export declare function optPositions(prompt: string, offsets: [number, number][]): number[];
/** `optPositions` plus a check: exactly one position per option (`nOptions`; default = number of `<opt>` after `<q>`), else throws (never returns shifted positions). */
export declare function optPositionsFor(prompt: string, offsets: [number, number][], nOptions?: number): number[];
/**
 * ids + decoded tokens + offsets (UTF-16 units, like the `matchAll` above).
 * transformers.js does not expose offset_mapping, so offsets are derived by cumulative decoding
 * of prefixes (a token completing a multi-byte character gets the right end; partial bytes
 * leave a trailing U+FFFD, which is dropped), then the reconstruction is checked against the prompt.
 */
export declare function tokenOffsets(tok: any, prompt: string): {
    ids: number[];
    tokens: string[];
    offsets: [number, number][];
};
