/**
/**
 * systemone.ts — System-1 via the paper's bilinear head, in plain JavaScript.
 *
 * The head is recovered from the truncated `head.pt` of the official repo (89% of the `k`
 * weights; `k.bias` is zeroed). Architecture derived from the pickle:
 *
 *     score = (Wq·h_q + bq) · (Wk·h_k + bk) / temperature
 *     Wq, Wk: [256, 2560]   bq, bk: [256]   temperature: 2.0
 *
 * Measured spec (200+ combinations swept against the logit-based `decide()` as oracle, with 8
 * holdout probes the head never saw):
 *   - `q` reads the hidden at the LAST token of `<decide>`
 *   - `k` reads the hidden at the LAST prompt position (`noul`) or at each option token (`choice`)
 *   - pre-norm and post-norm give the same result; normalizing the hidden hurts (corr 0.984 -> 0.962)
 *   - temperature only scales; it does not change the sign
 *   Result: 16/16 (8 original + 8 holdout), correlation +0.984 with the logit-based `decide()`.
 *
 * `choice` is a pointer head: `q` is the hidden at `<decide>` (one per question) and each option
 * is a `k`, read at the option token's hidden. This scores 8/8 on the 8 calibration probes and
 * 5/8 on 8 new ones (`k` has 29/256 zeroed rows, so comparing MAGNITUDE across options stays lossy).
 *
 * Modes (only the head supports all; the logit-based `decide()` only does noul):
 *   - `noul`   — P(yes) for a yes/no question
 *   - `choice` — ranks N options against a criteria map (one score per option)
 *   - `score`  — ordinal grade 1..K
 *
 * The head is tiny (2.6 MB fp16) and runs in ~0.1 ms per call on a 2560-dim hidden.
 */
export type QuestionType = "noul" | "choice" | "score";
export interface HeadWeights {
    /** [256, 2560] query projection */
    q: Float32Array;
    /** [256] query bias */
    qb: Float32Array;
    /** [256, 2560] key projection (89% recovered; 29 zeroed rows) */
    k: Float32Array;
    /** [256] key bias (lost in the truncated file; zeros) */
    kb: Float32Array;
    headDim: number;
    hiddenSize: number;
    temperature: number;
    /** Offset added to the final score; compensates for the lost `k.bias`.
     *  Measured by bootstrap (median over base probes, no labels) and validated 16/16 on holdout.
     *  Not a per-dimension bias: adding 34.3 to each of the 256 dimensions would give +8780. */
    scoreOffset: number;
}
export interface Question {
    type: QuestionType;
    /** `noul`: the yes/no question */
    instructions?: string;
    /** `choice`: the options and each one's criterion */
    options?: string[];
    criteria?: Record<string, string>;
    /** `score`: K ordinal levels */
    levels?: number;
}
export interface Answer {
    type: QuestionType;
    /** `noul`: P(yes) in [0,1] */
    noul?: number;
    /** `choice`: one probability per option */
    choice?: Record<string, number>;
    /** `score`: the grade 1..K */
    score?: number;
    /** raw score before the sigmoid (diagnostics) */
    raw?: number;
}
/** Loads the head weights from an ArrayBuffer (JSON header + fp16 blocks). */
export declare function loadHead(buf: ArrayBuffer): HeadWeights;
/**
 * Raw head score for a pair of hidden states.
 * `hq` = hidden at `<decide>`; `hk` = hidden at the last prompt position.
 */
export declare function headScore(hd: HeadWeights, hq: Float32Array, hk: Float32Array): number;
/**
 * Answers a typed question. For `choice`, runs the head once per option; each option is
 * wrapped as `<opt>…</opt>`, as in the paper.
 */
export declare function answer(hd: HeadWeights, q: Question, hiddens: {
    atDecide: Float32Array;
    atLast: Float32Array;
    perOption?: Float32Array[];
}): Answer;
