export type QuestionType = "noul" | "choice" | "score";
export interface HeadWeights {
    q: Float32Array;
    qb: Float32Array;
    k: Float32Array;
    kb: Float32Array;
    headDim: number;
    hiddenSize: number;
    temperature: number;
    scoreOffset: number;
}
export interface Question {
    type: QuestionType;
    instructions?: string;
    options?: string[];
    criteria?: Record<string, string>;
    levels?: number;
}
export interface Answer {
    type: QuestionType;
    noul?: number;
    choice?: Record<string, number>;
    score?: number;
    raw?: number;
}
export declare function loadHead(buf: ArrayBuffer): HeadWeights;
export declare function headScore(hd: HeadWeights, hq: Float32Array, hk: Float32Array): number;
export declare function answer(hd: HeadWeights, q: Question, hiddens: {
    atDecide: Float32Array;
    atLast: Float32Array;
    perOption?: Float32Array[];
}): Answer;
