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
export declare function assertProbe(p: Probe): void;
export declare function promptFor(p: Probe): string;
export declare function decidePos(tokens: string[]): number;
export declare function optPositions(prompt: string, offsets: [number, number][]): number[];
export declare function optPositionsFor(prompt: string, offsets: [number, number][], nOptions?: number): number[];
export declare function tokenOffsets(tok: any, prompt: string): {
    ids: number[];
    tokens: string[];
    offsets: [number, number][];
};
