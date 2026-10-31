export interface GenerateOptions {
    maxNewTokens?: number;
    temperature?: number;
    topP?: number;
    topK?: number;
    repetitionPenalty?: number;
    seed?: number;
    system?: string;
    think?: boolean;
    signal?: AbortSignal;
}
export interface ResolvedGenerateOptions {
    maxNewTokens: number;
    temperature: number;
    topP: number;
    topK: number;
    repetitionPenalty: number;
    seed: number;
    signal?: AbortSignal;
}
export declare const GENERATE_DEFAULTS: {
    readonly maxNewTokens: 512;
    readonly temperature: 0.7;
    readonly topP: 0.9;
    readonly topK: 40;
    readonly repetitionPenalty: 1.1;
    readonly seed: 0;
};
export declare function resolveGenerateOptions(o?: GenerateOptions): ResolvedGenerateOptions;
export interface Backend {
    kind: "decode-only" | "scan" | "genai" | "webllm" | (string & {});
    hiddenName: string | null;
    prefill(ids: number[], want: {
        positions: number[];
    }): Promise<{
        hiddens: Float32Array[];
        lastLogits: Float32Array;
    }>;
    dispose(): Promise<void>;
    generate(ids: number[], opts: GenerateOptions): AsyncGenerator<number>;
    snapshot?(ids: number[]): Promise<StateHandle>;
    prefillFrom?(h: StateHandle, ids: number[], want: {
        positions: number[];
    }): Promise<{
        hiddens: Float32Array[];
        lastLogits: Float32Array;
    }>;
}
export interface StateHandle {
    readonly ids: readonly number[];
    dispose(): void | Promise<void>;
}
export declare function checkContinuation(name: string, h: StateHandle, ids: number[], positions: number[]): number;
export interface Gate {
    readonly disposed: boolean;
    acquire(): Promise<() => void>;
    assertLive(): void;
    dispose(release: () => unknown): Promise<void>;
}
export declare function createGate(name: string): Gate;
export declare function gateFor(shared: object, name: string): Gate;
