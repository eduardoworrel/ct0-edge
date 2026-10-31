import { type Backend } from "./backend.js";
export interface LogitCapture {
    processor: {
        processLogits(l: Float32Array): Float32Array;
        processSampledToken(t: number): void;
        resetState(): void;
    };
    arm(): void;
    take(): Float32Array | null;
    drain(): number[];
    reset(): void;
}
export declare function logitCapture(): LogitCapture;
export interface Recovery {
    rows: Int32Array;
    P: Float32Array | Uint16Array;
}
interface RecoveryF32 {
    rows: Int32Array;
    P: Float32Array;
    D: number;
}
export declare function recoverHidden(rec: Recovery | RecoveryF32, logits: Float32Array): Float32Array;
export declare function loadRecovery(rowsUrl: string, pUrl: string, format?: "f32" | "f16", fetchFn?: typeof fetch): Promise<Recovery>;
export declare function tryLoadRecovery(base: string, files: {
    rows: string;
    P: string;
}, fetchFn?: typeof fetch): Promise<{
    recovery?: Recovery;
    reason?: string;
}>;
export interface WebllmBackendOptions {
    tok: any;
    recovery?: Recovery;
    eos?: Set<number>;
}
export declare function webllmBackend(engine: any, capture: LogitCapture, opts: WebllmBackendOptions): Backend;
export interface CreateWebllmOptions extends WebllmBackendOptions {
    modelId: string;
    modelUrl?: string;
    libUrl?: string;
    appConfig?: any;
    initProgressCallback?: (p: any) => void;
    recoveryUrls?: {
        rows: string;
        P: string;
        format?: "f32" | "f16";
    };
}
export declare function createWebllm(webllm: any, opts: CreateWebllmOptions): Promise<Backend & {
    engine: any;
    capture: LogitCapture;
}>;
export declare function webllmModel(engine: any, opts?: {
    stop?: string[];
    noThink?: boolean;
    capture?: LogitCapture;
    maxSafeTokens?: number;
}): {
    countTokens: (t: string) => number;
    fitContext: (p: string) => string;
    maxSafeTokens: number;
    stream(prompt: string, o?: {
        temperature?: number;
        maxNewTokens?: number;
    }): AsyncGenerator<any, void, unknown>;
    decide(state: string, question: string): Promise<number>;
};
export {};
