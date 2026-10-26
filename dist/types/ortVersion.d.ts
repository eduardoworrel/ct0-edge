/**
 * ortVersion.ts — single onnxruntime-web version check, used by EVERY path that loads ONNX in the
 * browser (CT0.load / createDecodeOnly / createGenai). Below 1.30 WebGPU returns NaN (and genai lacks the
 * LinearAttention/CausalConvWithState kernels).
 */
/** Verified: onnxruntime-web 1.30.0 (npm) and 1.31.0-dev.20260914 load and run the graph. Below 1.30 it throws.
 *  A missing/unreadable version (e.g. onnxruntime-node) does not throw. */
export declare function checkOrtVersion(v: string | undefined, min?: number[]): void;
