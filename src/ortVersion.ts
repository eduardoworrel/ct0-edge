

export function checkOrtVersion(v: string | undefined, min: number[] = [1, 30]): void {
  const m = /^(\d+)\.(\d+)/.exec(v ?? "");
  if (!m) return;
  if (Number(m[1]) < min[0] || (Number(m[1]) === min[0] && Number(m[2]) < min[1]))
    throw new Error(`ct0-edge: onnxruntime-web ${v} is too old; requires >= ${min.join(".")} (WebGPU returns NaN below that; the genai backend needs the LinearAttention/CausalConvWithState kernels). Pass skipVersionCheck to ignore.`);
}
