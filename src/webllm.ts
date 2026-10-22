/** WebLLM (MLC) adapter for ct0-edge — wrap an MLC `MLCEngine` so the ct0-cai `Agent` can drive it.
 *  Covers the WebLLM mode alongside the native ONNX path:
 *   - stream():  System-2 generation.
 *   - decide():  System-1 — reads the logit over the Yes/No answer token via completions `logprobs`
 *                (one forward, softmax → P(yes)); NOT greedy generation.
 *  The engine is injected (no WebLLM import here), so this stays dependency-free. */
export function webllmModel(engine: any, opts: { stop?: string[] } = {}) {
  const stop = opts.stop ?? ["<|im_end|>"];
  return {
    countTokens: (t: string) => (t.length / 4) | 0,
    fitContext: (p: string) => p,
    maxSafeTokens: 3200,
    async *stream(prompt: string, o: { temperature?: number; maxNewTokens?: number } = {}) {
      const it = await engine.completions.create({ prompt, temperature: o.temperature ?? 0, max_tokens: o.maxNewTokens ?? 300, stream: true, stop });
      for await (const ch of it) { const d = ch.choices?.[0]?.text || ""; if (d) yield d; }
    },
    async decide(state: string, question: string): Promise<number> {
      const prompt = `<|im_start|>system\nYou are a security decision function. Answer with exactly one word: Yes or No.<|im_end|>\n` +
        `<|im_start|>user\nSTATE:\n${String(state).slice(0, 2000)}\n\nQUESTION: ${question}\nAnswer Yes or No.<|im_end|>\n` +
        `<|im_start|>assistant\n<think>\n\n</think>\n\n`;
      try {
        const r = await engine.completions.create({ prompt, max_tokens: 1, temperature: 0, logprobs: true, top_logprobs: 5 });
        const tl = r.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs || r.choices?.[0]?.logprobs?.top_logprobs;
        if (tl && tl.length) {
          let y = -1e9, n = -1e9;
          for (const e of tl) { const t = (e.token || "").trim().toLowerCase(); if (t === "yes") y = Math.max(y, e.logprob); if (t === "no") n = Math.max(n, e.logprob); }
          if (y > -1e8 || n > -1e8) { const m = Math.max(y, n); return Math.exp(y - m) / (Math.exp(y - m) + Math.exp(n - m)); }
        }
        return /^\s*yes/i.test(r.choices?.[0]?.text || "") ? 0.9 : 0.1;  // fallback: greedy token
      } catch { return 0.5; }
    },
  };
}
