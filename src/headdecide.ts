
import { renderPrompt, promptFor, tokenOffsets, decidePos, optPositionsFor } from "./positions.js";
import type { Backend } from "./backend.js";
import { headScore, answer, type HeadWeights, type Answer } from "./systemone.js";

export interface HiddenRuntime {

  hiddens(prompt: string): Promise<{ atDecide: Float32Array; atLast: Float32Array }>;

  optHiddens?(prompt: string, nOptions?: number): Promise<{ atDecide: Float32Array; atLast: Float32Array; atOpts: Float32Array[] }>;
}

export function backendRuntime(be: Backend, tok: any): HiddenRuntime {
  const run = async (prompt: string, withOpts: boolean, nOptions?: number) => {
    if (!be.hiddenName) throw new Error("backendRuntime: backend has no hidden output");
    const t = tokenOffsets(tok, prompt);
    const dp = decidePos(t.tokens);
    if (!withOpts) {
      const { hiddens } = await be.prefill(t.ids, { positions: [dp, t.ids.length - 1] });
      return { atDecide: hiddens[0], atLast: hiddens[1], atOpts: [] as Float32Array[] };
    }

    const { hiddens } = await be.prefill(t.ids, { positions: [dp, ...optPositionsFor(prompt, t.offsets, nOptions)] });
    const atOpts = hiddens.slice(1);
    return { atDecide: hiddens[0], atLast: atOpts[atOpts.length - 1], atOpts };
  };
  return {
    hiddens: async (prompt) => { const r = await run(prompt, false); return { atDecide: r.atDecide, atLast: r.atLast }; },
    optHiddens: (prompt, nOptions) => run(prompt, true, nOptions),
  };
}

export interface HeadDecideOptions {

  render?: (state: string, question: string) => string;
}

const DEFAULT_RENDER = renderPrompt;

export function headDecide(hd: HeadWeights, rt: HiddenRuntime, opts: HeadDecideOptions = {}) {
  const render = opts.render || DEFAULT_RENDER;
  return async (state: string, question: string): Promise<number> => {

    try {
      const h = await rt.hiddens(render(state, question));
      if (!h || !h.atDecide || !h.atLast) return 0.5;
      const score = headScore(hd, h.atDecide, h.atLast);
      if (!Number.isFinite(score)) return 0.5;   
      return 1 / (1 + Math.exp(-score));
    } catch { return 0.5; }
  };
}

export async function headAnswer(
  hd: HeadWeights, rt: HiddenRuntime, q: { type: "noul" | "choice" | "score"; state: string; question: string; options?: string[]; levels?: number },
  opts: HeadDecideOptions = {}
): Promise<Answer> {
  const render = opts.render || DEFAULT_RENDER;
  if (q.type !== "noul" && q.type !== "choice" && q.type !== "score") throw new Error(`headAnswer: invalid type ${JSON.stringify((q as any).type)}; use "noul", "choice" or "score"`);
  if (q.type === "noul" || q.type === "score") {
    const h = await rt.hiddens(render(q.state, q.question));
    return answer(hd, { type: q.type, levels: q.levels }, h);
  }

  if (!rt.optHiddens) throw new Error("headAnswer: choice needs a runtime with optHiddens (see backendRuntime)");
  const options = q.options || [];
  const h = await rt.optHiddens(promptFor({ kind: "choice", state: q.state, options }), options.length);
  return answer(hd, { type: "choice", options },
    { atDecide: h.atDecide, atLast: h.atLast, perOption: h.atOpts });
}
