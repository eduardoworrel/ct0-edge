/**
/**
 * headdecide.ts — adapts the bilinear head (systemone.ts) to the `decide()` contract.
 *
 * `recon()` and `Agent` expect `decide(state, question) => Promise<number>`. This adapter
 * provides it using the HEAD instead of the vocab logit, which changes two things:
 *
 *   1. `choice`/`score` — modes the logit cannot do (the real gain).
 *   2. Higher margin — measured (16 cases): the head's minimum margin is 0.498 vs 0.054 for the
 *      logit. The head is more decisive, not better calibrated: it saturates at 0.000/1.000 while
 *      the logit has 6 intermediate cases. For binary judgments that widens the margin; for
 *      triage with fine confidence the logit is more informative.
 *   Binary agreement between the two is 16/16, so swapping one for the other in `recon()` does
 *   not change the verdict, only the reported confidence.
 *
 * Cost: 1.18 ms for the head on top of the model prefill (same as the logit-based decide), so
 * the head is free next to the forward pass and much cheaper than generating tokens.
 */
import { renderPrompt, promptFor, tokenOffsets, decidePos, optPositionsFor } from "./positions.js";
import type { Backend } from "./backend.js";
import { headScore, answer, type HeadWeights, type Answer } from "./systemone.js";

/** What the adapter needs from the runtime: run a prompt and return the two hidden states. */
export interface HiddenRuntime {
  /** Runs the prefill and returns the hidden at `<decide>` and at the last position. */
  hiddens(prompt: string): Promise<{ atDecide: Float32Array; atLast: Float32Array }>;
  /** For `choice`: hiddens of a prompt with several `<opt>…</opt>` (one hidden per `</opt>`, at the
   *  positions of positions.ts). Required by headAnswer with `choice`. `nOptions` = how many options the
   *  prompt must have; if the number of `</opt>` after `<q>` differs, it throws (never returns shifted hiddens). */
  optHiddens?(prompt: string, nOptions?: number): Promise<{ atDecide: Float32Array; atLast: Float32Array; atOpts: Float32Array[] }>;
}

/** HiddenRuntime over a Backend: same positions as ask(). */
export function backendRuntime(be: Backend, tok: any): HiddenRuntime {
  const run = async (prompt: string, withOpts: boolean, nOptions?: number) => {
    if (!be.hiddenName) throw new Error("backendRuntime: backend has no hidden output");
    const t = tokenOffsets(tok, prompt);
    const dp = decidePos(t.tokens);
    if (!withOpts) {
      const { hiddens } = await be.prefill(t.ids, { positions: [dp, t.ids.length - 1] });
      return { atDecide: hiddens[0], atLast: hiddens[1], atOpts: [] as Float32Array[] };
    }
    // choice: same positions as ask(): [decide, ...opt]; `atLast` = hidden of the last option.
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
  /** Prompt prefix with the `<>` markers. Default: the measured format. */
  render?: (state: string, question: string) => string;
}

const DEFAULT_RENDER = renderPrompt;

/** Creates a head-backed `decide(state, question)`. Drop-in for `recon({ decide })`. */
export function headDecide(hd: HeadWeights, rt: HiddenRuntime, opts: HeadDecideOptions = {}) {
  const render = opts.render || DEFAULT_RENDER;
  return async (state: string, question: string): Promise<number> => {
    // Graceful degradation: if the runtime fails (model gone, OOM, missing hidden), return 0.5
    // (uncertain) instead of throwing, so a loop of many decides survives a transient error.
    try {
      const h = await rt.hiddens(render(state, question));
      if (!h || !h.atDecide || !h.atLast) return 0.5;
      const score = headScore(hd, h.atDecide, h.atLast);
      if (!Number.isFinite(score)) return 0.5;   // invalid hidden -> NaN -> uncertain
      return 1 / (1 + Math.exp(-score));
    } catch { return 0.5; }
  };
}

/** Answers a typed question (`noul`, `choice`, `score`), which `decide()` cannot. */
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
  // choice: ONE prompt with all the `<opt>…</opt>`; the query is the hidden at `<decide>` and each
  // option is a key at the hidden of the token covering its `</opt>`.
  // (The fixed CHOICE_Q comes from promptFor; `q.question` is not used.)
  if (!rt.optHiddens) throw new Error("headAnswer: choice needs a runtime with optHiddens (see backendRuntime)");
  const options = q.options || [];
  const h = await rt.optHiddens(promptFor({ kind: "choice", state: q.state, options }), options.length);
  return answer(hd, { type: "choice", options },
    { atDecide: h.atDecide, atLast: h.atLast, perOption: h.atOpts });
}
