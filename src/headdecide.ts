/**
 * headdecide.ts — adapta o head bilinear (systemone.ts) ao contrato `decide()` do cai-lite.
 *
 * O `recon()` e o `Agent` esperam `decide(state, question) => Promise<number>`. Este adaptador
 * fornece isso usando o HEAD em vez do logit de vocab, o que muda duas coisas:
 *
 *   1. **`choice`/`score`** — modos que o logit NAO faz (este e o ganho real).
 *   2. **Margem mais alta** — medido (16 casos): margem minima do head = 0.498 vs 0.054
 *      do logit. O head e MAIS decisivo, nao mais calibrado: satura em 0.000/1.000
 *      enquanto o logit tem 6 casos intermediarios. Para juizo binario isso aumenta a
 *      margem; para triagem com confianca fina, o logit e mais informativo.
 *   A concordancia binaria entre os dois e 16/16, entao trocar um pelo outro no `recon()`
 *   nao muda o veredito — muda a confianca reportada.
 *
 * Custo: 1.18 ms para o head + o prefill do modelo (o mesmo custo do decide atual).
 * Ou seja, o head é gratuito em relação ao forward, e muito mais barato que gerar tokens.
 */
import { headScore, answer, type HeadWeights, type Answer } from "./systemone";

/** O que o adaptador precisa do runtime: rodar um prompt e devolver os dois hidden states. */
export interface HiddenRuntime {
  /** Executa o prefill e devolve o hidden em `<decide>` e na última posição. */
  hiddens(prompt: string): Promise<{ atDecide: Float32Array; atLast: Float32Array }>;
}

export interface HeadDecideOptions {
  /** Prefixo do prompt com os marcadores `<>`. Default: o formato medido no EVIDENCE §20c. */
  render?: (state: string, question: string) => string;
}

const DEFAULT_RENDER = (state: string, question: string) =>
  `<|im_start|>system\nYou are a security decision function. Answer with exactly one word: Yes or No.<|im_end|>\n` +
  `<|im_start|>user\n<state>${state}</state>\n<q>${question}</q><decide><|im_end|>\n` +
  `<|im_start|>assistant\n`;

/** Cria um `decide(state, question)` apoiado no head. Drop-in para `recon({ decide })`. */
export function headDecide(hd: HeadWeights, rt: HiddenRuntime, opts: HeadDecideOptions = {}) {
  const render = opts.render || DEFAULT_RENDER;
  return async (state: string, question: string): Promise<number> => {
    // Degradacao graciosa: se o runtime falhar (modelo fora, OOM, hidden ausente), devolve 0.5
    // (incerteza) em vez de propagar. `recon()` ja tolera `decide` que rejeita, mas um adaptador
    // que estoura obriga o chamador a envolver cada call em try/catch — e um laco de 38 decides
    // nao deve morrer no primeiro erro transitorio.
    try {
      const h = await rt.hiddens(render(state, question));
      if (!h || !h.atDecide || !h.atLast) return 0.5;
      const score = headScore(hd, h.atDecide, h.atLast);
      return 1 / (1 + Math.exp(-score));
    } catch { return 0.5; }
  };
}

/** Responde uma pergunta tipada (`noul`, `choice`, `score`) — o que o `decide()` não faz. */
export async function headAnswer(
  hd: HeadWeights, rt: HiddenRuntime, q: { type: "noul" | "choice" | "score"; state: string; question: string; options?: string[]; levels?: number },
  opts: HeadDecideOptions = {}
): Promise<Answer> {
  const render = opts.render || DEFAULT_RENDER;
  if (q.type === "noul" || q.type === "score") {
    const h = await rt.hiddens(render(q.state, q.question));
    return answer(hd, { type: q.type, levels: q.levels }, h);
  }
  // choice: um prefill por opção (cada `<opt>…</opt>` no mesmo slot)
  const perOption: Float32Array[] = [];
  let last: Float32Array | null = null;
  for (const o of q.options || []) {
    const h = await rt.hiddens(render(q.state, `${q.question}\n<opt>${o}</opt>`));
    perOption.push(h.atDecide);
    last = h.atLast;
  }
  return answer(hd, { type: "choice", options: q.options }, { atDecide: perOption[0], atLast: last! , perOption });
}
