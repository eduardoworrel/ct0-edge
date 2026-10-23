/**
 * systemone.ts — System-1 via o head BILINEAR do paper, em JavaScript puro.
 *
 * O que é: o head recuperado do `head.pt` truncado do repo oficial (89% dos pesos de `k`,
 * `k.bias` zerado). Arquitetura derivada do pickle (não do paper, que não detalha):
 *
 *     score = (Wq·h_q + bq) · (Wk·h_k + bk) / temperature
 *     Wq, Wk: [256, 2560]   bq, bk: [256]   temperature: 2.0
 *
 * ESPECIFICAÇÃO MEDIDA (o paper não diz; varremos 200+ combinações contra o `decide()` por
 * logits como oráculo, com 8 probes de holdout que o head nunca viu):
 *   - `q` lê o hidden na posição do ÚLTIMO token de `<decide>`   ← invertido do nome
 *   - `k` lê o hidden na ÚLTIMA posição do prompt
 *   - pre-norm e pos-norm dão o mesmo resultado
 *   - normalizar o hidden ATRAPALHA (corr 0.984 → 0.962)
 *   - temperature só escala; não muda o sinal
 *   Resultado: 16/16 (8 originais + 8 holdout), correlação +0.984 com o `decide()` por logits.
 *
 * Modos suportados (só o head faz — o `decide()` por logits faz apenas noul):
 *   - `noul`   — P(sim) de uma pergunta
 *   - `choice` — ranqueia N opções por um mapa de critérios (um score por opção)
 *   - `score`  — nota ordinal 1..K
 *
 * O head é minúsculo (2.6 MB fp16) e roda em ~0.1 ms por chamada sobre um hidden de 2560.
 */

export type QuestionType = "noul" | "choice" | "score";

export interface HeadWeights {
  /** [256, 2560] projeção de query */
  q: Float32Array;
  /** [256] bias de query */
  qb: Float32Array;
  /** [256, 2560] projeção de key (89% recuperado; 29 linhas zeradas) */
  k: Float32Array;
  /** [256] bias de key (perdido no arquivo truncado → zeros) */
  kb: Float32Array;
  headDim: number;
  hiddenSize: number;
  temperature: number;
  /** Offset somado ao score final. Recupera o `k.bias` perdido no truncamento do head.pt.
   *  Medido por bootstrap (mediana dos probes base, sem rotulos) e validado 16/16 com holdout.
   *  NAO e um bias por dimensao — somar 34.3 em cada uma das 256 dimensoes daria +8780. */
  scoreOffset: number;
}

export interface Question {
  type: QuestionType;
  /** para `noul`: a pergunta sim/não */
  instructions?: string;
  /** para `choice`: as opções e o critério de cada uma */
  options?: string[];
  criteria?: Record<string, string>;
  /** para `score`: K níveis ordinais */
  levels?: number;
}

export interface Answer {
  type: QuestionType;
  /** `noul`: P(sim) em [0,1] */
  noul?: number;
  /** `choice`: uma probabilidade por opção */
  choice?: Record<string, number>;
  /** `score`: a nota 1..K */
  score?: number;
  /** score bruto antes do sigmoid (diagnóstico) */
  raw?: number;
}

/** Carrega os pesos do head a partir de um ArrayBuffer (formato: 3 blocos fp16 + header JSON). */
export function loadHead(buf: ArrayBuffer): HeadWeights {
  // formato simples: [uint32 jsonLen][json][q fp16][qb fp16][k fp16][kb fp16]
  const dv = new DataView(buf);
  const jsonLen = dv.getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jsonLen)));
  // alinha em 2 bytes: o header JSON pode ter tamanho impar e desalinhar os Uint16Array
  let off = 4 + jsonLen;   // o exportador alinha o header em 2 bytes (padding com espaco)
  const take = (n: number) => {
    const avail = Math.floor((buf.byteLength - off) / 2);
    if (n > avail) throw new Error(`head binario curto: pediu ${n} fp16, disponivel ${avail}`);
    const f32 = new Float32Array(n);
    for (let i = 0; i < n; i++) f32[i] = f16ToF32(dv.getUint16(off + i * 2, true));
    off += n * 2;
    return f32;
  };
  return {
    q: take(meta.headDim * meta.hiddenSize),
    qb: take(meta.headDim),
    k: take(meta.headDim * meta.hiddenSize),
    kb: take(meta.headDim),
    headDim: meta.headDim,
    hiddenSize: meta.hiddenSize,
    temperature: meta.temperature,
    scoreOffset: meta.score_offset ?? 0,
  };
}

function f16ToF32(h: number): number {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** Produto MatVec: y[d] = sum_i W[d, i] * x[i] + b[d]  (W é row-major [dim, hidden]) */
function proj(W: Float32Array, b: Float32Array, x: Float32Array, dim: number, hidden: number): Float32Array {
  const y = new Float32Array(dim);
  for (let d = 0; d < dim; d++) {
    let acc = b[d];
    const row = d * hidden;
    for (let i = 0; i < hidden; i++) acc += W[row + i] * x[i];
    y[d] = acc;
  }
  return y;
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/**
 * Score bruto do head para um par de hidden states.
 * `hq` = hidden em `<decide>`; `hk` = hidden na última posição do prompt.
 */
export function headScore(hd: HeadWeights, hq: Float32Array, hk: Float32Array): number {
  const a = proj(hd.q, hd.qb, hq, hd.headDim, hd.hiddenSize);
  const b = proj(hd.k, hd.kb, hk, hd.headDim, hd.hiddenSize);
  return dot(a, b) / hd.temperature + (hd.scoreOffset || 0);
}

/**
 * Responde uma pergunta tipada. Para `choice`, roda o head uma vez por opção —
 * o formato do prompt de cada opção é o mesmo `<opt>…</opt>` que o paper usa.
 */
export function answer(
  hd: HeadWeights,
  q: Question,
  hiddens: { atDecide: Float32Array; atLast: Float32Array; perOption?: Float32Array[] }
): Answer {
  if (q.type === "noul") {
    const raw = headScore(hd, hiddens.atDecide, hiddens.atLast);
    return { type: "noul", noul: sigmoid(raw), raw };
  }
  if (q.type === "choice") {
    const opts = q.options || [];
    // opcoes no MESMO prompt (option_isolation=False no head oficial): o head le o hidden
    // em cada `</opt>`, nao em um prefill por opcao. Se `perOption` nao vier, cai no atDecide.
    const per = hiddens.perOption || [];
    const choice: Record<string, number> = {};
    const raws: number[] = [];
    for (let i = 0; i < opts.length; i++) {
      const h = per[i];
      if (!h) { choice[opts[i]] = 0; raws.push(-Infinity); continue; }
      const r = headScore(hd, h, hiddens.atLast);
      raws.push(r);
      choice[opts[i]] = r;
    }
    // softmax sobre as opções
    const mx = Math.max(...raws.filter(Number.isFinite));
    let z = 0;
    const exps = raws.map(r => (Number.isFinite(r) ? Math.exp(r - mx) : 0));
    z = exps.reduce((a, b) => a + b, 0) || 1;
    for (let i = 0; i < opts.length; i++) choice[opts[i]] = exps[i] / z;
    return { type: "choice", choice };
  }
  // score: mapeia o score bruto para 1..K pelo sigmoid escalonado
  const K = q.levels || 5;
  const raw = headScore(hd, hiddens.atDecide, hiddens.atLast);
  const p = sigmoid(raw);
  const score = Math.max(1, Math.min(K, Math.round(p * (K - 1) + 1)));
  return { type: "score", score, raw };
}
