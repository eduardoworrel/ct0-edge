/**
/**
 * systemone.ts — System-1 via the paper's bilinear head, in plain JavaScript.
 *
 * The head is recovered from the truncated `head.pt` of the official repo (89% of the `k`
 * weights; `k.bias` is zeroed). Architecture derived from the pickle:
 *
 *     score = (Wq·h_q + bq) · (Wk·h_k + bk) / temperature
 *     Wq, Wk: [256, 2560]   bq, bk: [256]   temperature: 2.0
 *
 * Measured spec (200+ combinations swept against the logit-based `decide()` as oracle, with 8
 * holdout probes the head never saw):
 *   - `q` reads the hidden at the LAST token of `<decide>`
 *   - `k` reads the hidden at the LAST prompt position (`noul`) or at each option token (`choice`)
 *   - pre-norm and post-norm give the same result; normalizing the hidden hurts (corr 0.984 -> 0.962)
 *   - temperature only scales; it does not change the sign
 *   Result: 16/16 (8 original + 8 holdout), correlation +0.984 with the logit-based `decide()`.
 *
 * `choice` is a pointer head: `q` is the hidden at `<decide>` (one per question) and each option
 * is a `k`, read at the option token's hidden. This scores 8/8 on the 8 calibration probes and
 * 5/8 on 8 new ones (`k` has 29/256 zeroed rows, so comparing MAGNITUDE across options stays lossy).
 *
 * Modes (only the head supports all; the logit-based `decide()` only does noul):
 *   - `noul`   — P(yes) for a yes/no question
 *   - `choice` — ranks N options against a criteria map (one score per option)
 *   - `score`  — ordinal grade 1..K
 *
 * The head is tiny (2.6 MB fp16) and runs in ~0.1 ms per call on a 2560-dim hidden.
 */

export type QuestionType = "noul" | "choice" | "score";

export interface HeadWeights {
  /** [256, 2560] query projection */
  q: Float32Array;
  /** [256] query bias */
  qb: Float32Array;
  /** [256, 2560] key projection (89% recovered; 29 zeroed rows) */
  k: Float32Array;
  /** [256] key bias (lost in the truncated file; zeros) */
  kb: Float32Array;
  headDim: number;
  hiddenSize: number;
  temperature: number;
  /** Offset added to the final score; compensates for the lost `k.bias`.
   *  Measured by bootstrap (median over base probes, no labels) and validated 16/16 on holdout.
   *  Not a per-dimension bias: adding 34.3 to each of the 256 dimensions would give +8780. */
  scoreOffset: number;
}

export interface Question {
  type: QuestionType;
  /** `noul`: the yes/no question */
  instructions?: string;
  /** `choice`: the options and each one's criterion */
  options?: string[];
  criteria?: Record<string, string>;
  /** `score`: K ordinal levels */
  levels?: number;
}

export interface Answer {
  type: QuestionType;
  /** `noul`: P(yes) in [0,1] */
  noul?: number;
  /** `choice`: one probability per option */
  choice?: Record<string, number>;
  /** `score`: the grade 1..K */
  score?: number;
  /** raw score before the sigmoid (diagnostics) */
  raw?: number;
}

/** Loads the head weights from an ArrayBuffer (JSON header + fp16 blocks). */
export function loadHead(buf: ArrayBuffer): HeadWeights {
  // layout: [uint32 jsonLen][json][q fp16][qb fp16][k fp16][kb fp16]
  const dv = new DataView(buf);
  const jsonLen = dv.getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jsonLen)));
  // the exporter pads the JSON header to an even size so the fp16 blocks stay 2-byte aligned
  let off = 4 + jsonLen;
  const take = (n: number) => {
    const avail = Math.floor((buf.byteLength - off) / 2);
    if (n > avail) throw new Error(`head binary too short: needs ${n} fp16, ${avail} available`);
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

/** MatVec: y[d] = sum_i W[d, i] * x[i] + b[d]  (W is row-major [dim, hidden]) */
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
 * Raw head score for a pair of hidden states.
 * `hq` = hidden at `<decide>`; `hk` = hidden at the last prompt position.
 */
export function headScore(hd: HeadWeights, hq: Float32Array, hk: Float32Array): number {
  // A wrong-sized hidden would give a silent NaN (reads past the end); return NaN explicitly
  // instead. headDecide converts it to 0.5.
  if (!hq || !hk || hq.length < hd.hiddenSize || hk.length < hd.hiddenSize) return NaN;
  const a = proj(hd.q, hd.qb, hq, hd.headDim, hd.hiddenSize);
  const b = proj(hd.k, hd.kb, hk, hd.headDim, hd.hiddenSize);
  const s = dot(a, b) / hd.temperature + (hd.scoreOffset || 0);
  return Number.isFinite(s) ? s : NaN;
}

/**
 * Answers a typed question. For `choice`, runs the head once per option; each option is
 * wrapped as `<opt>…</opt>`, as in the paper.
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
    // Pointer head: the query is the single hidden at `<decide>`; each option is a key (the hidden
    // at the option token). Roles are the same as in `noul`. Measured: q@<decide>, k@</opt> gets
    // 8/8 (5/8 on unseen probes); swapping the roles gets 3/8.
    // All options share ONE prompt, with one hidden per `</opt>`.
    const per = hiddens.perOption || [];
    if (!per.length) return { type: "choice", choice: {} };
    const choice: Record<string, number> = {};
    const raws: number[] = [];
    for (let i = 0; i < opts.length; i++) {
      const h = per[i];
      if (!h) { choice[opts[i]] = 0; raws.push(-Infinity); continue; }
      const r = headScore(hd, hiddens.atDecide, h);
      raws.push(r);
      choice[opts[i]] = r;
    }
    // softmax over the options
    const mx = Math.max(...raws.filter(Number.isFinite));
    let z = 0;
    const exps = raws.map(r => (Number.isFinite(r) ? Math.exp(r - mx) : 0));
    z = exps.reduce((a, b) => a + b, 0) || 1;
    for (let i = 0; i < opts.length; i++) choice[opts[i]] = exps[i] / z;
    return { type: "choice", choice };
  }
  // score: map the raw score to 1..K through the scaled sigmoid
  const K = q.levels || 5;
  const raw = headScore(hd, hiddens.atDecide, hiddens.atLast);
  const p = sigmoid(raw);
  const score = Math.max(1, Math.min(K, Math.round(p * (K - 1) + 1)));
  return { type: "score", score, raw };
}
