

export type QuestionType = "noul" | "choice" | "score";

export interface HeadWeights {

  q: Float32Array;

  qb: Float32Array;

  k: Float32Array;

  kb: Float32Array;
  headDim: number;
  hiddenSize: number;
  temperature: number;

  scoreOffset: number;
}

export interface Question {
  type: QuestionType;

  instructions?: string;

  options?: string[];
  criteria?: Record<string, string>;

  levels?: number;
}

export interface Answer {
  type: QuestionType;

  noul?: number;

  choice?: Record<string, number>;

  score?: number;

  raw?: number;
}

export function loadHead(buf: ArrayBuffer): HeadWeights {

  const dv = new DataView(buf);
  const jsonLen = dv.getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jsonLen)));

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

export function headScore(hd: HeadWeights, hq: Float32Array, hk: Float32Array): number {

  if (!hq || !hk || hq.length < hd.hiddenSize || hk.length < hd.hiddenSize) return NaN;
  const a = proj(hd.q, hd.qb, hq, hd.headDim, hd.hiddenSize);
  const b = proj(hd.k, hd.kb, hk, hd.headDim, hd.hiddenSize);
  const s = dot(a, b) / hd.temperature + (hd.scoreOffset || 0);
  return Number.isFinite(s) ? s : NaN;
}

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

    const mx = Math.max(...raws.filter(Number.isFinite));
    let z = 0;
    const exps = raws.map(r => (Number.isFinite(r) ? Math.exp(r - mx) : 0));
    z = exps.reduce((a, b) => a + b, 0) || 1;
    for (let i = 0; i < opts.length; i++) choice[opts[i]] = exps[i] / z;
    return { type: "choice", choice };
  }

  const K = q.levels || 5;
  const raw = headScore(hd, hiddens.atDecide, hiddens.atLast);
  const p = sigmoid(raw);
  const score = Math.max(1, Math.min(K, Math.round(p * (K - 1) + 1)));
  return { type: "score", score, raw };
}
