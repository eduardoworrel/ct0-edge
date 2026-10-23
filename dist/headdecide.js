// src/systemone.ts
function proj(W, b, x, dim, hidden) {
  const y = new Float32Array(dim);
  for (let d = 0; d < dim; d++) {
    let acc = b[d];
    const row = d * hidden;
    for (let i = 0; i < hidden; i++) acc += W[row + i] * x[i];
    y[d] = acc;
  }
  return y;
}
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
var sigmoid = (z) => 1 / (1 + Math.exp(-z));
function headScore(hd, hq, hk) {
  if (!hq || !hk || hq.length < hd.hiddenSize || hk.length < hd.hiddenSize) return NaN;
  const a = proj(hd.q, hd.qb, hq, hd.headDim, hd.hiddenSize);
  const b = proj(hd.k, hd.kb, hk, hd.headDim, hd.hiddenSize);
  const s = dot(a, b) / hd.temperature + (hd.scoreOffset || 0);
  return Number.isFinite(s) ? s : NaN;
}
function answer(hd, q, hiddens) {
  if (q.type === "noul") {
    const raw2 = headScore(hd, hiddens.atDecide, hiddens.atLast);
    return { type: "noul", noul: sigmoid(raw2), raw: raw2 };
  }
  if (q.type === "choice") {
    const opts = q.options || [];
    const per = hiddens.perOption || [];
    const choice = {};
    const raws = [];
    for (let i = 0; i < opts.length; i++) {
      const h = per[i];
      if (!h) {
        choice[opts[i]] = 0;
        raws.push(-Infinity);
        continue;
      }
      const r = headScore(hd, h, hiddens.atLast);
      raws.push(r);
      choice[opts[i]] = r;
    }
    const mx = Math.max(...raws.filter(Number.isFinite));
    let z = 0;
    const exps = raws.map((r) => Number.isFinite(r) ? Math.exp(r - mx) : 0);
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

// src/headdecide.ts
var DEFAULT_RENDER = (state, question) => `<|im_start|>system
You are a security decision function. Answer with exactly one word: Yes or No.<|im_end|>
<|im_start|>user
<state>${state}</state>
<q>${question}</q><decide><|im_end|>
<|im_start|>assistant
`;
function headDecide(hd, rt, opts = {}) {
  const render = opts.render || DEFAULT_RENDER;
  return async (state, question) => {
    try {
      const h = await rt.hiddens(render(state, question));
      if (!h || !h.atDecide || !h.atLast) return 0.5;
      const score = headScore(hd, h.atDecide, h.atLast);
      if (!Number.isFinite(score)) return 0.5;
      return 1 / (1 + Math.exp(-score));
    } catch {
      return 0.5;
    }
  };
}
async function headAnswer(hd, rt, q, opts = {}) {
  const render = opts.render || DEFAULT_RENDER;
  if (q.type === "noul" || q.type === "score") {
    const h = await rt.hiddens(render(q.state, q.question));
    return answer(hd, { type: q.type, levels: q.levels }, h);
  }
  const perOption = [];
  let last = null;
  for (const o of q.options || []) {
    const h = await rt.hiddens(render(q.state, `${q.question}
<opt>${o}</opt>`));
    perOption.push(h.atDecide);
    last = h.atLast;
  }
  return answer(hd, { type: "choice", options: q.options }, { atDecide: perOption[0], atLast: last, perOption });
}
export {
  headAnswer,
  headDecide
};
