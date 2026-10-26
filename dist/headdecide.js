// src/positions.ts
var CHOICE_Q = "Which vulnerability class does this code contain?";
var SEV_Q = "How severe is the security risk in this content? (1=none, 5=critical)";
function renderPrompt(state, question) {
  return "<|im_start|>system\nYou are a security decision function. Answer with exactly one word: Yes or No.<|im_end|>\n<|im_start|>user\n<state>" + state + "</state>\n<q>" + question + "</q><decide><|im_end|>\n<|im_start|>assistant\n";
}
var KINDS = ["noul", "choice", "score"];
function assertProbe(p) {
  const q = p;
  if (!q || typeof q !== "object") throw new Error("probe: expected an object {kind, state, ...}");
  if (!KINDS.includes(q.kind)) {
    const hint = q.kind === void 0 && "type" in q ? ` (the field is \`kind\`, not \`type\`: {kind: ${JSON.stringify(q.type)}, state, q})` : "";
    throw new Error(`probe: invalid kind ${JSON.stringify(q.kind)}; use "noul", "choice" or "score"${hint}`);
  }
  if (typeof q.state !== "string") throw new Error("probe: state must be a string");
  if (q.kind === "choice") {
    const o = q.options;
    if (!Array.isArray(o) || !o.length) throw new Error("probe: choice requires options (non-empty list)");
    const seen = /* @__PURE__ */ new Set();
    for (const x of o) {
      if (typeof x !== "string" || !x.trim()) throw new Error("probe: option label empty or not a string: " + JSON.stringify(x));
      if (/<\/?opt>/.test(x)) throw new Error("probe: option label cannot contain <opt> / </opt>: " + JSON.stringify(x));
      if (seen.has(x)) throw new Error("probe: duplicate option label: " + JSON.stringify(x));
      seen.add(x);
    }
  }
}
function promptFor(p) {
  assertProbe(p);
  if (p.kind === "noul") return renderPrompt(p.state, p.q ?? "");
  if (p.kind === "choice") {
    return renderPrompt(p.state, CHOICE_Q + " " + (p.options ?? []).map((o) => "<opt>" + o + "</opt>").join(""));
  }
  return renderPrompt(p.state, SEV_Q);
}
function decidePos(tokens) {
  const imEnd = [];
  tokens.forEach((s, i) => {
    if (s === "<|im_end|>") imEnd.push(i);
  });
  if (!imEnd.length) throw new Error("decidePos: no <|im_end|> in " + JSON.stringify(tokens.slice(-8)));
  let lastAsst = -1;
  tokens.forEach((s, i) => {
    if (s.trim() === "assistant") lastAsst = i;
  });
  const cand = lastAsst >= 0 ? imEnd.filter((i) => i < lastAsst) : [];
  const end = cand.length ? cand[cand.length - 1] : imEnd[imEnd.length - 1];
  if (end === 0) throw new Error("decidePos: <|im_end|> at position 0, no preceding token");
  if (tokens[end - 1] !== ">") {
    throw new Error("decidePos: token before <|im_end|> is " + JSON.stringify(tokens[end - 1]) + ", expected '>' (end of <decide>)");
  }
  return end - 1;
}
function optPositions(prompt, offsets) {
  const out = [];
  const from = Math.max(0, prompt.lastIndexOf("<q>"));
  for (const m of prompt.slice(from).matchAll(/<\/opt>/g)) {
    const c = from + m.index + m[0].length - 1;
    const hit = offsets.findIndex(([s, e]) => s <= c && c < e);
    if (hit < 0) throw new Error("optPositions: no token covers char " + c);
    out.push(hit);
  }
  return out;
}
function optPositionsFor(prompt, offsets, nOptions) {
  const pos = optPositions(prompt, offsets);
  if (nOptions === void 0) nOptions = (prompt.slice(Math.max(0, prompt.lastIndexOf("<q>"))).match(/<opt>/g) ?? []).length;
  if (pos.length !== nOptions) throw new Error(`positions: ${pos.length} </opt> markers after <q> for ${nOptions} options (label containing </opt>?)`);
  return pos;
}
function tokenOffsets(tok, prompt) {
  const ids = Array.from(tok.encode(prompt, { add_special_tokens: false }), (x) => Number(x));
  const dec = (a) => tok.decode(a, { skip_special_tokens: false, clean_up_tokenization_spaces: false });
  const tokens = ids.map((i) => dec([i]));
  const offsets = [];
  let prev = 0;
  let full = "";
  for (let i = 0; i < ids.length; i++) {
    full = dec(ids.slice(0, i + 1));
    let end = full.length;
    while (end > prev && full.charCodeAt(end - 1) === 65533) end--;
    offsets.push([prev, end]);
    prev = end;
  }
  if (full !== prompt) throw new Error("tokenOffsets: reconstruction != prompt (offsets not reliable)");
  return { ids, tokens, offsets };
}

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
    if (!per.length) return { type: "choice", choice: {} };
    const choice = {};
    const raws = [];
    for (let i = 0; i < opts.length; i++) {
      const h = per[i];
      if (!h) {
        choice[opts[i]] = 0;
        raws.push(-Infinity);
        continue;
      }
      const r = headScore(hd, hiddens.atDecide, h);
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
function backendRuntime(be, tok) {
  const run = async (prompt, withOpts, nOptions) => {
    if (!be.hiddenName) throw new Error("backendRuntime: backend has no hidden output");
    const t = tokenOffsets(tok, prompt);
    const dp = decidePos(t.tokens);
    if (!withOpts) {
      const { hiddens: hiddens2 } = await be.prefill(t.ids, { positions: [dp, t.ids.length - 1] });
      return { atDecide: hiddens2[0], atLast: hiddens2[1], atOpts: [] };
    }
    const { hiddens } = await be.prefill(t.ids, { positions: [dp, ...optPositionsFor(prompt, t.offsets, nOptions)] });
    const atOpts = hiddens.slice(1);
    return { atDecide: hiddens[0], atLast: atOpts[atOpts.length - 1], atOpts };
  };
  return {
    hiddens: async (prompt) => {
      const r = await run(prompt, false);
      return { atDecide: r.atDecide, atLast: r.atLast };
    },
    optHiddens: (prompt, nOptions) => run(prompt, true, nOptions)
  };
}
var DEFAULT_RENDER = renderPrompt;
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
  if (q.type !== "noul" && q.type !== "choice" && q.type !== "score") throw new Error(`headAnswer: invalid type ${JSON.stringify(q.type)}; use "noul", "choice" or "score"`);
  if (q.type === "noul" || q.type === "score") {
    const h2 = await rt.hiddens(render(q.state, q.question));
    return answer(hd, { type: q.type, levels: q.levels }, h2);
  }
  if (!rt.optHiddens) throw new Error("headAnswer: choice needs a runtime with optHiddens (see backendRuntime)");
  const options = q.options || [];
  const h = await rt.optHiddens(promptFor({ kind: "choice", state: q.state, options }), options.length);
  return answer(
    hd,
    { type: "choice", options },
    { atDecide: h.atDecide, atLast: h.atLast, perOption: h.atOpts }
  );
}
export {
  backendRuntime,
  headAnswer,
  headDecide
};
