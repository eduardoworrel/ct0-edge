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

// src/ask.ts
var YES_ID = 9175;
var NO_ID = 2665;
var DIGIT_IDS = [16, 17, 18, 19, 20];
var DEFAULT_METHOD = { noul: "logits", choice: "head", score: "head" };
var sigmoid2 = (z) => 1 / (1 + Math.exp(-z));
function softmax(xs) {
  const mx = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - mx));
  const z = e.reduce((a, b) => a + b, 0) || 1;
  return e.map((v) => v / z);
}
function argmax(xs) {
  let b = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[b]) b = i;
  return b;
}
function toChoice(options, probs) {
  const choice = {};
  options.forEach((o, i) => {
    choice[o] = probs[i];
  });
  return { choice, top: options[argmax(probs)] };
}
function levelFromCuts(x, cuts) {
  let l = 1;
  for (const c of cuts) if (x >= c) l++;
  return l;
}
function checkCuts(cuts, n, read) {
  if (cuts === void 0) return void 0;
  if (!Array.isArray(cuts) || cuts.length !== n || !cuts.every(Number.isFinite) || cuts.some((c, i) => i > 0 && c <= cuts[i - 1]))
    throw new Error(`ask: scoreCuts for the ${read} read must be ${n} finite ascending numbers, got ${JSON.stringify(cuts)}`);
  return cuts;
}
function planAsk(be, tok, heads, q, opts) {
  const maxSafe = opts.maxSafeTokens ?? 2048;
  const prompt = promptFor(q);
  const type = q.kind;
  const method = opts.method ?? DEFAULT_METHOD[type];
  if (method !== "head" && method !== "logits" && method !== "both") throw new Error(`ask: unknown method ${JSON.stringify(method)}; use "head", "logits" or "both"`);
  const options = q.options ?? [];
  const t = tokenOffsets(tok, prompt);
  if (t.ids.length > maxSafe) throw new Error(`ask: prompt has ${t.ids.length} tokens, over maxSafeTokens (${maxSafe})`);
  let headCuts, logitCuts;
  if (type === "score" && opts.scoreCuts !== void 0) {
    const sc = opts.scoreCuts;
    if (Array.isArray(sc)) {
      if (method === "both") throw new Error('ask: method "both" reads two scales; pass scoreCuts as { head, logits }');
      if (method === "head") headCuts = sc;
      else logitCuts = sc;
    } else ({ head: headCuts, logits: logitCuts } = sc ?? {});
    headCuts = checkCuts(headCuts, (q.levels || 5) - 1, "head");
    logitCuts = checkCuts(logitCuts, DIGIT_IDS.length - 1, "logits");
  }
  const optFirst = type === "choice" ? options.map((o) => Number(tok.encode(o, { add_special_tokens: false })[0])) : [];
  const fromLogits = (L) => {
    if (type === "noul") {
      const raw = L[YES_ID] - L[NO_ID];
      return { type, method: "logits", raw, p: sigmoid2(raw) };
    }
    if (type === "choice") return { type, method: "logits", ...toChoice(options, softmax(optFirst.map((id) => L[id]))) };
    const d = DIGIT_IDS.map((i) => L[i]);
    const expected = softmax(d).reduce((a, p, i) => a + (i + 1) * p, 0);
    return { type, method: "logits", level: logitCuts ? levelFromCuts(expected, logitCuts) : argmax(d) + 1, expected };
  };
  if (method === "logits") return { ids: t.ids, positions: [], read: (r) => fromLogits(r.lastLogits) };
  if (!be.hiddenName) throw new Error(`ask: method '${method}' needs a graph with a hidden output (backend.hiddenName is null); use method 'logits'`);
  const hd = type === "score" ? heads.score : heads.noulChoice;
  if (!hd) throw new Error(type === "score" ? `ask: method '${method}' for score needs the score head (head_score.bin); load it via loadHeads({score})` : `ask: method '${method}' needs a loaded head for '${type}'`);
  const dp = decidePos(t.tokens);
  const withLogits = (r, L) => method === "both" ? { ...r, logits: fromLogits(L) } : r;
  if (type === "choice") {
    const op = optPositionsFor(prompt, t.offsets, options.length);
    return { ids: t.ids, positions: [dp, ...op], read: ({ hiddens, lastLogits }) => {
      const perOption = hiddens.slice(1);
      const a = answer(hd, { type: "choice", options }, { atDecide: hiddens[0], atLast: perOption[perOption.length - 1], perOption });
      return withLogits({ type, method: "head", ...toChoice(options, options.map((o) => a.choice[o])) }, lastLogits);
    } };
  }
  return { ids: t.ids, positions: [dp, t.ids.length - 1], read: ({ hiddens, lastLogits }) => {
    const a = answer(hd, { type, levels: q.levels }, { atDecide: hiddens[0], atLast: hiddens[1] });
    if (type === "noul") return withLogits({ type, method: "head", raw: a.raw, p: a.noul }, lastLogits);
    const level = headCuts && Number.isFinite(a.raw) ? levelFromCuts(a.raw, headCuts) : a.score;
    return withLogits({ type, method: "head", raw: a.raw, level }, lastLogits);
  } };
}
async function ask(be, tok, heads, q, opts = {}) {
  const p = planAsk(be, tok, heads, q, opts);
  return p.read(await be.prefill(p.ids, { positions: p.positions }));
}
function sharedPrefix(plans) {
  let k = Math.min(...plans.map((p) => p.ids.length - 1), ...plans.flatMap((p) => p.positions));
  const a = plans[0].ids;
  for (const p of plans.slice(1)) {
    let i = 0;
    while (i < k && p.ids[i] === a[i]) i++;
    k = i;
  }
  return k;
}
function askMany(be, tok, heads, state, qs, opts = {}) {
  return askManyWith(be, tok, heads, state, qs, () => opts);
}
async function askManyWith(be, tok, heads, state, qs, optsFor) {
  if (typeof state !== "string") throw new Error("askMany: state must be a string");
  const plans = qs.map((q0) => {
    if (q0?.state !== void 0 && q0.state !== state) throw new Error("askMany: a question carries a state different from the shared state");
    const q = { ...q0, state };
    return planAsk(be, tok, heads, q, optsFor(q));
  });
  const out = [];
  const k = plans.length > 1 && be.snapshot && be.prefillFrom ? sharedPrefix(plans) : 0;
  if (k < 1) {
    for (const p of plans) out.push(p.read(await be.prefill(p.ids, { positions: p.positions })));
    return out;
  }
  const h = await be.snapshot(plans[0].ids.slice(0, k));
  try {
    for (const p of plans) out.push(p.read(await be.prefillFrom(h, p.ids, { positions: p.positions })));
  } finally {
    await h.dispose();
  }
  return out;
}
export {
  DEFAULT_METHOD,
  DIGIT_IDS,
  NO_ID,
  YES_ID,
  ask,
  askMany,
  askManyWith,
  levelFromCuts
};
