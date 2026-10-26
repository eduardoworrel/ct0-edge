// parity_full.mjs — FULL parity (all 172 probes of the test set) of the transformers.js backend (Node/CPU)
// against the aggregates of the Python reference JSON. Outside `npm test` (slow: tens of minutes to hours).
//
//   node test/parity_full.mjs [--backend transformers.js] [--limit N] [--out results.json] [--resume]
//
// Run order: low discrepancy (van der Corput) within each kind, interleaving noul/choice/score,
// so ANY completed prefix is a stratified sample; --limit N runs only the first N of that order.
// Each probe does ONE prefill (positions = union) reused by ask() for the head and logits methods (and head.bin on
// score) through a caching backend. Checkpoint in <out>.ckpt.jsonl (--resume continues where it stopped).
// The aggregates (hits, exact/±1, Spearman with AVERAGE RANKS for ties) are computed over the SAME set of probes
// for Python and JS. Reference: the CUDA scan q4f16 rows (CT0_PARITY_REF).
// Required env: CT0_MODELS_DIR (dir with ct0-scan/), CT0_PARITY_REF, CT0_TJS_REF, CT0_HEAD, CT0_HEAD_SCORE.
import { createTransformersJs } from "../dist/transformers.js";
import { ask } from "../dist/ask.js";
import { loadHead } from "../dist/systemone.js";
import { promptFor, tokenOffsets, decidePos, optPositions } from "../dist/positions.js";
import { readFileSync, existsSync, writeFileSync, appendFileSync } from "node:fs";
import { AutoTokenizer, env } from "@huggingface/transformers";

const args = process.argv.slice(2);
const arg = (n, d) => { const i = args.indexOf("--" + n); return i < 0 ? d : (args[i + 1]?.startsWith("--") || args[i + 1] === undefined ? true : args[i + 1]); };
const backendName = arg("backend", "transformers.js");
if (backendName !== "transformers.js") {
  console.error(`backend '${backendName}' is not supported here: decode-only on CPU takes 3-5 min/probe (see test/parity_decode_only.mjs); genai/webllm run in a browser (test/browser/).`);
  process.exit(2);
}
const unset = ["CT0_MODELS_DIR", "CT0_PARITY_REF", "CT0_TJS_REF", "CT0_HEAD", "CT0_HEAD_SCORE"].filter((n) => !process.env[n]);
if (unset.length) { console.error("usage: set " + unset.join(", ") + " (see the header comment)"); process.exit(2); }
const MODELS = (process.env.CT0_MODELS_DIR).replace(/\/?$/, "/");
const REF = process.env.CT0_PARITY_REF;
const TJS = process.env.CT0_TJS_REF;
const HEAD = process.env.CT0_HEAD;
const HEAD_S = process.env.CT0_HEAD_SCORE;
const OUT = arg("out", "parity_full_" + backendName + ".json");
const LIMIT = Number(arg("limit", 0)) || Infinity;
const CKPT = OUT + ".ckpt.jsonl";
const missing = [MODELS + "ct0-scan/onnx", REF, HEAD, HEAD_S].filter((f) => !existsSync(f));
if (missing.length) { console.error("missing files:", missing); process.exit(1); }

// ---- statistics ----
/** Average ranks (1-based); ties get the mean of their positions. */
export function avgRanks(xs) {
  const idx = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
  const r = new Array(xs.length);
  for (let i = 0; i < idx.length;) {
    let j = i; while (j + 1 < idx.length && xs[idx[j + 1]] === xs[idx[i]]) j++;
    const m = (i + j) / 2 + 1; for (let k = i; k <= j; k++) r[idx[k]] = m; i = j + 1;
  }
  return r;
}
/** Spearman = Pearson of the average ranks (tie-aware). NaN if a series is constant. */
export function spearman(a, b) {
  const x = avgRanks(a), y = avgRanks(b), n = x.length;
  const mx = x.reduce((s, v) => s + v, 0) / n, my = y.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return sxy / Math.sqrt(sxx * syy);
}

// ---- backend with prefill cache (one pass per probe, several ask calls) ----
function cached(be) {
  let key = null, byPos = new Map(), last = null;
  return Object.assign(Object.create(be), {
    async prime(ids, positions) { const r = await be.prefill(ids, { positions }); key = ids.join(","); byPos = new Map(positions.map((p, i) => [p, r.hiddens[i]])); last = r.lastLogits; },
    async prefill(ids, want) {
      if (ids.join(",") !== key || want.positions.some((p) => !byPos.has(p))) throw new Error("cache miss");
      return { hiddens: want.positions.map((p) => byPos.get(p)), lastLogits: last };
    },
  });
}

env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = MODELS;
const tok = await AutoTokenizer.from_pretrained("ct0-scan");
const fx = JSON.parse(readFileSync(new URL("./fixtures/positions.json", import.meta.url)));
const ref = JSON.parse(readFileSync(REF));
const ld = (p) => { const b = readFileSync(p); return loadHead(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const heads = { noulChoice: ld(HEAD), score: ld(HEAD_S) };

const seen = {};
const items = fx.map((f, g) => { const k = f.probe.kind; const i = seen[k] = (seen[k] ?? -1) + 1; return { g, kind: k, i, f, row: ref[k].rows.find((r) => r.i === i) }; });

// stratified order: van der Corput per kind, interleaved
const vdc = (n) => { let r = 0, d = 0.5; for (; n > 0; n >>= 1, d /= 2) if (n & 1) r += d; return r; };
const perKind = {};
for (const it of items) (perKind[it.kind] ??= []).push(it);
for (const k in perKind) perKind[k].sort((a, b) => vdc(a.i) - vdc(b.i));
const order = [];
for (let n = 0; order.length < items.length; n++) for (const k of ["noul", "choice", "score"]) if (perKind[k][n]) order.push(perKind[k][n]);
const todo = order.slice(0, LIMIT);

const done = new Map();
if (arg("resume", false) && existsSync(CKPT)) for (const l of readFileSync(CKPT, "utf8").split("\n").filter(Boolean)) { const r = JSON.parse(l); done.set(r.g, r); }
console.log(`probes: ${todo.length} of ${items.length} (already done: ${done.size})`);

const t0 = Date.now();
const raw = await createTransformersJs("ct0-scan", { dtype: "q4f16", device: "cpu", localModelPath: MODELS });
const be = cached(raw);
console.log(`model loaded in ${((Date.now() - t0) / 1000).toFixed(0)}s; hidden=${raw.hiddenName}`);

for (const it of todo) {
  if (done.has(it.g)) continue;
  const q = it.f.probe, prompt = promptFor(q), t = tokenOffsets(tok, prompt), t1 = Date.now();
  const positions = [decidePos(t.tokens), t.ids.length - 1, ...(q.kind === "choice" ? optPositions(prompt, t.offsets) : [])];
  await be.prime(t.ids, positions);
  const r = { g: it.g, kind: it.kind, i: it.i, n_tok: t.ids.length };
  r.head = await ask(be, tok, heads, q, { method: "head" });
  r.logits = await ask(be, tok, heads, q, { method: "logits" });
  if (it.kind === "score") r.headbin = await ask(be, tok, { noulChoice: heads.noulChoice, score: heads.noulChoice }, q, { method: "head" });
  r.sec = (Date.now() - t1) / 1000;
  done.set(it.g, r); appendFileSync(CKPT, JSON.stringify(r) + "\n");
  console.log(`[${done.size}/${todo.length}] #${it.g} ${it.kind} ${t.ids.length} tok ${r.sec.toFixed(0)}s`);
}

// ---- aggregates (completed probes within `todo` only) ----
const rows = todo.map((it) => ({ it, r: done.get(it.g) })).filter((x) => x.r);
const sel = (k) => rows.filter((x) => x.it.kind === k);
const out = { backend: backendName, ref: REF, n_done: rows.length, n_total: items.length, stratified_subset: rows.length < items.length };
const pct = (a, n) => `${a}/${n}`;
const maxd = (xs) => xs.length ? Math.max(...xs) : null;

{ // noul
  const R = sel("noul"), n = R.length;
  const jsHead = R.filter(({ it, r }) => (r.head.p >= 0.5) === (it.f.probe.want === "yes")).length;
  const jsLog = R.filter(({ it, r }) => (r.logits.p >= 0.5) === (it.f.probe.want === "yes")).length;
  const pyHead = R.filter(({ it }) => it.row.ok_head).length, pyLog = R.filter(({ it }) => it.row.ok_logits).length;
  out.noul = { n, js: { logits_hits: jsLog, head_hits: jsHead }, py: { logits_hits: pyLog, head_hits: pyHead },
    decisions_equal_head: R.filter(({ it, r }) => ((r.head.p >= 0.5) ? "yes" : "no") === it.row.pred_head).length,
    decisions_equal_logits: R.filter(({ it, r }) => ((r.logits.p >= 0.5) ? "yes" : "no") === it.row.pred_logits).length,
    max_abs_d_raw_head: maxd(R.map(({ it, r }) => Math.abs(r.head.raw - it.row.raw))) };
}
{ // choice
  const R = sel("choice"), n = R.length;
  out.choice = { n, js: { logits_hits: R.filter(({ it, r }) => r.logits.top === it.f.probe.want).length, head_hits: R.filter(({ it, r }) => r.head.top === it.f.probe.want).length },
    py: { logits_hits: R.filter(({ it }) => it.row.ok_logits).length, head_hits: R.filter(({ it }) => it.row.ok_head).length },
    decisions_equal_head: R.filter(({ it, r }) => r.head.top === it.row.pred_head).length,
    decisions_equal_logits: R.filter(({ it, r }) => r.logits.top === it.row.pred_logits).length };
}
{ // score
  const R = sel("score"), n = R.length;
  const sev = R.map(({ it }) => it.row.sev);
  const stats = (lv) => ({ exact: lv.filter((l, i) => l === sev[i]).length, within1: lv.filter((l, i) => Math.abs(l - sev[i]) <= 1).length, spearman_level: spearman(lv, sev) });
  const js = { logits: stats(R.map(({ r }) => r.logits.level)), headbin: stats(R.map(({ r }) => r.headbin.level)), head_score: stats(R.map(({ r }) => r.head.level)) };
  const py = { logits: stats(R.map(({ it }) => it.row.level_logits)), headbin: stats(R.map(({ it }) => it.row.level_headbin)), head_score: stats(R.map(({ it }) => it.row.level_scorehead)) };
  out.score = { n, js, py,
    levels_equal_head_score: R.filter(({ it, r }) => r.head.level === it.row.level_scorehead).length,
    levels_equal_logits: R.filter(({ it, r }) => r.logits.level === it.row.level_logits).length,
    max_abs_d_raw_head_score: maxd(R.map(({ it, r }) => Math.abs(r.head.raw - it.row.raw_scorehead))) };
}
out.mean_sec = rows.reduce((s, x) => s + x.r.sec, 0) / Math.max(1, rows.length);
writeFileSync(OUT, JSON.stringify(out, null, 2));
console.log("\n" + JSON.stringify(out, null, 2));
console.log(`\n${rows.length} probes; output in ${OUT}`);
process.exit(0);
