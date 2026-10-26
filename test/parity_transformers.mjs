// parity_transformers.mjs — transformers.js backend (Node/CPU) with the REAL model (ct0-scan q4f16).
// Slow (CPU): outside the test_*.mjs glob. `npm run test:parity:transformers`; CT0_PARITY_PICKS=14,0 to choose gidx.
// Default: the 4 shortest probes of each kind (12) via ask() with the head method.
// Primary reference: rows of the transformers.js run (CT0_TJS_REF), |Δ|<1e-2; against the Python/CUDA reference only
// the same decision and |Δraw|<1 (known CPU-vs-CUDA gap: noul up to 0.61, score up to 0.72).
// Then a greedy 12-token generate() == CT0_GEN_REF.gen. Fails if files are missing unless CT0_PARITY_SKIP=1.
// Required env: CT0_MODELS_DIR (dir with ct0-scan/), CT0_PARITY_REF, CT0_TJS_REF, CT0_HEAD, CT0_HEAD_SCORE, CT0_GEN_REF.
import { createTransformersJs } from "../dist/transformers.js";
import { ask } from "../dist/ask.js";
import { loadHead } from "../dist/systemone.js";
import { promptFor, tokenOffsets } from "../dist/positions.js";
import { readFileSync, existsSync } from "node:fs";
import { AutoTokenizer, env } from "@huggingface/transformers";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };

const unset = ["CT0_MODELS_DIR", "CT0_PARITY_REF", "CT0_TJS_REF", "CT0_HEAD", "CT0_HEAD_SCORE", "CT0_GEN_REF"].filter((n) => !process.env[n]);
if (unset.length) { console.error("usage: set " + unset.join(", ") + " (see the header comment)"); process.exit(2); }
const MODELS = (process.env.CT0_MODELS_DIR).replace(/\/?$/, "/");
const REF = process.env.CT0_PARITY_REF;
const HEAD = process.env.CT0_HEAD;
const HEAD_S = process.env.CT0_HEAD_SCORE;
const TJS = process.env.CT0_TJS_REF;
const GEN = process.env.CT0_GEN_REF;
const need = [MODELS + "ct0-scan/onnx", REF, TJS, HEAD, HEAD_S, GEN];
const missing = need.filter((f) => !existsSync(f));
if (missing.length) {
  if (process.env.CT0_PARITY_SKIP === "1") console.log("\n(missing files; parity skipped by CT0_PARITY_SKIP=1)", missing);
  else ok(false, "missing files: " + missing.join(", ") + " (CT0_PARITY_SKIP=1 to skip)");
} else {
  env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = MODELS;
  const tok = await AutoTokenizer.from_pretrained("ct0-scan");
  const fx = JSON.parse(readFileSync(new URL("./fixtures/positions.json", import.meta.url)));
  const ref = JSON.parse(readFileSync(REF)), tjs = JSON.parse(readFileSync(TJS));
  const ld = (p) => { const b = readFileSync(p); return loadHead(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
  const heads = { noulChoice: ld(HEAD), score: ld(HEAD_S) };
  // gidx -> (kind, index within kind) -> reference row
  const seen = {}, meta = fx.map((f) => { const k = f.probe.kind; const i = seen[k] = (seen[k] ?? -1) + 1; return { kind: k, row: ref[k].rows.find((r) => r.i === i), tj: tjs.rows.find((r) => r.kind === k && r.i === i) }; });

  let picks;
  if (process.env.CT0_PARITY_PICKS) picks = process.env.CT0_PARITY_PICKS.split(",").map(Number);
  else {
    picks = [];
    for (const kind of ["noul", "choice", "score"]) {
      const lens = fx.map((f, g) => [g, f.ids.length]).filter(([g]) => meta[g].kind === kind).sort((a, b) => a[1] - b[1]);
      picks.push(...lens.slice(0, 4).map(([g]) => g));
    }
  }

  const t0 = Date.now();
  const be = await createTransformersJs("ct0-scan", { dtype: "q4f16", device: "cpu", localModelPath: MODELS });
  console.log(`\nloaded in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  ok(be.kind === "scan", "kind");
  ok(be.hiddenName === "mul_13450", "hidden detected: " + be.hiddenName);

  console.log("\nparity (picks=" + picks.join(",") + "): transformers.js reference |Δ|<1e-2; Python/CUDA reference: same decision, |Δraw|<1.0");
  const { headScore } = await import("../dist/systemone.js");
  const pos = await import("../dist/positions.js");
  for (const g of picks) {
    const f = fx[g], { kind, row, tj } = meta[g];
    const t1 = Date.now();
    const r = await ask(be, tok, heads, f.probe, { method: "head" });
    const sec = ((Date.now() - t1) / 1000).toFixed(0);
    if (kind === "noul") {
      ok(Math.abs(r.raw - tj.raw) < 1e-2, `#${g} noul raw=${r.raw.toFixed(4)} tjs=${tj.raw.toFixed(4)} |Δ|=${Math.abs(r.raw - tj.raw).toExponential(2)} (${sec}s)`);
      ok((r.p >= 0.5) === (row.pred_head === "yes") && Math.abs(r.raw - row.raw) < 1.0, `#${g} noul vs reference: same decision, |Δraw|=${Math.abs(r.raw - row.raw).toFixed(3)}`);
    } else if (kind === "score") {
      ok(Math.abs(r.raw - tj.raw_scorehead) < 1e-2 && r.level === tj.level_scorehead, `#${g} score raw=${r.raw.toFixed(4)} tjs=${tj.raw_scorehead.toFixed(4)} level ${r.level} (${sec}s)`);
      ok(r.level === row.level_scorehead && Math.abs(r.raw - row.raw_scorehead) < 1.0, `#${g} score vs reference: level ${r.level}/${row.level_scorehead}, |Δraw|=${Math.abs(r.raw - row.raw_scorehead).toFixed(3)}`);
    } else {
      const t = tokenOffsets(tok, promptFor(f.probe));
      const { hiddens } = await be.prefill(t.ids, { positions: [pos.decidePos(t.tokens), ...pos.optPositions(promptFor(f.probe), t.offsets)] });
      const sc = hiddens.slice(1).map((k) => headScore(heads.noulChoice, hiddens[0], k));
      const d = Math.max(...sc.map((v, i) => Math.abs(v - tj.scores_head[i])));
      ok(d < 1e-2, `#${g} choice scores_head vs tjs max|Δ|=${d.toExponential(2)} (${sec}s)`);
      ok(r.top === row.pred_head, `#${g} choice vs reference: top=${r.top} ref=${row.pred_head}`);
    }
  }

  console.log("\ngenerate (greedy 12 tokens vs the generation reference):");
  const gref = JSON.parse(readFileSync(GEN));
  const got = [];
  for await (const id of be.generate(gref.prompt_ids, { maxNewTokens: 12, temperature: 0 })) got.push(id);
  ok(JSON.stringify(got) === JSON.stringify(gref.gen), `gen == ref (${got.length} tokens): ${JSON.stringify(got)}`);
  await be.dispose();
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
