// parity_decode_only.mjs — decode-only backend parity with the Python reference, real model in Node (onnxruntime-node, CPU).
// Slow (3-5 min/probe): outside the test_*.mjs glob. `npm run test:parity`; CT0_PARITY_PICKS=14 for a single probe.
// Fails if files are missing unless CT0_PARITY_SKIP=1. Required env: CT0_DECODE_ONNX, CT0_PARITY_REF, CT0_HEAD, CT0_TOK_DIR.
// Parity with the reference rows (172-question test set): |Δraw| < 1e-2 (CPU vs CUDA).
import { createDecodeOnly } from "../dist/decodeOnly.js";
import { headScore, loadHead } from "../dist/systemone.js";
import { promptFor, tokenOffsets } from "../dist/positions.js";
import { readFileSync, existsSync } from "node:fs";
import { dirname, basename, resolve } from "node:path";
import * as ort from "onnxruntime-node";
import { AutoTokenizer, env } from "@huggingface/transformers";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };

const unset = ["CT0_DECODE_ONNX", "CT0_PARITY_REF", "CT0_HEAD", "CT0_TOK_DIR"].filter((n) => !process.env[n]);
if (unset.length) { console.error("usage: set " + unset.join(", ") + " (see the header comment)"); process.exit(2); }
const MODEL = process.env.CT0_DECODE_ONNX;
const REF = process.env.CT0_PARITY_REF;
const HEAD = process.env.CT0_HEAD;
if (!existsSync(MODEL) || !existsSync(REF) || !existsSync(HEAD)) {
  const missing = [MODEL, REF, HEAD].filter((f) => !existsSync(f));
  if (process.env.CT0_PARITY_SKIP === "1") console.log("\n(missing files; parity skipped by CT0_PARITY_SKIP=1)", missing);
  else ok(false, "missing files: " + missing.join(", ") + " (CT0_PARITY_SKIP=1 to skip)");
} else {
  const tokDir = resolve(process.env.CT0_TOK_DIR);
  env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = dirname(tokDir) + "/";
  const tok = await AutoTokenizer.from_pretrained(basename(tokDir));
  const fx = JSON.parse(readFileSync(new URL("./fixtures/positions.json", import.meta.url)));
  const ref = JSON.parse(readFileSync(REF));
  const hb = readFileSync(HEAD); const hd = loadHead(hb.buffer.slice(hb.byteOffset, hb.byteOffset + hb.byteLength));

  console.log("\ngraph:");
  const session = await ort.InferenceSession.create(MODEL);
  const be = await createDecodeOnly(session, tok, { ort });
  ok(be.kind === "decode-only", "kind");
  ok(be.hiddenName === "/model/norm/Mul_1_output_0", "hidden detected: " + be.hiddenName);

  // 2 per kind, the shortest (token-by-token CPU is slow): gidx = fixture index
  const picks = (process.env.CT0_PARITY_PICKS || "14,0,79,81,141,156").split(",").map(Number);
  console.log("\nparity with the reference (|Δ| < 1e-2):");
  for (const g of picks) {
    const f = fx[g], row = ref.rows.find((r) => r.gidx === g);
    const t = tokenOffsets(tok, promptFor(f.probe));
    const positions = f.probe.kind === "choice" ? [f.pos_decide, ...f.opt_pos] : [f.pos_decide, f.pos_last];
    const t0 = Date.now();
    const { hiddens, lastLogits } = await be.prefill(t.ids, { positions });
    ok(hiddens.length === positions.length && hiddens.every((h) => h.length === 2560 && h.every(Number.isFinite)), `#${g} ${f.probe.kind}: ${hiddens.length} finite hiddens [2560] (${t.ids.length} tok, ${((Date.now() - t0) / 1000).toFixed(0)}s)`);
    ok(lastLogits.length === 248320, `#${g} last-position logits`);
    if (f.probe.kind === "choice") {
      const sc = hiddens.slice(1).map((k) => headScore(hd, hiddens[0], k));
      const d = Math.max(...sc.map((s, i) => Math.abs(s - row.scores_head[i])));
      ok(d < 1e-2, `#${g} choice scores_head max|Δ|=${d.toExponential(2)}`);
    } else {
      const raw = headScore(hd, hiddens[0], hiddens[1]);
      ok(Math.abs(raw - row.raw) < 1e-2, `#${g} ${f.probe.kind} raw=${raw.toFixed(4)} ref=${row.raw.toFixed(4)} |Δ|=${Math.abs(raw - row.raw).toExponential(2)}`);
    }
  }

  console.log("\ngenerate:");
  const ids = [];
  for await (const id of be.generate(tokenOffsets(tok, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").ids, { maxNewTokens: 3, temperature: 0 })) ids.push(id);
  ok(ids.length >= 1 && ids.every(Number.isInteger), "generates ids: " + JSON.stringify(ids));
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
