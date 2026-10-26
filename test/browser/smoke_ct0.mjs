// smoke_ct0.mjs — CT0 (dist bundle) in Chrome with onnxruntime-web WebGPU, decode-only q4 graph served LOCALLY.
// Outside `npm test` (3 GB, needs GPU/Chrome): `npm run test:browser`.
// Required env: CT0_DECODE_DIR (decode-only model directory), CT0_TOK_DIR, CT0_HEAD (head.bin, noul/choice),
//      CT0_HEAD_SCORE (head_score.bin), CT0_ROWS (Python reference rows for the decode-only q4 model),
//      CT0_PLAYWRIGHT (path to the playwright module).
// Optional env: CT0_DECODE_MODEL (default model_q4_decode.onnx), CT0_HEADED=1 for a window,
//      CT0_PROBES (fixture indices, default "0,14,79,141": 2 noul + 1 choice + 1 score, all short; decode-only runs
//      token by token, so this smoke does NOT cover all 172 probes; that is test:parity).
//      Besides load/decide/prefillHiddens, it calls ask() in the 3 modes (head and logits), compares the DECISIONS with the
//      Python reference rows (CUDA; |Δraw| < 1.5 on noul, WebGPU tolerance) and measures tok/s of a short generate.
import http from "node:http";
import { createReadStream, statSync, existsSync, readFileSync } from "node:fs";
import { resolve, join, extname, basename } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const unset = ["CT0_DECODE_DIR", "CT0_TOK_DIR", "CT0_HEAD", "CT0_HEAD_SCORE", "CT0_ROWS", "CT0_PLAYWRIGHT"].filter((n) => !process.env[n]);
if (unset.length) { console.error("usage: set " + unset.join(", ") + " (see the header comment)"); process.exit(2); }
const DECODE_DIR = process.env.CT0_DECODE_DIR;
const MODEL = process.env.CT0_DECODE_MODEL || "model_q4_decode.onnx";
const TOK_DIR = process.env.CT0_TOK_DIR;
const HEAD = process.env.CT0_HEAD;
const HEAD_SCORE = process.env.CT0_HEAD_SCORE;
const ROWS = process.env.CT0_ROWS;
const PW = process.env.CT0_PLAYWRIGHT;
let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };

for (const f of [join(DECODE_DIR, MODEL), join(DECODE_DIR, MODEL + ".data"), join(TOK_DIR, "tokenizer.json"), PW, join(root, "dist/ct0.js"), HEAD, HEAD_SCORE, ROWS]) {
  if (!existsSync(f)) { console.log("  ✗ missing file:", f); process.exit(1); }
}
const { chromium } = await import(pathToFileURL(PW).href);
const fx = JSON.parse(readFileSync(join(root, "test/fixtures/positions.json")));
const probe = fx[14];  // short noul (64 tok)
const rows = JSON.parse(readFileSync(ROWS));
const gidxs = (process.env.CT0_PROBES || "0,14,79,141").split(",").map(Number);
const probes = gidxs.map((g) => ({ gidx: g, probe: fx[g].probe, ntok: fx[g].ids.length }));
const rowOf = (arr, g) => arr.find((r) => r.gidx === g);

const MIME = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".html": "text/html" };
const ROUTES = [
  ["/dist/", join(root, "dist")], ["/nm/ort/", join(root, "node_modules/onnxruntime-web/dist")],
  ["/nm/common/", join(root, "node_modules/onnxruntime-common/dist/esm")],
  ["/nm/tjs/", join(root, "node_modules/@huggingface/transformers/dist")],
  ["/heads/noul/", resolve(HEAD, "..")], ["/heads/score/", resolve(HEAD_SCORE, "..")],
  ["/files/tok/", TOK_DIR], ["/files/", DECODE_DIR],
];
const page = `<!doctype html><meta charset=utf-8><script type="importmap">{"imports":{
"onnxruntime-web/webgpu":"/nm/ort/ort.webgpu.bundle.min.mjs","onnxruntime-web":"/nm/ort/ort.webgpu.bundle.min.mjs","onnxruntime-common":"/nm/common/index.js","@huggingface/transformers":"/nm/tjs/transformers.web.js"}}</script><body>ct0</body>`;
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(req.url.split("?")[0]);
  if (process.env.CT0_DEBUG && !u.startsWith("/nm")) console.log("  REQ", u);
  res.setHeader("Access-Control-Allow-Origin", "*");
  if (u === "/") { res.setHeader("content-type", "text/html"); return res.end(page); }
  for (const [pre, dir] of ROUTES) {
    if (!u.startsWith(pre)) continue;
    const f = resolve(dir, u.slice(pre.length));
    if (!f.startsWith(resolve(dir)) || !existsSync(f) || !statSync(f).isFile()) continue;
    const size = statSync(f).size, m = /bytes=(\d+)-(\d*)/.exec(req.headers.range || "");
    res.setHeader("content-type", MIME[extname(f)] || "application/octet-stream"); res.setHeader("accept-ranges", "bytes");
    if (m) { const s = +m[1], e = m[2] ? +m[2] : size - 1; res.writeHead(206, { "content-range": `bytes ${s}-${e}/${size}`, "content-length": e - s + 1 }); return createReadStream(f, { start: s, end: e }).pipe(res); }
    res.setHeader("content-length", size); return createReadStream(f).pipe(res);
  }
  res.statusCode = 404; res.end("404");
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const origin = "http://127.0.0.1:" + server.address().port;

const browser = await chromium.launch({ channel: "chrome", headless: process.env.CT0_HEADED !== "1",
  args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU,Vulkan", "--ignore-gpu-blocklist"] });
let code = 1;
try {
  const p = await browser.newPage();
  p.on("pageerror", (e) => console.log("  PAGEERR", e.message));
  p.on("console", (m) => { if (process.env.CT0_DEBUG && m.type() === "warning") console.log("  CWARN", m.text().slice(0, 300));
  if (m.type() === "error" && !/404|constant folding|VerifyEach|Rerunning/.test(m.text())) console.log("  CERR", m.text().slice(0, 200)); });
  await p.goto(origin + "/");
  console.log("webgpu:", await p.evaluate(async () => !!navigator.gpu && !!(await navigator.gpu.requestAdapter())));
  const res = await p.evaluate(async ({ origin, model, prompt, probes, HEADURL, SCOREURL }) => {
    const { CT0 } = await import("/dist/ct0.js");
    const st = [];
    const ct0 = await CT0.load({ baseUrl: origin + "/files/", decodeModel: model, externalDataFile: model + ".data", providers: ["webgpu"], onStatus: (s) => st.push(s) });
    const t = performance.now();
    const pd = await ct0.decide("User-Agent: curl/8.0; GET /admin HTTP/1.1; src 203.0.113.9", "Is this an IOC?");
    const t1 = performance.now() - t;
    const hs = await ct0.prefillHiddens(prompt);
    const fin = (a) => Array.from(a).every(Number.isFinite);
    // ask() in the 3 modes (head + logits) + tok/s of a short generate
    const BASE = { noul: HEADURL, score: SCOREURL };
    await ct0.loadHeads({ noul: location.origin + BASE.noul, score: location.origin + BASE.score });
    const asks = [];
    for (const { gidx, probe: pr } of probes) {
      for (const method of ["head", "logits"]) {
        const t0 = performance.now(); let r, err;
        try { r = await ct0.ask(pr, { method }); } catch (e) { err = String(e.message); }
        asks.push({ gidx, kind: pr.kind, method, r, err, ms: Math.round(performance.now() - t0) });
      }
    }
    const tg = performance.now(); let n = 0, tFirst = 0, tLast = 0, text = "";
    for await (const piece of ct0.generate("Count from 1 to 30, separated by commas.", { maxNewTokens: 24, temperature: 0, think: false })) { n++; text += piece; tLast = performance.now(); if (n === 1) tFirst = tLast; }
    const gen = { tokens: n, text, firstMs: Math.round(tFirst - tg), tokPerSec: n > 1 ? (n - 1) / ((tLast - tFirst) / 1000) : 0 };
    return { asks, gen, status: st, hiddenOutput: ct0.hiddenOutput, decide: pd, decideMs: Math.round(t1),
      atDecide: hs && [hs.atDecide.length, fin(hs.atDecide)], atLast: hs && [hs.atLast.length, fin(hs.atLast)], atOpts: hs?.atOpts.length };
  }, { origin, model: MODEL, prompt: probe.prompt, probes, HEADURL: "/heads/noul/" + basename(HEAD), SCOREURL: "/heads/score/" + basename(HEAD_SCORE) });
  console.log("  ", JSON.stringify({ ...res, asks: undefined }));
  ok(res.hiddenOutput === "/model/norm/Mul_1_output_0", "hidden detected in ort-web: " + res.hiddenOutput);
  ok(Number.isFinite(res.decide) && res.decide > 0 && res.decide < 1, "decide() finite in (0,1): " + res.decide);
  ok(res.atDecide?.[0] === 2560 && res.atDecide[1] && res.atLast?.[0] === 2560 && res.atLast[1], "prefillHiddens: finite hiddens [2560]");
  // ---- ask() vs the Python reference rows
  console.log("ask() (probes " + gidxs.join(",") + "; ntok " + probes.map((x) => x.ntok).join(",") + "):");
  const fin = (x) => typeof x === "number" && Number.isFinite(x);
  for (const a of res.asks) {
    const tag = `${a.kind}#${a.gidx} ${a.method} (${a.ms} ms)`;
    if (a.err || !a.r) { ok(false, tag + ": error " + a.err); continue; }
    if (a.kind === "noul") {
      const row = rowOf(rows.rows, a.gidx);
      ok(fin(a.r.p) && fin(a.r.raw) && a.r.p >= 0 && a.r.p <= 1, tag + ": finite p/raw (p=" + a.r.p.toFixed(4) + ")");
      const pred = a.r.p >= 0.5 ? "yes" : "no";
      if (a.method === "head") ok(pred === row.pred_head && Math.abs(a.r.raw - row.raw) < 1.5, `${tag}: decision ${pred} == reference ${row.pred_head}; raw ${a.r.raw.toFixed(3)} vs ${row.raw.toFixed(3)}`);
      else ok(pred === row.pred_logits, `${tag}: decision ${pred} == reference ${row.pred_logits}`);
    } else if (a.kind === "choice") {
      const row = rowOf(rows.rows, a.gidx), vals = Object.values(a.r.choice);
      ok(vals.length === row.n_options && vals.every(fin) && Math.abs(vals.reduce((x, y) => x + y, 0) - 1) < 1e-3, tag + ": " + vals.length + " finite probabilities sum to 1");
      const want = a.method === "head" ? row.pred_head : row.pred_logits;
      ok(a.r.top === want, `${tag}: top ${a.r.top} == reference ${want}`);
    } else {
      const row = a.method === "head" ? rowOf(rows.rows_head_score, a.gidx) : rowOf(rows.rows, a.gidx);
      const lv = a.method === "head" ? row.level_head : row.level_logits;
      ok(Number.isInteger(a.r.level) && a.r.level >= 1 && a.r.level <= 5 && (a.method === "logits" || fin(a.r.raw)), tag + ": level 1..5 (" + a.r.level + ")");
      ok(a.r.level === lv, `${tag}: level ${a.r.level} == reference ${lv}` + (a.method === "head" ? " (head_score.bin)" : ""));
    }
  }
  console.log("short generate:", JSON.stringify(res.gen));
  ok(res.gen.tokens >= 8 && res.gen.tokens <= 24 && Number.isFinite(res.gen.tokPerSec) && res.gen.tokPerSec > 0, `generate: ${res.gen.tokens} tokens, ${res.gen.tokPerSec.toFixed(2)} tok/s (first token in ${res.gen.firstMs} ms)`);
  code = fail ? 1 : 0;
} catch (e) { console.log("  ✗ ERROR:", String(e.message).slice(0, 400)); } finally { await browser.close(); server.close(); }
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(code);
