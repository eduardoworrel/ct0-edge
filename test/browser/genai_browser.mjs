// genai_browser.mjs — genai backend (dist/genai.js) in Chrome with onnxruntime-web WebGPU, REAL int4 graph (2.44 GB) served LOCALLY.
// Outside `npm test` (GPU/Chrome/2.4 GB): `npm run test:browser:genai`.
// TOLERANCE (reported, not silent): |Δraw|<2e-2 is UNREACHABLE between WebGPU and CUDA. The WebGPU fp16 hidden has systematic
// noise in the massive-activation dims (0, 24, 186, 835: |h|=13-30; chunk 7 vs 128 of the SAME graph already differs by 1.6 in dim 835), and the
// bilinear head amplifies it: vs genai/CUDA the head |Δraw| is ~1.0 (noul), 0.08-0.24 (score), choice scores up to 8.8. Here: identical decision (12/12),
// 'logits' path (L[Yes]-L[No]) |Δ|<0.25 vs CUDA, head |Δraw| < CT0_GENAI_TOL (default 1.5; WebGPU's own noise floor is ~0.4),
// choice with no numeric bound (same top + finite only; distribution printed).
// Tests: (1) 24 greedy tokens == PyTorch reference for the spike prompt; (2) 12 probes of the test set via ask() (head method)
// vs the onnxruntime-genai/CUDA reference rows: same decision and |Δraw| < TOL; (3) duplicate hidden; (4) abort.
// Required env: CT0_GENAI_DIR (int4 model directory), CT0_TOK_DIR, CT0_GENAI_REF, CT0_HEAD, CT0_HEAD_SCORE, CT0_PLAYWRIGHT.
// Optional env: CT0_ORT_DIR (onnxruntime-web dist; default the one nested in transformers.js, 1.31-dev), CT0_HEADED=1,
//      CT0_GENAI_TOL (default 1.5; see TOLERANCE), CT0_GENAI_PICKS (comma-separated gidx), CT0_PARITY_SKIP=1 skips if files are missing.
import http from "node:http";
import { createReadStream, statSync, existsSync, readFileSync } from "node:fs";
import { resolve, join, extname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const unset = ["CT0_GENAI_DIR", "CT0_TOK_DIR", "CT0_GENAI_REF", "CT0_HEAD", "CT0_HEAD_SCORE", "CT0_PLAYWRIGHT"].filter((n) => !process.env[n]);
if (unset.length) { console.error("usage: set " + unset.join(", ") + " (see the header comment)"); process.exit(2); }
const GENAI = process.env.CT0_GENAI_DIR;
const TOK_DIR = process.env.CT0_TOK_DIR;
const nested = join(root, "node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist");
const ORT_DIR = process.env.CT0_ORT_DIR || (existsSync(nested) ? nested : join(root, "node_modules/onnxruntime-web/dist"));
const REF = process.env.CT0_GENAI_REF;
const HEAD = process.env.CT0_HEAD;
const HEAD_S = process.env.CT0_HEAD_SCORE;
const PW = process.env.CT0_PLAYWRIGHT;
const BUNDLE = process.env.CT0_ORT_BUNDLE || "ort.webgpu.bundle.min.mjs"; // CT0_ORT_BUNDLE=ort.bundle.min.mjs (JSEP) already fails in InferenceSession.create (before the EP assertion, which the mock covers)
const TOL = Number(process.env.CT0_GENAI_TOL || 1.5), TOL_LOGITS = 0.25, TOL_CHUNK = 0.75;
let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };

const need = [join(GENAI, "model_webgpu.onnx"), join(GENAI, "model.onnx.data"), join(TOK_DIR, "tokenizer.json"), join(ORT_DIR, BUNDLE), REF, HEAD, HEAD_S, PW, join(root, "dist/genai.js"), join(root, "dist/ask.js")];
const missing = need.filter((f) => !existsSync(f));
if (missing.length) {
  if (process.env.CT0_PARITY_SKIP === "1") { console.log("\n(missing files; test skipped by CT0_PARITY_SKIP=1)", missing); process.exit(0); }
  console.log("  ✗ missing files:", missing.join(", "), "(CT0_PARITY_SKIP=1 to skip)"); process.exit(1);
}
const { chromium } = await import(pathToFileURL(PW).href);
const fx = JSON.parse(readFileSync(join(root, "test/fixtures/positions.json")));
const spike = JSON.parse(readFileSync(join(root, "test/fixtures/genai_spike_prompt.json")));
const ref = JSON.parse(readFileSync(REF));

// picks: the 4 shortest probes of each kind (12)
let picks;
if (process.env.CT0_GENAI_PICKS) picks = process.env.CT0_GENAI_PICKS.split(",").map(Number);
else { picks = []; for (const kind of ["noul", "choice", "score"]) picks.push(...fx.map((f, g) => [g, f.ids.length, f.probe.kind]).filter((x) => x[2] === kind).sort((a, b) => a[1] - b[1]).slice(0, 4).map((x) => x[0])); }
const refRow = (g) => { const k = fx[g].probe.kind; const r = (k === "score" ? ref.rows_head_score : ref.rows).find((x) => x.gidx === g && x.kind === k); if (!r) throw new Error("no reference row for #" + g); return r; };
const b64 = (p) => readFileSync(p).toString("base64");

const MIME = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".html": "text/html" };
const ROUTES = [["/dist/", join(root, "dist")], ["/nm/ort/", ORT_DIR], ["/nm/common/", join(root, "node_modules/onnxruntime-common/dist/esm")],
  ["/nm/tjs/", join(root, "node_modules/@huggingface/transformers/dist")], ["/files/tok/", TOK_DIR], ["/genai/", GENAI]];
const page = `<!doctype html><meta charset=utf-8><script type="importmap">{"imports":{
"onnxruntime-web/webgpu":"/nm/ort/${BUNDLE}","onnxruntime-web":"/nm/ort/${BUNDLE}","onnxruntime-common":"/nm/common/index.js","@huggingface/transformers":"/nm/tjs/transformers.web.js"}}</script><body>genai</body>`;
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(req.url.split("?")[0]);
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
  args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--ignore-gpu-blocklist", "--js-flags=--max-old-space-size=8192"] });
let code = 1;
try {
  const p = await browser.newPage();
  p.on("pageerror", (e) => console.log("  PAGEERR", e.message));
  p.on("console", (m) => { const t = m.text(); if (process.env.CT0_DEBUG) console.log("  CONSOLE", m.type(), t.slice(0, 300)); else if (m.type() === "error" && !/404|VerifyEach|Rerunning|onnxruntime/.test(t)) console.log("  CERR", t.slice(0, 200)); });
  await p.goto(origin + "/");
  console.log("webgpu:", await p.evaluate(async () => !!navigator.gpu && !!(await navigator.gpu.requestAdapter())));
  const res = await p.evaluate(async ({ origin, spike, picks, fx, heads, verifyEp, skipVersionCheck }) => {
    const out = { timings: {} };
    const { createGenai } = await import("/dist/genai.js");
    const { ask, askMany } = await import("/dist/ask.js");
    const { loadHead, headScore } = await import("/dist/systemone.js");
    const pos = await import("/dist/positions.js");
    const tjs = await import("@huggingface/transformers");
    tjs.env.allowRemoteModels = false; tjs.env.allowLocalModels = true; tjs.env.localModelPath = "/files/"; tjs.env.useBrowserCache = false;
    const tok = await tjs.AutoTokenizer.from_pretrained("tok");
    const d64 = (s) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u.buffer; };
    const hd = { noulChoice: loadHead(d64(heads.h)), score: loadHead(d64(heads.s)) };
    let t = performance.now();
    const be = await createGenai(origin + "/genai/model_webgpu.onnx", origin + "/genai/model.onnx.data", { verifyEp, skipVersionCheck });
    out.timings.loadMs = Math.round(performance.now() - t);
    out.kind = be.kind; out.hiddenName = be.hiddenName;
    // (1) greedy
    t = performance.now(); const got = [], stamps = [];
    for await (const id of be.generate(spike.ids, { maxNewTokens: 24, temperature: 0, repetitionPenalty: 1 })) { got.push(id); stamps.push(performance.now()); }
    out.gen = got; out.timings.gen24Ms = Math.round(performance.now() - t);
    const perTok = (st) => (st.at(-1) - st[5]) / (st.length - 6);
    const samples = [stamps.length > 6 ? perTok(stamps) : NaN];
    for (let k = 0; k < 2; k++) { const st2 = []; for await (const _ of be.generate(spike.ids, { maxNewTokens: 24, temperature: 0, repetitionPenalty: 1 })) st2.push(performance.now()); samples.push(st2.length > 6 ? perTok(st2) : NaN); }
    out.timings.decodeMsPerTok = Math.round(samples.sort((a, b) => a - b)[1]);   // median of 3 (a single-sample timing was unstable)
    // (4) abort + session reuse
    const ac = new AbortController(); const ab = [];
    for await (const id of be.generate(spike.ids, { maxNewTokens: 50, temperature: 0, repetitionPenalty: 1, signal: ac.signal })) { ab.push(id); if (ab.length === 3) ac.abort(); }
    out.abortLen = ab.length; out.abortPrefix = ab;
    // (2) probes
    out.rows = [];
    for (const g of picks) {
      const f = fx[g]; t = performance.now();
      const tk = pos.tokenOffsets(tok, pos.promptFor(f.probe));
      const row = { g, kind: f.probe.kind, idsEq: JSON.stringify(tk.ids) === JSON.stringify(f.ids) };
      if (f.probe.kind === "choice") {
        const { hiddens } = await be.prefill(tk.ids, { positions: [pos.decidePos(tk.tokens), ...pos.optPositions(pos.promptFor(f.probe), tk.offsets)] });
        row.scores = hiddens.slice(1).map((k) => headScore(hd.noulChoice, hiddens[0], k));
        row.top = f.probe.options[row.scores.indexOf(Math.max(...row.scores))];
        row.fin = hiddens.every((h) => Array.from(h).every(Number.isFinite));
      } else { const r = await ask(be, tok, hd, f.probe, { method: "head" }); row.raw = r.raw; row.p = r.p; row.level = r.level;
        be.chunk = 7; const r7 = await ask(be, tok, hd, f.probe, { method: "head" }); be.chunk = 128; row.rawChunk7 = r7.raw;
        if (f.probe.kind === "noul") { const l = await ask(be, tok, hd, f.probe, { method: "logits" }); row.logitRaw = l.raw; } }
      row.ms = Math.round(performance.now() - t); out.rows.push(row);
    }
    // (3) duplicate hidden
    const f0 = fx[picks[0]]; const dd = await be.prefill(f0.ids, { positions: [f0.pos_decide, f0.pos_last, f0.pos_decide] });
    out.dup = { same: dd.hiddens[0].every((v, i) => v === dd.hiddens[2][i]), diff: dd.hiddens[0].some((v, i) => v !== dd.hiddens[1][i]), len: dd.hiddens[0].length };
    // (5) internal consistency: prefill in chunks of 7 vs 128 (same graph, same GPU)
    be.chunk = 7; const c7 = await be.prefill(f0.ids, { positions: [f0.pos_decide, f0.pos_last] }); be.chunk = 128;
    const full = await be.prefill(f0.ids, { positions: [f0.pos_decide, f0.pos_last] });
    const am = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
    let md = 0, mx = 0, mdim = -1; const big = []; for (let k = 0; k < 2; k++) for (let i = 0; i < full.hiddens[k].length; i++) { const d = Math.abs(full.hiddens[k][i] - c7.hiddens[k][i]); if (d > md) { md = d; mdim = i; } mx = Math.max(mx, Math.abs(full.hiddens[k][i])); if (Math.abs(full.hiddens[k][i]) > 10 && k === 1) big.push([i, full.hiddens[k][i], c7.hiddens[k][i]]); }
    let ld = 0; for (let i = 0; i < full.lastLogits.length; i++) ld = Math.max(ld, Math.abs(full.lastLogits[i] - c7.lastLogits[i]));
    out.chunkConsistency = { mdim, big, hiddenMaxDiff: md, hiddenMaxAbs: mx, logitsMaxDiff: ld, argmaxEq: am(full.lastLogits) === am(c7.lastLogits) };
    // (6) askMany (GPU snapshot of the shared prefix, re-fed to every continuation) vs ask one by one, method both
    const nf = Object.values(fx).find((f) => f.probe.kind === "noul"), cf = Object.values(fx).find((f) => f.probe.kind === "choice");
    const qs = [{ kind: "noul", q: nf.probe.q }, { kind: "choice", options: cf.probe.options }, { kind: "score" }];
    t = performance.now(); const one = []; for (const q of qs) one.push(await ask(be, tok, hd, { ...q, state: nf.probe.state }, { method: "both" }));
    const oneMs = performance.now() - t;
    t = performance.now(); const many = await askMany(be, tok, hd, nf.probe.state, qs, { method: "both" });
    const manyMs = performance.now() - t;
    const again = await askMany(be, tok, hd, nf.probe.state, qs, { method: "both" });
    out.askMany = { one, many, again, oneMs: Math.round(oneMs), manyMs: Math.round(manyMs) };
    await be.dispose();
    return out;
  }, { origin, spike, picks, fx: picks.map((g) => ({ g })).reduce((a, { g }) => (a[g] = fx[g], a), {}), heads: { h: b64(HEAD), s: b64(HEAD_S) }, verifyEp: true, skipVersionCheck: process.env.CT0_SKIPVER === "1" });
  // decode on the default path (verifyEp:true) must stay <= 1.15x (median of 3 samples each side) of the unverified path (verification uses a throwaway session)
  const t2 = await p.evaluate(async ({ origin, spike, skipVersionCheck }) => {
    const { createGenai } = await import("/dist/genai.js");
    const be = await createGenai(origin + "/genai/model_webgpu.onnx", origin + "/genai/model.onnx.data", { verifyEp: false, skipVersionCheck });
    const samples = [];
    for (let k = 0; k < 3; k++) { const stamps = []; for await (const id of be.generate(spike.ids, { maxNewTokens: 24, temperature: 0, repetitionPenalty: 1 })) stamps.push(performance.now()); samples.push((stamps.at(-1) - stamps[5]) / (stamps.length - 6)); }
    await be.dispose(); return samples.sort((a, b) => a - b)[1];   // median of 3
  }, { origin, spike, skipVersionCheck: process.env.CT0_SKIPVER === "1" });
  res.timings.decodeMsPerTokUnverified = Math.round(t2 * 10) / 10;
  // EP verdict cache: the verified load above stored the key (with the real adapter); a second verified load does not probe
  const c = await p.evaluate(async ({ origin, skipVersionCheck }) => {
    const { createGenai } = await import("/dist/genai.js");
    const keys = Object.keys(localStorage).filter((k) => k.startsWith("ct0-edge:genai-ep:"));
    let t = performance.now();
    const be = await createGenai(origin + "/genai/model_webgpu.onnx", origin + "/genai/model.onnx.data", { verifyEp: true, skipVersionCheck, warmup: false });
    const ms = Math.round(performance.now() - t); await be.dispose(); return { keys, ms };
  }, { origin, skipVersionCheck: process.env.CT0_SKIPVER === "1" });
  console.log("  epCache:", JSON.stringify(c), "first verified load:", res.timings.loadMs, "ms");
  ok(c.keys.length === 1 && /^ct0-edge:genai-ep:[^:]+:[0-9a-f]{8}:/.test(c.keys[0]), "EP verdict stored in localStorage with the adapter hash: " + c.keys[0]);
  ok(c.ms < res.timings.loadMs, `second verified load (cache) ${c.ms} ms < first ${res.timings.loadMs} ms (no probe)`);
  console.log("  timings:", JSON.stringify(res.timings));
  if (process.env.CT0_GENAI_OUT) (await import("node:fs")).writeFileSync(process.env.CT0_GENAI_OUT, JSON.stringify(res));
  console.log("  chunk 7 vs 128:", JSON.stringify(res.chunkConsistency));
  ok(res.chunkConsistency.argmaxEq && res.chunkConsistency.logitsMaxDiff < 1.0, "chunked(7) prefill == single prefill: same argmax, logits max|Δ|<1.0");
  ok(res.timings.decodeMsPerTok <= 1.15 * res.timings.decodeMsPerTokUnverified, `default (verified) decode ${res.timings.decodeMsPerTok} ms/tok <= 1.15 x unverified (medians of 3) ${res.timings.decodeMsPerTokUnverified}`);
  ok(res.kind === "genai" && res.hiddenName === "hidden_states", `kind/hidden: ${res.kind}/${res.hiddenName}`);
  ok(JSON.stringify(res.gen) === JSON.stringify(spike.ref), `24 greedy tokens == PyTorch reference: ${JSON.stringify(res.gen)}`);
  ok(res.abortLen === 3 && JSON.stringify(res.abortPrefix) === JSON.stringify(spike.ref.slice(0, 3)), "abort after 3 tokens; session reused");
  ok(res.dup.same && res.dup.diff && res.dup.len === 2560, "duplicate hiddens equal, distinct positions differ");
  { // askMany: same decisions as ask one by one; numbers within the chunk-boundary tolerance (the snapshot moves chunk boundaries)
    const { one, many, again, oneMs, manyMs } = res.askMany;
    const dec = (r) => [r.p >= 0.5, r.top, r.level, r.logits.p >= 0.5, r.logits.top, r.logits.level].map(String).join();
    const dRaw = Math.max(...one.map((r, i) => Math.max(Math.abs((r.raw ?? 0) - (many[i].raw ?? 0)), Math.abs((r.logits.raw ?? 0) - (many[i].logits.raw ?? 0)))));
    const dProb = (a, b) => Math.max(0, ...Object.keys(a ?? {}).map((k) => Math.abs(a[k] - b[k])));
    const dChoice = Math.max(...one.map((r, i) => Math.max(dProb(r.choice, many[i].choice), dProb(r.logits.choice, many[i].logits.choice))));
    const dExp = Math.max(...one.map((r, i) => Math.abs((r.logits.expected ?? 0) - (many[i].logits.expected ?? 0))));
    console.log(`  askMany vs ask x3: ${manyMs} ms vs ${oneMs} ms; max|Δraw| ${dRaw.toFixed(4)}, max|Δchoice p| ${dChoice.toFixed(4)}, |Δexpected| ${dExp.toFixed(4)}`);
    ok(one.every((r, i) => dec(r) === dec(many[i])) && dRaw < TOL_CHUNK, `askMany == ask x3: same decisions, max|Δraw|=${dRaw.toFixed(4)} < ${TOL_CHUNK}`);
    ok(dChoice < TOL_CHUNK && dExp < TOL_CHUNK, `askMany == ask x3: choice probabilities max|Δ|=${dChoice.toFixed(4)}, score expected |Δ|=${dExp.toFixed(4)} (both < ${TOL_CHUNK})`);
    ok(JSON.stringify(again) === JSON.stringify(many), "askMany twice: identical (the snapshot is rebuilt and released each call)");
  }
  console.log(`\nparity vs genai/CUDA (same decision; head |Δraw| < ${TOL}; logits |Δ| < ${TOL_LOGITS}):`);
  const deltas = [], logitDeltas = [], choiceDeltas = [], chunkDeltas = [];
  for (const r of res.rows) {
    const rr = refRow(r.g);
    ok(r.idsEq, `#${r.g} JS tokenizer ids == Python fixture`);
    if (r.kind === "noul") {
      const d = Math.abs(r.raw - rr.raw); deltas.push(d);
      ok((r.p >= 0.5) === (rr.pred_head === "yes") && d < TOL, `#${r.g} noul raw=${r.raw.toFixed(4)} ref=${rr.raw.toFixed(4)} |Δ|=${d.toFixed(3)} (${r.ms}ms)`);
      const ref = Math.log(rr.p_logits / (1 - rr.p_logits)), dl = Math.abs(r.logitRaw - ref); logitDeltas.push(dl);
      ok(dl < TOL_LOGITS, `#${r.g} noul logits L[Yes]-L[No]=${r.logitRaw.toFixed(3)} ref=${ref.toFixed(3)} |Δ|=${dl.toFixed(3)}`);
    } else if (r.kind === "score") { const d = Math.abs(r.raw - rr.raw); deltas.push(d); ok(r.level === rr.level_head && d < TOL, `#${r.g} score raw=${r.raw.toFixed(4)} ref=${rr.raw.toFixed(4)} level ${r.level}/${rr.level_head} |Δ|=${d.toFixed(3)} (${r.ms}ms)`); }
    else { const d = Math.max(...r.scores.map((v, i) => Math.abs(v - rr.scores_head[i]))); choiceDeltas.push(d); ok(r.top === rr.pred_head && r.fin, `#${r.g} choice top=${r.top}/${rr.pred_head} max|Δscores|=${d.toFixed(2)} (informativo) (${r.ms}ms)`); }
    if (r.rawChunk7 !== undefined) { const dc = Math.abs(r.raw - r.rawChunk7); chunkDeltas.push(dc); ok(dc < TOL_CHUNK, `#${r.g} head raw chunk7 vs 128: |Δ|=${dc.toFixed(3)}`); }
  }
  const dist = (n, a) => { a.sort((x, y) => x - y); console.log(`  ${n} (n=${a.length}): min=${a[0]?.toFixed(3)} median=${a[a.length >> 1]?.toFixed(3)} max=${a.at(-1)?.toFixed(3)}`); };
  dist("|Δraw| head noul+score vs CUDA", deltas); dist("|Δ| logits noul vs CUDA", logitDeltas); dist("max|Δscores| choice vs CUDA", choiceDeltas); dist("|Δraw| head chunk7 vs 128", chunkDeltas);
  code = fail ? 1 : 0;
} catch (e) { console.log("  ✗ ERROR:", String(e.stack || e.message).slice(0, 1500)); } finally { await browser.close(); server.close(); }
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(code);
