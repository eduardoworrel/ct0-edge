// webllm_browser.mjs — webllm backend (dist/webllm.js) in Chrome/WebGPU with the REAL MLC q4f16_1 build served LOCALLY.
// Outside `npm test` (GPU/Chrome/2 GB): `npm run test:browser:webllm`.
// Tests: (1) webllmModel decide() on a contrastive pair => probability strictly in (0,1), not 0/0.5/1 (one-hot top_logprobs
//   bug); (2) ask() on 12 probes of the test set (4 noul + 4 choice + 4 score) via the head and logits methods vs
//   the MLC reference rows: |Δraw| < 2e-2 (head noul/score), logits noul, choice top; (3) duplicate hiddens; (4) generate + abort.
// Required env: CT0_MLC_DIR (q4f16_1 build), CT0_REC_DIR (directory with P_q4f16_8192.*), CT0_TOK_DIR,
//   CT0_MLC_REF, CT0_HEAD_SCORE, CT0_PLAYWRIGHT.
// Optional env: CT0_HEAD (default <CT0_MLC_DIR>/head.bin), CT0_WEBLLM_JS (lib/index.js of @mlc-ai/web-llm 0.2.86; default jsdelivr CDN),
//   CT0_REC_FMT (f32|f16; default f32), CT0_MLC_TOL (default 2e-2), CT0_MLC_PICKS (gidx,...), CT0_PORT (fixed; needed to reuse the cache:
//   WebLLM keys it by URL, otherwise every run writes ~2.7 GB of new data to the profile), CT0_PROFILE (persistent profile dir: reuses the
//   WebLLM cache), CT0_HEADED=1, CT0_PARITY_SKIP=1 skips if files are missing.
import http from "node:http";
import { createReadStream, statSync, existsSync, readFileSync } from "node:fs";
import { resolve, join, extname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const unset = ["CT0_MLC_DIR", "CT0_REC_DIR", "CT0_TOK_DIR", "CT0_MLC_REF", "CT0_HEAD_SCORE", "CT0_PLAYWRIGHT"].filter((n) => !process.env[n]);
if (unset.length) { console.error("usage: set " + unset.join(", ") + " (see the header comment)"); process.exit(2); }
const MLC = process.env.CT0_MLC_DIR;
const REC = process.env.CT0_REC_DIR;
const FMT = process.env.CT0_REC_FMT || "f32";
const TOK_DIR = process.env.CT0_TOK_DIR;
const REF = process.env.CT0_MLC_REF;
const HEAD = process.env.CT0_HEAD || join(MLC, "head.bin");
const HEAD_S = process.env.CT0_HEAD_SCORE;
const PW = process.env.CT0_PLAYWRIGHT;
const WL_JS = process.env.CT0_WEBLLM_JS || "";
const TOL = Number(process.env.CT0_MLC_TOL || 2e-2);
let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };

const need = [join(MLC, "mlc-chat-config.json"), join(MLC, "model-lib.wasm"), join(MLC, "ndarray-cache.json"), join(REC, "P_q4f16_rows.i32"), join(REC, `P_q4f16_8192.${FMT}`),
  join(TOK_DIR, "tokenizer.json"), REF, HEAD, HEAD_S, PW, join(root, "dist/webllm.js"), join(root, "dist/ask.js"), ...(WL_JS ? [WL_JS] : [])];
const missing = need.filter((f) => !existsSync(f));
if (missing.length) {
  if (process.env.CT0_PARITY_SKIP === "1") { console.log("\n(missing files; test skipped by CT0_PARITY_SKIP=1)", missing); process.exit(0); }
  console.log("  ✗ missing files:", missing.join(", "), "(CT0_PARITY_SKIP=1 to skip)"); process.exit(1);
}
const { chromium } = await import(pathToFileURL(PW).href);
const fx = JSON.parse(readFileSync(join(root, "test/fixtures/positions.json")));
const ref = JSON.parse(readFileSync(REF));
let picks;
if (process.env.CT0_MLC_PICKS) picks = process.env.CT0_MLC_PICKS.split(",").map(Number);
else { picks = []; for (const kind of ["noul", "choice", "score"]) picks.push(...fx.map((f, g) => [g, f.ids.length, f.probe.kind]).filter((x) => x[2] === kind).sort((a, b) => a[1] - b[1]).slice(0, 4).map((x) => x[0])); }
const refRow = (g) => { const k = fx[g].probe.kind; const r = (k === "score" ? ref.rows_head_score : ref.rows).find((x) => x.gidx === g && x.kind === k); if (!r) throw new Error("no reference row for #" + g); return r; };
const b64 = (p) => readFileSync(p).toString("base64");

const MIME = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".html": "text/html" };
const nested = join(root, "node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist");
const ORT_DIR = process.env.CT0_ORT_DIR || (existsSync(nested) ? nested : join(root, "node_modules/onnxruntime-web/dist"));
const ROUTES = [["/dist/", join(root, "dist")], ["/nm/ort/", ORT_DIR], ["/nm/common/", join(root, "node_modules/onnxruntime-common/dist/esm")], ["/nm/tjs/", join(root, "node_modules/@huggingface/transformers/dist")], ["/files/tok/", TOK_DIR],
  ["/rec/", REC], ["/m/resolve/main/", MLC], ["/m/", MLC], ...(WL_JS ? [["/wl/", resolve(WL_JS, "..")]] : [])];
const WL_URL = WL_JS ? "/wl/" + WL_JS.split("/").pop() : "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.86/+esm";
const page = `<!doctype html><meta charset=utf-8><script type="importmap">{"imports":{"onnxruntime-web/webgpu":"/nm/ort/ort.webgpu.bundle.min.mjs","onnxruntime-web":"/nm/ort/ort.webgpu.bundle.min.mjs","onnxruntime-common":"/nm/common/index.js","@huggingface/transformers":"/nm/tjs/transformers.web.js"}}</script><body>webllm</body>`;
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(req.url.split("?")[0]);
  res.setHeader("Access-Control-Allow-Origin", "*");
  if (u === "/") { res.setHeader("content-type", "text/html"); return res.end(page); }
  for (const [pre, dir] of ROUTES) {
    if (!u.startsWith(pre)) continue;
    const f = resolve(dir, u.slice(pre.length));
    if (!f.startsWith(resolve(dir)) || !existsSync(f) || !statSync(f).isFile()) continue;
    const size = statSync(f).size;
    res.setHeader("content-type", MIME[extname(f)] || "application/octet-stream"); res.setHeader("content-length", size);
    return createReadStream(f).pipe(res);
  }
  res.statusCode = 404; res.end("404");
});
await new Promise((r) => server.listen(Number(process.env.CT0_PORT || 0), "127.0.0.1", r));
const origin = "http://127.0.0.1:" + server.address().port;

const launchArgs = ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--ignore-gpu-blocklist", "--js-flags=--max-old-space-size=8192"];
let ctx, browser;
if (process.env.CT0_PROFILE) { ctx = await chromium.launchPersistentContext(process.env.CT0_PROFILE, { channel: "chrome", headless: process.env.CT0_HEADED !== "1", args: launchArgs }); browser = ctx; }
else { browser = await chromium.launch({ channel: "chrome", headless: process.env.CT0_HEADED !== "1", args: launchArgs }); }
let code = 1;
try {
  const p = await (ctx ? (ctx.pages()[0] || ctx.newPage()) : browser.newPage());
  p.on("pageerror", (e) => console.log("  PAGEERR", e.message));
  p.on("console", (m) => { const t = m.text(); if (process.env.CT0_DEBUG) console.log("  CONSOLE", m.type(), t.slice(0, 300)); else if (m.type() === "error" && !/404/.test(t)) console.log("  CERR", t.slice(0, 200)); });
  p.setDefaultTimeout(0);
  await p.goto(origin + "/");
  console.log("webgpu:", await p.evaluate(async () => !!navigator.gpu && !!(await navigator.gpu.requestAdapter())));
  const res = await p.evaluate(async ({ origin, wlUrl, fmt, picks, fx, heads }) => {
    const out = { timings: {} };
    const { createWebllm, webllmModel } = await import("/dist/webllm.js");
    const { ask } = await import("/dist/ask.js");
    const { loadHead } = await import("/dist/systemone.js");
    const pos = await import("/dist/positions.js");
    const tjs = await import("@huggingface/transformers");
    tjs.env.allowRemoteModels = false; tjs.env.allowLocalModels = true; tjs.env.localModelPath = "/files/"; tjs.env.useBrowserCache = false;
    const tok = await tjs.AutoTokenizer.from_pretrained("tok");
    const webllm = await import(wlUrl);
    const d64 = (s) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u.buffer; };
    const hd = { noulChoice: loadHead(d64(heads.h)), score: loadHead(d64(heads.s)) };
    let t = performance.now();
    const be = await createWebllm(webllm, { modelId: "ct0-q4f16-test", modelUrl: origin + "/m/", libUrl: origin + "/m/model-lib.wasm", tok,
      recoveryUrls: { rows: origin + "/rec/P_q4f16_rows.i32", P: origin + `/rec/P_q4f16_8192.${fmt}`, format: fmt } });
    out.timings.loadMs = Math.round(performance.now() - t);
    out.kind = be.kind; out.hiddenName = be.hiddenName;
    // (1) contrastive decide()
    const model = webllmModel(be.engine, { capture: be.capture });
    out.decide = [];
    for (const [state, q] of [
      ['<input name="password" type="password" autocomplete="on">', "Does this excerpt itself contain a form submitting sensitive data insecurely?"],
      ["<p>Welcome to our documentation. Read the guide below.</p>", "Does this excerpt itself contain a form submitting sensitive data insecurely?"],
      ["<script>document.write(location.hash.slice(1))</script>", "Does this code contain a DOM XSS sink?"],
      ["const total = items.reduce((a, b) => a + b.price, 0);", "Does this code contain a DOM XSS sink?"]]) out.decide.push(await model.decide(state, q));
    // (1b) without capture: error
    try { await webllmModel(be.engine).decide("s", "q"); out.noCapThrows = false; } catch (e) { out.noCapThrows = /logitProcessorRegistry/.test(e.message); }
    // (2) probes
    out.rows = [];
    for (const g of picks) {
      const f = fx[g]; t = performance.now();
      const tk = pos.tokenOffsets(tok, pos.promptFor(f.probe));
      const row = { g, kind: f.probe.kind, idsEq: JSON.stringify(tk.ids) === JSON.stringify(f.ids) };
      const h = await ask(be, tok, hd, f.probe, { method: "head" });
      const l = await ask(be, tok, hd, f.probe, { method: "logits" });
      Object.assign(row, { head: h, logits: l });
      if (f.probe.kind === "choice") {
        const { hiddens } = await be.prefill(tk.ids, { positions: [pos.decidePos(tk.tokens), ...pos.optPositions(pos.promptFor(f.probe), tk.offsets)] });
        const { headScore } = await import("/dist/systemone.js");
        row.scores = hiddens.slice(1).map((k) => headScore(hd.noulChoice, hiddens[0], k));
        row.fin = hiddens.every((x) => Array.from(x).every(Number.isFinite));
      }
      row.ms = Math.round(performance.now() - t); out.rows.push(row);
    }
    // (3) duplicate hidden
    const f0 = fx[picks[0]]; const dd = await be.prefill(f0.ids, { positions: [f0.pos_decide, f0.pos_last, f0.pos_decide] });
    out.dup = { same: dd.hiddens[0].every((v, i) => v === dd.hiddens[2][i]), diff: dd.hiddens[0].some((v, i) => v !== dd.hiddens[1][i]), len: dd.hiddens[0].length, logits: dd.lastLogits.length };
    // (4) generate + abort (engine reused afterwards)
    const prompt = tok.encode("<|im_start|>user\nCount from 1 to 40, separated by commas.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", { add_special_tokens: false }).map(Number);
    const gen = []; for await (const id of be.generate(prompt, { maxNewTokens: 16, temperature: 0 })) gen.push(id);
    out.gen = gen; out.genText = tok.decode(gen, { skip_special_tokens: true });
    const ac = new AbortController(), ab = [];
    for await (const id of be.generate(prompt, { maxNewTokens: 50, temperature: 0, signal: ac.signal })) { ab.push(id); if (ab.length === 3) ac.abort(); }
    out.abortLen = ab.length;
    const again = []; for await (const id of be.generate(prompt, { maxNewTokens: 16, temperature: 0 })) again.push(id);
    out.reuseSame = JSON.stringify(again) === JSON.stringify(gen);
    await be.dispose();
    return out;
  }, { origin, wlUrl: WL_URL.startsWith("/") ? origin + WL_URL : WL_URL, fmt: FMT, picks, fx: picks.reduce((a, g) => (a[g] = fx[g], a), {}), heads: { h: b64(HEAD), s: b64(HEAD_S) } });
  console.log("  timings:", JSON.stringify(res.timings));
  ok(res.kind === "webllm" && res.hiddenName === "recovered-from-logits", `kind/hidden: ${res.kind}/${res.hiddenName}`);
  // (1)
  console.log("  decide():", res.decide.map((x) => x.toFixed(4)).join(" "));
  ok(res.decide.every((x) => x > 0 && x < 1 && x !== 0.5), "decide(): all probabilities strictly in (0,1), none 0/0.5/1");
  ok(new Set(res.decide.map((x) => x.toFixed(6))).size === res.decide.length, "decide(): distinct values (continuous)");
  ok(res.decide[0] > res.decide[1], `decide(): contrast password form (${res.decide[0].toFixed(4)}) > neutral text (${res.decide[1].toFixed(4)})`);
  ok(res.noCapThrows, "decide() without capture throws an explanatory error");
  // (2)
  console.log(`\nparity vs the MLC reference rows (head |Δraw| < ${TOL}):`);
  const dh = [], dl = [], dc = [];
  for (const r of res.rows) {
    const rr = refRow(r.g);
    ok(r.idsEq, `#${r.g} JS tokenizer ids == Python fixture`);
    if (r.kind === "noul") {
      const d = Math.abs(r.head.raw - rr.raw); dh.push(d);
      ok(d < TOL && (r.head.p >= 0.5) === (rr.pred_head === "yes"), `#${r.g} noul head raw=${r.head.raw.toFixed(4)} ref=${rr.raw.toFixed(4)} |Δ|=${d.toFixed(4)} (${r.ms}ms)`);
      const refl = Math.log(rr.p_logits / (1 - rr.p_logits)), e = Math.abs(r.logits.raw - refl); dl.push(e);
      ok(e < TOL, `#${r.g} noul logits L[Yes]-L[No]=${r.logits.raw.toFixed(4)} ref=${refl.toFixed(4)} |Δ|=${e.toFixed(4)}`);
    } else if (r.kind === "score") {
      const d = Math.abs(r.head.raw - rr.raw); dh.push(d);
      ok(d < TOL && r.head.level === rr.level_head, `#${r.g} score head raw=${r.head.raw.toFixed(4)} ref=${rr.raw.toFixed(4)} level ${r.head.level}/${rr.level_head} |Δ|=${d.toFixed(4)} (${r.ms}ms)`);
      ok(r.logits.level === rr.level_logits, `#${r.g} score logits level ${r.logits.level}/${rr.level_logits}`);
    } else {
      const d = Math.max(...r.scores.map((v, i) => Math.abs(v - rr.scores_head[i]))); dc.push(d);
      ok(r.fin && r.head.top === rr.pred_head && d < TOL, `#${r.g} choice head top=${r.head.top}/${rr.pred_head} max|Δscores|=${d.toFixed(4)} (${r.ms}ms)`);
      ok(r.logits.top === rr.pred_logits, `#${r.g} choice logits top=${r.logits.top}/${rr.pred_logits}`);
    }
  }
  const dist = (n, a) => { a.sort((x, y) => x - y); console.log(`  ${n} (n=${a.length}): min=${a[0]?.toFixed(4)} median=${a[a.length >> 1]?.toFixed(4)} max=${a.at(-1)?.toFixed(4)}`); };
  dist("|Δraw| head noul+score", dh); dist("|Δ| logits noul", dl); dist("max|Δscores| choice head", dc);
  ok(res.dup.same && res.dup.diff && res.dup.len === 2560 && res.dup.logits > 240000, `duplicate hiddens equal, distinct positions differ (D=${res.dup.len}, V=${res.dup.logits})`);
  ok(res.gen.length === 16, `generate: ${res.gen.length} ids, text ${JSON.stringify(res.genText)}`);
  ok(res.abortLen === 3 && res.reuseSame, `abort after 3 ids; engine reused (identical generation): abortLen=${res.abortLen}`);
  code = fail ? 1 : 0;
} catch (e) { console.log("  ✗ ERROR:", String(e.stack || e.message).slice(0, 1500)); } finally { await browser.close(); server.close(); }
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(code);
