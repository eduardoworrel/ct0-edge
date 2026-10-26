#!/usr/bin/env node
// bench_figures.mjs — README benchmark figures (SVG) + the text table, generated from bench.json.
// Deterministic: same bench.json -> byte-identical SVGs. No dependencies, no external fonts.
//
//   node scripts/bench_figures.mjs [path/to/bench.json] [outDir]
//   bench.json: argument or $CT0_BENCH_JSON (required); outDir default: docs/bench/
//
// Writes docs/bench/{pick-a-model,accuracy,size-vs-time}.svg and prints the README markdown table on stdout.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const BENCH = process.argv[2] ?? process.env.CT0_BENCH_JSON;
if (!BENCH) { console.error("usage: node scripts/bench_figures.mjs <bench.json> [outDir]  (or set CT0_BENCH_JSON)"); process.exit(2); }
const OUT = resolve(process.argv[3] ?? "docs/bench");
const bench = JSON.parse(readFileSync(BENCH, "utf8"));
const byId = new Map(bench.rows.map((r) => [r.id, r]));
const row = (id) => { const r = byId.get(id); if (!r) throw new Error("bench.json: row not found: " + id); return r; };
const BROWSER_LIMIT_GB = 4;

// ---------------------------------------------------------------- browser timings
// Per-probe timings come from the result files listed in bench.json "sources" (never wall time / probe count: the
// browser runs also timed the 30 sev30 probes, 202 in all).
const srcFile = (suffix) => {
  const hit = Object.keys(bench.sources).find((p) => (suffix instanceof RegExp ? suffix.test(p) : p.endsWith(suffix)));
  if (!hit) throw new Error("bench.json sources: no file ending in " + suffix);
  return JSON.parse(readFileSync(hit, "utf8"));
};
const median = (xs) => { const a = [...xs].sort((x, y) => x - y), m = a.length >> 1; return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2; };
/** Seconds per decision and how it was obtained. */
const TIMES = {
  // ct0-edge decode-only in Chrome: per-probe "ms" of the test-set rows -> median
  "web/decode-q4": { s: median(srcFile(/modes_web_decode_q4_n\d+\.json$/).rows.map((r) => r.ms)) / 1000, how: "median" },
  "web/decode-q4f16": { s: median(srcFile(/modes_web_decode_q4f16_n\d+\.json$/).rows.map((r) => r.ms)) / 1000, how: "median" },
  // transformers.js in Chrome: median per probe, uncontended run
  "tjs-webgpu/canon-q4": { s: srcFile("browser_q4_system1.json").timing_uncontended.median_sec, how: "median" },
  "tjs-webgpu/canon-q4f16": { s: srcFile("browser_q4f16_system1.json").timing_uncontended.median_sec, how: "median" },
};
// WebLLM stores no per-probe time: total run time / probes actually run (test-set rows + severity-set rows) = mean per probe
for (const [q, id] of [["q4f16", "mlc/q4f16_1"], ["q4f32", "mlc/q4f32_1"]]) {
  const d = srcFile(new RegExp(`modes_mlc_${q}_n\\d+\\.json$`));
  TIMES[id] = { s: d.mlc.total_s / (d.rows.length + d.sev_rows.length), how: "mean" };
}
function perDecision(id) { return TIMES[id] ?? null; }

// genai int4 in Chrome (Apple M2 Pro): prefill of a ~70-token probe, 0.43-0.55 s hot. Source: bench.json
// "genai_webgpu.source": the genai WebGPU timing run (browser, Chrome).
// the ct0-edge v0.4 genai backend browser test (test/browser/genai_browser.mjs).
const GENAI_WEBGPU_S = 0.5;

/** Decisions (default reads) in Chrome equal to the same graph on the server: "k/n", or null. */
function agree(spec) {
  if (spec.q === "genai/int4-cuda" && spec.sec === "browser") { const d = bench.genai_webgpu.decisions_vs_cuda; return `${d.identical}/${d.n} (vs genai CUDA int4)`; }
  if (!spec.pq) return null;
  const p = row(spec.q).paired;
  const k = p.noul_agree.logits + p.choice_agree.head + p.score_agree_scorehead.levels;
  return `${k}/${p.noul_agree.n + p.choice_agree.n + p.score_agree_scorehead.n}`;
}

/** Each displayed row: what it is, where its quality numbers come from, how it is timed. */
const SPEC = [
  // --- browser (Chrome WebGPU, Apple M2 Pro)
  { sec: "browser", label: "genai int4", runtime: "onnxruntime-web WebGPU", q: "genai/int4-cuda", size: "genai/int4-cuda", time: GENAI_WEBGPU_S, browser: "yes", tag: "DEFAULT", dagger: true },
  { sec: "browser", label: "decode-only q4", runtime: "onnxruntime-web WebGPU", q: "web/decode-q4", pq: "decode/q4", size: "decode/q4", time: "web/decode-q4", browser: "yes", tag: "FALLBACK" },
  { sec: "browser", label: "decode-only q4f16", runtime: "onnxruntime-web WebGPU", q: "web/decode-q4f16", pq: "decode/q4f16", size: "decode/q4f16", time: "web/decode-q4f16", browser: "yes" },
  { sec: "browser", label: "optimum/Scan q4", runtime: "transformers.js WebGPU", q: "tjs-webgpu/canon-q4", pq: "canon/q4", size: "canon/q4", time: "tjs-webgpu/canon-q4", browser: "yes" },
  { sec: "browser", label: "optimum/Scan q4f16", runtime: "transformers.js WebGPU", q: "tjs-webgpu/canon-q4f16", pq: "canon/q4f16", size: "canon/q4f16", time: "tjs-webgpu/canon-q4f16", browser: "yes" },
  // WebLLM: head reads through the hidden state recovered from the logits (recovery/ in each MLC repo)
  { sec: "browser", label: "MLC q4f16_1", runtime: "WebLLM", q: "mlc/q4f16_1", size: "mlc/q4f16_1", time: "mlc/q4f16_1", browser: "yes" },
  { sec: "browser", label: "MLC q4f32_1", runtime: "WebLLM", q: "mlc/q4f32_1", size: "mlc/q4f32_1", time: "mlc/q4f32_1", browser: "yes" },
  { sec: "browser", label: "decode-only q4", runtime: "onnxruntime-web WASM", q: null, size: "decode/q4", time: null, browser: "yes", tag: "NO WEBGPU", unmeasured: true },
  // --- Node / server (onnxruntime CUDA)
  ...["fp16-full", "q8", "uint8", "q4", "q4f16", "bnb4-nf4"].map((d) => ({ sec: "server", label: "decode-only " + d.replace("-full", "").replace("-nf4", ""), runtime: "onnxruntime", q: "decode/" + d, size: "decode/" + d })),
  ...["fp16", "q8", "uint8", "q4", "q4f16", "bnb4"].map((d) => ({ sec: "server", label: "optimum/Scan " + d, runtime: "transformers.js / onnxruntime", q: "canon/" + d, size: "canon/" + d })),
  { sec: "server", label: "genai fp16", runtime: "onnxruntime-genai", q: "genai/fp16-cuda", size: "genai/fp16-cuda" },
  { sec: "server", label: "genai int4", runtime: "onnxruntime-genai", q: "genai/int4-cuda", size: "genai/int4-cuda" },
];

// System-1 reference: the original model as released (PyTorch bf16); every row carries `paired_orig` (McNemar vs it)
const ORIG = "original/pytorch-bf16";
// default read per mode (ct0-edge ask() defaults): noul logits, choice head, score head_score.bin (exact level)
const READ_DEFAULT = {
  noul: { get: (m) => m.noul.logits, pair: "noul_logits" },
  choice: { get: (m) => m.choice.head, pair: "choice_head" },
  score: { get: (m) => m.score50.scorehead.exact, pair: "score_exact_scorehead" },
};
// logits read for every mode (WebLLM default), compared with the original model under the same read
const READ_LOGITS = {
  noul: READ_DEFAULT.noul,
  choice: { get: (m) => m.choice.logits, pair: "choice_logits" },
  score: { get: (m) => m.score50.logits.exact, pair: "score_exact_logits" },
};
const readOf = (spec) => (spec.read === "logits" ? READ_LOGITS : READ_DEFAULT);

/** green: within 2 answers of the original model (or better) and not significant; amber: 3+ below, not significant; red: p < 0.05 loss. */
function grade(spec, mode) {
  if (spec.unmeasured) return { status: "na", k: null, n: null, acc: null };
  const q = row(spec.q), qp = row(spec.pq ?? spec.q);
  const READ = readOf(spec);
  const v = READ[mode].get(q.metrics);
  const ref = READ[mode].get(row(ORIG).metrics);
  const diff = v.k - ref.k;
  const p = qp.paired_orig?.[READ[mode].pair]?.p ?? 1;
  const status = diff < 0 && p < 0.05 ? "red" : diff <= -3 ? "amber" : "green";
  return { ...v, status, diff, ref: ref.acc };
}

function browserCell(spec, gb) {
  if (spec.browser === "yes") return "yes";
  if (gb > BROWSER_LIMIT_GB) return "no";
  return SPEC.some((s) => s.sec === "browser" && s.q && (s.pq ?? s.q) === spec.q) || spec.q === "genai/int4-cuda" ? "above" : "untested";
}

const ROWS = SPEC.map((s) => {
  const gb = row(s.size).bytes / 1e9;
  const tm = typeof s.time === "number" ? { s: s.time, how: "prefill" } : s.time ? perDecision(s.time) : null;
  const t = tm?.s ?? null, tHow = tm?.how ?? null;
  return { ...s, gb, t, tHow, agree: agree(s), browserCell: browserCell(s, gb), noul: grade(s, "noul"), choice: grade(s, "choice"), score: grade(s, "score") };
});

// ---------------------------------------------------------------- runtime logos
// docs/bench/logos (sources and trademark notes: docs/bench/logos/SOURCES.md), inlined once per figure as <symbol>s:
// a README shows the figures through <img>, which loads no external files.
const LOGO_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "../docs/bench/logos");
const LOGO_FILES = { ort: "ort_mark_64.png", onnx: "onnx_mark.svg", hf: "hf_logo.svg", mlc: "mlc_mark.svg" };
const LOGO_NAMES = { ort: "ONNX Runtime GenAI", onnx: "ONNX Runtime (decode-only)", hf: "transformers.js", mlc: "WebLLM (MLC)" };
function logoSymbol(key) {
  const file = join(LOGO_DIR, LOGO_FILES[key]);
  if (file.endsWith(".png")) {
    const b64 = readFileSync(file).toString("base64");
    return `<symbol id="lg-${key}" viewBox="0 0 64 64"><image href="data:image/png;base64,${b64}" width="64" height="64"/></symbol>`;
  }
  const src = readFileSync(file, "utf8").replace(/<\?xml[^>]*>/, "").trim();
  const open = /<svg\b[^>]*>/.exec(src)[0];
  const attr = (n) => (new RegExp(`\\s${n}="([^"]*)"`).exec(open) ?? [])[1];
  const viewBox = attr("viewBox") ?? `0 0 ${attr("width")} ${attr("height")}`;
  const inner = src.slice(src.indexOf(open) + open.length, src.lastIndexOf("</svg>")).replace(/\s*\n\s*/g, " ").trim();
  const fill = attr("fill");
  return `<symbol id="lg-${key}" viewBox="${viewBox}">${fill ? `<g fill="${fill}">${inner}</g>` : inner}</symbol>`;
}
const LOGO_DEFS = `<defs>${Object.keys(LOGO_FILES).map(logoSymbol).join("")}</defs>`;
/** Runtime (logo) of a displayed row. */
const logoOf = (r) => (r.label.startsWith("genai") ? "ort" : r.label.startsWith("decode-only") ? "onnx" : r.label.startsWith("MLC") ? "mlc" : "hf");
/** Logo centred at (cx, cy), `size` px, on a white circular backplate (optional ring). */
function logo(key, cx, cy, size, o = {}) {
  const r = size / 2 + (o.pad ?? 4);
  const plate = o.plate === false ? "" : `<circle cx="${f(cx)}" cy="${f(cy)}" r="${f(r)}" fill="#ffffff" stroke="${o.ring ?? C.rule}" stroke-width="${o.ringW ?? 1.5}"/>`;
  const title = o.title ? `<title>${esc(o.title)}</title>` : "";
  return `<g${o.opacity ? ` opacity="${o.opacity}"` : ""}>${title}${plate}<use href="#lg-${key}" x="${f(cx - size / 2)}" y="${f(cy - size / 2)}" width="${size}" height="${size}"/></g>`;
}
/** One-line logo legend starting at (x, y); returns the markup. */
function logoLegend(x, y) {
  const out = [];
  for (const k of ["ort", "onnx", "hf", "mlc"]) {
    out.push(logo(k, x + 13, y - 6, 20, { pad: 3 }));
    out.push(txt(x + 32, y, LOGO_NAMES[k], { size: 17, fill: C.ink2 }));
    x += 32 + LOGO_NAMES[k].length * 8.6 + 30;
  }
  return out.join("\n");
}

// ---------------------------------------------------------------- svg helpers
const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";
const C = {
  surface: "#fcfcfb", ink: "#0b0b0b", ink2: "#52514e", muted: "#8a8984", rule: "#d9d8d4", band: "#f2f1ee",
  accent: "#2a78d6", s1: "#2a78d6", s2: "#eb6834", s3: "#1baf7a",
  good: "#0ca30c", warn: "#fab219", bad: "#d03b3b",
  tGood: "#dbf0db", tWarn: "#fdedc8", tBad: "#f5d5d5", tGrey: "#ebeae6", tRef: "#e3edf9",
};
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const f = (x, d = 1) => Number(x.toFixed(d)).toString();
const txt = (x, y, s, o = {}) => `<text x="${f(x)}" y="${f(y)}" font-size="${o.size ?? 20}"${o.weight ? ` font-weight="${o.weight}"` : ""} fill="${o.fill ?? C.ink}"${o.anchor ? ` text-anchor="${o.anchor}"` : ""}>${esc(s)}</text>`;
const rect = (x, y, w, h, fill, o = {}) => `<rect x="${f(x)}" y="${f(y)}" width="${f(w)}" height="${f(h)}" fill="${fill}"${o.rx ? ` rx="${o.rx}"` : ""}${o.stroke ? ` stroke="${o.stroke}" stroke-width="${o.sw ?? 2}"` : ""}/>`;
const line = (x1, y1, x2, y2, stroke, w = 1, extra = "") => `<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}" stroke="${stroke}" stroke-width="${w}"${extra}/>`;
const svg = (w, h, title, body) => `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" font-family="${FONT}" role="img" aria-label="${esc(title)}">
<title>${esc(title)}</title>
${LOGO_DEFS}
${rect(0, 0, w, h, C.surface, { rx: 12 })}
${body}
</svg>
`;
const gbStr = (gb) => f(gb, 2) + " GB";
const tStr = (t, how) => (t == null ? "—" : (t < 1 ? "~" + f(t, 1) + " s" : f(t, 1) + " s") + (how === "mean" ? " (mean)" : how === "prefill" ? " (prefill)" : ""));

// ---------------------------------------------------------------- (a) pick a model
function pickAModel() {
  const W = 1600, X = { name: 32, size: 640, br: 780, time: 930, noul: 1100, choice: 1265, score: 1430 }, CW = 145;
  const RH = 46, top = 150;
  const out = [];
  out.push(txt(32, 52, "Pick a model", { size: 34, weight: 700 }));
  out.push(txt(32, 88, "System-1 hits on the CT0 decision test set: 172 security questions with known answers (65 yes/no, 57 multiple choice, 50 severity 1–5). Color: vs the original model.", { size: 19, fill: C.ink2 }));
  const hy = top - 18;
  [["Variant · runtime", X.name, null], ["Size", X.size, null], ["Browser", X.br, null], ["Time / question", X.time, null],
    ["yes/no (65)", X.noul, null], ["choice (57)", X.choice, null], ["severity (50)", X.score, null]].forEach(([h, x]) => out.push(txt(x, hy, h, { size: 19, weight: 700, fill: C.ink2 })));
  out.push(line(24, top - 6, W - 24, top - 6, C.rule, 2));
  let y = top;
  let sec = null;
  for (const r of ROWS) {
    if (r.sec !== sec) {
      sec = r.sec;
      out.push(rect(24, y + 4, W - 48, 38, C.band, { rx: 6 }));
      out.push(txt(36, y + 30, sec === "browser" ? "In the browser — Chrome WebGPU, Apple M2 Pro" : "Node / server — onnxruntime (CUDA run)", { size: 20, weight: 700 }));
      y += 46;
    }
    const cy = y + RH / 2 + 7;
    if (r.tag === "DEFAULT") out.push(rect(24, y + 2, W - 48, RH - 4, "#eef4fc", { rx: 8, stroke: C.accent, sw: 3 }));
    const label = r.label + (r.dagger ? " †" : "");
    out.push(logo(logoOf(r), X.name + 14, y + RH / 2, 24, { title: LOGO_NAMES[logoOf(r)] }));
    out.push(txt(X.name + 38, cy - 1, label, { size: 21, weight: r.tag ? 700 : 400 }));
    let nx = X.name + 38 + label.length * (r.tag ? 12.2 : 11.4) + 12;
    if (r.tag) {
      const pw = r.tag.length * 11 + 18;
      out.push(rect(nx, y + 10, pw, 26, r.tag === "DEFAULT" ? C.accent : "#52514e", { rx: 13 }));
      out.push(txt(nx + pw / 2, y + 29, r.tag, { size: 15, weight: 700, fill: "#ffffff", anchor: "middle" }));
      nx += pw + 12;
    }
    out.push(txt(nx, cy - 1, r.runtime, { size: 17, fill: C.ink2 }));
    out.push(txt(X.size, cy, gbStr(r.gb), { size: 20 }));
    const b = { yes: ["✓ yes", C.tGood], no: ["✗ > 4 GB", C.tGrey], above: ["✓ (above)", C.tGrey], untested: ["– untested", C.tGrey] }[r.browserCell];
    out.push(rect(X.br - 8, y + 5, 128, RH - 10, b[1], { rx: 6 }));
    out.push(txt(X.br, cy, b[0], { size: 19 }));
    out.push(txt(X.time, cy, r.unmeasured ? "slow, not measured" : tStr(r.t, r.tHow), { size: r.unmeasured ? 17 : 20, weight: r.tag === "DEFAULT" ? 700 : 400 }));
    for (const m of ["noul", "choice", "score"]) {
      const g = r[m];
      const [fill, glyph] = { ref: [C.tRef, "ref"], green: [C.tGood, "✓"], amber: [C.tWarn, "▼"], red: [C.tBad, "✗"], na: [C.tGrey, "–"] }[g.status];
      out.push(rect(X[m] - 8, y + 5, CW, RH - 10, fill, { rx: 6 }));
      out.push(txt(X[m] + 4, cy, g.status === "na" ? "– not measured" : `${glyph} ${g.k}/${g.n}`, { size: g.status === "na" ? 17 : 20 }));
    }
    out.push(line(24, y + RH, W - 24, y + RH, C.rule, 1));
    y += RH;
  }
  y += 34;
  const o = row(ORIG).metrics;
  const leg = [[C.tGood, "✓", "within 2 answers of the original model"], [C.tWarn, "▼", "3+ below it, not significant"], [C.tBad, "✗", "significantly below it (McNemar p < 0.05)"]];
  let lx = 32;
  for (const [fill, g, s] of leg) {
    out.push(rect(lx, y - 22, 44, 30, fill, { rx: 6 }));
    out.push(txt(lx + 22, y - 1, g, { size: 18, anchor: "middle" }));
    out.push(txt(lx + 54, y, s, { size: 18, fill: C.ink2 }));
    lx += 54 + s.length * 9.4 + 34;
  }
  y += 34;
  out.push(txt(32, y, "Reads: yes/no from the logits, multiple choice from the head, severity from head_score.bin (exact level); MLC: heads on the hidden state recovered from the logits.", { size: 17, fill: C.ink2 }));
  y += 26;
  out.push(txt(32, y, `Original model (PyTorch bf16, as released): ${READ_DEFAULT.noul.get(o).k}/65 · ${READ_DEFAULT.choice.get(o).k}/57 · ${READ_DEFAULT.score.get(o).k}/50. Time: median per question in Chrome (MLC: mean; genai: hot prefill).`, { size: 17, fill: C.ink2 }));
  y += 26;
  out.push(txt(32, y, "† accuracy of the same int4 build on onnxruntime-genai CUDA; in Chrome its decisions match it on every question checked (12/12).", { size: 17, fill: C.ink2 }));
  return svg(W, y + 26, "Pick a model: CT0 variants, size, browser support, time per decision and System-1 accuracy", out.join("\n"));
}

// ---------------------------------------------------------------- (b) accuracy, grouped bars
function accuracy() {
  const W = 1480, L = 470, R = W - 60, top = 150, bar = 11, gap = 2, group = 3 * bar + 2 * gap, GH = group + 18;
  const x = (v) => L + v * (R - L);
  const out = [];
  out.push(txt(32, 52, "System-1 accuracy per variant", { size: 34, weight: 700 }));
  out.push(txt(32, 88, "Share of the CT0 decision test set answered right (172 security questions with known answers), default read per type. Black tick: the original model.", { size: 19, fill: C.ink2 }));
  const modes = [["noul", C.s1, "yes/no (65)"], ["choice", C.s2, "multiple choice (57)"], ["score", C.s3, "severity, exact level (50)"]];
  let lx = L;
  for (const [, col, s] of modes) { out.push(rect(lx, 108, 18, 18, col, { rx: 4 })); out.push(txt(lx + 26, 124, s, { size: 19 })); lx += 26 + s.length * 10 + 36; }
  out.push(line(lx, 106, lx, 128, C.ink, 3)); out.push(txt(lx + 10, 124, "original model", { size: 19 }));
  const AR = ROWS.filter((r) => !r.unmeasured);
  const Hc = top + AR.length * GH + 2 * 40 + 60, H = Hc + 44;
  for (let v = 0; v <= 1.0001; v += 0.25) {
    out.push(line(x(v), top - 6, x(v), Hc - 64, C.rule, 1));
    out.push(txt(x(v), Hc - 38, f(v * 100, 0) + "%", { size: 18, fill: C.ink2, anchor: "middle" }));
  }
  let y = top, sec = null;
  for (const r of AR) {
    if (r.sec !== sec) { sec = r.sec; out.push(txt(32, y + 26, sec === "browser" ? "In the browser" : "Node / server", { size: 20, weight: 700, fill: C.ink2 })); y += 40; }
    const refM = row(ORIG).metrics;
    if (r.tag === "DEFAULT") out.push(rect(24, y - 4, W - 48, group + 8, "#eef4fc", { rx: 8 }));
    out.push(txt(L - 46, y + group / 2 + 7, r.label + (r.tag ? "  (" + r.tag.toLowerCase() + ")" : ""), { size: 19, weight: r.tag ? 700 : 400, anchor: "end" }));
    out.push(logo(logoOf(r), L - 22, y + group / 2, 22, { pad: 3, title: LOGO_NAMES[logoOf(r)] }));
    modes.forEach(([m, col], i) => {
      const v = r[m].acc, by = y + i * (bar + gap);
      out.push(`<path d="M${f(L)},${f(by)} H${f(x(v) - 4)} a4,4 0 0 1 4,4 V${f(by + bar - 4)} a4,4 0 0 1 -4,4 H${f(L)} Z" fill="${col}"><title>${esc(`${r.label} · ${m}: ${r[m].k}/${r[m].n} (${f(v * 100, 0)}%)`)}</title></path>`);
      const rv = readOf(r)[m].get(refM).acc;
      out.push(line(x(rv), by - 2, x(rv), by + bar + 2, C.ink, 3));
    });
    y += GH;
  }
  out.push(line(L, top - 6, L, Hc - 64, C.ink2, 2));
  out.push(logoLegend(32, H - 22));
  return svg(W, H, "System-1 accuracy per variant for yes/no, multiple-choice and severity questions, with ticks for the original model", out.join("\n"));
}

// ---------------------------------------------------------------- (c) size vs time per decision
function sizeVsTime() {
  const W = 1480, H = 900, L = 120, R = W - 240, T = 130, B = H - 230;
  const X0 = 2.2, X1 = 5.0, Y0 = Math.log10(0.2), Y1 = Math.log10(60);
  const x = (gb) => L + ((gb - X0) / (X1 - X0)) * (R - L);
  const y = (s) => B - ((Math.log10(s) - Y0) / (Y1 - Y0)) * (B - T);
  const out = [];
  out.push(txt(32, 52, "Size vs time per decision (browser)", { size: 34, weight: 700 }));
  out.push(txt(32, 88, "Chrome WebGPU on an Apple M2 Pro, median per question of the decision test set (MLC: mean; log scale). The browser fits graphs up to about 4 GB.", { size: 19, fill: C.ink2 }));
  // limit zone
  out.push(rect(x(BROWSER_LIMIT_GB), T, R - x(BROWSER_LIMIT_GB), B - T, C.band));
  out.push(line(x(BROWSER_LIMIT_GB), T, x(BROWSER_LIMIT_GB), B + 70, C.bad, 3));
  out.push(txt(x(BROWSER_LIMIT_GB) + 12, T + 28, "~4 GB browser limit", { size: 20, weight: 700, fill: C.bad }));
  out.push(txt(x(BROWSER_LIMIT_GB) + 12, T + 54, "too big for the browser", { size: 18, fill: C.ink2 }));
  for (const s of [0.5, 1, 2, 5, 10, 20, 50]) { out.push(line(L, y(s), x(BROWSER_LIMIT_GB), y(s), C.rule, 1)); out.push(txt(L - 12, y(s) + 6, s + " s", { size: 18, fill: C.ink2, anchor: "end" })); }
  for (let g = 2.5; g <= X1; g += 0.5) out.push(txt(x(g), B + 100, f(g, 1) + " GB", { size: 18, fill: C.ink2, anchor: "middle" }));
  out.push(line(L, T, L, B, C.ink2, 2)); out.push(line(L, B, R, B, C.ink2, 2));
  out.push(txt((L + R) / 2, B + 138, "download size", { size: 19, fill: C.ink2, anchor: "middle" }));
  // timed browser points; labels to the right, placed in fixed slots (no collisions)
  const pts = ROWS.filter((r) => r.sec === "browser" && r.t != null).map((r) => ({ r, px: x(r.gb), py: y(r.t) }));
  const slots = [...pts].sort((a, b) => a.py - b.py);
  const lx = Math.max(...pts.map((p) => p.px)) + 40;
  let prev = -1e9;
  slots.forEach((p) => { p.ly = Math.max(p.py, prev + 40); prev = p.ly; });
  // re-centre the stack if it ran past the bottom
  const over = Math.max(0, slots.at(-1).ly - (B - 20)); slots.forEach((p) => { p.ly -= over; });
  for (const p of slots) {
    const def = p.r.tag === "DEFAULT";
    out.push(`<path d="M${f(p.px + 18)},${f(p.py)} L${f(lx - 10)},${f(p.ly - 6)}" stroke="${C.muted}" stroke-width="1.5" fill="none"/>`);
    out.push(logo(logoOf(p.r), p.px, p.py, def ? 28 : 24, { ring: def ? C.accent : C.rule, ringW: def ? 3 : 1.5,
      title: `${p.r.label} · ${p.r.runtime}: ${gbStr(p.r.gb)}, ${tStr(p.r.t, p.r.tHow)}` }));
    out.push(txt(lx, p.ly, `${p.r.label} · ${tStr(p.r.t, p.r.tHow)}${def ? "  (default)" : ""}`, { size: 19, weight: def ? 700 : 400, fill: def ? C.accent : C.ink }));
  }
  // server-only variants: strip under the axis
  const sy = B + 48;
  out.push(txt(L - 12, sy + 6, "server", { size: 17, fill: C.ink2, anchor: "end" }));
  const server = ROWS.filter((r) => r.sec === "server" && r.browserCell !== "above");
  const beyond = server.filter((r) => r.gb > X1);
  for (const r of server.filter((r) => r.gb <= X1)) out.push(logo(logoOf(r), x(r.gb), sy, 18, { pad: 3, opacity: 0.6, title: `${r.label}: ${gbStr(r.gb)} (not timed in the browser)` }));
  out.push(txt(R + 20, sy + 6, `→ ${beyond.length} more at ${f(Math.min(...beyond.map((r) => r.gb)), 1)}–${f(Math.max(...beyond.map((r) => r.gb)), 1)} GB`, { size: 17, fill: C.ink2 }));
  out.push(logoLegend(32, H - 58));
  out.push(txt(32, H - 22, "Each point is the runtime's logo. Faded logos under the axis: variants measured on the server only (no browser time).", { size: 17, fill: C.ink2 }));
  return svg(W, H, "Download size versus time per decision in the browser, with the 4 GB browser limit", out.join("\n"));
}

// ---------------------------------------------------------------- markdown table (text fallback)
function table() {
  const dot = (g) => (g.status === "na" ? "not measured" : ({ ref: "⚪", green: "🟢", amber: "🟡", red: "🔴" }[g.status]) + ` ${g.k}/${g.n}`);
  const br = { yes: "✅", no: "❌ (> 4 GB)", above: "✅ (above)", untested: "–" };
  const lines = ["| variant | runtime | size | browser | median time / question | yes/no (65) | choice (57) | severity (50) | Chrome decisions = server |", "|---|---|---|---|---|---|---|---|---|"];
  for (const r of ROWS) {
    const name = r.tag === "DEFAULT" ? "**" + r.label + "** (default)" : r.tag === "FALLBACK" ? r.label + " (fallback)" : r.tag === "NO WEBGPU" ? r.label + " (fallback, no WebGPU)" : r.read === "logits" ? r.label + " (logits read)" : r.label;
    const time = r.unmeasured ? "slow, not measured" : tStr(r.t, r.tHow);
    lines.push(`| ${name} | ${r.runtime} | ${f(r.gb, 2)} GB | ${br[r.browserCell]} | ${time} | ${dot(r.noul)} | ${dot(r.choice)} | ${dot(r.score)} | ${r.agree ?? "—"} |`);
  }
  return lines.join("\n");
}

mkdirSync(OUT, { recursive: true });
const files = { "pick-a-model.svg": pickAModel(), "accuracy.svg": accuracy(), "size-vs-time.svg": sizeVsTime() };
for (const [n, s] of Object.entries(files)) writeFileSync(join(OUT, n), s);
console.log(table());
console.error(`wrote ${Object.keys(files).map((n) => join(OUT, n)).join(", ")} from ${BENCH}`);
