// test_webllm_mock.mjs — WebLLM/MLC backend and webllmModel adapter with a FAKE engine/tokenizer (no GPU, no WebLLM).
import { logitCapture, recoverHidden, webllmBackend, webllmModel } from "../dist/webllm.js";
import { YES_ID, NO_ID } from "../dist/ask.js";
let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };
const rejects = async (f) => { try { await f(); return false; } catch { return true; } };

const V = 9200, D = 4, N = 24;
// fake "tied" embedding: logits = E·h(prompt)
let seed = 7; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
const E = Float32Array.from({ length: V * D }, rnd);
const rows = Int32Array.from({ length: N }, (_, i) => 100 + i * 13);
// P = (EnᵀEn)⁻¹Eₙᵀ  [D x N] (row-major, D rows)
function solveP() {
  const A = Array.from({ length: D }, () => new Float64Array(2 * D)); // [EᵀE | I]
  for (let a = 0; a < D; a++) { for (let b = 0; b < D; b++) { let s = 0; for (const r of rows) s += E[r * D + a] * E[r * D + b]; A[a][b] = s; } A[a][D + a] = 1; }
  for (let c = 0; c < D; c++) { let p = c; for (let r = c + 1; r < D; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r; [A[c], A[p]] = [A[p], A[c]];
    const d = A[c][c]; for (let j = 0; j < 2 * D; j++) A[c][j] /= d;
    for (let r = 0; r < D; r++) if (r !== c) { const f = A[r][c]; for (let j = 0; j < 2 * D; j++) A[r][j] -= f * A[c][j]; } }
  const P = new Float32Array(D * N);
  for (let d = 0; d < D; d++) for (let n = 0; n < N; n++) { let s = 0; for (let a = 0; a < D; a++) s += A[d][D + a] * E[rows[n] * D + a]; P[d * N + n] = s; }
  return P;
}
const P = solveP();
const hOf = (text) => { let h = 0; for (const c of text) h = (h * 31 + c.charCodeAt(0)) >>> 0; const s0 = seed; seed = h; const v = Float32Array.from({ length: D }, rnd); seed = s0; return v; };
const logitsOf = (text) => { const h = hOf(text), L = new Float32Array(V); for (let v = 0; v < V; v++) { let s = 0; for (let d = 0; d < D; d++) s += E[v * D + d] * h[d]; L[v] = s; } return L; };

const tok = { decode: (ids) => ids.map((i) => `<${i}>`).join("") };
function mkEngine(cap, over = {}) {
  const e = { calls: [], resets: 0, interrupts: 0, running: 0, overlap: false,
    async resetChat() { e.resets++; },
    async interruptGenerate() { e.interrupts++; e.stop = true; },
    completions: { async create(req) {
      if (e.running++) e.overlap = true;
      await new Promise((r) => setTimeout(r, 1));
      e.calls.push(req);
      try {
        if (req.stream) return mkStream(e, cap, req);
        const L = over.logits ? over.logits(req.prompt) : logitsOf(req.prompt);
        cap.processor.processLogits(L);
        const n = (req.prompt.match(/</g) || []).length;
        return { usage: { prompt_tokens: over.promptTokens ? over.promptTokens(n) : n }, choices: [{ text: "x", logprobs: over.logprobs?.(req) }] };
      } finally { e.running--; }
    } } };
  return e;
}
function mkStream(e, cap, req) {
  e.stop = false; let i = 0; const toks = [11, 12, 13, 14, 15, 248046];
  return { [Symbol.asyncIterator]() { return this; },
    async next() { await new Promise((r) => setTimeout(r, 1));
      if (e.stop || i >= toks.length || i >= req.max_tokens) return { done: true, value: undefined };
      const t = toks[i++]; cap.processor.processSampledToken(t); return { done: false, value: { choices: [{ text: "t" + t }] } }; },
    async return() { e.streamReturned = true; return { done: true, value: undefined }; } };
}

// ---- recoverHidden ----
{
  const h = hOf("abc"), L = logitsOf("abc");
  const r = recoverHidden({ rows, P }, L);
  ok(r.length === D && Math.max(...Array.from(r, (v, i) => Math.abs(v - h[i]))) < 1e-4, "recoverHidden: h = P·logits[rows] (exact least squares for logits = E·h)");
}

// ---- prefill ----
{
  const cap = logitCapture(), eng = mkEngine(cap);
  const be = webllmBackend(eng, cap, { tok, recovery: { rows, P } });
  ok(be.kind === "webllm" && be.hiddenName === "recovered-from-logits", "kind webllm + hiddenName");
  const ids = [3, 4, 5, 6, 7, 8];
  const r = await be.prefill(ids, { positions: [2, 0, 2, 5] });
  const prompts = eng.calls.map((c) => c.prompt);
  ok(prompts.length === 3 && new Set(prompts).size === 3, "one prefill per distinct position (0, 2) + the full prompt: " + prompts.join(" | "));
  ok(prompts.includes("<3><4><5>") && prompts.includes("<3>") && prompts.includes("<3><4><5><6><7><8>"), "prompts truncated at ids[0..p]");
  ok(eng.calls.every((c) => c.max_tokens === 1 && c.temperature === 0) && eng.resets === 3, "max_tokens 1, T=0, resetChat before each run");
  ok(r.hiddens.length === 4 && r.hiddens[0] === r.hiddens[2] || r.hiddens[0].every((v, i) => v === r.hiddens[2][i]), "duplicate position returns a duplicate hidden");
  const want = (txt) => Array.from(hOf(txt));
  const err = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
  ok(err(Array.from(r.hiddens[0]), want("<3><4><5>")) < 1e-4 && err(Array.from(r.hiddens[1]), want("<3>")) < 1e-4 && err(Array.from(r.hiddens[3]), want("<3><4><5><6><7><8>")) < 1e-4, "recovered hiddens match hiddens[i] <-> positions[i]");
  const Lf = logitsOf("<3><4><5><6><7><8>");
  ok(r.lastLogits.length === V && r.lastLogits[YES_ID] === Lf[YES_ID] && r.lastLogits[5] === Lf[5], "lastLogits = logits of the full prompt (copy)");
  const e0 = eng.calls.length;
  const r2 = await be.prefill(ids, { positions: [] });
  ok(eng.calls.length === e0 + 1 && r2.hiddens.length === 0 && r2.lastLogits[7] === Lf[7], "positions [] => a single run (logits only)");
  ok(await rejects(() => be.prefill(ids, { positions: [6] })) && await rejects(() => be.prefill(ids, { positions: [-1] })) && await rejects(() => be.prefill([], { positions: [] })), "out-of-range position / empty ids reject");
  const arr = await Promise.all([be.prefill(ids, { positions: [1] }), be.prefill(ids, { positions: [3] })]);
  ok(!eng.overlap && arr.length === 2, "runs serialized (no concurrent runs on the engine)");
}
{
  const cap = logitCapture(), eng = mkEngine(cap, { promptTokens: (n) => n + 1 });
  const be = webllmBackend(eng, cap, { tok, recovery: { rows, P } });
  ok(await rejects(() => be.prefill([3, 4, 5], { positions: [1] })), "prompt_tokens != position+1 (different re-tokenization) rejects");
  const cap2 = logitCapture(), eng2 = mkEngine(cap2), be2 = webllmBackend(eng2, cap2, { tok });
  ok(be2.hiddenName === null && await rejects(() => be2.prefill([3, 4, 5], { positions: [1] })), "no recovery: hiddenName null and positions != [] rejects");
  ok((await be2.prefill([3, 4, 5], { positions: [] })).lastLogits.length === V, "no recovery: logits work");
}

// ---- generate ----
{
  const cap = logitCapture(), eng = mkEngine(cap), be = webllmBackend(eng, cap, { tok });
  const got = []; for await (const id of be.generate([3, 4], { maxNewTokens: 10, temperature: 0 })) got.push(id);
  ok(got.join() === "11,12,13,14,15", "generate emits the sampled ids (via processSampledToken), EOS not emitted: " + got);
  ok(eng.calls.at(-1).prompt === "<3><4>" && eng.calls.at(-1).stream === true && eng.calls.at(-1).max_tokens === 10 && eng.calls.at(-1).temperature === 0, "generate: raw prompt, stream, max_tokens, temperature");
  const ac = new AbortController(), ab = [];
  for await (const id of be.generate([3, 4], { maxNewTokens: 10, signal: ac.signal })) { ab.push(id); if (ab.length === 2) ac.abort(); }
  ok(ab.join() === "11,12" && eng.interrupts >= 1 && eng.streamReturned, "abort: stops after 2 ids, interrupts the engine and closes the stream");
  const early = []; for await (const id of be.generate([3, 4], { maxNewTokens: 10 })) { early.push(id); break; }
  ok(early.length === 1 && eng.running === 0, "early break: engine idle when the generator ends");
  const g = await (async () => { const out = []; for await (const id of be.generate([3, 4], { maxNewTokens: 3 })) out.push(id); return out; })();
  ok(g.join() === "11,12,13", "maxNewTokens respected; backend reusable after abort");
  const ac2 = new AbortController(); ac2.abort(); const none = []; for await (const id of be.generate([3, 4], { signal: ac2.signal })) none.push(id);
  ok(none.length === 0, "signal already aborted: nothing emitted");
}

// ---- dispose ----
{
  const cap = logitCapture(), eng = mkEngine(cap); eng.unload = async () => { eng.unloaded = true; };
  const be = webllmBackend(eng, cap, { tok, recovery: { rows, P } });
  await be.prefill([3, 4], { positions: [0] });
  await be.dispose();
  ok(eng.unloaded, "dispose unloads the engine");
  ok(await rejects(() => be.prefill([3, 4], { positions: [] })) && await rejects(async () => { for await (const _ of be.generate([3, 4], {})) break; }), "prefill/generate after dispose reject");
  await be.dispose(); ok(true, "idempotent dispose");
}

// ---- decide (webllmModel) ----
{
  const cap = logitCapture();
  const mk = (dy) => (prompt) => { const L = new Float32Array(V); L[YES_ID] = dy(prompt); L[NO_ID] = 0; return L; };
  const eng = mkEngine(cap, { logits: mk((p) => (p.includes("EVIL") ? 3.2 : -2.5)) });
  const m = webllmModel(eng, { capture: cap });
  const pe = await m.decide("EVIL state", "q?"), pb = await m.decide("ok state", "q?");
  ok(pe > 0 && pe < 1 && pb > 0 && pb < 1 && Math.abs(pe - 1 / (1 + Math.exp(-3.2))) < 1e-6 && Math.abs(pb - 1 / (1 + Math.exp(2.5))) < 1e-6, `decide: continuous sigmoid(L[Yes]-L[No]) (${pe.toFixed(4)}, ${pb.toFixed(4)}), not 0/0.5/1`);
  ok(pe !== 1 && pe !== 0 && pe !== 0.5 && pb !== 0.5 && pe > pb, "decide: correct contrast (evil > ok)");
  const sat = webllmModel(mkEngine(cap, { logits: mk(() => 300) }), { capture: cap });
  const ps = await sat.decide("a", "b"); ok(ps > 0 && ps < 1, "decide: saturation stays strictly within (0,1): " + ps);
  const fail = webllmModel(mkEngine(cap, { logits: () => { throw new Error("boom"); } }), { capture: cap });
  ok((await fail.decide("a", "b")) === 0.5, "decide: engine failure => 0.5 (previous behavior)");
  // no capture: clear error (no fallback)
  let msg = ""; try { await webllmModel(mkEngine(logitCapture())).decide("s", "q"); } catch (e) { msg = String(e.message); }
  ok(/logitProcessorRegistry/.test(msg) && /capture/.test(msg), "decide without capture throws an error explaining how to enable it");
  // capture given but nothing captured in this call (processor not registered)
  const orphan = logitCapture(), eng3 = mkEngine(logitCapture());
  msg = ""; try { await webllmModel(eng3, { capture: orphan }).decide("s", "q"); } catch (e) { msg = String(e.message); }
  ok(/no logits were captured/.test(msg), "decide with capture but no captured logits throws (no degradation)");
  msg = ""; try { await webllmModel(mkEngine(cap, { logits: () => new Float32Array(V).fill(NaN) }), { capture: cap }).decide("s", "q"); } catch (e) { msg = String(e.message); }
  ok(/NaN/.test(msg), "decide with NaN logits throws");
  ok(webllmModel(eng).maxSafeTokens === 2048 && webllmModel(eng, { maxSafeTokens: 999 }).maxSafeTokens === 999, "maxSafeTokens defaults to 2048, overridable");
  const s = []; for await (const d of webllmModel(mkEngine(cap), { capture: cap }).stream("p", {})) s.push(d);
  ok(s.join("") === "t11t12t13t14t15t248046", "stream() preserved");
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
