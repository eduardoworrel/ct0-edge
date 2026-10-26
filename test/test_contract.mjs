// test_contract.mjs — Backend contract (src/backend.ts) on all four backends, with mocks (no model/GPU):
// serialized calls, idempotent dispose that rejects afterwards, single GenerateOptions normalization,
// maxNewTokens 0, ort version check in decode-only, and webllmModel.decide() on the backend's queue.
import { resolveGenerateOptions, GENERATE_DEFAULTS, createDecodeOnly } from "../dist/decodeOnly.js";
import { transformersJsFromModel } from "../dist/transformers.js";
import { genaiFromSession, checkOrtVersion } from "../dist/genai.js";
import { checkOrtVersion as ct0Check } from "../dist/decodeOnly.js";
import { logitCapture, webllmBackend, webllmModel } from "../dist/webllm.js";
import { YES_ID, NO_ID } from "../dist/ask.js";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };
const rejects = async (f, re = /dispose/) => { try { await f(); return false; } catch (e) { return re.test(String(e && e.message)); } };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const drain = async (g) => { const o = []; for await (const x of g) o.push(x); return o; };

// ---------------- GenerateOptions normalization
console.log("resolveGenerateOptions:");
{
  const d = resolveGenerateOptions({});
  ok(JSON.stringify(d) === JSON.stringify({ ...GENERATE_DEFAULTS }) || (d.maxNewTokens === 512 && d.temperature === 0.7 && d.topP === 0.9 && d.topK === 40 && d.repetitionPenalty === 1.1 && d.seed === 0), "single set of defaults: 512 / 0.7 / 0.9 / 40 / 1.1 / seed 0");
  const u = resolveGenerateOptions({ temperature: undefined, maxNewTokens: undefined, topP: undefined, seed: undefined });
  ok(u.temperature === 0.7 && u.maxNewTokens === 512 && u.topP === 0.9 && u.seed === 0, "undefined fields do NOT override defaults");
  ok(resolveGenerateOptions({ maxNewTokens: 0 }).maxNewTokens === 0, "maxNewTokens 0 => 0");
  ok(resolveGenerateOptions({ temperature: 0 }).repetitionPenalty === 1, "temperature 0 => pure greedy (repetitionPenalty 1)");
  ok(resolveGenerateOptions({ temperature: 0, repetitionPenalty: 1.3 }).repetitionPenalty === 1.3, "temperature 0 + explicit repetitionPenalty => kept");
  ok(resolveGenerateOptions({}).repetitionPenalty === 1.1, "sampled default => repetitionPenalty 1.1");
  ok(resolveGenerateOptions({ maxNewTokens: 3.9 }).maxNewTokens === 3 && resolveGenerateOptions({ maxNewTokens: -5 }).maxNewTokens === 0, "integer >= 0");
  ok(resolveGenerateOptions({ temperature: 0 }).temperature === 0 && resolveGenerateOptions({ temperature: NaN }).temperature === 0.7, "temperature 0 (greedy) respected; NaN => default");
  const ac = new AbortController();
  ok(resolveGenerateOptions({ signal: ac.signal, seed: 42, topK: 5, repetitionPenalty: 1 }).seed === 42 && resolveGenerateOptions({ signal: ac.signal }).signal === ac.signal, "seed/topK/rep/signal pass through");
}

// ---------------- mocks
class Tensor { constructor(type, data, dims, location = "cpu") { this.type = type; this.data = data; this._d = data; this.dims = dims; this.location = location; } async getData() { return this._d; } dispose() {} }
const V = 8, H = 3;

function decodeSession(delay = 1) {
  const s = { running: 0, overlap: false, released: 0, runs: 0,
    inputNames: ["input_ids", "past_conv.0"], outputNames: ["logits", "present_conv.0", "hidden_states"],
    inputMetadata: [{ name: "input_ids", type: "int64", shape: [1, 1] }, { name: "past_conv.0", type: "float32", shape: [1, 2] }],
    async run(feed, names) {
      if (s.running++) s.overlap = true; s.runs++;
      await sleep(delay); s.running--;
      const l = new Float32Array(V); l[(s.runs % 5) + 1] = 9;
      const r = { logits: new Tensor("float32", l, [1, 1, V]), "present_conv.0": new Tensor("float32", new Float32Array(2), [1, 2]) };
      if (names.includes("hidden_states")) r.hidden_states = new Tensor("float32", new Float32Array(H).fill(1), [1, 1, H]);
      return r;
    },
    async release() { await sleep(1); s.released++; } };
  return s;
}
function tjsModel(delay = 1) {
  const m = { running: 0, overlap: false, disposed: 0, generates: 0,
    sessions: { model: { outputNames: ["logits", "type_as_80"] } },
    async forward(f) { if (m.running++) m.overlap = true; await sleep(delay); m.running--;
      const S = f.input_ids.dims[1];
      return { logits: new Tensor("float32", new Float32Array(S * V).fill(1), [1, S, V]), type_as_80: new Tensor("float32", new Float32Array(S * H), [1, S, H]) }; },
    async generate(args) { m.generates++; m.lastArgs = args; if (m.running++) m.overlap = true; await sleep(delay * 3);
      args.streamer?.put([[1n, 2n]]); for (const t of [7n, 8n, 9n]) { await sleep(delay); args.streamer?.put([[t]]); } args.streamer?.end?.(); m.running--; return null; },
    async dispose() { await sleep(1); m.disposed++; } };
  return m;
}
function genaiSession(delay = 1) {
  const meta = [{ name: "input_ids", type: "int64", shape: [1, "s"] }, { name: "attention_mask", type: "int64", shape: [1, "t"] },
    { name: "position_ids", type: "int64", shape: [3, 1, "s"] }, { name: "past.0.conv", type: "float16", shape: [1, 2] }];
  const s = { running: 0, overlap: false, released: 0, inputNames: meta.map((m) => m.name), inputMetadata: meta,
    outputNames: ["logits", "present.0.conv", "hidden_states"],
    async run(feeds, fetches) {
      if (s.running++) s.overlap = true; await sleep(delay); s.running--;
      const S = feeds.input_ids.dims[1], out = { "present.0.conv": new Tensor("float16", new Uint16Array(2), [1, 2], "cpu") };
      if (fetches.includes("logits")) { const l = new Float32Array(S * V); l[3] = 5; out.logits = new Tensor("float32", l, [1, S, V]); }
      if (fetches.includes("hidden_states")) out.hidden_states = new Tensor("float32", new Float32Array(S * H).fill(1), [1, S, H]);
      return out; },
    async release() { await sleep(1); s.released++; } };
  return s;
}
function webllmEngine(cap, delay = 1) {
  const e = { running: 0, overlap: false, unloaded: 0, calls: [], async resetChat() {}, async interruptGenerate() { e.stop = true; },
    async unload() { await sleep(1); e.unloaded++; },
    completions: { async create(req) {
      if (e.running++) e.overlap = true; e.calls.push(req.prompt);
      try {
        await sleep(delay);
        if (req.stream) { e.stop = false; let i = 0; const toks = [11, 12, 13];
          return { [Symbol.asyncIterator]() { return this; }, async next() { await sleep(1); if (e.stop || i >= toks.length || i >= req.max_tokens) return { done: true }; const t = toks[i++]; cap.processor.processSampledToken(t); return { done: false, value: { choices: [{ text: "t" }] } }; }, async return() { return { done: true }; } }; }
        // logits depend only on the prompt: L[5] = length; Yes-No = 3 for the decide prompt
        const L = new Float32Array(9200); L[5] = req.prompt.length; if (req.prompt.includes("STATE:")) { L[YES_ID] = 3; L[NO_ID] = 0; }
        cap.processor.processLogits(L);
        return { usage: { prompt_tokens: (req.prompt.match(/</g) || []).length }, choices: [] };
      } finally { e.running--; } } } };
  return e;
}
const wtok = { decode: (ids) => ids.map((i) => `<${i}>`).join("") };

const mkers = {
  "decode-only": async () => { const s = decodeSession(); const be = await createDecodeOnly(s, null, { ort: { Tensor } }); return { be, st: s, released: () => s.released, gen: { maxNewTokens: 3, temperature: 0 } }; },
  "transformers.js": async () => { const m = tjsModel(); const be = transformersJsFromModel(m, Tensor); return { be, st: m, released: () => m.disposed, gen: { maxNewTokens: 3, temperature: 0 } }; },
  genai: async () => { const s = genaiSession(); const be = genaiFromSession(s, { Tensor }, { chunk: 2 }); return { be, st: s, released: () => s.released, gen: { maxNewTokens: 3, temperature: 0 } }; },
  webllm: async () => { const cap = logitCapture(), e = webllmEngine(cap); const be = webllmBackend(e, cap, { tok: wtok }); return { be, st: e, released: () => e.unloaded, gen: { maxNewTokens: 3, temperature: 0 } }; },
};

for (const [name, mk] of Object.entries(mkers)) {
  console.log(`\n${name}:`);
  {
    const { be, st, gen } = await mk();
    // prefill/prefill, prefill/generate, generate/prefill: never interleaved
    const g = drain(be.generate([3, 4, 5], gen));
    const ps = [be.prefill([3, 4, 5], { positions: [] }), be.prefill([6, 7], { positions: [] })];
    const g2 = drain(be.generate([3, 4], gen));
    await Promise.all([g, ...ps, g2]);
    ok(!st.overlap, "overlapping calls (generate + prefill + prefill + generate) are serialized (no concurrent run)");
  }
  {
    const { be, st, released, gen } = await mk();
    const inflight = be.prefill([1, 2, 3], { positions: [] });
    const d = be.dispose();
    ok((await inflight).lastLogits.length > 0, "dispose waits for in-flight work (the in-flight prefill completes)");
    await d;
    ok(released() === 1, "resource released exactly once");
    ok(await rejects(() => be.prefill([1], { positions: [] })), "prefill after dispose rejects with a clear error");
    ok(await rejects(() => drain(be.generate([1], gen))), "generate after dispose rejects with a clear error");
    await be.dispose(); await Promise.all([be.dispose(), be.dispose()]);
    ok(released() === 1, "idempotent dispose: does not release again");
    ok(!st.overlap, "no overlap in dispose");
  }
  {
    const { be, st, gen } = await mk();
    const before = st.runs ?? st.generates ?? st.calls?.length ?? 0;
    const none = await drain(be.generate([3, 4, 5], { ...gen, maxNewTokens: 0 }));
    ok(none.length === 0, "maxNewTokens 0 => 0 tokens");
    const after = st.runs ?? st.generates ?? st.calls?.length ?? 0;
    ok(after === before, "maxNewTokens 0 does not even run the model/prompt");
    const und = await drain(be.generate([3, 4, 5], { maxNewTokens: 2, temperature: undefined, topP: undefined, topK: undefined, repetitionPenalty: undefined, seed: undefined }));
    ok(und.length <= 2 && und.every((x) => Number.isFinite(x)), "undefined options do not become NaN (finite tokens: " + und + ")");
  }
}

// transformers.js: library args come from the single normalization
console.log("\ntransformers.js (library args):");
{
  const m = tjsModel(); const be = transformersJsFromModel(m, Tensor);
  await drain(be.generate([3, 4], { maxNewTokens: 2 }));
  const a = m.lastArgs;
  ok(a.do_sample === true && a.temperature === 0.7 && a.top_p === 0.9 && a.top_k === 40 && a.repetition_penalty === 1.1 && a.max_new_tokens === 2, "defaults 0.7/0.9/40/1.1 reach model.generate");
  await drain(be.generate([3, 4], { maxNewTokens: 2, temperature: 0 }));
  ok(m.lastArgs.do_sample === false && m.lastArgs.top_k === undefined, "temperature 0 => greedy");
}

// decode-only: reproducible seed (JS sampler)
console.log("\nseed:");
{
  const mkS = async () => createDecodeOnly(decodeSession(0), null, { ort: { Tensor } });
  const a = await drain((await mkS()).generate([3, 4], { maxNewTokens: 6, seed: 7, temperature: 1.5 }));
  const b = await drain((await mkS()).generate([3, 4], { maxNewTokens: 6, seed: 7, temperature: 1.5 }));
  ok(JSON.stringify(a) === JSON.stringify(b) && a.length === 6, "decode-only: same seed => same sampling");
}

// ---------------- ort version in decode-only / facade
console.log("\nshared checkOrtVersion:");
{
  { const m = (f) => { try { f("1.20.0"); return ""; } catch (e) { return e.message; } };
  ok(m(checkOrtVersion) !== "" && m(checkOrtVersion) === m(ct0Check), "genai e decode-only compartilham a mesma checagem (mesma mensagem)"); }
  ok(await rejects(() => createDecodeOnly(decodeSession(), null, { ort: { Tensor, env: { versions: { web: "1.20.1" } } } }), /1\.20\.1.*1\.30/), "decode-only rejects ort-web 1.20");
  ok(!(await rejects(() => createDecodeOnly(decodeSession(), null, { ort: { Tensor, env: { versions: { web: "1.30.0" } } } }), /./)), "decode-only accepts 1.30.0");
  ok(!(await rejects(() => createDecodeOnly(decodeSession(), null, { ort: { Tensor, env: { versions: { web: "1.20.1" } } }, skipVersionCheck: true }), /./)), "skipVersionCheck skips the check");
  ok(!(await rejects(() => createDecodeOnly(decodeSession(), null, { ort: { Tensor } }), /./)), "no version (ort-node) is not rejected");
}

// ---------------- webllm: decide() and prefill() share engine + capture => same queue
console.log("\nwebllm decide + backend on the same queue:");
{
  const cap = logitCapture(), e = webllmEngine(cap), be = webllmBackend(e, cap, { tok: wtok }), m = webllmModel(e, { capture: cap });
  const ids = [3, 4, 5, 6, 7];
  const [p, r, p2, r2] = await Promise.all([m.decide("a", "q?"), be.prefill(ids, { positions: [] }), m.decide("b", "q?"), be.prefill([3, 4], { positions: [] })]);
  const sig = 1 / (1 + Math.exp(-3));
  ok(!e.overlap, "no overlapping engine call");
  ok(Math.abs(p - sig) < 1e-6 && Math.abs(p2 - sig) < 1e-6, "decide() le os logits do SEU prompt (" + p.toFixed(4) + ")");
  ok(r.lastLogits[5] === "<3><4><5><6><7>".length && r2.lastLogits[5] === "<3><4>".length && r.lastLogits[YES_ID] === 0, "prefill() reads the logits of ITS prompt (no crossover with decide)");
  const s = []; const st = drain(m.stream("p", {})); const pf = be.prefill([3], { positions: [] }); await Promise.all([st, pf]);
  ok(!e.overlap, "the adapter's stream() also uses the queue");
  await be.dispose();
  ok(await rejects(() => m.decide("a", "q?")), "decide() after backend dispose rejects");
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
