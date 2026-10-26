// test_genai_mock.mjs — genai backend with FAKE session/ort (no onnxruntime-web, no GPU).
import { genaiFromSession, parsePlacement, assertWebGpuPlacement, createGenai, checkOrtVersion } from "../dist/genai.js";
import { pastNameFor } from "../dist/decodeOnly.js";
let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };

const V = 16, H = 3;
function thr0(f) { try { f(); return false; } catch { return true; } }
class Tensor {
  constructor(type, data, dims, location = "cpu") { this.type = type; this.dims = dims; this._d = data; this.location = location; this.disposed = false; }
  async getData() { if (this.location === "cpu") return this._d; return this._d; }
  dispose() { this.disposed = true; }
}
const ort = { Tensor };
// fake graph: 2 "linear" layers (conv/recurrent) + 1 "full attention" (key/value); names/dtypes only via metadata.
function mkSession(opts = {}) {
  const meta = [
    { name: "input_ids", type: "int64", shape: [1, "s"] }, { name: "attention_mask", type: "int64", shape: [1, "t"] },
    { name: "position_ids", type: "int64", shape: [3, 1, "s"] },
    { name: "past_key_values.1.key", type: "float16", shape: [1, 4, "p", 8] }, { name: "past_key_values.1.value", type: "float16", shape: [1, 4, "p", 8] },
    { name: "past.0.conv", type: "float16", shape: [1, 6, 3] }, { name: "past.0.recurrent", type: "float16", shape: [1, 2, 4, 4] },
  ];
  const s = { calls: [], released: false, running: 0, overlap: false,
    inputNames: meta.map((m) => m.name), inputMetadata: meta,
    outputNames: ["logits", "present.1.key", "present.1.value", "present.0.conv", "present.0.recurrent", "hidden_states"],
    async run(feeds, fetches) {
      if (s.running++) s.overlap = true;
      await new Promise((r) => setTimeout(r, opts.delay ?? 1));
      const S = feeds.input_ids.dims[1], ids = Array.from(feeds.input_ids._d, Number), pos = Array.from(feeds.position_ids._d.slice(0, S), Number);
      const past = feeds["past_key_values.1.key"].dims[2];
      s.calls.push({ S, ids, pos, mask: feeds.attention_mask.dims[1], posDims: feeds.position_ids.dims, past, fetches: [...fetches], feeds });
      const out = {};
      if (fetches.includes("logits")) { const l = new Float32Array(S * V); for (let i = 0; i < S; i++) l[i * V + ((ids[i] + 1) % V)] = 5; out.logits = new Tensor("float32", l, [1, S, V], "gpu-buffer"); }
      if (fetches.includes("hidden_states")) { const h = new Float32Array(S * H); for (let i = 0; i < S; i++) for (let j = 0; j < H; j++) h[i * H + j] = pos[i] * 10 + j; out.hidden_states = new Tensor("float32", h, [1, S, H], "gpu-buffer"); }
      out["present.1.key"] = new Tensor("float16", new Uint16Array(4 * (past + S) * 8), [1, 4, past + S, 8], "gpu-buffer");
      out["present.1.value"] = new Tensor("float16", new Uint16Array(4 * (past + S) * 8), [1, 4, past + S, 8], "gpu-buffer");
      out["present.0.conv"] = new Tensor("float16", new Uint16Array(18), [1, 6, 3], "gpu-buffer");
      out["present.0.recurrent"] = new Tensor("float16", new Uint16Array(32), [1, 2, 4, 4], "gpu-buffer");
      s.running--; return out;
    },
    async release() { s.released = true; } };
  return s;
}

ok(pastNameFor("present.3.conv") === "past.3.conv" && pastNameFor("present.3.recurrent") === "past.3.recurrent", "pastNameFor covers present.N.conv/recurrent");

const s = mkSession(); const be = genaiFromSession(s, ort, { chunk: 4 });
ok(be.kind === "genai" && be.hiddenName === "hidden_states", "kind + hiddenName");
const ids = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const r = await be.prefill(ids, { positions: [9, 3, 3, 0] });
ok(s.calls.map((c) => c.S).join() === "4,4,2", "prefill em chunks 4,4,2: " + s.calls.map((c) => c.S));
ok(s.calls.map((c) => c.past).join() === "0,4,8", "past grows per chunk");
ok(s.calls.every((c) => c.mask === c.past + c.S && c.posDims.join() === "3,1," + c.S), "attention_mask [1,past+S] and position_ids [3,1,S]");
ok(JSON.stringify(s.calls[1].pos) === "[4,5,6,7]", "absolute position_ids");
{ const f = s.calls[1].feeds.position_ids._d; ok(Array.from(f.slice(0, 4)).join() === Array.from(f.slice(4, 8)).join() && Array.from(f.slice(8, 12)).join() === "4,5,6,7", "3 linhas de position_ids iguais"); }
ok(r.hiddens.length === 4 && r.hiddens[0][0] === 90 && r.hiddens[1][1] === 31 && r.hiddens[2][1] === 31 && r.hiddens[3][2] === 2, "hiddens in requested order, duplicates");
ok(!s.calls[0].fetches.includes("logits") && s.calls[2].fetches.includes("logits"), "logits only in the last chunk");
ok(s.calls[0].fetches.includes("hidden_states") && !s.calls[1].fetches.includes("hidden_states") && s.calls[2].fetches.includes("hidden_states"), "hidden only in chunks with a requested position");
ok(r.lastLogits.length === V && r.lastLogits[(12 + 1) % V] === 5, "lastLogits = last position");
const first = s.calls[0].feeds;
ok(first["past.0.conv"].dims.join() === "1,6,3" && first["past.0.recurrent"].dims.join() === "1,2,4,4" && first["past_key_values.1.key"].dims.join() === "1,4,0,8" && first["past_key_values.1.key"].type === "float16", "initial caches zeroed, shape/dtype from metadata");
ok(s.calls[1].feeds["past_key_values.1.key"] === undefined ? false : s.calls[1].feeds["past_key_values.1.key"].dims[2] === 4 && s.calls[1].feeds["past.0.conv"].location === "gpu-buffer", "present -> past (GPU)");
ok(s.calls[0].feeds["past.0.conv"].disposed === false && s.calls[1].feeds["past.0.conv"].disposed === true && s.calls[2].feeds["past.0.recurrent"].disposed === true, "previous GPU past disposed; initial zeros (CPU) not");
let bad = false; try { await be.prefill(ids, { positions: [10] }); } catch { bad = true; } ok(bad, "out-of-range position throws");
const r0 = await be.prefill([1, 2], { positions: [] }); ok(r0.hiddens.length === 0 && r0.lastLogits.length === V, "empty positions");
ok(!s.calls.at(-1).fetches.includes("hidden_states"), "no positions, no hidden fetch");

// generate
{
  const s2 = mkSession(), b = genaiFromSession(s2, ort, { chunk: 8 }); const g = [];
  for await (const t of b.generate([1, 2, 3], { maxNewTokens: 5, temperature: 0, repetitionPenalty: 1 })) g.push(t);
  ok(JSON.stringify(g) === "[4,5,6,7,8]", "greedy: " + JSON.stringify(g));
  ok(s2.calls.length === 5 && s2.calls[0].S === 3 && s2.calls[1].S === 1 && s2.calls[1].past === 3 && s2.calls[1].pos[0] === 3, "prefill S=3 + decode S=1 with correct past/position");
  ok(!s2.calls.some((c) => c.fetches.includes("hidden_states")), "generate does not fetch hidden");
  const g2 = []; for await (const t of b.generate([13], { maxNewTokens: 9, temperature: 0, repetitionPenalty: 1 })) g2.push(t);
  ok(JSON.stringify(g2) === "[14,15,0,1,2,3,4,5,6]", "generates again in the same session: " + JSON.stringify(g2));
}
{ // EOS 248046 is never emitted (V=16 cannot reach it; the sampler returns EOS via the larger logit)
  const s3 = mkSession(); const b = genaiFromSession(s3, ort, { chunk: 8, eos: new Set([6]) }); const g = [];
  for await (const t of b.generate([1, 2, 3], { maxNewTokens: 9, temperature: 0, repetitionPenalty: 1 })) g.push(t);
  ok(JSON.stringify(g) === "[4,5]", "stops at EOS without emitting it: " + JSON.stringify(g));
}
{ // abort + break: generator settles (no in-flight run) before the next prefill
  const s4 = mkSession({ delay: 5 }), b = genaiFromSession(s4, ort, {}); const ac = new AbortController(), g = [];
  for await (const t of b.generate([1], { maxNewTokens: 100, temperature: 0, repetitionPenalty: 1, signal: ac.signal })) { g.push(t); if (g.length === 3) ac.abort(); }
  ok(g.length === 3 && s4.calls.length === 3, `abort: stopped cleanly (got=${g.length}, runs=${s4.calls.length})`);
  const n = s4.calls.length; await b.prefill([1, 2], { positions: [] }); ok(!s4.overlap && s4.calls.length === n + 1, "prefill after abort does not overlap");
  const it = b.generate([1], { maxNewTokens: 50, temperature: 0, repetitionPenalty: 1 }); await it.next(); await it.return();
  await b.prefill([1], { positions: [] }); ok(!s4.overlap, "early break: no overlap");
  // prefill concurrent with an in-flight generate is serialized
  const gp = (async () => { for await (const _ of b.generate([1], { maxNewTokens: 4, temperature: 0, repetitionPenalty: 1 })) { /* */ } })();
  const pp = b.prefill([1, 2, 3], { positions: [1] }); await Promise.all([gp, pp]); ok(!s4.overlap, "concurrent generate and prefill are serialized");
  await b.dispose(); ok(s4.released, "dispose releases the session");
}

// abort during a long prefill: stops between chunks
{
  const s5 = mkSession(), b = genaiFromSession(s5, ort, { chunk: 2 }); const ac = new AbortController();
  const orig = s5.run.bind(s5); s5.run = async (f, x) => { const r = await orig(f, x); if (s5.calls.length === 2) ac.abort(); return r; };
  const g = []; for await (const t of b.generate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], { maxNewTokens: 5, temperature: 0, repetitionPenalty: 1, signal: ac.signal })) g.push(t);
  ok(g.length === 0 && s5.calls.length === 2, `abort in prefill: ${s5.calls.length} of 5 chunks, 0 tokens`);
  ok(s5.calls[1].feeds["past.0.conv"].disposed && !s5.overlap, "abort in prefill releases the state");
}
ok(thr0(() => checkOrtVersion("1.29.2")) && !thr0(() => checkOrtVersion("1.30.0")) && !thr0(() => checkOrtVersion("1.31.0-dev.20260914-8d85527a0")) && !thr0(() => checkOrtVersion("1.32.1")) && !thr0(() => checkOrtVersion(undefined)), "checkOrtVersion: <1.30 rejects, >=1.30 passes");
// EP assertion
const goodLog = ["[V:onnxruntime:, session_state.cc:1377 VerifyEachNodeIsAssignedToAnEp] Node placements",
  "[V:onnxruntime:, session_state.cc:1383 VerifyEachNodeIsAssignedToAnEp]  Node(s) placed on [WebGpuExecutionProvider]. Number of nodes: 900",
  "[V:onnxruntime:, session_state.cc:1383 VerifyEachNodeIsAssignedToAnEp]  Node(s) placed on [CPUExecutionProvider]. Number of nodes: 6"];
const jsepLog = ["Node(s) placed on [WebGpuExecutionProvider]. Number of nodes: 300", "Node(s) placed on [CPUExecutionProvider]. Number of nodes: 591"];
const p = parsePlacement(goodLog);
ok(p.WebGpuExecutionProvider === 900 && p.CPUExecutionProvider === 6, "parsePlacement");
ok(assertWebGpuPlacement(p) === undefined, "assert: 900 webgpu / 6 cpu passes");
const thr = (f) => { try { f(); return null; } catch (e) { return String(e.message); } };
ok(/CPU|webgpu/i.test(thr(() => assertWebGpuPlacement(parsePlacement(jsepLog))) ?? ""), "assert: JSEP bundle (591 on CPU) fails");
ok(thr(() => assertWebGpuPlacement(parsePlacement([]))) !== null, "assert: no placement log fails (unverifiable)");
ok(thr(() => assertWebGpuPlacement(parsePlacement(["Node(s) placed on [CPUExecutionProvider]. Number of nodes: 900"]))) !== null, "assert: all-CPU fails");

// createGenai rejects non-webgpu providers
let rej = null; try { await createGenai("m.onnx", "d.data", { providers: ["wasm"], ort: { ...ort, InferenceSession: { create() { throw new Error("should not create"); } } } }); } catch (e) { rej = String(e.message); }
ok(rej && /webgpu/i.test(rej), "createGenai rejects providers without webgpu: " + rej);

// verifyEp: verdict cached per (ort version, model URL) -> no second probe
{
  const mem = new Map(); const store = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v) };
  let creates = []; // log level of each created session
  const mkOrt = (ver, placeCpu = 6) => ({ Tensor, env: { versions: { web: ver } }, InferenceSession: { async create(url, o) {
    creates.push(o.logSeverityLevel);
    if (o.logSeverityLevel === 0) console.log(`Node(s) placed on [WebGpuExecutionProvider]. Number of nodes: 900`), console.log(`Node(s) placed on [CPUExecutionProvider]. Number of nodes: ${placeCpu}`);
    const s = mkSession(); s.release = async () => {}; return s; } } });
  const o = { ort: mkOrt("1.30.0"), epCache: store, warmup: false };
  const origLog = console.log; 
  try {
    await createGenai("https://x/m.onnx", "https://x/d.data", o);
    const first = creates.slice();
    creates = []; await createGenai("https://x/m.onnx", "https://x/d.data", o);
    const second = creates.slice();
    creates = []; await createGenai("https://x/m.onnx", "https://x/d.data", { ...o, verifyEp: "force" });
    const forced = creates.slice();
    creates = []; await createGenai("https://x/other.onnx", "https://x/d.data", o);
    const otherUrl = creates.slice();
    creates = []; await createGenai("https://x/m.onnx", "https://x/d.data", { ...o, ort: mkOrt("1.31.0") });
    const otherVer = creates.slice();
    creates = []; await createGenai("https://x/m.onnx", "https://x/d.data", { ...o, verifyEp: false });
    const off = creates.slice();
    let failed = null; const bad = { ...o, ort: mkOrt("1.30.0", 900), epCache: store };
    try { await createGenai("https://x/bad.onnx", "https://x/d.data", bad); } catch (e) { failed = e.message; }
    ok(JSON.stringify(first) === "[0,3]", "verifyEp: first load probes (level 0) and creates the session (level 3)");
    ok(JSON.stringify(second) === "[3]", "verifyEp: second load uses the cached verdict (no probe)");
    ok(JSON.stringify(forced) === "[0,3]", "verifyEp 'force' re-probes");
    ok(JSON.stringify(otherUrl) === "[0,3]", "verifyEp: a different model URL probes again");
    ok(JSON.stringify(otherVer) === "[0,3]", "verifyEp: a different ort version probes again");
    ok(JSON.stringify(off) === "[3]", "verifyEp false never probes");
    ok(failed && /CPU/.test(failed) && ![...mem.keys()].some((k) => k.includes("/bad.onnx")), "verifyEp: a failed verdict is not cached");
    creates = []; await createGenai("https://x/m.onnx", "https://x/other.data", o);
    ok(JSON.stringify(creates) === "[0,3]", "verifyEp: a different data URL probes again");
    creates = []; await createGenai("https://x/m.onnx", "https://x/d.data", { ...o, maxCpuNodes: 8 });
    ok(JSON.stringify(creates) === "[0,3]", "verifyEp: a different maxCpuNodes probes again");
    creates = []; await createGenai("https://x/m.onnx", "https://x/d.data", { ...o, adapterTag: "gpuB" });
    ok(JSON.stringify(creates) === "[0,3]", "verifyEp: a different GPU adapter probes again");
    creates = []; await createGenai("https://x/m.onnx", "https://x/d.data", { ...o, adapterTag: "gpuB" });
    ok(JSON.stringify(creates) === "[3]", "verifyEp: same adapter uses the cache");
    ok([...mem.keys()].every((k) => k.startsWith("ct0-edge:genai-ep:")) && [...mem.keys()].some((k) => k.includes(":gpuB:")), "verifyEp: key contains the adapter");
  } finally { console.log = origLog; }
  // no localStorage (Node): always probes, never throws
  creates = []; const o2 = { ort: mkOrt("1.30.0"), epCache: null, warmup: false };
  await createGenai("m", "d", o2); await createGenai("m", "d", o2);
  ok(JSON.stringify(creates) === "[0,3,0,3]", "verifyEp: no storage, probes every time");
}

// session created but the backend fails (warmup / construction): the session is released before the error propagates
{
  let released = 0;
  const mk = (sessFn) => ({ Tensor, env: { versions: { web: "1.30.0" } }, InferenceSession: { async create() { const s = sessFn(); s.release = async () => { released++; }; return s; } } });
  let e1 = null;
  try { await createGenai("https://x/m.onnx", "https://x/d.data", { ort: mk(() => { const s = mkSession(); s.run = async () => { throw new Error("warmup boom"); }; return s; }), verifyEp: false }); } catch (e) { e1 = e.message; }
  ok(/warmup boom/.test(e1) && released === 1, "warmup fails -> session.release() called and error propagated");
  released = 0; let e2 = null;
  try { await createGenai("https://x/m.onnx", "https://x/d.data", { ort: mk(() => ({ inputNames: [], outputNames: [] })), verifyEp: false }); } catch (e) { e2 = e.message; }
  ok(/input_ids/.test(e2) && released === 1, "genaiFromSession throws -> session.release() called");
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
