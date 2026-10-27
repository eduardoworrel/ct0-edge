// test_litert_mock.mjs — createLiteRt (EXPERIMENTAL runtime "litert") with a FAKE LiteRT.js module (no model):
// prefill chunking/padding, decode, fresh state per call (no cross-call leak, unlike LiteRT-LM #3165), snapshots,
// generate, helpers, and the runtime plan (never chosen by "auto").
import { createLiteRt, buildMask, prefillChunks, embedFromF16Reader, httpRangeReader, LITERT_WARNING } from "../dist/litert.js";
import { planRuntime, defaultMethods, packagesFor, RUNTIMES } from "../dist/ct0.js";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };
const throws = async (f, re) => { try { await f(); return false; } catch (e) { return re.test(String(e.message)); } };

const H = 2, V = 6, CACHE = 64, P = 4;
class Tensor {
  constructor(data, shape, dtype) { this.data = data; this.shape = shape; this.dtype = dtype; this.deleted = false; }
  toTypedArray() { return this.data; } async moveTo() { return this; } async copyTo() { return new Tensor(this.data.slice(), this.shape); }
  delete() { this.deleted = true; }
}
// Toy recurrent model with the real signature layout: state r (1 float) = 0.5*r + emb[0] for every VALID row
// (row 0, or input_pos increasing), exactly how the litert-torch graphs derive the valid mask from input_pos.
// decode logits[i] = r * (i - 2): argmax = V-1 when r > 0.
function fakeModel(log) {
  const io = (rows) => ({ inputs: [{ name: "embeddings", shape: [1, rows, H], dtype: "float32" }, { name: "input_pos", shape: [rows], dtype: "int32" },
    { name: "mask", shape: [1, 1, rows, CACHE], dtype: "float32" }, { name: "kv_cache_r_0", shape: [1, 1], dtype: "float32" }],
    outputs: [{ name: "kv_cache_r_0", shape: [1, 1], dtype: "float32" }] });
  const manifest = { format: "ct0-litert-bundle/1", vocab: V, hidden: H, signatures: { [`prefill_${P}`]: io(P), decode: io(1) } };
  manifest.signatures.decode.outputs.push({ name: "logits", shape: [1, 1, V], dtype: "float32" });
  const model = {
    deleted: false, delete() { this.deleted = true; },
    async run(sig, feed) {
      const pos = feed.input_pos.data, e = feed.embeddings.data;
      log.push({ sig, pos: Array.from(pos), mask0: feed.mask.data[0], maskLen: feed.mask.data.length });
      let r = feed.kv_cache_r_0.data[0];
      for (let i = 0; i < pos.length; i++) if (i === 0 || pos[i] > pos[i - 1]) r = 0.5 * r + e[i * H];
      const out = { kv_cache_r_0: new Tensor(Float32Array.of(r), [1, 1]) };
      if (sig === "decode") out.logits = new Tensor(Float32Array.from({ length: V }, (_, i) => r * (i - 2)), [1, 1, V]);
      return out;
    },
  };
  return { manifest, model };
}
const embed = { async rows(ids) { const o = new Float32Array(ids.length * H); ids.forEach((id, r) => o.fill(id, r * H, r * H + H)); return o; } };
const refState = (ids) => ids.reduce((r, x) => 0.5 * r + x, 0);
function mk(opts = {}) {
  const log = [], { manifest, model } = fakeModel(log);
  const lrt = { Tensor, async loadAndCompile() { return model; }, async loadModelAndWeights(g, w) { lrt.stream = w; return model; } };
  return { log, manifest, model, lrt, be: () => createLiteRt(lrt, new Uint8Array(4), manifest, embed, { quiet: true, ...opts }) };
}

console.log("helpers:");
ok(JSON.stringify(prefillChunks(9, 4)) === '[{"start":0,"len":4},{"start":4,"len":4},{"start":8,"len":1}]', "prefillChunks 9/4");
ok(prefillChunks(0, 4).length === 0, "prefillChunks of 0 tokens is empty");
{ const m = buildMask(Int32Array.of(2, 0), 4, "float32"); ok(m[0] === 0 && m[2] === 0 && m[3] < -1e29 && m[4] === 0 && m[5] < -1e29, "additive mask: row sees cols <= pos (pads at pos 0 see col 0)"); }
{ const m = buildMask(Int32Array.of(1), 3, "bool"); ok(m[0] === 1 && m[1] === 1 && m[2] === 0, "bool mask"); }
{ const f16 = new Uint16Array([0x3c00, 0xc000, 0x0000, 0x3800]); const b = new Uint8Array(f16.buffer);
  const e = embedFromF16Reader(async (o, n) => b.subarray(o, o + n), 2); const r = await e.rows([1, 0, 1]);
  ok(Array.from(r).join() === "0,0.5,1,-2,0,0.5", "fp16 embedding rows read by offset"); }
{ const calls = []; const f = async (u, o) => { calls.push(o.headers.Range); return { ok: true, status: 206, arrayBuffer: async () => new Uint8Array([7, 8]).buffer }; };
  const r = await httpRangeReader("u", f)(10, 2); ok(calls[0] === "bytes=10-11" && r[0] === 7, "HTTP Range reader"); }

console.log("\nprefill (wasm, all chunks through prefill_N):");
{
  const t = mk({ accelerator: "wasm" }); const be = await t.be();
  ok(be.kind === "litert" && be.hiddenName === null, "kind litert, no hidden output");
  const ids = [3, 1, 4, 1, 5, 9, 2];
  const { lastLogits } = await be.prefill(ids, { positions: [] });
  const r = refState(ids);
  ok(Math.abs(lastLogits[V - 1] - r * (V - 3)) < 1e-6, "last logits = toy reference over all 7 tokens");
  ok(JSON.stringify(t.log.map((l) => l.sig)) === JSON.stringify(["prefill_4", "prefill_4", "decode"]), "6 prompt tokens -> 2 prefill chunks, last token -> decode");
  ok(JSON.stringify(t.log[1].pos) === "[4,5,0,0]", "second chunk right-padded with position 0");
  ok(t.log[0].maskLen === P * CACHE && t.log[2].maskLen === CACHE, "mask shapes [1,1,rows,cache]");
  const again = await be.prefill(ids, { positions: [] });
  ok(Array.from(again.lastLogits).join() === Array.from(lastLogits).join(), "second identical call -> identical logits (fresh state, no leak)");
  const other = await be.prefill([2, 7], { positions: [] });
  ok(Math.abs(other.lastLogits[V - 1] - refState([2, 7]) * (V - 3)) < 1e-6, "a different prompt after it is unaffected by the previous one");
  ok(await throws(() => be.prefill([1, 2], { positions: [0] }), /no hidden-state output/), "positions requested -> clear error");
  ok(await throws(() => be.prefill([], { positions: [] }), /empty ids/), "empty ids rejected");
}

console.log("\nWebGPU: streamed weights, remainder through decode:");
{
  const t = mk({ accelerator: "webgpu", weights: () => "STREAM" }); const be = await t.be();
  ok(typeof t.lrt.stream?.getReader === "function", "loadModelAndWeights gets an object whose getReader() reopens the stream");
  const ids = [3, 1, 4, 1, 5, 9, 2];
  const { lastLogits } = await be.prefill(ids, { positions: [] });
  ok(JSON.stringify(t.log.map((l) => l.sig)) === JSON.stringify(["prefill_4", "decode", "decode", "decode"]), "full chunk via prefill, 2-token remainder + last via decode");
  ok(Math.abs(lastLogits[V - 1] - refState(ids) * (V - 3)) < 1e-6, "same result as the all-prefill path");
}

console.log("\nsnapshot / prefillFrom / generate:");
{
  const t = mk({ accelerator: "wasm" }); const be = await t.be();
  const pre = [5, 6, 7], ids = [5, 6, 7, 1, 2];
  const h = await be.snapshot(pre);
  const a = await be.prefillFrom(h, ids, { positions: [] }), b = await be.prefill(ids, { positions: [] });
  ok(Array.from(a.lastLogits).join() === Array.from(b.lastLogits).join(), "prefillFrom(snapshot) == prefill");
  const a2 = await be.prefillFrom(h, [5, 6, 7, 3], { positions: [] });
  ok(Math.abs(a2.lastLogits[V - 1] - refState([5, 6, 7, 3]) * (V - 3)) < 1e-6, "snapshot reusable (not consumed)");
  h.dispose(); ok(await throws(() => be.prefillFrom(h, ids, { positions: [] }), /released snapshot/), "released snapshot rejected");
  const gen = []; for await (const x of be.generate([1, 2], { maxNewTokens: 3, temperature: 0 })) gen.push(x);
  ok(gen.length === 3 && gen.every((x) => x === V - 1), "greedy generate: 3 tokens, argmax");
  await be.dispose(); ok(t.model.deleted, "dispose deletes the compiled model");
  ok(await throws(() => be.prefill([1], { positions: [] }), /already disposed/), "calls after dispose reject");
}

console.log("\nwarning + runtime plan:");
{
  const msgs = []; const w = console.warn; console.warn = (m) => msgs.push(m);
  const t = mk({ accelerator: "wasm" }); await createLiteRt(t.lrt, new Uint8Array(4), t.manifest, embed, { accelerator: "wasm" });
  console.warn = w;
  ok(msgs.length === 1 && msgs[0] === LITERT_WARNING && /EXPERIMENTAL/.test(LITERT_WARNING), "experimental warning printed once");
  ok(RUNTIMES.includes("litert"), "litert is a known runtime");
  const auto = planRuntime("auto", "auto", { platform: "browser", webgpu: true, cuda: false });
  ok(!auto.some((p) => p.runtime === "litert"), "auto never picks litert");
  ok(JSON.stringify(planRuntime("litert", "auto", { platform: "browser", webgpu: true, cuda: false })) === '[{"runtime":"litert","device":"webgpu"}]', "explicit litert -> webgpu");
  let e1 = ""; try { planRuntime("litert", "auto", { platform: "node", webgpu: false, cuda: false }); } catch (e) { e1 = e.message; }
  ok(/runs in the browser/.test(e1), "litert in Node -> clear error");
  let e2 = ""; try { planRuntime("litert", "wasm", { platform: "browser", webgpu: true, cuda: false }); } catch (e) { e2 = e.message; }
  ok(/needs WebGPU/.test(e2), "litert on wasm -> clear error");
  ok(JSON.stringify(defaultMethods("litert")) === '{"noul":"logits","choice":"logits","score":"logits"}', "litert reads logits for every mode");
  ok(packagesFor("litert", "browser").includes("@litertjs/core"), "needs @litertjs/core");
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
