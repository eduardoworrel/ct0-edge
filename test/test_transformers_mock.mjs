// test_transformers_mock.mjs — transformers.js backend with a FAKE model (no real transformers.js).
import { transformersJsFromModel } from "../dist/transformers.js";
let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };
class Tensor { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }
const V = 5, H = 3;
function mk(hid = "type_as_80") {
  const m = { forwards: [], disposed: false,
    sessions: { model: { outputNames: ["logits", hid, "present.0.key"], inputNames: ["input_ids", "attention_mask"] } },
    async forward(f) { m.forwards.push(f); const S = f.input_ids.dims[1];
      const h = new Float32Array(S * H); for (let i = 0; i < S; i++) for (let j = 0; j < H; j++) h[i * H + j] = i * 10 + j;
      const l = new Float32Array(S * V); for (let i = 0; i < S; i++) l[i * V + (i % V)] = 1 + i;
      return { logits: new Tensor("float32", l, [1, S, V]), [hid]: new Tensor("float32", h, [1, S, H]) }; },
    async generate({ input_ids, streamer, max_new_tokens }) {
      streamer?.put([[1n, 2n]]); const out = [7n, 8n, 248046n, 9n].slice(0, max_new_tokens); for (const t of out) streamer?.put([[t]]); streamer?.end?.(); return null; },
    async dispose() { m.disposed = true; } };
  return m;
}
const m = mk(); const be = transformersJsFromModel(m, Tensor);
ok(be.kind === "scan" && be.hiddenName === "type_as_80", "kind + hiddenName detected by name");
const r = await be.prefill([1, 2, 3, 4], { positions: [2, 0, 2] });
ok(r.hiddens.length === 3 && r.hiddens[0][1] === 21 && r.hiddens[1][0] === 0 && r.hiddens[2][2] === 22, "hiddens per position, duplicates");
ok(r.lastLogits.length === V && r.lastLogits[3] === 4, "lastLogits = last position");
const f = m.forwards[0];
ok(f.input_ids.dims.join() === "1,4" && f.attention_mask.data.every((x) => x === 1n), "forward with input_ids + attention_mask");
const r0 = await be.prefill([1, 2], { positions: [] });
ok(r0.hiddens.length === 0 && r0.lastLogits.length === V, "empty positions");
const be2 = transformersJsFromModel(mk("mul_13450"), Tensor);
ok(be2.hiddenName === "mul_13450", "mul_13450 too");
const noHid = transformersJsFromModel({ ...mk(), sessions: { model: { outputNames: ["logits"] } } }, Tensor);
ok(noHid.hiddenName === null, "no hidden -> null");
let bad = false; try { await be.prefill([1], { positions: [5] }); } catch { bad = true; }
ok(bad, "out-of-range position throws");
const g = []; for await (const id of be.generate([1, 2], { maxNewTokens: 4, temperature: 0 })) g.push(id);
ok(JSON.stringify(g) === "[7,8]", "generate stops at EOS (without emitting it): " + JSON.stringify(g));
await be.dispose(); ok(m.disposed, "dispose");
// long async generation that honors stopping_criteria; tracks the active run
function mkLong() {
  const m = mk(); m.running = false; m.steps = 0; m.overlap = false;
  m.generate = async ({ streamer, max_new_tokens, stopping_criteria }) => {
    m.running = true; streamer.put([[1n, 2n]]);
    for (let i = 0; i < max_new_tokens; i++) {
      await new Promise((r) => setTimeout(r, 2)); m.steps++;
      streamer.put([[BigInt(10 + i)]]);
      if (stopping_criteria.some((c) => c([[0]])[0])) break;
    }
    streamer.end(); await new Promise((r) => setTimeout(r, 5)); m.running = false;
  };
  const origFwd = m.forward; m.forward = async (f) => { if (m.running) m.overlap = true; return origFwd(f); };
  return m;
}
{ // (a) abort via signal
  const m = mkLong(), b = transformersJsFromModel(m, Tensor), ac = new AbortController(), got = [];
  for await (const id of b.generate([1], { maxNewTokens: 200, signal: ac.signal })) { got.push(id); if (got.length === 3) ac.abort(); }
  ok(!m.running && m.steps < 20, `abort: library stopped and settled (steps=${m.steps}, got=${got.length})`);
  await b.prefill([1, 2], { positions: [] }); ok(!m.overlap, "abort: next prefill does not overlap the run");
}
{ // (b) early break
  const m = mkLong(), b = transformersJsFromModel(m, Tensor); let n = 0;
  for await (const id of b.generate([1], { maxNewTokens: 200 })) { if (++n === 2) break; }
  ok(!m.running && m.steps < 20, `break: library stopped and settled (steps=${m.steps})`);
  await b.prefill([1, 2], { positions: [] }); ok(!m.overlap, "break: next prefill does not overlap the run");
}
{ // library error propagates
  const m = mk(); m.generate = async () => { throw new Error("boom"); }; const b = transformersJsFromModel(m, Tensor);
  let msg = ""; try { for await (const _ of b.generate([1], {})) {} } catch (e) { msg = e.message; }
  ok(msg === "boom", "generate error propagates");
}
console.log(`\n${pass} passed, ${fail} failed`); process.exit(fail ? 1 : 0);
