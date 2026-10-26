// test_decode_only_mock.mjs — createDecodeOnly with a FAKE session (no model): metadata, mapping, dtype, errors.
import { createDecodeOnly, tensorToF32, halfToNumber, pastNameFor, cacheDims } from "../dist/decodeOnly.js";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };
const throws = async (f, re) => { try { await f(); return false; } catch (e) { return re.test(String(e.message)); } };

class Tensor { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }
const ort = { Tensor };
const V = 6, H = 4;

// fake session: echoes each present_* tagged with the step, deterministic logits, hidden = [step, ...]
function mockSession({ inputs, outputs, meta, metaAsObject = false }) {
  const s = { inputNames: inputs, outputNames: outputs, feeds: [], fetches: [], released: false,
    release: async () => { s.released = true; },
    async run(feed, names) {
      s.feeds.push(feed); s.fetches.push(names);
      const step = s.feeds.length;
      const r = {};
      for (const n of names) {
        if (n === "logits") { const d = new Float32Array(V); d[step % V] = 5; r[n] = new Tensor("float32", d, [1, 1, V]); }
        else if (/hidden|norm|mul_|type_as/.test(n)) r[n] = new Tensor("float16", new Uint16Array(H).fill(0x3c00), [1, 1, H]);
        else r[n] = new Tensor("float32", new Float32Array([step]), [1]);
      }
      return r;
    } };
  s.inputMetadata = metaAsObject ? Object.fromEntries(meta.map(({ name, ...m }) => [name, m])) : meta;
  return s;
}
const M = (name, type, shape) => ({ name, isTensor: true, type, shape });
const opt = () => ({
  inputs: ["input_ids", "past_conv.0", "past_recurrent.0", "past_key_values.3.key", "past_key_values.3.value"],
  outputs: ["logits", "present_conv.0", "present_recurrent.0", "present.3.key", "present.3.value", "/model/norm/Mul_1_output_0"],
  meta: [M("input_ids", "int64", ["batch_size", "sequence_length"]), M("past_conv.0", "float32", ["batch_size", 8, 4]),
    M("past_recurrent.0", "float32", ["batch_size", 2, 3, 3]), M("past_key_values.3.key", "float32", ["batch_size", 4, "past_sequence_length", 5]),
    M("past_key_values.3.value", "float32", ["batch_size", 4, "past_sequence_length", 5])],
});

console.log("cache shapes (pastLen):");
ok(JSON.stringify(cacheDims("past_key_values.3.key", ["batch_size", 4, "past_sequence_length", 256], 1)) === "[1,4,1,256]", "KV pastLen=1");
ok(JSON.stringify(cacheDims("past_key_values.3.key", ["batch_size", 4, "past_sequence_length", 256], 0)) === "[1,4,0,256]", "KV pastLen=0 (batch stays 1)");
ok(JSON.stringify(cacheDims("past_conv.0", ["batch_size", 8192, 4], 0)) === "[1,8192,4]", "conv: pastLen has no effect");
ok(JSON.stringify(cacheDims("past_recurrent.0", ["batch_size", 32, 128, 128], 0)) === "[1,32,128,128]", "numeric recurrent unchanged");
for (const pl of [undefined, 0, 1]) {
  const s = mockSession(opt());
  const be = await createDecodeOnly(s, null, { ort, ...(pl === undefined ? {} : { pastLen: pl }) });
  await be.prefill([7], { positions: [] });
  const f = s.feeds[0], want = pl ?? 1;
  ok(JSON.stringify(f["past_key_values.3.key"].dims) === JSON.stringify([1, 4, want, 5]) && f["past_key_values.3.key"].data.length === 4 * want * 5, `feed KV pastLen=${pl ?? "default"} -> dims [1,4,${want},5]`);
  ok(JSON.stringify(f["past_conv.0"].dims) === "[1,8,4]" && f.input_ids.dims.join() === "1,1" && f.input_ids.type === "int64", "conv/input_ids");
}

console.log("\nmetadata as object + present->past:");
{
  const o = opt(); o.metaAsObject = true;
  const s = mockSession(o);
  const be = await createDecodeOnly(s, null, { ort });
  ok(be.kind === "decode-only" && be.hiddenName === "/model/norm/Mul_1_output_0", "object form accepted; hidden detected");
  await be.prefill([1, 2], { positions: [] });
  ok(s.feeds[1]["past_conv.0"].data[0] === 1 && s.feeds[1]["past_key_values.3.value"].data[0] === 1, "step 2 receives step 1's present (past_conv and KV)");
  ok(!s.fetches[0].includes("/model/norm/Mul_1_output_0"), "hidden is not fetched when not requested");
  await be.dispose(); ok(s.released, "dispose calls session.release");
}

console.log("\nlegacy so/si graph:");
{
  const s = mockSession({ inputs: ["input_ids", "cache_position", "si0", "si1"], outputs: ["logits", "type_as_80", "so0", "so1"],
    meta: [M("input_ids", "int64", [1, 1]), M("cache_position", "int64", [1]), M("si0", "float16", [1, 4, 0, 8]), M("si1", "float32", [1, 2])] });
  const be = await createDecodeOnly(s, null, { ort });
  ok(be.hiddenName === "type_as_80", "hidden type_as_80");
  const r = await be.prefill([3, 4, 5], { positions: [2, 0, 2] });
  ok(r.hiddens.length === 3 && r.hiddens[0] !== r.hiddens[1] && r.hiddens[0][0] === 1 && r.lastLogits.length === V, "hiddens in requested order, duplicate returns a duplicate entry");
  const f = s.feeds[0];
  ok(f.si0.type === "float16" && f.si0.dims.join() === "1,4,0,8", "fp16 cache (dtype from metadata, numeric dims)");
  ok(f.cache_position.type === "int64" && s.feeds[2].cache_position.data[0] === 2n, "cache_position fed");
  ok(s.feeds[1].si0.data[0] === 1, "so0 -> si0");
}

console.log("\nalternative hidden outputs:");
for (const hn of ["mul_13450", "hidden_states"]) {
  const o = opt(); o.outputs = ["logits", "present_conv.0", "present_recurrent.0", "present.3.key", "present.3.value", hn];
  const be = await createDecodeOnly(mockSession(o), null, { ort });
  ok(be.hiddenName === hn, "detecta " + hn);
}

console.log("\nerror paths:");
{
  const o = opt(); o.outputs = o.outputs.slice(0, 5);
  const be = await createDecodeOnly(mockSession(o), null, { ort });
  ok(be.hiddenName === null, "no hidden output -> hiddenName null");
  ok(await throws(() => be.prefill([1], { positions: [0] }), /does not expose a hidden/), "prefill with positions but no hidden throws");
  const r = await be.prefill([1], { positions: [] }); ok(r.hiddens.length === 0 && r.lastLogits.length === V, "works without hidden when no positions");
  const be2 = await createDecodeOnly(mockSession(opt()), null, { ort });
  ok(await throws(() => be2.prefill([1, 2], { positions: [5] }), /out of/), "out-of-range position throws");
  ok(await throws(() => be2.prefill([], { positions: [] }), /empty/), "empty ids throws");
  const u = opt(); u.inputs.push("foo_state"); u.meta.push(M("foo_state", "float32", [1, 2]));
  ok(await throws(() => createDecodeOnly(mockSession(u), null, { ort }), /without a matching present/), "unknown cache throws");
  const nm = opt(); nm.meta = nm.meta.slice(0, 2);
  ok(await throws(() => createDecodeOnly(mockSession(nm), null, { ort }), /no shape metadata/), "missing shape metadata throws");
  const am = opt(); am.inputs.push("attention_mask");
  ok(await throws(() => createDecodeOnly(mockSession(am), null, { ort }), /attention_mask/), "attention_mask throws");
  const dt = opt(); dt.meta[1] = M("past_conv.0", "int8", ["batch_size", 8, 4]);
  ok(await throws(() => createDecodeOnly(mockSession(dt), null, { ort }).then((b) => b.prefill([1], { positions: [] })), /dtype/), "unsupported cache dtype throws");
}

console.log("\nfp16 / conversion:");
{
  const u = new Uint16Array([0x3c00, 0xc000, 0x3800, 0x7bff, 0x7c00, 0x0001]);
  const f = tensorToF32(u, "float16");
  ok(f[0] === 1 && f[1] === -2 && f[2] === 0.5 && f[3] === 65504 && f[4] === Infinity, "Uint16Array fp16 decoded bit by bit (0x3c00 -> 1, not 15360)");
  ok(Math.abs(f[5] - Math.pow(2, -24)) < 1e-12 && halfToNumber(0x3c00) === 1, "subnormal / halfToNumber");
  ok(tensorToF32(new Float32Array([1.5, 2.5]), "float32")[1] === 2.5, "float32 direct");
  if (typeof Float16Array !== "undefined") ok(tensorToF32(new Float16Array([1.5, -2]), "float16")[1] === -2, "native Float16Array");
  const hs = await (await createDecodeOnly(mockSession(opt()), null, { ort })).prefill([1], { positions: [0] });
  ok(hs.hiddens[0].every((x) => x === 1), "fp16 hidden from the graph arrives as 1.0 (numeric)");
  ok(pastNameFor("present_conv.3") === "past_conv.3" && pastNameFor("present.7.value") === "past_key_values.7.value"
    && pastNameFor("present_recurrent.0") === "past_recurrent.0" && pastNameFor("so5") === "si5" && pastNameFor("logits") === null, "present -> past");
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
