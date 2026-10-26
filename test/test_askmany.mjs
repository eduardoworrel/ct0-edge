// test_askmany.mjs — askMany (shared-prefix snapshot), method "both", score `expected` and `scoreCuts`, plus the
// snapshot / prefillFrom of the decode-only and genai backends over FAKE sessions whose outputs depend on the carried state.
import { ask, askMany, levelFromCuts, DIGIT_IDS } from "../dist/ask.js";
import { createDecodeOnly } from "../dist/decodeOnly.js";
import { genaiFromSession } from "../dist/genai.js";
import { promptFor } from "../dist/positions.js";
import { readFileSync, existsSync } from "node:fs";
import { AutoTokenizer, env } from "@huggingface/transformers";
import { dirname, basename, resolve } from "node:path";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };
const rejects = async (fn, re) => { try { await fn(); return false; } catch (e) { return re.test(String(e && e.message || e)); } };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

if (!process.env.CT0_TOK_DIR || !existsSync(resolve(process.env.CT0_TOK_DIR))) {
  console.log("SKIP %s: set CT0_TOK_DIR to a local tokenizer directory", import.meta.url.split("/").pop());
  process.exit(0);
}
const tokDir = resolve(process.env.CT0_TOK_DIR);
env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = dirname(tokDir) + "/";
const tok = await AutoTokenizer.from_pretrained(basename(tokDir));
const fx = JSON.parse(readFileSync(new URL("./fixtures/ask_expected.json", import.meta.url)));
const mkHead = (j) => ({ q: Float32Array.from(j.q), qb: Float32Array.from(j.qb), k: Float32Array.from(j.k), kb: Float32Array.from(j.kb),
  headDim: j.headDim, hiddenSize: j.hiddenSize, temperature: j.temperature, scoreOffset: j.scoreOffset });
const heads = { noulChoice: mkHead(fx.headNoulChoice), score: mkHead(fx.headScore) };
const D = fx.headNoulChoice.hiddenSize, VOC = 248320;

// Deterministic "model": the hidden at position p depends on ids[0..p], the logits on the whole prompt.
const mix = (h, id) => { h = Math.imul(h ^ (id + 0x9e37), 0x85ebca6b) >>> 0; return (h ^ (h >>> 13)) >>> 0; };
function rng(seed) { let a = seed >>> 0 || 1; return () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const vec = (seed, n, scale = 1) => { const r = rng(seed), v = new Float32Array(n); for (let i = 0; i < n; i++) v[i] = (r() * 2 - 1) * scale; return v; };
function model(ids, positions) {
  const hs = []; let h = 7; for (const id of ids) { h = mix(h, id); hs.push(h); }
  return { hiddens: positions.map((p) => vec(hs[p], D)), lastLogits: vec(hs[ids.length - 1] ^ 0xabcdef, VOC, 4) };
}
function fakeBackend({ snapshot = true, hiddenName = "h" } = {}) {
  const calls = { prefill: [], snapshot: [], prefillFrom: [], disposed: 0 };
  const be = { kind: "fake", hiddenName, calls,
    async prefill(ids, want) { calls.prefill.push({ n: ids.length, positions: want.positions }); return model(ids, want.positions); },
    async dispose() {}, async *generate() {} };
  if (snapshot) {
    be.snapshot = async (ids) => { calls.snapshot.push(ids.length); const h = { ids: ids.slice(), dispose() { calls.disposed++; } }; return h; };
    be.prefillFrom = async (h, ids, want) => {
      if (!h.ids.every((x, i) => ids[i] === x)) throw new Error("fake: ids do not extend the snapshot");
      if (want.positions.some((p) => p < h.ids.length)) throw new Error("fake: position inside the snapshot");
      calls.prefillFrom.push({ n: ids.length - h.ids.length, positions: want.positions });
      return model(ids, want.positions);
    };
  }
  return be;
}

const state = "function render(q){ document.getElementById('out').innerHTML = q; fetch('/api?x='+q); }";
const qs = [
  { kind: "noul", q: "Does this code write untrusted input into the DOM?" },
  { kind: "choice", options: ["dom-xss", "sqli", "open-redirect"] },
  { kind: "score" },
];
const full = qs.map((q) => ({ ...q, state }));
const ntok = full.map((q) => tok.encode(promptFor(q), { add_special_tokens: false }).length);

console.log("askMany == ask one by one:");
for (const method of [undefined, "head", "logits", "both"]) {
  const opts = method ? { method } : {};
  const one = []; for (const q of full) one.push(await ask(fakeBackend(), tok, heads, q, opts));
  const be = fakeBackend();
  const many = await askMany(be, tok, heads, state, qs, opts);
  ok(same(many, one), `method ${method ?? "default"}: askMany results identical to ask x${qs.length} (noul, choice, score)`);
  ok(be.calls.snapshot.length === 1 && be.calls.prefillFrom.length === qs.length && be.calls.prefill.length === 0 && be.calls.disposed === 1,
    `method ${method ?? "default"}: 1 prefix snapshot + ${qs.length} continuations, no full prefill, snapshot released`);
  const k = be.calls.snapshot[0];
  ok(be.calls.prefillFrom.every((c, i) => c.n === ntok[i] - k), `method ${method ?? "default"}: each continuation feeds only its suffix (prefix ${k} tok)`);
}
{
  const be = fakeBackend(); await askMany(be, tok, heads, state, qs);
  const k = be.calls.snapshot[0];
  ok(k > tok.encode(state, { add_special_tokens: false }).length, `prefix covers the system turn and the whole state (${k} tok)`);
  ok(be.calls.prefillFrom.every((c) => c.positions.every((p) => p >= k)), "every wanted position is after the prefix");
}

console.log("\nfallback (backend without snapshot):");
{
  const one = []; for (const q of full) one.push(await ask(fakeBackend({ snapshot: false }), tok, heads, q));
  const be = fakeBackend({ snapshot: false });
  const many = await askMany(be, tok, heads, state, qs);
  ok(same(many, one), "results identical to ask one by one");
  ok(be.calls.prefill.length === qs.length && same(be.calls.prefill.map((c) => c.n), ntok), "one full prefill per question");
  const b1 = fakeBackend(); const r1 = await askMany(b1, tok, heads, state, [qs[0]]);
  ok(b1.calls.snapshot.length === 0 && b1.calls.prefill.length === 1 && same(r1, [await ask(fakeBackend(), tok, heads, full[0])]), "a single question: plain prefill, no snapshot");
  ok(same(await askMany(fakeBackend(), tok, heads, state, []), []), "no questions -> []");
  const dup = fakeBackend(); const rd = await askMany(dup, tok, heads, state, [qs[0], qs[0]], { method: "head" });
  ok(same(rd[0], rd[1]) && dup.calls.snapshot[0] <= Math.min(...dup.calls.prefillFrom.flatMap((c) => c.positions)), "identical questions: prefix stops before the first wanted position");
}

console.log("\naskMany validation:");
{
  const be = fakeBackend();
  ok(await rejects(() => askMany(be, tok, heads, state, [{ ...qs[0], state: "other" }]), /state/), "question with a different state rejects");
  ok(same(await askMany(fakeBackend(), tok, heads, state, [{ ...qs[0], state }]), [await ask(fakeBackend(), tok, heads, full[0])]), "question repeating the same state is accepted");
  ok(await rejects(() => askMany(be, tok, heads, state, [qs[0], { kind: "bogus" }]), /kind/), "invalid question rejects");
  ok(await rejects(() => askMany(be, tok, heads, state, qs, { maxSafeTokens: 10 }), /maxSafeTokens/), "over maxSafeTokens rejects");
  ok(be.calls.snapshot.length === 0 && be.calls.prefill.length === 0, "nothing runs when a question is invalid (all validated first)");
  const failing = fakeBackend(); failing.prefillFrom = async () => { throw new Error("boom"); };
  ok(await rejects(() => askMany(failing, tok, heads, state, qs), /boom/) && failing.calls.disposed === 1, "an error in a continuation still releases the snapshot");
}

console.log("\nmethod \"both\" (one forward, two reads):");
{
  for (const q of full) {
    const be = fakeBackend();
    const b = await ask(be, tok, heads, q, { method: "both" });
    const h = await ask(fakeBackend(), tok, heads, q, { method: "head" });
    const l = await ask(fakeBackend(), tok, heads, q, { method: "logits" });
    const { logits, ...headPart } = b;
    ok(same(headPart, h) && same(logits, l), `${q.kind}: head part == method head, .logits == method logits`);
    ok(be.calls.prefill.length === 1 && be.calls.prefill[0].positions.length > 0, `${q.kind}: a single prefill`);
  }
  ok(await rejects(() => ask(fakeBackend({ hiddenName: null }), tok, heads, full[0], { method: "both" }), /hidden output/), "both without a hidden output rejects");
  ok(await rejects(() => ask(fakeBackend(), tok, {}, full[0], { method: "both" }), /head/), "both without a head rejects");
}

console.log("\nscore expected (logits) by hand:");
{
  // digit logits ln[1,2,3,4,10] -> p = [.05,.1,.15,.2,.5] -> expected = .05+.2+.45+.8+2.5 = 4.0; argmax level 5
  const L = new Float32Array(VOC); [1, 2, 3, 4, 10].forEach((w, i) => { L[DIGIT_IDS[i]] = Math.log(w); });
  const be = { kind: "fixed", hiddenName: null, async prefill() { return { hiddens: [], lastLogits: L }; }, async dispose() {}, async *generate() {} };
  const r = await ask(be, tok, heads, full[2], { method: "logits" });
  ok(Math.abs(r.expected - 4.0) < 1e-5, `expected = ${r.expected.toFixed(6)} == 4.0`);
  ok(r.level === 5, "level without cuts = argmax digit (5)");
  const c = await ask(be, tok, heads, full[2], { method: "logits", scoreCuts: [1.5, 2.5, 3.5, 4.5] });
  ok(c.level === 4 && Math.abs(c.expected - 4.0) < 1e-5, "cuts [1.5,2.5,3.5,4.5]: expected 4.0 -> level 4");
  const c2 = await ask(be, tok, heads, full[2], { method: "logits", scoreCuts: [1.5, 2.5, 3.5, 3.9] });
  ok(c2.level === 5, "cuts [1.5,2.5,3.5,3.9]: expected 4.0 -> level 5");
  const fxs = fx.cases.find((x) => x.name === "score");
  const Lf = new Float32Array(VOC); for (const [k, v] of Object.entries(fxs.logits)) Lf[+k] = v;
  const d = DIGIT_IDS.map((i) => Lf[i]), mx = Math.max(...d), e = d.map((x) => Math.exp(x - mx)), z = e.reduce((a, b) => a + b, 0);
  const want = e.reduce((a, x, i) => a + (i + 1) * x / z, 0);
  const bf = { ...be, async prefill() { return { hiddens: [], lastLogits: Lf }; } };
  const rf = await ask(bf, tok, heads, full[2], { method: "logits" });
  ok(Math.abs(rf.expected - want) < 1e-6 && rf.level === fxs.expected.level_logits, `fixture logits: expected ${rf.expected.toFixed(4)}, level unchanged (${rf.level})`);
  const n = await ask(bf, tok, heads, full[0], { method: "logits" });
  ok(n.expected === undefined, "noul has no expected");
}

console.log("\nscoreCuts on the head (raw):");
{
  const sig = (z) => 1 / (1 + Math.exp(-z)), logit = (p) => Math.log(p / (1 - p));
  const defaultCuts = [0.125, 0.375, 0.625, 0.875].map(logit);
  let agree = true;
  for (let x = -6; x <= 6; x += 0.01) if (levelFromCuts(x, defaultCuts) !== Math.max(1, Math.min(5, Math.round(sig(x) * 4 + 1)))) agree = false;
  ok(agree, "levelFromCuts with logit([.125,.375,.625,.875]) == the default head mapping on [-6, 6]");
  ok(levelFromCuts(-1, [0, 1, 2, 3]) === 1 && levelFromCuts(0, [0, 1, 2, 3]) === 2 && levelFromCuts(2.5, [0, 1, 2, 3]) === 4 && levelFromCuts(9, [0, 1, 2, 3]) === 5, "levelFromCuts = 1 + number of cuts <= value");
  const c = fx.cases.find((x) => x.name === "score");
  const be = { kind: "fx", hiddenName: "h", async prefill(ids, want) {
    const L = new Float32Array(VOC); for (const [k, v] of Object.entries(c.logits)) L[+k] = v;
    return { hiddens: want.positions.map((i) => Float32Array.from(c.hseq[i])), lastLogits: L }; }, async dispose() {}, async *generate() {} };
  const raw = c.expected.raw; // -3.84 -> default level 1
  const base = await ask(be, tok, heads, { ...c.probe }, { method: "head" });
  ok(base.level === c.expected.level_head && base.expected === undefined, `no cuts: head level ${base.level} (default mapping), no expected on the head`);
  const up = await ask(be, tok, heads, { ...c.probe }, { method: "head", scoreCuts: [raw - 3, raw - 2, raw - 1, raw + 1] });
  ok(up.level === 4 && Math.abs(up.raw - raw) < 1e-4, "cuts below raw lift the level (1 -> 4)");
  const both = await ask(be, tok, heads, { ...c.probe }, { method: "both", scoreCuts: { head: [raw - 3, raw - 2, raw - 1, raw + 1], logits: [1.5, 2.5, 3.5, 4.5] } });
  ok(both.level === 4 && both.logits.level === Math.min(5, 1 + [1.5, 2.5, 3.5, 4.5].filter((x) => both.logits.expected >= x).length), "both: {head, logits} cuts apply to each read");
  ok(await rejects(() => ask(be, tok, heads, { ...c.probe }, { method: "both", scoreCuts: [0, 1, 2, 3] }), /scoreCuts/), "both + plain array rejects (two scales)");
  ok(await rejects(() => ask(be, tok, heads, { ...c.probe }, { method: "head", scoreCuts: [0, 1, 2] }), /scoreCuts/), "wrong number of cuts rejects");
  ok(await rejects(() => ask(be, tok, heads, { ...c.probe }, { method: "head", scoreCuts: [0, 2, 1, 3] }), /scoreCuts/), "non-ascending cuts reject");
  ok(await rejects(() => ask(be, tok, heads, { ...c.probe }, { method: "logits", scoreCuts: [0, 1, NaN, 3] }), /scoreCuts/), "non-finite cut rejects");
  const k3 = await ask(be, tok, heads, { ...c.probe, levels: 3 }, { method: "head", scoreCuts: [raw - 1, raw + 1] });
  ok(k3.level === 2, "levels 3 takes 2 cuts");
  const nc = await ask(be, tok, heads, { kind: "noul", state: "x", q: "y" }, { method: "logits", scoreCuts: [0, 1] });
  ok(nc.type === "noul", "scoreCuts are ignored by noul / choice");
}

// ------------------------------------------------------------------ backend snapshots over fake sessions
class Tensor {
  constructor(type, data, dims, location = "cpu") { this.type = type; this.data = data; this._d = data; this.dims = dims; this.location = location; this.disposed = false; }
  async getData() { return this._d; }
  dispose() { this.disposed = true; }
}
const ort = { Tensor };
const V = 8, H = 4;
const nxt = (h, id) => (h * 31 + id + 1) % 1000003; // fits a float32 exactly
const outVec = (h, n, salt) => Float32Array.from({ length: n }, (_, j) => ((h * (j + salt)) % 977) / 977);

console.log("\ndecode-only snapshot / prefillFrom:");
{
  const mk = () => {
    const s = { steps: 0, inputNames: ["input_ids", "past_conv.0", "past_key_values.0.key"],
      outputNames: ["logits", "present_conv.0", "present.0.key", "/model/norm/Mul_1_output_0"],
      inputMetadata: [{ name: "input_ids", type: "int64", shape: [1, 1] }, { name: "past_conv.0", type: "float32", shape: [1, 1] },
        { name: "past_key_values.0.key", type: "float32", shape: ["b", 1, "p", 1] }],
      async run(feed, names) {
        s.steps++;
        const h = nxt(feed["past_conv.0"].data[0], Number(feed.input_ids.data[0])), p = feed["past_key_values.0.key"].dims[2];
        const r = { logits: new Tensor("float32", outVec(h, V, 3), [1, 1, V]), "present_conv.0": new Tensor("float32", new Float32Array([h]), [1, 1]),
          "present.0.key": new Tensor("float32", new Float32Array(p + 1), [1, 1, p + 1, 1]) };
        if (names.includes("/model/norm/Mul_1_output_0")) r["/model/norm/Mul_1_output_0"] = new Tensor("float32", outVec(h, H, 5), [1, 1, H]);
        return r;
      } };
    return s;
  };
  const s = mk(), be = await createDecodeOnly(s, null, { ort });
  ok(typeof be.snapshot === "function" && typeof be.prefillFrom === "function", "decode-only exposes snapshot + prefillFrom");
  const pre = [11, 12, 13, 14], a = [...pre, 21, 22, 23], b = [...pre, 31, 32];
  const ra = await be.prefill(a, { positions: [5, 6] }), rb = await be.prefill(b, { positions: [4, 5] });
  s.steps = 0;
  const h = await be.snapshot(pre);
  ok(s.steps === pre.length && same(h.ids, pre), "snapshot runs the prefix once");
  const fa = await be.prefillFrom(h, a, { positions: [5, 6] }), fb = await be.prefillFrom(h, b, { positions: [4, 5] });
  ok(s.steps === pre.length + 3 + 2, `continuations run only their suffix (${s.steps} steps = 4 + 3 + 2)`);
  ok(same([...fa.lastLogits], [...ra.lastLogits]) && same(fa.hiddens.map((x) => [...x]), ra.hiddens.map((x) => [...x])), "A: hiddens + logits == full prefill");
  ok(same([...fb.lastLogits], [...rb.lastLogits]) && same(fb.hiddens.map((x) => [...x]), rb.hiddens.map((x) => [...x])), "B (after A, same snapshot): == full prefill");
  ok(await rejects(() => be.prefillFrom(h, [11, 12, 99, 14, 5], { positions: [] }), /prefix/), "ids that do not extend the prefix reject");
  ok(await rejects(() => be.prefillFrom(h, pre, { positions: [] }), /prefix/), "ids equal to the prefix (nothing to run) reject");
  ok(await rejects(() => be.prefillFrom(h, a, { positions: [2] }), /position/), "a position inside the prefix rejects");
  await h.dispose();
  ok(await rejects(() => be.prefillFrom(h, a, { positions: [] }), /snapshot/), "a released handle rejects");
  ok(await rejects(() => be.prefillFrom({ ids: pre, dispose() {} }, a, { positions: [] }), /snapshot/), "a foreign handle rejects");
  ok(await rejects(() => be.snapshot([]), /empty/), "empty snapshot rejects");
  await be.dispose();
  ok(await rejects(() => be.snapshot(pre), /dispose/), "snapshot after dispose rejects");
}

console.log("\ngenai snapshot / prefillFrom:");
{
  const mk = () => {
    const meta = [{ name: "input_ids", type: "int64", shape: [1, "s"] }, { name: "attention_mask", type: "int64", shape: [1, "t"] },
      { name: "position_ids", type: "int64", shape: [3, 1, "s"] }, { name: "past.0.conv", type: "float32", shape: [1, 1] },
      { name: "past_key_values.1.key", type: "float32", shape: [1, 1, "p", 1] }];
    const s = { calls: [], inputNames: meta.map((m) => m.name), inputMetadata: meta,
      outputNames: ["logits", "present.0.conv", "present.1.key", "hidden_states"],
      async run(feeds, fetches) {
        const S = feeds.input_ids.dims[1], ids = Array.from(feeds.input_ids._d, Number), past = feeds["past_key_values.1.key"].dims[2];
        for (const t of Object.values(feeds)) if (t.disposed) throw new Error("fed a disposed tensor");
        s.calls.push({ S, past, fetches: [...fetches], feeds });
        let h = feeds["past.0.conv"]._d[0]; const hs = [];
        for (const id of ids) { h = nxt(h, id); hs.push(h); }
        const out = { "present.0.conv": new Tensor("float32", new Float32Array([h]), [1, 1], "gpu-buffer"),
          "present.1.key": new Tensor("float32", new Float32Array(past + S), [1, 1, past + S, 1], "gpu-buffer") };
        if (fetches.includes("logits")) { const l = new Float32Array(S * V); hs.forEach((x, i) => l.set(outVec(x, V, 3), i * V)); out.logits = new Tensor("float32", l, [1, S, V], "gpu-buffer"); }
        if (fetches.includes("hidden_states")) { const d = new Float32Array(S * H); hs.forEach((x, i) => d.set(outVec(x, H, 5), i * H)); out.hidden_states = new Tensor("float32", d, [1, S, H], "gpu-buffer"); }
        return out;
      }, async release() {} };
    return s;
  };
  const s = mk(), be = genaiFromSession(s, ort, { chunk: 3 });
  ok(typeof be.snapshot === "function" && typeof be.prefillFrom === "function", "genai exposes snapshot + prefillFrom");
  const pre = [11, 12, 13, 14, 15], a = [...pre, 21, 22, 23, 24], b = [...pre, 31];
  const ra = await be.prefill(a, { positions: [5, 8] }), rb = await be.prefill(b, { positions: [5] });
  s.calls.length = 0;
  const h = await be.snapshot(pre);
  ok(s.calls.map((c) => c.S).join() === "3,2" && !s.calls.some((c) => c.fetches.includes("logits")), "snapshot: prefix in chunks, no logits read back");
  s.calls.length = 0;
  const fa = await be.prefillFrom(h, a, { positions: [5, 8] });
  ok(s.calls.map((c) => `${c.past}+${c.S}`).join() === "5+3,8+1", "continuation A starts at past = prefix length");
  const fb = await be.prefillFrom(h, b, { positions: [5] });
  ok(same([...fa.lastLogits], [...ra.lastLogits]) && same(fa.hiddens.map((x) => [...x]), ra.hiddens.map((x) => [...x])), "A: hiddens + logits == full prefill");
  ok(same([...fb.lastLogits], [...rb.lastLogits]) && same(fb.hiddens.map((x) => [...x]), rb.hiddens.map((x) => [...x])), "B (after A): == full prefill (snapshot not consumed)");
  const first = s.calls[0].feeds;
  ok(first["past.0.conv"].disposed === false && first["past_key_values.1.key"].disposed === false, "the snapshot's GPU tensors survive the continuations");
  ok(s.calls[1].feeds["past.0.conv"].disposed === true, "a continuation's own intermediate state is still freed");
  await h.dispose();
  ok(first["past.0.conv"].disposed === true && first["past_key_values.1.key"].disposed === true, "handle.dispose frees the snapshot's GPU tensors");
  ok(await rejects(() => be.prefillFrom(h, a, { positions: [] }), /snapshot/), "a released handle rejects");
  ok(await rejects(() => be.prefillFrom(h, [1, 2], { positions: [] }), /snapshot|prefix/), "bad ids reject");
}

console.log("\naskMany over decode-only (fake session, real tokenizer prompts):");
{
  const mkS = () => { const s = { steps: 0, inputNames: ["input_ids", "past_conv.0"], outputNames: ["logits", "present_conv.0", "/model/norm/Mul_1_output_0"],
    inputMetadata: [{ name: "input_ids", type: "int64", shape: [1, 1] }, { name: "past_conv.0", type: "float32", shape: [1, 1] }],
    async run(feed, names) {
      s.steps++;
      const h = nxt(feed["past_conv.0"].data[0], Number(feed.input_ids.data[0]));
      const L = new Float32Array(VOC); const v = outVec(h, 16, 3); [9175, 2665, ...DIGIT_IDS, ...Array.from({ length: 9 }, (_, i) => i)].forEach((id, j) => { L[id] = v[j % 16] * 4; });
      const r = { logits: new Tensor("float32", L, [1, 1, VOC]), "present_conv.0": new Tensor("float32", new Float32Array([h]), [1, 1]) };
      if (names.includes("/model/norm/Mul_1_output_0")) r["/model/norm/Mul_1_output_0"] = new Tensor("float32", outVec(h, D, 5), [1, 1, D]);
      return r;
    } }; return s; };
  const s1 = mkS(), s2 = mkS();
  const b1 = await createDecodeOnly(s1, tok, { ort }), b2 = await createDecodeOnly(s2, tok, { ort });
  const one = []; for (const q of full) one.push(await ask(b1, tok, heads, q, { method: "both" }));
  const many = await askMany(b2, tok, heads, state, qs, { method: "both" });
  ok(same(many, one), "askMany (snapshot) == ask x3 on the decode-only backend");
  ok(s1.steps === ntok.reduce((a, b) => a + b, 0) && s2.steps < s1.steps, `steps: ask x3 ${s1.steps}, askMany ${s2.steps}`);
}

console.log(`\n=== ${pass} pass · ${fail} fail ===`);
process.exit(fail ? 1 : 0);
