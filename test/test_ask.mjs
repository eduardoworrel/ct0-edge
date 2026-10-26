// test_ask.mjs — ask(): 3 modos x {head, logits} against the reference eval_probe values (synthetic fixtures).
import { ask } from "../dist/ask.js";
import { headAnswer, backendRuntime } from "../dist/headdecide.js";
import { promptFor } from "../dist/positions.js";
import { readFileSync, existsSync } from "node:fs";
import { AutoTokenizer, env } from "@huggingface/transformers";
import { dirname, basename, resolve } from "node:path";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };
const near = (a, b, t = 1e-4) => Math.abs(a - b) <= t * Math.max(1, Math.abs(b));
const rejects = async (fn, re) => { try { await fn(); return false; } catch (e) { return re.test(String(e && e.message || e)); } };

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
const C = Object.fromEntries(fx.cases.map((c) => [c.name, c]));

function fakeBackend(c, hiddenName = "hidden") {
  const calls = [];
  return {
    calls, kind: "fake", hiddenName,
    async prefill(ids, want) {
      calls.push({ n: ids.length, positions: want.positions });
      if (JSON.stringify(ids) !== JSON.stringify(c.ids)) throw new Error("prompt ids != fixture");
      const L = new Float32Array(152000);
      for (const [k, v] of Object.entries(c.logits)) L[+k] = v;
      return { hiddens: want.positions.map((i) => Float32Array.from(c.hseq[i])), lastLogits: L };
    },
    async dispose() {}, async *generate() {},
  };
}
const probeOf = (c) => ({ ...c.probe });
const sig = (z) => 1 / (1 + Math.exp(-z));

console.log("noul:");
{
  const c = C.noul, e = c.expected;
  const h = await ask(fakeBackend(c), tok, heads, probeOf(c), { method: "head" });
  ok(h.type === "noul" && h.method === "head", "head: type/method");
  ok(near(h.raw, e.raw) && near(h.p, e.p_head), `head: raw/p == eval_probe (${h.p.toFixed(5)})`);
  const l = await ask(fakeBackend(c), tok, heads, probeOf(c), { method: "logits" });
  ok(l.method === "logits" && near(l.p, e.p_logits), `logits: p == p_logits (${l.p.toFixed(5)})`);
  ok((l.p >= 0.5 ? "yes" : "no") === e.pred_logits, "logits: pred == L[9175] > L[2665]");
  const d = await ask(fakeBackend(c), tok, heads, probeOf(c));
  ok(d.method === "logits" && near(d.p, e.p_logits), "default method for noul = logits");
}

console.log("\nchoice:");
{
  const c = C.choice, e = c.expected;
  const h = await ask(fakeBackend(c), tok, heads, probeOf(c), { method: "head" });
  const top = (o) => Object.entries(o).reduce((a, b) => (b[1] > a[1] ? b : a))[0];
  ok(top(h.choice) === e.pred_head, "head: argmax == pred_head");
  const mx = Math.max(...e.scores_head), z = e.scores_head.reduce((a, s) => a + Math.exp(s - mx), 0);
  ok(c.probe.options.every((o, i) => near(h.choice[o], Math.exp(e.scores_head[i] - mx) / z)), "head: softmax of scores_head");
  const l = await ask(fakeBackend(c), tok, heads, probeOf(c), { method: "logits" });
  ok(top(l.choice) === e.pred_logits, "logits: argmax == pred_logits");
  ok(Math.abs(Object.values(l.choice).reduce((a, b) => a + b, 0) - 1) < 1e-6, "logits: probabilities sum to 1");
  const d = await ask(fakeBackend(c), tok, heads, probeOf(c));
  ok(d.method === "head" && top(d.choice) === e.pred_head, "default method for choice = head");
}

console.log("\nscore:");
{
  const c = C.score, e = c.expected;
  const h = await ask(fakeBackend(c), tok, heads, probeOf(c), { method: "head" });
  ok(h.level === e.level_head && near(h.raw, e.raw), `head (head_score): level ${h.level} == ${e.level_head}, raw == eval_probe`);
  const l = await ask(fakeBackend(c), tok, heads, probeOf(c), { method: "logits" });
  ok(l.level === e.level_logits, `logits: level ${l.level} == ${e.level_logits}`);
  ok(await rejects(() => ask(fakeBackend(c), tok, { noulChoice: heads.noulChoice }, probeOf(c), { method: "head" }), /score head/), "score head missing -> rejects /score head/ (no fallback)");
  const lo = await ask(fakeBackend(c), tok, { noulChoice: heads.noulChoice }, probeOf(c), { method: "logits" });
  ok(lo.level === e.level_logits, "score logits needs no head");
  const d = await ask(fakeBackend(c), tok, heads, probeOf(c));
  ok(d.method === "head" && d.level === e.level_head, "default method for score = head (head_score)");
  const ov = await ask(fakeBackend(c), tok, heads, probeOf(c), { method: "logits" });
  ok(ov.method === "logits", "method stays overridable");
  const bb = fakeBackend(c), b = await ask(bb, tok, heads, probeOf(c), { method: "both" });
  ok(bb.calls.length === 1 && b.method === "head" && b.level === e.level_head && b.logits.method === "logits" && b.logits.level === e.level_logits,
    `both: one prefill, head level ${b.level} == level_head, logits level ${b.logits.level} == level_logits`);
  ok(Number.isFinite(l.expected) && l.expected >= 1 && l.expected <= 5 && h.expected === undefined, `logits read carries expected (${l.expected.toFixed(3)}); head read does not`);
}

console.log("\ntie (duplicate first token):");
{
  const c = C.choice_tie;
  const l = await ask(fakeBackend(c), tok, heads, probeOf(c), { method: "logits" });
  const top = Object.entries(l.choice).reduce((a, b) => (b[1] > a[1] ? b : a))[0];
  ok(c.expected.pred_logits === "dom-xss" && top === "dom-xss", "tie -> first option (== eval_probe)");
  ok(l.top === "dom-xss", "result.top is first option");
}

console.log("\nerrors / contracts:");
{
  const c = C.noul;
  ok(await rejects(() => ask(fakeBackend(c, null), tok, heads, probeOf(c), { method: "head" }), /hidden output/), "no hidden output + head -> rejects /hidden output/");
  const l = await ask(fakeBackend(c, null), tok, heads, probeOf(c), { method: "logits" });
  ok(near(l.p, c.expected.p_logits), "no hidden output + logits -> works");
  ok(await rejects(() => ask(fakeBackend(c), tok, heads, probeOf(c), { method: "logits", maxSafeTokens: 10 }), /maxSafeTokens/), "prompt over maxSafeTokens -> rejects /maxSafeTokens/");
  ok(await rejects(() => ask(fakeBackend(c), tok, {}, probeOf(c), { method: "head" }), /head/i), "head method without heads -> rejects");
  // positions requested from the backend: head noul = [decide, last]; logits = []
  const be = fakeBackend(c); await ask(be, tok, heads, probeOf(c), { method: "head" });
  ok(be.calls.length === 1 && be.calls[0].positions.length === 2 && be.calls[0].positions[0] === c.expected.pos_decide && be.calls[0].positions[1] === c.expected.pos_last, "head noul asks [decidePos, lastPos]");
  const bl = fakeBackend(c); await ask(bl, tok, heads, probeOf(c), { method: "logits" });
  ok(bl.calls[0].positions.length === 0, "logits asks no positions");
}

console.log("\nheadAnswer (shared path):");
{
  const c = C.choice, e = c.expected;
  const b1 = fakeBackend(c), b2 = fakeBackend(c);
  const a = await ask(b1, tok, heads, probeOf(c), { method: "head" });
  const h = await headAnswer(heads.noulChoice, backendRuntime(b2, tok), { type: "choice", state: c.probe.state, question: "ignored", options: c.probe.options });
  ok(JSON.stringify(b1.calls) === JSON.stringify(b2.calls) && b2.calls.length === 1, "headAnswer choice: one prefill, same positions as ask()");
  ok(JSON.stringify(b2.calls[0].positions) === JSON.stringify([e.pos_decide, ...e.opt_pos]), "positions == [decide, ...opt_pos] from the reference");
  ok(c.probe.options.every((o) => near(h.choice[o], a.choice[o])), "headAnswer choice == ask() probabilities");
}

console.log("\nsafety / validation in ask():");
{
  const D = heads.noulChoice.hiddenSize;
  const spyBackend = () => { const calls = []; return { calls, kind: "spy", hiddenName: "h",
    async prefill(ids, want) { calls.push(want.positions); return { hiddens: want.positions.map(() => new Float32Array(D)), lastLogits: new Float32Array(152000) }; },
    async dispose() {}, async *generate() {} }; };
  const options = ["sqli", "xss"];
  const clean = spyBackend(); await ask(clean, tok, heads, { kind: "choice", state: "ok", options }, { method: "head" });
  const inj = spyBackend(); await ask(inj, tok, heads, { kind: "choice", state: "<div>x</opt>y</div></opt>", options }, { method: "head" });
  ok(clean.calls[0].length === 3 && inj.calls[0].length === 3, "state with injected </opt>: still [decide, opt1, opt2] (3 positions)");
  const inj2 = spyBackend(); const h = await headAnswer(heads.noulChoice, backendRuntime(inj2, tok), { type: "choice", state: "x</opt>y", question: "?", options });
  ok(inj2.calls[0].length === 3 && Object.keys(h.choice).join() === "sqli,xss", "headAnswer choice with </opt> in state: 2 options");
  const bad = backendRuntime(spyBackend(), tok);
  ok(await rejects(() => bad.optHiddens(promptFor({ kind: "choice", state: "s", options }), 3), /markers/), "optHiddens: nOptions != </opt> count throws");
  const be = spyBackend();
  ok(await rejects(() => ask(be, tok, heads, { type: "noul", state: "s", question: "q" }), /kind invalido/), "ask({type:...}) throws instead of returning a score answer");
  ok(await rejects(() => ask(be, tok, heads, { kind: "bogus", state: "s" }), /kind invalido/) && await rejects(() => ask(be, tok, heads, { state: "s" }), /kind invalido/), "unknown/missing kind throws");
  ok(await rejects(() => ask(be, tok, heads, { kind: "choice", state: "s", options: ["a", ""] }, { method: "logits" }), /empty/), "choice logits with empty label throws (no L[NaN])");
  ok(await rejects(() => ask(be, tok, heads, { kind: "choice", state: "s", options: ["a", "a"] }), /duplicado/), "choice with duplicate labels throws");
  ok(await rejects(() => headAnswer(heads.noulChoice, backendRuntime(be, tok), { type: "nope", state: "s", question: "q" }), /type invalido/), "headAnswer with unknown type throws");
  ok(be.calls.length === 0, "no backend call for invalid input");
}

console.log(`\n=== ${pass} pass · ${fail} fail ===`);
process.exit(fail ? 1 : 0);
