/**
 * ask.ts — typed System-1 (noul / choice / score) over any Backend, by two methods:
 *
 *  - `head`:   bilinear head (systemone.answer) over the prefill hiddens. Needs a hidden output in the graph.
 *  - `logits`: reads the logits of the LAST position directly:
 *      noul   -> L[Yes=9175] > L[No=2665]; p = sigmoid(L[Yes]-L[No])
 *      choice -> logit of the FIRST token of each label; argmax; tie -> first option
 *      score  -> argmax over the digits '1'..'5' (ids 16..20) + 1; `expected` = sum of level x softmax(digit logits)
 *  - `both`:   the head read plus the logits read (`result.logits`) from the SAME prefill.
 *
 * `askMany` asks several questions about one state: it prefills the shared token prefix once (`Backend.snapshot`) and
 * runs only each question's suffix (`Backend.prefillFrom`); backends without snapshots get independent `ask` calls.
 *
 * Prompt and positions come from positions.ts.
 */
import type { Backend, StateHandle } from "./backend.js";
import { promptFor, tokenOffsets, decidePos, optPositionsFor, type Probe } from "./positions.js";
import { answer, type HeadWeights } from "./systemone.js";

export const YES_ID = 9175, NO_ID = 2665;
export const DIGIT_IDS = [16, 17, 18, 19, 20];

export interface AskHeads {
  /** Original head (head.bin): noul and choice. */
  noulChoice?: HeadWeights;
  /** Score head (head_score.bin); required for score with method 'head' (no fallback). */
  score?: HeadWeights;
}

export type AskQuestion = Probe & { levels?: number };

/** Default read per mode (measured on a 50-probe set): noul from the logits (the head ties them on accuracy and its
 *  p saturates), choice from the head (0.84 vs 0.68 on q4), score from head_score.bin (exact 0.52 vs 0.40). */
export const DEFAULT_METHOD: Readonly<Record<"noul" | "choice" | "score", "head" | "logits">> = { noul: "logits", choice: "head", score: "head" };

export type AskManyQuestion = Omit<AskQuestion, "state"> & { state?: string };

export interface AskOptions {
  /** Read to use. Default per mode: see `DEFAULT_METHOD` (noul: logits; choice and score: head).
   *  `"both"`: the head result, with the logits read of the same prefill in `result.logits`. */
  method?: "head" | "logits" | "both";
  /** Prompt token ceiling (default 2048). */
  maxSafeTokens?: number;
  /** score: map a value to a level with K-1 ascending cuts (K = `levels`, default 5; the logits read always has K = 5):
   *  level = 1 + number of cuts <= value. The value is `raw` for the head and `expected` for the logits.
   *  Default: head round(sigmoid(raw)·(K-1)+1), logits argmax of the digits. With `"both"` the two reads have
   *  different scales, so pass `{ head, logits }`. Ignored by noul and choice. */
  scoreCuts?: number[] | { head?: number[]; logits?: number[] };
}

export interface AskResult {
  type: "noul" | "choice" | "score";
  method: "head" | "logits";
  /** noul: raw score (head) or L[Yes]-L[No] (logits). score/head: raw score. */
  raw?: number;
  /** noul: P(yes). */
  p?: number;
  /** choice: probability (softmax) per option. */
  choice?: Record<string, number>;
  /** choice: winning option (tie -> the first). */
  top?: string;
  /** score: level 1..K. */
  level?: number;
  /** score, logits read: expected level, sum over l=1..5 of l x softmax(logits of the digits '1'..'5'). */
  expected?: number;
  /** method "both": the logits read of the same prefill. */
  logits?: AskResult;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

function softmax(xs: number[]): number[] {
  const mx = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - mx));
  const z = e.reduce((a, b) => a + b, 0) || 1;
  return e.map((v) => v / z);
}

/** Index of the largest value; tie -> the first. */
function argmax(xs: number[]): number {
  let b = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[b]) b = i;
  return b;
}

function toChoice(options: string[], probs: number[]) {
  const choice: Record<string, number> = {};
  options.forEach((o, i) => { choice[o] = probs[i]; });
  return { choice, top: options[argmax(probs)] };
}

/** Level from ascending cuts: 1 + number of cuts <= x. */
export function levelFromCuts(x: number, cuts: readonly number[]): number {
  let l = 1;
  for (const c of cuts) if (x >= c) l++;
  return l;
}

function checkCuts(cuts: number[] | undefined, n: number, read: string): number[] | undefined {
  if (cuts === undefined) return undefined;
  if (!Array.isArray(cuts) || cuts.length !== n || !cuts.every(Number.isFinite) || cuts.some((c, i) => i > 0 && c <= cuts[i - 1]))
    throw new Error(`ask: scoreCuts for the ${read} read must be ${n} finite ascending numbers, got ${JSON.stringify(cuts)}`);
  return cuts;
}

/** One question ready to run: the ids, the hidden positions it needs, and how to read the prefill result. */
interface AskPlan {
  ids: number[];
  positions: number[];
  read(r: { hiddens: Float32Array[]; lastLogits: Float32Array }): AskResult;
}

function planAsk(be: Backend, tok: any, heads: AskHeads, q: AskQuestion, opts: AskOptions): AskPlan {
  const maxSafe = opts.maxSafeTokens ?? 2048;
  const prompt = promptFor(q); // validates kind / options (throws on unknown kind, empty or duplicate options)
  const type = q.kind;
  const method = opts.method ?? DEFAULT_METHOD[type];
  if (method !== "head" && method !== "logits" && method !== "both") throw new Error(`ask: unknown method ${JSON.stringify(method)}; use "head", "logits" or "both"`);
  const options = q.options ?? [];
  const t = tokenOffsets(tok, prompt);
  if (t.ids.length > maxSafe) throw new Error(`ask: prompt has ${t.ids.length} tokens, over maxSafeTokens (${maxSafe})`);

  let headCuts: number[] | undefined, logitCuts: number[] | undefined;
  if (type === "score" && opts.scoreCuts !== undefined) {
    const sc = opts.scoreCuts;
    if (Array.isArray(sc)) {
      if (method === "both") throw new Error('ask: method "both" reads two scales; pass scoreCuts as { head, logits }');
      if (method === "head") headCuts = sc; else logitCuts = sc;
    } else ({ head: headCuts, logits: logitCuts } = sc ?? {});
    headCuts = checkCuts(headCuts, (q.levels || 5) - 1, "head");
    logitCuts = checkCuts(logitCuts, DIGIT_IDS.length - 1, "logits");
  }

  const optFirst = type === "choice" ? options.map((o) => Number(tok.encode(o, { add_special_tokens: false })[0])) : [];
  const fromLogits = (L: Float32Array): AskResult => {
    if (type === "noul") {
      const raw = L[YES_ID] - L[NO_ID];
      return { type, method: "logits", raw, p: sigmoid(raw) };
    }
    if (type === "choice") return { type, method: "logits", ...toChoice(options, softmax(optFirst.map((id) => L[id]))) };
    const d = DIGIT_IDS.map((i) => L[i]);
    const expected = softmax(d).reduce((a, p, i) => a + (i + 1) * p, 0);
    return { type, method: "logits", level: logitCuts ? levelFromCuts(expected, logitCuts) : argmax(d) + 1, expected };
  };
  if (method === "logits") return { ids: t.ids, positions: [], read: (r) => fromLogits(r.lastLogits) };

  // method "head" | "both"
  if (!be.hiddenName) throw new Error(`ask: method '${method}' needs a graph with a hidden output (backend.hiddenName is null); use method 'logits'`);
  const hd = type === "score" ? heads.score : heads.noulChoice;
  if (!hd) throw new Error(type === "score" ? `ask: method '${method}' for score needs the score head (head_score.bin); load it via loadHeads({score})` : `ask: method '${method}' needs a loaded head for '${type}'`);
  const dp = decidePos(t.tokens);
  const withLogits = (r: AskResult, L: Float32Array): AskResult => (method === "both" ? { ...r, logits: fromLogits(L) } : r);
  if (type === "choice") {
    const op = optPositionsFor(prompt, t.offsets, options.length);
    return { ids: t.ids, positions: [dp, ...op], read: ({ hiddens, lastLogits }) => {
      const perOption = hiddens.slice(1);
      const a = answer(hd, { type: "choice", options }, { atDecide: hiddens[0], atLast: perOption[perOption.length - 1], perOption });
      return withLogits({ type, method: "head", ...toChoice(options, options.map((o) => a.choice![o])) }, lastLogits);
    } };
  }
  return { ids: t.ids, positions: [dp, t.ids.length - 1], read: ({ hiddens, lastLogits }) => {
    const a = answer(hd, { type, levels: q.levels }, { atDecide: hiddens[0], atLast: hiddens[1] });
    if (type === "noul") return withLogits({ type, method: "head", raw: a.raw, p: a.noul }, lastLogits);
    const level = headCuts && Number.isFinite(a.raw) ? levelFromCuts(a.raw!, headCuts) : a.score;
    return withLogits({ type, method: "head", raw: a.raw, level }, lastLogits);
  } };
}

export async function ask(
  be: Backend, tok: any, heads: AskHeads, q: AskQuestion, opts: AskOptions = {},
): Promise<AskResult> {
  const p = planAsk(be, tok, heads, q, opts);
  return p.read(await be.prefill(p.ids, { positions: p.positions }));
}

/** Longest token prefix all plans share, cut before the first wanted hidden position and before each prompt's last token
 *  (every question needs at least one token of its own to produce its last logits). */
function sharedPrefix(plans: AskPlan[]): number {
  let k = Math.min(...plans.map((p) => p.ids.length - 1), ...plans.flatMap((p) => p.positions));
  const a = plans[0].ids;
  for (const p of plans.slice(1)) { let i = 0; while (i < k && p.ids[i] === a[i]) i++; k = i; }
  return k;
}

/**
 * Several questions about ONE state: same results as `ask` for each question, in order. When the backend implements
 * `snapshot` / `prefillFrom` (decode-only, genai), the shared prompt prefix (system turn + state) runs once and each
 * question runs only its own suffix; otherwise it falls back to independent `ask` calls. A question may omit `state`
 * (the shared one is used); a different `state` rejects. Every question is validated before anything runs.
 */
export function askMany(
  be: Backend, tok: any, heads: AskHeads, state: string, qs: AskManyQuestion[], opts: AskOptions = {},
): Promise<AskResult[]> {
  return askManyWith(be, tok, heads, state, qs, () => opts);
}

/** `askMany` with per-question options (the facade uses it for its per-kind default method). */
export async function askManyWith(
  be: Backend, tok: any, heads: AskHeads, state: string, qs: AskManyQuestion[], optsFor: (q: AskQuestion) => AskOptions,
): Promise<AskResult[]> {
  if (typeof state !== "string") throw new Error("askMany: state must be a string");
  const plans = qs.map((q0) => {
    if (q0?.state !== undefined && q0.state !== state) throw new Error("askMany: a question carries a state different from the shared state");
    const q = { ...q0, state } as AskQuestion;
    return planAsk(be, tok, heads, q, optsFor(q));
  });
  const out: AskResult[] = [];
  const k = plans.length > 1 && be.snapshot && be.prefillFrom ? sharedPrefix(plans) : 0;
  if (k < 1) {
    for (const p of plans) out.push(p.read(await be.prefill(p.ids, { positions: p.positions })));
    return out;
  }
  const h: StateHandle = await be.snapshot!(plans[0].ids.slice(0, k));
  try {
    for (const p of plans) out.push(p.read(await be.prefillFrom!(h, p.ids, { positions: p.positions })));
  } finally { await h.dispose(); }
  return out;
}
