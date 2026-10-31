

export type Probe = {
  kind: "noul" | "choice" | "score";
  state: string;
  q?: string;
  options?: string[];
  sev?: number;
  want?: string;
};

export const CHOICE_Q = "Which vulnerability class does this code contain?";
export const SEV_Q = "How severe is the security risk in this content? (1=none, 5=critical)";

export function renderPrompt(state: string, question: string): string {
  return (
    "<|im_start|>system\nYou are a security decision function. Answer with exactly one word: Yes or No.<|im_end|>\n" +
    "<|im_start|>user\n<state>" + state + "</state>\n<q>" + question + "</q><decide><|im_end|>\n<|im_start|>assistant\n"
  );
}

const KINDS = ["noul", "choice", "score"];

export function assertProbe(p: Probe): void {
  const q: any = p;
  if (!q || typeof q !== "object") throw new Error("probe: expected an object {kind, state, ...}");
  if (!KINDS.includes(q.kind)) {
    const hint = q.kind === undefined && "type" in q ? ` (the field is \`kind\`, not \`type\`: {kind: ${JSON.stringify(q.type)}, state, q})` : "";
    throw new Error(`probe: invalid kind ${JSON.stringify(q.kind)}; use "noul", "choice" or "score"${hint}`);
  }
  if (typeof q.state !== "string") throw new Error("probe: state must be a string");
  if (q.kind === "choice") {
    const o = q.options;
    if (!Array.isArray(o) || !o.length) throw new Error("probe: choice requires options (non-empty list)");
    const seen = new Set<string>();
    for (const x of o) {
      if (typeof x !== "string" || !x.trim()) throw new Error("probe: option label empty or not a string: " + JSON.stringify(x));
      if (/<\/?opt>/.test(x)) throw new Error("probe: option label cannot contain <opt> / </opt>: " + JSON.stringify(x));
      if (seen.has(x)) throw new Error("probe: duplicate option label: " + JSON.stringify(x));
      seen.add(x);
    }
  }
}

export function promptFor(p: Probe): string {
  assertProbe(p);
  if (p.kind === "noul") return renderPrompt(p.state, p.q ?? "");
  if (p.kind === "choice") {
    return renderPrompt(p.state, CHOICE_Q + " " + (p.options ?? []).map((o) => "<opt>" + o + "</opt>").join(""));
  }
  return renderPrompt(p.state, SEV_Q);
}

export function decidePos(tokens: string[]): number {
  const imEnd: number[] = [];
  tokens.forEach((s, i) => { if (s === "<|im_end|>") imEnd.push(i); });
  if (!imEnd.length) throw new Error("decidePos: no <|im_end|> in " + JSON.stringify(tokens.slice(-8)));
  let lastAsst = -1;
  tokens.forEach((s, i) => { if (s.trim() === "assistant") lastAsst = i; });
  const cand = lastAsst >= 0 ? imEnd.filter((i) => i < lastAsst) : [];
  const end = cand.length ? cand[cand.length - 1] : imEnd[imEnd.length - 1];
  if (end === 0) throw new Error("decidePos: <|im_end|> at position 0, no preceding token");
  if (tokens[end - 1] !== ">") {
    throw new Error("decidePos: token before <|im_end|> is " + JSON.stringify(tokens[end - 1]) + ", expected '>' (end of <decide>)");
  }
  return end - 1;
}

export function optPositions(prompt: string, offsets: [number, number][]): number[] {
  const out: number[] = [];
  const from = Math.max(0, prompt.lastIndexOf("<q>"));
  for (const m of prompt.slice(from).matchAll(/<\/opt>/g)) {
    const c = from + (m.index as number) + m[0].length - 1;
    const hit = offsets.findIndex(([s, e]) => s <= c && c < e);
    if (hit < 0) throw new Error("optPositions: no token covers char " + c);
    out.push(hit);
  }
  return out;
}

export function optPositionsFor(prompt: string, offsets: [number, number][], nOptions?: number): number[] {
  const pos = optPositions(prompt, offsets);
  if (nOptions === undefined) nOptions = (prompt.slice(Math.max(0, prompt.lastIndexOf("<q>"))).match(/<opt>/g) ?? []).length;
  if (pos.length !== nOptions) throw new Error(`positions: ${pos.length} </opt> markers after <q> for ${nOptions} options (label containing </opt>?)`);
  return pos;
}

export function tokenOffsets(tok: any, prompt: string): { ids: number[]; tokens: string[]; offsets: [number, number][] } {
  const ids: number[] = Array.from(tok.encode(prompt, { add_special_tokens: false }), (x: any) => Number(x));
  const dec = (a: number[]): string => tok.decode(a, { skip_special_tokens: false, clean_up_tokenization_spaces: false });
  const tokens = ids.map((i) => dec([i]));
  const offsets: [number, number][] = [];
  let prev = 0;
  let full = "";
  for (let i = 0; i < ids.length; i++) {
    full = dec(ids.slice(0, i + 1));
    let end = full.length;
    while (end > prev && full.charCodeAt(end - 1) === 0xfffd) end--;
    offsets.push([prev, end]);
    prev = end;
  }
  if (full !== prompt) throw new Error("tokenOffsets: reconstruction != prompt (offsets not reliable)");
  return { ids, tokens, offsets };
}
