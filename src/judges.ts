/**
 * judges.ts — System-2 as a JUDGE, in short structured steps.
 *
 * Measured context (EVIDENCE §18q–18u):
 *   - One forward (decide) is cheap and stable, but a 1-forward judge has no view of the whole.
 *   - The deterministic tools have high recall (0.769) but mis-attribute the signal: `sinks` fires and
 *     gets called `postmessage`, `storage` fires and gets called `exfil-egress`. That is a
 *     classification-with-context problem, not a detection problem.
 *   - Correlating findings into an exploit chain is reasoning over a set — out of reach of 1 forward.
 *
 * Every judge here follows the same contract and the same cost discipline: a SHORT generation over a
 * small, already-filtrated state (findings, snippets, candidate labels) — never over raw page HTML and
 * never in a loop. Latency of a short step is measured separately; a judge that is not affordable should
 * be left off, and the caller can do exactly that.
 */
import type { ReconFinding, Chain, Specialty, SeverityHint } from "./recon";

/** The model surface a judge needs: a plain short generation. */
export interface GenModel {
  stream: (prompt: string, opts?: any) => AsyncIterable<string>;
}

/** Render used by every judge. Plain, deterministic, no chat template assumptions. */
export function judgePrompt(instruction: string, state: string): string {
  return (
    `<|im_start|>system\nYou are a precise web-security reviewer. Answer only in the requested format. Do not explain.<|im_end|>\n` +
    `<|im_start|>user\n${instruction}\n\nSTATE:\n${state}<|im_end|>\n<|im_start|>assistant\n`
  );
}

async function generate(model: GenModel, prompt: string, maxTokens: number): Promise<string> {
  let out = "";
  for await (const p of model.stream(prompt, { temperature: 0, maxNewTokens: maxTokens })) out += p;
  return out;
}

/** Parse a `LABEL: <name>` reply into one of the allowed labels (case-insensitive, first match). */
export function parseLabel(reply: string, allowed: string[]): string | undefined {
  const r = (reply || "").toLowerCase();
  for (const a of allowed) if (r.includes(a.toLowerCase())) return a;
  return undefined;
}

/** Parse a yes/no reply. Accepts "yes", "REFUTED: yes", "ANSWER: no", "COMPOSES: yes", etc.
 *  Takes the LAST yes/no token so a prefixed label does not shadow the answer. */
export function parseYes(reply: string): boolean | undefined {
  const r = (reply || "").toLowerCase();
  const hits = [...r.matchAll(/\b(yes|true|no|false)\b/g)].map(m => m[1]);
  if (!hits.length) return undefined;
  const last = hits[hits.length - 1];
  return last === "yes" || last === "true";
}

/**
 * H1 — attribution judge. The tools fired; does the signal ALSO belong to a neighbouring specialty?
 *
 * Medido (§18w): a primeira versão perguntava "a qual destas concern isto pertence?" — pergunta de
 * EXCLUSÃO MÚTUA. Resultado: apagou `postmessage` em 3 casos onde ele estava CERTO, porque o snippet
 * mostra `innerHTML` e o modelo escolhe o rótulo mais óbvio. Duas especialidades podem coexistir no
 * mesmo trecho (um handler `message` que escreve em `innerHTML` é postmessage E dom-xss).
 * Aqui a pergunta é ADITIVA: "há sinal de X aqui?" — pode acrescentar, nunca remover.
 */
export async function attribute(
  model: GenModel,
  finding: ReconFinding,
  candidates: Specialty[],
  opts: { maxTokens?: number } = {}
): Promise<string[] | undefined> {
  const missing = candidates.filter(c => c.key !== finding.specialty);
  if (!missing.length) return undefined;
  const allowed = missing.map(c => c.key);
  const instruction =
    `The excerpt below was flagged as "${finding.specialty}". Two concerns can apply to the SAME code.\n` +
    `Answer only whether a DIFFERENT concern from this list is ALSO clearly present: ${allowed.join(", ")}.\n` +
    `If none is clearly present, answer NONE.\n` +
    `Reply with exactly one line: ALSO: <name or NONE>`;
  const state = `EXCERPT:\n${finding.snippet}\n\nALREADY FLAGGED AS: ${finding.specialty} (${finding.signal})`;
  const reply = await generate(model, judgePrompt(instruction, state), opts.maxTokens ?? 12);
  const got = parseLabel(reply, allowed);
  return got ? [got] : undefined;
}

/**
 * H4 — adversarial refuter. Tries to knock the finding down with the code visible. Refuting is easier
 * than proving, which is why this is a plausible small-model task (the earlier failure mode, §18l, was
 * a model *claiming* success, not failing to refute).
 */
export async function refute(
  model: GenModel,
  finding: ReconFinding,
  opts: { maxTokens?: number } = {}
): Promise<boolean | undefined> {
  const instruction =
    `Below is a claimed finding. Try to REFUTE it. It is refuted if the excerpt does not actually ` +
    `support the claim (e.g. the sink receives only constant data, the value is not a real secret, ` +
    `the endpoint is same-origin, the form action is HTTPS with a token, the message listener does validate origin).\n` +
    `Reply with exactly one line: REFUTED: yes  or  REFUTED: no`;
  const state = `CLAIM: ${finding.specialty} — ${finding.signal}\nEXCERPT:\n${finding.snippet}`;
  const reply = await generate(model, judgePrompt(instruction, state), opts.maxTokens ?? 8);
  return parseYes(reply);
}

/**
 * H2 — finding writer. One short generation per finding, in prose a bug-bounty report can use.
 * Not an exploratory loop: a single rewrite of evidence that already exists.
 */
export async function writeFinding(
  model: GenModel,
  finding: ReconFinding,
  opts: { maxTokens?: number } = {}
): Promise<string | undefined> {
  const instruction =
    `Write ONE bug-bounty finding in at most 3 short lines. Format exactly:\n` +
    `IMPACT: <one line>\nREPRO: <one line>\nFIX: <one line>`;
  const state = `TYPE: ${finding.specialty}\nSEVERITY: ${finding.severity}\nSIGNAL: ${finding.signal}\nLOCATOR: ${finding.locator}\nEXCERPT:\n${finding.snippet}`;
  const out = (await generate(model, judgePrompt(instruction, state), opts.maxTokens ?? 90)).trim();
  return out || undefined;
}

/**
 * H5 — next-step planner. Given an accepted finding, name the single verification that would close it.
 * This is the Mode-B navigation decision, but now it runs over a handful of findings, not the page.
 */
export async function planNextStep(
  model: GenModel,
  finding: ReconFinding,
  opts: { maxTokens?: number } = {}
): Promise<string | undefined> {
  const instruction =
    `Name the SINGLE cheapest concrete verification that would prove or disprove this finding in a ` +
    `browser sandbox. One line, at most 15 words, starting with a verb.`;
  const state = `TYPE: ${finding.specialty}\nSIGNAL: ${finding.signal}\nLOCATOR: ${finding.locator}\nEXCERPT:\n${finding.snippet}`;
  const out = (await generate(model, judgePrompt(instruction, state), opts.maxTokens ?? 30)).trim();
  return out || undefined;
}

/**
 * H6 — chain judge. Two findings each passed on their own; do they COMPOSE into a real exploit path?
 * This is the reasoning step a 1-forward decide cannot do, and the one no single tool can see.
 */
export async function judgeChain(
  model: GenModel,
  chain: Chain,
  parts: ReconFinding[],
  opts: { maxTokens?: number } = {}
): Promise<number | undefined> {
  const instruction =
    `Two findings were made on the same page. Decide whether they COMPOSE into a single exploitable ` +
    `path ("${chain.name}"), or are unrelated.\n` +
    `Reply with exactly one line: COMPOSES: yes  or  COMPOSES: no`;
  const state = parts.map(p => `[${p.id}] ${p.specialty} @ ${p.locator}\n     ${p.snippet.slice(0, 240)}`).join("\n\n") +
    `\n\nPROPOSED CHAIN: ${chain.rationale}`;
  const reply = await generate(model, judgePrompt(instruction, state), opts.maxTokens ?? 8);
  const y = parseYes(reply);
  return y === undefined ? undefined : y ? 0.9 : 0.1;
}

/** Run the full judge pass over findings. Every step is optional and individually switchable, so cost
 *  is always attributable to one decision. Returns new findings (attribution may move them) + chains. */
export async function judgePass(
  model: GenModel,
  findings: ReconFinding[],
  opts: {
    candidates?: Specialty[];
    attribute?: boolean;
    refute?: boolean;
    write?: boolean;
    plan?: boolean;
    chains?: boolean;
    maxFindings?: number;
  } = {}
): Promise<{ findings: ReconFinding[]; chains: Chain[]; log: Record<string, number> }> {
  const log: Record<string, number> = { attributed: 0, refuted: 0, written: 0, planned: 0, chains: 0 };
  let out = findings.map(f => ({ ...f }));
  const cap = opts.maxFindings ?? out.length;

  if (opts.attribute && opts.candidates) {
    // ADITIVO: acrescenta uma especialidade ao mesmo trecho, nunca renomeia (medido §18w: renomear
    // apagava postmessage onde ele estava certo). Dedup no fim remove repetições por especialidade.
    for (const f of out.slice(0, cap)) {
      const also = await attribute(model, f, opts.candidates);
      if (also && also.length) {
        log.attributed++;
        for (const key of also) {
          out.push({ ...f, id: `R${out.length + 1}`, specialty: key, signal: `${f.signal} (also: ${f.specialty})` });
        }
      }
    }
  }
  if (opts.refute) {
    const kept: ReconFinding[] = [];
    for (const f of out) {
      const refuted = await refute(model, f);
      if (refuted === true && !f.corroborated) { log.refuted++; continue; }   // só descarta sem prova dura
      kept.push(f);
    }
    out = kept;
  }
  if (opts.plan) {
    for (const f of out.slice(0, cap)) {
      const n = await planNextStep(model, f);
      if (n) { f.nextStep = n; log.planned++; }
    }
  }
  if (opts.write) {
    for (const f of out.slice(0, cap)) {
      const w = await writeFinding(model, f);
      if (w) { (f as any).written = w; log.written++; }
    }
  }
  let chains: Chain[] = [];
  if (opts.chains) {
    chains = await correlateWith(model, out);
    log.chains = chains.length;
  }
  // dedup por especialidade após re-atribuição (duas findings podem convergir para o mesmo rótulo)
  const seen = new Set<string>();
  out = out.filter(f => (seen.has(f.specialty) ? false : (seen.add(f.specialty), true)));
  return { findings: out, chains, log };
}

async function correlateWith(model: GenModel, findings: ReconFinding[]): Promise<Chain[]> {
  const { correlate } = await import("./recon");
  return correlate(findings, async (chain, parts) => {
    const p = await judgeChain(model, chain, parts);
    return p ?? 0.5;
  });
}
