/**
 * recon.ts — System-1 as the RECON ENGINE (not a triage gate in front of a System-2 router).
 *
 * The measured fit (EVIDENCE.md §18n) is: a System-1 decision costs ~0.6 s and is stable over
 * dozens of consecutive calls, while a generation-heavy agent step costs ~17 s. So the cheap,
 * reliable primitive should carry the WORK — locating the interesting spans of a page and deciding
 * which specialty applies — instead of being used as a one-shot gate before an unaffordable loop.
 *
 * This module is pure and injected:
 *   - `decide(state, question) -> Promise<P>` is the only model surface. Any System-1 works
 *     (CT0 ONNX logits, WebLLM logprobs, or a test double). No heuristics gate entry.
 *   - `tools` is optional: deterministic extractors used only to CORROBORATE a located span.
 *     They never route.
 *
 * Flow (programmed mode, "A"):
 *   segment(html)                        -> candidate regions with locators
 *   funnel: coarse System-1 sweep        -> "is there anything here at all?"
 *           zoom hot regions             -> subdivide + re-ask (bounded depth/budget)
 *   cross: region x specialty            -> which specialty lights up where
 *   confirm: deterministic tools         -> corroboration + strong evidence label
 *   verdict: System-1 on evidence        -> final confidence per finding
 *   report                              -> bug-bounty-style recon
 *
 * Escalation to an agentic red-team pass is DESIGNED, not required, and lives in `reconToolset`
 * (the same navigation steps exposed as ReAct tools so a conductor model can drive mode "B"
 * through the existing `Agent.run` guards). See README for the stage-2 hardware disclosure.
 */

/** A security specialty: key + the System-1 yes/no probe + deterministic tools that corroborate it. */
export interface Specialty {
  key: string;
  probe: string;
  tools?: string[];
  /** Presence probe for the LOCALISATION pass: "does this excerpt itself contain X's raw signal?"
   *  Kept separate from `probe` on purpose — asking presence AND exploitability in one question makes
   *  the model answer "no" for a span that plainly contains the signal but not yet the exploit. */
  presence?: string;
  /** Suggested manual/agentic verification if a span lights up (recon is signals, not proofs). */
  nextStep?: string;
  /** Default severity hint when corroborated evidence exists. */
  severity?: SeverityHint;
}

export type SeverityHint = "info" | "low" | "medium" | "high";

/** Specialties that fit the browser sandbox. Deliberately broad probes: the funnel refines. */
export const BROWSER_SPECIALTIES: Specialty[] = [
  {
    key: "dom-xss",
    probe: "Does this page render user-controllable input into the DOM (a preview, comment, search echo, template) such that a stored/reflected/DOM XSS is plausible?",
    presence: "Does this excerpt itself contain markup, code or text that turns user-controllable input into DOM content — a preview/sink (innerHTML, insertAdjacentHTML, document.write), a template, a comment box, a search echo, or user input rendered as HTML?",
    tools: ["sinks","source","snapshot","scan","type","readHTML","events"],
    nextStep: "inject a benign marker through the input and confirm it reaches a sink (readHTML) and executes (events).",
    severity: "high",
  },
  {
    key: "prompt-injection",
    probe: "Does this page contain untrusted document/content that an AI assistant might read and be manipulated by (hidden instructions, fake system notes)?",
    presence: "Does this excerpt itself contain text that reads like instructions aimed at an AI (ignore/disregard previous instructions, fake system notes, hidden text directing an assistant), or untrusted content an assistant would ingest?",
    tools: ["source","inspect","scan"],
    nextStep: "quote the exact instruction text and show the path by which untrusted content reaches an assistant prompt.",
    severity: "medium",
  },
  {
    key: "secret-scan",
    probe: "Might this page expose secrets or tokens client-side (in localStorage, cookies, inline scripts, or comments)?",
    presence: "Does this excerpt itself contain a secret or token material — a hardcoded key/credential, a value read from localStorage/cookies, an inline token, or a comment carrying one?",
    tools: ["storage","inspect","cookies","scan"],
    nextStep: "read the value with JS (document.cookie / localStorage) and confirm it is not HttpOnly-only.",
    severity: "high",
  },
  {
    key: "postmessage",
    probe: "Does this page use cross-frame messaging (iframes, window.postMessage, message listeners) that could be exploited across origins?",
    presence: "Does this excerpt itself contain cross-frame messaging mechanics — postMessage() calls, addEventListener('message') / onmessage listeners, or embedded iframes that exchange messages?",
    tools: ["messages","inspect","sinks","source","scan"],
    nextStep: "send a cross-origin message to the listener and observe whether origin is validated before use.",
    severity: "medium",
  },
  {
    key: "csp-misconfig",
    probe: "Is this page likely missing a Content-Security-Policy or using a weak one (unsafe-inline/eval, wildcards)?",
    presence: "Does this excerpt itself indicate a CSP problem — a meta Content-Security-Policy, or inline scripts/handlers that a strict policy would have to allow (unsafe-inline/unsafe-eval, wildcard)?",
    tools: ["csp","source"],
    nextStep: "show the response policy and which directive is weak; demonstrate an inline payload it fails to block.",
    severity: "low",
  },
  {
    key: "exfil-egress",
    probe: "Does this page make outbound/cross-origin network requests or beacons that could carry data off-origin?",
    presence: "Does this excerpt itself contain outbound network code — fetch/XHR/sendBeacon/Image().src to an external URL, or a tracking/beacon endpoint?",
    tools: ["resources","events","cookies","storage"],
    nextStep: "capture the outbound request and decode its payload to show what data leaves the origin.",
    severity: "medium",
  },
  {
    key: "sensitive-path",
    probe: "Does this page reference sensitive paths that look publicly reachable — environment files, VCS metadata, database dumps, backups, admin or debug endpoints?",
    presence: "Does this excerpt itself reference a sensitive path (.env, .git, .svn, backup, dump, .sql, .bak, config.json, id_rsa, admin, debug, swagger, graphql)?",
    tools: ["sensitivePaths", "links", "source"],
    nextStep: "request the referenced path and confirm whether it is publicly served (do not download user data).",
    severity: "high",
  },
  {
    key: "error-disclosure",
    probe: "Does this page display a stack trace or detailed internal error to the user, exposing paths, queries or credentials?",
    presence: "Does this excerpt itself contain a stack trace, framework exception, SQL error or absolute internal path?",
    tools: ["errorLeak", "inspect", "source"],
    nextStep: "capture the exact error text and any credential it leaks, and show which request triggers it.",
    severity: "high",
  },
  {
    key: "supply-chain",
    probe: "Does this page load third-party scripts or stylesheets without Subresource Integrity, allowing a compromised CDN to run code here?",
    presence: "Does this excerpt itself reference an external (cross-origin) script or stylesheet, and does it lack an integrity attribute?",
    tools: ["thirdPartyNoSri", "resources"],
    nextStep: "list the third-party origins without integrity and confirm which one can execute in this origin.",
    severity: "medium",
  },
  {
    key: "info-disclosure",
    probe: "Does this page disclose information useful to an attacker — published source maps, debug mode, framework version, or hidden debug/admin fields?",
    presence: "Does this excerpt itself contain a sourceMappingURL, a DEBUG/verbose flag, a version string, or a hidden admin/debug field?",
    tools: ["infoDisclosure", "inspect", "source"],
    nextStep: "report the specific disclosure and what an attacker learns from it (version → known CVEs, debug → internals).",
    severity: "low",
  },
  {
    key: "insecure-forms",
    probe: "Does this page have forms handling credentials or sensitive data insecurely (non-HTTPS action, autocomplete on, no CSRF token)?",
    presence: "Does this excerpt itself contain a form or credential field — a <form> (any action), a password/email input, or auth-related markup?",
    tools: ["forms","inspect","cookies"],
    nextStep: "submit with a non-secret probe and confirm the action URL/scheme and missing token.",
    severity: "medium",
  },
];

/** Deterministic strong-evidence label from a tool observation ("" = none). General and testable. */
import {
  detectSecret, detectJsReadableCookie, detectInsecureForm, detectDangerousLink, detectUnsandboxedIframe,
  detectClobber, detectSink, detectCspWeakness, detectSensitivePath, detectErrorLeak, detectNoSri,
  detectInfoDisclosure, detectExfil, detectExecution, detectPromptInjection, detectMessageSurface,
  detectEgressInSource, detectFromTool, SIGNAL_OWNER, signalFits,
  type Detection,
} from "./detectors";

export type Decide = (state: string, question: string) => Promise<number>;
/** Roda uma tool neutra e aplica os detectores sobre o FATO devolvido. Devolve as Detection que
 *  pertencem à especialidade pedida. É o único ponto de contato tool → julgamento. */
async function runTool(tools: ToolMap, toolName: string, specialtyKey: string): Promise<Detection[]> {
  const fn = tools[toolName]; if (typeof fn !== "function") return [];
  try {
    const facts = await fn({});
    // a Detection já traz a especialidade dona; basta exigir que seja a pedida.
    return detectFromTool(toolName, facts).filter(d => d.key === specialtyKey);
  } catch { return []; }
}


export type ToolMap = Record<string, (a: any) => any>;

/** A candidate span of the page. Locators are approximations (tag + id/class + char offsets),
 *  not CSS-validated selectors — recon output, not a proof artifact. */
export interface Region {
  id: string;
  locator: string;
  kind: "form" | "script" | "textarea" | "iframe" | "handler" | "comment" | "container" | "block";
  tag?: string;
  start: number;
  end: number;
  depth: number;
  text: string;
  /** System-1 coarse relevance (0..1). */
  p?: number;
}

const VOID_TAGS = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"]);
const INTERESTING = new Set(["form", "script", "textarea", "iframe", "object", "embed", "template", "table", "main", "section", "article", "nav", "aside", "header", "footer"]);
const BLOCK_TAGS = new Set(["div", "p", "li", "td", "label", "button", "a", "span", "pre", "code"]);
const MAX_REGION_TEXT = 1500;
const MAX_REGIONS = 400;

interface TagTok { kind: "open" | "close" | "comment"; tag?: string; start: number; end: number; attrs?: string; }

function tokenize(html: string): TagTok[] {
  const re = /<!--[\s\S]*?-->|<\/([a-zA-Z][\w:-]*)\s*>|<([a-zA-Z][\w:-]*)((?:"[^"]*"|'[^']*'|[^>"'])*)>/g;
  const out: TagTok[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(html))) {
    const start = m.index, end = start + m[0].length;
    if (m[0].startsWith("<!--")) out.push({ kind: "comment", start, end });
    else if (m[1]) out.push({ kind: "close", tag: m[1].toLowerCase(), start, end });
    else out.push({ kind: "open", tag: (m[2] || "").toLowerCase(), attrs: m[3] || "", start, end });
    if (out.length > MAX_REGIONS * 8) break; // bounded: pathological pages must not stall
  }
  return out;
}

function locatorFor(tok: TagTok, ordinal: number): string {
  if (tok.kind === "comment" || !tok.tag) return `comment[${ordinal}]`;
  const a = tok.attrs || "";
  const idm = a.match(/\sid\s*=\s*["']?([\w:.-]+)/i);
  if (idm) return `${tok.tag}#${idm[1]}`;
  const cl = a.match(/\sclass\s*=\s*["']?([\w:.-]+)/i);
  if (cl) return `${tok.tag}.${cl[1]}`;
  return `${tok.tag}[${ordinal}]`;
}

/**
 * Segment HTML into candidate regions with locators. Deterministic, no model, no heuristics about
 * what is *vulnerable* — only structural candidates (forms, scripts, textareas, iframes, inline
 * handlers, comments, containers, blocks). The funnel decides what matters, not this function.
 */
export function segment(html: string, opts: { maxRegions?: number } = {}): Region[] {
  const h = html || "";
  const toks = tokenize(h);
  const cap = opts.maxRegions ?? MAX_REGIONS;
  const regions: Region[] = [];
  const seen = new Map<string, number>();
  const stack: { tag: string; start: number; tok: TagTok }[] = [];
  let depth = 0;

  const push = (kind: Region["kind"], tag: string | undefined, start: number, end: number, tok: TagTok, d: number) => {
    if (regions.length >= cap || end <= start) return;
    const n = (seen.get(tag || kind) || 0) + 1;
    seen.set(tag || kind, n);
    const text = h.slice(start, end).slice(0, MAX_REGION_TEXT);
    if (!text.trim()) return;
    regions.push({ id: `r${regions.length + 1}`, locator: tok ? locatorFor(tok, n) : `${kind}[${n}]`, kind, tag, start, end, depth: d, text });
  };

  for (let i = 0; i < toks.length; i++) {
    const t = toks[i];
    if (t.kind === "comment") {
      const text = h.slice(t.start, t.end);
      if (text.length > 12) push("comment", undefined, t.start, t.end, t, depth);
      continue;
    }
    if (t.kind === "open") {
      const tag = t.tag!;
      const inlineHandler = /\son[a-z]+\s*=/i.test(t.attrs || "");
      const isPassword = /type\s*=\s*["']?password/i.test(t.attrs || "");
      if (inlineHandler) {
        // an inline handler is a span in its own right: element start..end of tag
        push("handler", tag, t.start, t.end, t, depth);
      }
      if (isPassword) push("form", tag, t.start, t.end, t, depth);
      if (VOID_TAGS.has(tag)) continue;
      stack.push({ tag, start: t.start, tok: t });
      depth++;
      continue;
    }
    // close: pop to the matching tag if present
    const tag = t.tag!;
    let idx = -1;
    for (let k = stack.length - 1; k >= 0; k--) if (stack[k].tag === tag) { idx = k; break; }
    if (idx < 0) { depth = Math.max(0, depth - 1); continue; }
    const opened = stack[idx];
    stack.length = idx;
    depth = stack.length;
    const span = { start: opened.start, end: t.end };
    if (INTERESTING.has(tag)) {
      const kind: Region["kind"] = tag === "form" ? "form" : tag === "script" ? "script" : tag === "textarea" ? "textarea" : tag === "iframe" ? "iframe" : "container";
      push(kind, tag, span.start, span.end, opened.tok, opened.tok === opened.tok ? Math.max(0, depth) : 0);
    } else if (BLOCK_TAGS.has(tag) && t.end - opened.start >= 40) {
      push("block", tag, span.start, span.end, opened.tok, depth);
    }
  }
  return regions;
}

export interface ReconOptions {
  html: string;
  decide: Decide;
  /** Which specialties to engage. "all" (default) uses BROWSER_SPECIALTIES. */
  specialties?: Specialty[] | "all";
  /** Optional deterministic extractors, used only to corroborate (never to route). */
  tools?: ToolMap;
  /** P at/above which a region is "hot" in the coarse pass. */
  coarseThreshold?: number;
  /** P at/above which a (region, specialty) cell is kept. */
  threshold?: number;
  /** Subdivision rounds for hot containers. */
  zoomDepth?: number;
  /** Hard cap on System-1 calls, so cost is bounded and honest. */
  maxDecides?: number;
  /** Skip the coarse pass and cross every region with every specialty (expensive, exhaustive). */
  exhaustive?: boolean;
  /** Use the presence probes for the cross pass (default true). false = use the full `probe`. */
  usePresence?: boolean;
  /** Drop findings whose contrastive verdict rejected the span and that lack deterministic corroboration
   *  (default true). false = report everything that crossed. */
  verdictGate?: boolean;
  /** "tools-first": the deterministic tools locate (high recall, free, no model), and System-1 only judges
   *  the signal they found (precision + locator). Measured better in bench §18r for recall at equal FP.
   *  "decide-first": the original funnel, System-1 does the locating. */
  strategy?: "decide-first" | "tools-first";
  target?: string;
  onProgress?: (msg: string) => void;
}

export interface ReconFinding {
  id: string;
  specialty: string;
  locator: string;
  kind: Region["kind"];
  offsets: [number, number];
  snippet: string;
  /** System-1 relevance of the span for ANY specialty in scope. */
  relevance: number;
  /** System-1 answer to this specialty's probe on this span. */
  confidence: number;
  /** Final System-1 verdict on the corroborated evidence (or = confidence when nothing to corroborate with). */
  verdict: number;
  /** Strong-evidence label produced by a deterministic tool ("" if none). */
  signal: string;
  corroborated: boolean;
  severity: SeverityHint;
  nextStep: string;
}

export interface ReconReport {
  target?: string;
  specialties: string[];
  mode: "programmed" | "conducted";
  regionsTotal: number;
  regionsScanned: number;
  regionsHot: number;
  decides: number;
  budgetExhausted: boolean;
  findings: ReconFinding[];
  /** Spans the funnel located but where no specialty crossed the threshold (still useful recon). */
  locatedOnly: { locator: string; relevance: number; snippet: string }[];
  /** H6: correlated chains — findings that combine into something stronger than any one alone. */
  chains?: Chain[];
  notes: string[];
}

/** H6 — a chain is two or more findings whose combination is exploitable in a way neither is alone.
 *  Canonical case: a sink (innerHTML/document.write) + an unvalidated source (message w/o origin
 *  check) = a real XSS path. No single tool sees this, and a 1-forward decide has no view of the
 *  whole — correlating findings is genuinely System-2 work. */
export interface Chain {
  id: string;
  parts: string[];             // finding ids
  name: string;                // e.g. "unvalidated postMessage → DOM sink"
  rationale: string;
  /** System-2's judgement that the chain is real (0..1), when a generator is provided. */
  confidence?: number;
  severity: SeverityHint;
  nextStep: string;
}

/** Known source→sink correlations. Conservative and explicit: only combinations that actually
 *  compose into an exploit path in the browser.
 *  `require` is checked against the signal text of the two findings — a chain only fires when the
 *  underlying evidence is really there. Medido (§18x): sem isso, `insecure-forms + dom-xss` disparava
 *  em página com `<form>` SEM action (não é form-over-HTTP), fabricando uma cadeia que não existe. */
const CHAIN_RULES: {
  name: string; from: string; to: string; rationale: string;
  severity: SeverityHint; nextStep: string; require?: RegExp;
}[] = [
  {
    name: "unvalidated postMessage → DOM sink",
    from: "postmessage",
    to: "dom-xss",
    rationale: "A message listener that uses e.data without validating e.origin, feeding a DOM sink, lets any framing origin drive script execution in this page.",
    severity: "high",
    nextStep: "send a cross-origin message carrying a benign marker and confirm it reaches the sink and executes.",
    require: /sink|postmessage|surface/i,
  },
  {
    name: "client-side secret reachable by XSS",
    from: "dom-xss",
    to: "secret-scan",
    rationale: "An XSS path plus a secret readable from JS means the secret is exfiltratable by an attacker who lands the XSS.",
    severity: "high",
    nextStep: "confirm the secret is readable (not HttpOnly) and that the sink executes attacker-controlled input.",
    require: /secret|sink|surface/i,
  },
  {
    name: "exfil endpoint + client-side secret",
    from: "secret-scan",
    to: "exfil-egress",
    rationale: "A secret readable from JS plus a cross-origin beacon/egress endpoint means data can leave the origin — the skimmer pattern.",
    severity: "high",
    nextStep: "decode the outbound request payload and show which secret it carries.",
    require: /beacon|egress|secret/i,
  },
  {
    name: "credential form over HTTP + XSS",
    from: "insecure-forms",
    to: "dom-xss",
    rationale: "Credentials submitted over plaintext plus script execution gives two independent ways to capture them.",
    severity: "high",
    nextStep: "submit with a non-secret probe and confirm the scheme; confirm the sink executes.",
    require: /over http|no csrf/i,
  },
  {
    name: "prompt-injection targeting client-side secrets",
    from: "prompt-injection",
    to: "secret-scan",
    rationale: "Untrusted content instructing an assistant to reveal secrets, in a page that actually holds a JS-readable secret, is a viable exfiltration path through the assistant.",
    severity: "medium",
    nextStep: "show the instruction text and the secret it targets, and the path by which content reaches the assistant.",
    require: /secret/i,
  },
];

/** H6 — correlate accepted findings into chains. Deterministic pairing over CHAIN_RULES; the optional
 *  `judge` (System-2) then rules on each candidate chain, because composing two true findings into a
 *  real exploit is a reasoning step, not a lookup. */
export async function correlate(
  findings: ReconFinding[],
  judge?: (chain: Chain, parts: ReconFinding[]) => Promise<number>
): Promise<Chain[]> {
  const byKey: Record<string, ReconFinding> = {};
  for (const f of findings) if (!byKey[f.specialty]) byKey[f.specialty] = f;
  const chains: Chain[] = [];
  for (const rule of CHAIN_RULES) {
    const a = byKey[rule.from], b = byKey[rule.to];
    if (!a || !b) continue;
    // só forma a cadeia se a EVIDÊNCIA das duas realmente sustenta a composição (§18x)
    if (rule.require && !rule.require.test(`${a.signal} ${b.signal}`)) continue;
    const chain: Chain = {
      id: `C${chains.length + 1}`,
      parts: [a.id, b.id],
      name: rule.name,
      rationale: rule.rationale,
      severity: rule.severity,
      nextStep: rule.nextStep,
    };
    if (judge) {
      try { chain.confidence = await judge(chain, [a, b]); } catch { /* a failed judge keeps the candidate */ }
    }
    chains.push(chain);
  }
  return chains;
}

const SNIPPET = 220;

function snippetOf(text: string): string {
  return text.replace(/\s+/g, " ").trim().slice(0, SNIPPET);
}

function severityFor(sp: Specialty, corroborated: boolean, verdict: number): SeverityHint {
  const base = sp.severity || "medium";
  if (!corroborated) return base === "high" ? "medium" : "low";
  if (verdict >= 0.8) return base;
  return base === "high" ? "medium" : "low";
}

/**
 * Programmed recon: System-1 locates spans (funnel), crosses them with the selected specialties,
 * corroborates with deterministic tools, then takes a final System-1 verdict per finding.
 * No heuristic decides what to look at; the caller chooses which specialties to engage.
 */
export async function recon(o: ReconOptions): Promise<ReconReport> {
  const specialties = o.specialties === "all" || !o.specialties ? BROWSER_SPECIALTIES : o.specialties;
  const threshold = o.threshold ?? 0.5;
  const coarseThreshold = o.coarseThreshold ?? 0.35;
  const zoomDepth = o.zoomDepth ?? 1;
  const maxDecides = o.maxDecides ?? 120;
  const usePresence = o.usePresence ?? true;
  const verdictGate = o.verdictGate ?? true;
  const strategy = o.strategy ?? "decide-first";
  const notes: string[] = [];
  let decides = 0;
  let budgetExhausted = false;

  const ask = async (state: string, question: string, fallback = 0.5): Promise<number> => {
    if (decides >= maxDecides) { budgetExhausted = true; return fallback; }
    decides++;
    try {
      const p = await o.decide(state, question);
      return Number.isFinite(p) ? Math.max(0, Math.min(1, p)) : fallback;
    } catch {
      return fallback;
    }
  };

    // Uma tool pode servir VÁRIAS especialidades (ex. `inspect` é usada por prompt-injection,
    // secret-scan e insecure-forms). Sem casar o RÓTULO com a especialidade, o sinal de uma vaza para
    // todas — medido (§19d): o mesmo "cross-frame messaging surface present" virava prompt-injection,
    // secret-scan e insecure-forms ao mesmo tempo. Aqui o sinal só conta se pertencer à especialidade.
    const SIGNAL_OWNER: Record<string, string[]> = {
      "client-side secret exposed": ["secret-scan"],
      "untrusted content with AI-directed instructions": ["prompt-injection"],
      "cross-frame messaging surface present": ["postmessage"],
      "DOM-XSS sink present": ["dom-xss"],
      "insecure form issue": ["insecure-forms"],
      "CSP weakness": ["csp-misconfig"],
      "cross-origin beacon/egress endpoint": ["exfil-egress"],
      "outbound egress request in source": ["exfil-egress"],
      "dangerous link": ["dom-xss", "postmessage"],
      "exploitable surface found": ["dom-xss", "postmessage"],
      "execution/exfil proven": ["exfil-egress", "dom-xss"],
      "sensitive path referenced": ["sensitive-path"],
      "error detail exposed": ["error-disclosure"],
      "third-party without SRI": ["supply-chain"],
      "information disclosure": ["info-disclosure"],
    };
    const signalFits = (ev: string, key: string): boolean => {
      const owners = SIGNAL_OWNER[ev];
      return !owners || owners.includes(key);   // rótulo desconhecido: confia (comportamento antigo)
    };

  const keys = specialties.map(s => s.key).join(", ");
  const COARSE_Q = `Ignoring the page's overall purpose: does THIS specific region contain anything that could matter for any of these security concerns — ${keys}? Answer yes only if this region itself holds relevant markup, code, data or text.`;

  const regions = segment(o.html);
  let regionsTotal = regions.length;
  let regionsScanned = 0;

  // ---- PASS 1: locate ------------------------------------------------------------
  const hot: Region[] = [];
  if (strategy === "tools-first" && o.tools) {
    // TOOLS-FIRST (§18r): a tool determinística é barata e tem recall alto — ela localiza.
    // O System-1 entra só para JULGAR o sinal. Não gasta decide no coarse.
    for (const r of regions) {
      const sig: Record<string, string> = {};
      for (const sp of specialties) {
        for (const t of (sp.tools || []).filter(t => typeof o.tools![t] === "function")) {
          const ds = await runTool(o.tools!, t, sp.key);
          if (ds.length) { sig[sp.key] = ds[0].label; break; }
        }
      }
      regionsScanned++;
      if (Object.keys(sig).length) { r.p = 1; hot.push(r); (r as any)._sig = sig; }
    }
    notes.push(`tools-first: tools located ${hot.length}/${regions.length} region(s); System-1 used only to judge.`);
  } else if (o.exhaustive) {
    for (const r of regions) { r.p = 1; hot.push(r); regionsScanned++; }
    notes.push("exhaustive mode: coarse funnel skipped, every region crossed with every specialty.");
  } else {
    for (const r of regions) {
      const p = await ask(r.text, COARSE_Q, coarseThreshold);
      r.p = p; regionsScanned++;
      if (p >= coarseThreshold) hot.push(r);
      if (budgetExhausted) break;
    }
  }

  // ---- PASS 1b: zoom hot containers (bounded) ------------------------------------
  let pool = hot;
  for (let d = 0; d < zoomDepth && !budgetExhausted && pool.length; d++) {
    const next: Region[] = [];
    for (const r of pool) {
      const children = segment(r.text).filter(c => c.id !== "r1" || c.text !== r.text);
      if (children.length < 2) { next.push(r); continue; }
      const kept: Region[] = [];
      for (const c of children) {
        const p = await ask(c.text, COARSE_Q, coarseThreshold);
        c.p = p;
        c.locator = `${r.locator} >> ${c.locator}`;
        c.start += r.start; c.end += r.start;
        c.depth = r.depth + 1;
        regionsScanned++;
        if (p >= coarseThreshold) kept.push(c);
        if (budgetExhausted) break;
      }
      next.push(...(kept.length ? kept : [r]));
      if (budgetExhausted) break;
    }
    pool = next;
    regionsTotal = regionsScanned;   // honestidade do contador: regiões efetivamente consideradas
    o.onProgress?.(`zoom ${d + 1}: ${pool.length} hot region(s), ${decides} decides`);
  }

  // ---- PASS 2: cross located spans x selected specialties ------------------------
  const findings: ReconFinding[] = [];
  const locatedOnly: ReconReport["locatedOnly"] = [];
  for (const r of pool) {
    const cells: { sp: Specialty; p: number }[] = [];
    for (const sp of specialties) {
      // pergunta de PRESENÇA (não presença+explorabilidade): mediu-se que juntar as duas faz o modelo
      // responder "não" para um span que claramente contém o sinal mas ainda não o exploit.
      const p = await ask(r.text, (usePresence && sp.presence) || sp.probe, 0);
      cells.push({ sp, p });
      if (budgetExhausted) break;
    }
    const crossed = cells.filter(c => c.p >= threshold).sort((a, b) => b.p - a.p);
    if (!crossed.length) {
      const best = cells.reduce((a, b) => (b.p > a.p ? b : a), cells[0]);
      locatedOnly.push({ locator: r.locator, relevance: r.p || 0, snippet: snippetOf(r.text) });
      if (best) notes.push(`located ${r.locator} (coarse ${(r.p || 0).toFixed(2)}) but no specialty crossed ${threshold} (best ${best.sp.key} ${best.p.toFixed(2)}).`);
      if (budgetExhausted) break;
      continue;
    }
    // ---- PASS 2b (TOOLS-FIRST): a tool determinística ACHA o sinal (recall), o System-1 JULGA (precisão).
    // Medido no bench (§18q): o CT0 perguntado sobre HTML satura (tudo P=1.0) ou é ruído (greedy 0/18);
    // quem de fato discrimina é a tool. Então as tools rodam para TODAS as especialidades da região e,
    // se nenhuma disparar evidência, o span é localizado+descartado — não vira achado por "cross" do modelo.
    const toolSignals: Record<string, string> = {};
    const toolDetections: Record<string, Detection> = {};
    if (o.tools) {
      for (const sp of specialties) {
        const runnable = (sp.tools || []).filter(t => typeof o.tools![t] === "function");
        for (const t of runnable) {
          const ds = await runTool(o.tools!, t, sp.key);
          if (ds.length) { toolSignals[sp.key] = ds[0].label; toolDetections[sp.key] = ds[0]; break; }
        }
      }
    }
    const hasToolSignal = Object.keys(toolSignals).length > 0;
    if (o.tools && !hasToolSignal) {
      locatedOnly.push({ locator: `${r.locator} [hot by relevance, no tool evidence]`, relevance: r.p || 0, snippet: snippetOf(r.text) });
      continue;
    }
    // Em TOOLS-FIRST o conjunto de especialidades vem da TOOL, não do modelo: o `crossed` do decide
    // satura (P=1.0 em tudo) e reintroduzia as 7 especialidades em toda página (§19c). Aqui só segue
    // adiante o que a tool realmente sustentou.
    const accepted = (strategy === "tools-first" && o.tools)
      ? crossed.filter(c => toolSignals[c.sp.key])
      : crossed;
    if (!accepted.length) {
      locatedOnly.push({ locator: `${r.locator} [model crossed but tools found no evidence]`, relevance: r.p || 0, snippet: snippetOf(r.text) });
      continue;
    }
    for (const c of accepted) {
      // ---- PASS 3: signal comes from the TOOL; the model only judges it ----
      let signal = toolSignals[c.sp.key] || "";
      let corroborated = !!signal;
      if (!corroborated && !o.tools) {
        // sem tools injetadas, o sinal é a própria relevância do System-1 (modo degradado, menos preciso)
        corroborated = false;
      }
      // ---- PASS 4: final System-1 verdict — CONTRASTIVO ----
      // Mediu-se que perguntar "X está presente aqui?" sobre um span satura (P=1.00 em tudo, porque
      // qualquer trecho de página real contém algum sinal). Perguntar "este é O SÍTIO de X, comparado
      // ao resto da página?" força uma escolha relacional e é o que dá ordenação útil.
      const evidenceState = corroborated
        ? `This is region ${r.locator} of a larger page, with deterministic tool evidence.\nMARKUP:\n${r.text}\nTOOL EVIDENCE: ${signal}`
        : `This is region ${r.locator} of a larger page.\nMARKUP:\n${r.text}`;
      const verdict = await ask(
        evidenceState,
        `Compared to the rest of the page, is THIS region specifically the place where ${c.sp.key} lives${corroborated ? " (the tool evidence above confirms it)" : ""}? Answer yes only if this region — not merely the page as a whole — is the locus.`,
        c.p
      );
      // gate de veredito: se o System-1 contrastivo REJEITOU o span (verdict < threshold) e não há
      // corroboração determinística, não é achado — é "localizado, não atribuído". Sem isso o cross
      // saturado infla o relatório com regiões em que o próprio modelo, ao ser perguntado, disse não.
      if (verdictGate && !corroborated && verdict < threshold) {
        locatedOnly.push({ locator: `${r.locator} [${c.sp.key} rejected by verdict P=${verdict.toFixed(2)}]`, relevance: r.p || 0, snippet: snippetOf(r.text) });
        continue;
      }
      findings.push({
        id: `R${findings.length + 1}`,
        specialty: c.sp.key,
        locator: r.locator,
        kind: r.kind,
        offsets: [r.start, r.end],
        snippet: snippetOf(r.text),
        relevance: r.p || 0,
        confidence: c.p,
        verdict,
        signal: signal || "System-1 relevance (no deterministic corroboration)",
        corroborated,
        severity: severityFor(c.sp, corroborated, verdict),
        nextStep: c.sp.nextStep || "verify manually.",
      });
    }
    if (budgetExhausted) break;
  }

  findings.sort((a, b) => (Number(b.corroborated) - Number(a.corroborated)) || (b.verdict - a.verdict) || (b.confidence - a.confidence));

  // dedup de spans aninhados: quando um achado da mesma especialidade está CONTIDO noutro
  // (ex. `main` e o `form` dentro dele), manter o mais específico (menor span) e descartar o pai.
  // Geral: o zoom produz regiões aninhadas; sem isso o relatório repete o mesmo problema.
  const kept: ReconFinding[] = [];
  for (const f of findings) {
    const dup = kept.findIndex(k =>
      k.specialty === f.specialty &&
      ((k.offsets[0] <= f.offsets[0] && f.offsets[1] <= k.offsets[1]) ||
        (f.offsets[0] <= k.offsets[0] && k.offsets[1] <= f.offsets[1]))
    );
    if (dup === -1) { kept.push(f); continue; }
    const a = kept[dup], aLen = a.offsets[1] - a.offsets[0], fLen = f.offsets[1] - f.offsets[0];
    if (fLen < aLen || (fLen === aLen && Number(f.corroborated) > Number(a.corroborated))) kept[dup] = f;
  }
  findings.length = 0;
  findings.push(...kept.map((f, i) => ({ ...f, id: `R${i + 1}` })));

  // H6: encadeamento — combina achados que, juntos, exploram algo que nenhum explora sozinho.
  // Só faz sentido correlacionar o que passou pelo juiz, então é determinístico aqui e o System-2
  // pode refinar via judges.judgePass({chains:true}).
  const chains = await correlate(findings);

  if (budgetExhausted) notes.push(`System-1 budget of ${maxDecides} decides exhausted — coverage is INCOMPLETE; raise maxDecides or narrow specialties.`);
  if (!findings.length) notes.push("No specialty crossed the threshold on any located span. Absence of a signal is not proof of safety.");

  return {
    target: o.target,
    specialties: specialties.map(s => s.key),
    mode: "programmed",
    regionsTotal,
    regionsScanned,
    regionsHot: pool.length,
    decides,
    budgetExhausted,
    findings,
    chains,
    locatedOnly,
    notes,
  };
}

/**
 * Mode B support: the same navigation steps exposed as ReAct tools, so a conductor model can drive
 * the recon through `Agent.run` (which already carries the loop guards, tool resolution and the
 * evidence oracle). The conductor only chooses NAVIGATION; System-1 still does every judgement.
 * Stage 2 — see the README disclosure about on-device conductor latency.
 */
export function reconToolset(html: string, decide: Decide, specialties: Specialty[] | "all" = "all") {
  const sps = specialties === "all" ? BROWSER_SPECIALTIES : specialties;
  let regions = segment(html);
  const located: Region[] = [];
  return {
    regions: () => regions.slice(0, 60).map(r => `${r.id} ${r.locator} (${r.kind}, ${r.end - r.start}b)`).join("\n") || "no regions",
    zoom: (a: { id?: string }) => {
      const r = regions.find(x => x.id === String(a.id || "").trim());
      if (!r) return "unknown region id — call regions()";
      const kids = segment(r.text);
      if (kids.length < 2) return `region ${r.id} has no sub-structure (${r.text.length}b)`;
      kids.forEach((k, i) => { k.id = `${r.id}.${i + 1}`; k.locator = `${r.locator} >> ${k.locator}`; });
      regions = [...regions.filter(x => x.id !== r.id), ...kids];
      located.push(r);
      return kids.map(k => `${k.id} ${k.locator} (${k.kind}, ${k.end - k.start}b)`).join("\n");
    },
    cross: async (a: { id?: string; specialty?: string }) => {
      const r = regions.find(x => x.id === String(a.id || "").trim());
      if (!r) return "unknown region id — call regions()";
      const out: string[] = [];
      for (const sp of sps) {
        if (a.specialty && sp.key !== a.specialty) continue;
        const p = await decide(r.text, sp.probe);
        if (p >= 0.5) out.push(`${sp.key} P=${p.toFixed(2)}`);
      }
      located.push(r);
      return out.length ? out.join("\n") : "no specialty reached P>=0.50 in this region";
    },
    report: async (a: { summary?: string; findings?: string }) =>
      `recon report drafted: ${String(a.summary || "")} | located regions: ${located.map(r => r.locator).join(", ") || "none"} | findings: ${String(a.findings || "")}`,
  } as ToolMap;
}

/** Render a ReconReport as compact markdown — the bug-bounty-style recon artifact. */
export function formatRecon(r: ReconReport): string {
  const L: string[] = [];
  L.push(`# Recon${r.target ? ` — ${r.target}` : ""}`);
  L.push(`mode: ${r.mode} · specialties: ${r.specialties.join(", ")}`);
  L.push(`regions: ${r.regionsTotal} · scanned: ${r.regionsScanned} · hot: ${r.regionsHot} · System-1 decides: ${r.decides}${r.budgetExhausted ? " (budget exhausted)" : ""}`);
  L.push("");
  if (!r.findings.length) L.push("_No specialty crossed the threshold._");
  for (const f of r.findings) {
    L.push(`## ${f.id} · ${f.specialty} — ${f.severity}${f.corroborated ? " · corroborated" : ""}`);
    L.push(`- locator: \`${f.locator}\` (${f.kind}, ${f.offsets[0]}–${f.offsets[1]})`);
    L.push(`- confidence: P=${f.confidence.toFixed(2)} · verdict: P=${f.verdict.toFixed(2)} · relevance: P=${f.relevance.toFixed(2)}`);
    L.push(`- signal: ${f.signal}`);
    L.push(`- snippet: \`${f.snippet.slice(0, 120)}\``);
    L.push(`- next step: ${f.nextStep}`);
    L.push("");
  }
  if (r.chains && r.chains.length) {
    L.push("## Chains (composed findings)");
    for (const c of r.chains) {
      L.push(`### ${c.id} · ${c.name} — ${c.severity}${c.confidence !== undefined ? ` (P=${c.confidence.toFixed(2)})` : ""}`);
      L.push(`- parts: ${c.parts.join(" + ")}`);
      L.push(`- why: ${c.rationale}`);
      L.push(`- next step: ${c.nextStep}`);
    }
    L.push("");
  }
  if (r.locatedOnly.length) {
    L.push("## Located, not attributed");
    for (const l of r.locatedOnly) L.push(`- \`${l.locator}\` (relevance P=${l.relevance.toFixed(2)}) \`${l.snippet.slice(0, 90)}\``);
    L.push("");
  }
  if (r.notes.length) { L.push("## Notes"); for (const n of r.notes) L.push(`- ${n}`); }
  return L.join("\n");
}
