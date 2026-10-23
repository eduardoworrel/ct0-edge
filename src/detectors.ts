/**
 * detectors.ts — the SECURITY JUDGMENT layer of cai-lite.
 *
 * ── The contract ────────────────────────────────────────────────────────────────
 * `agent-browser-lite` OBSERVES. It returns raw facts about a page and nothing else:
 *     forms()   → [{ action: "http://x", method: "POST", hasPassword: true, hasCsrfToken: false }]
 *     links()   → ["/.env", "/backup.sql"]
 *     storage() → { admin_token: "ADM-77ac12ff90ee" }
 *
 * This module JUDGES those facts. Every string containing "⚠", a severity, or the name of a
 * vulnerability lives HERE, never in the browser library.
 *
 * The purity test: if a browser tool's output uses "insecure", "vulnerable", "dangerous",
 * "exposed" or "⚠", the judgment is in the wrong layer.
 *
 * ── Why this matters ────────────────────────────────────────────────────────────
 * A neutral tool can be reused by anything: an agent navigating, a test harness, a different
 * security policy. A tool that answers "is this dangerous?" hard-codes one policy. It also makes
 * the library lie: `forms()` said "⚠ action over HTTP" about a *fact* (the scheme), which is only
 * bad in a security context.
 *
 * ── Proof levels ────────────────────────────────────────────────────────────────
 * L0  inspection   — the artefact exists (script has a nonce). Not proof of exploitability.
 * L1  inference    — an error/exception is used as oracle. Ambiguous.
 * L2  effect proof — we CAUSE the effect and OBSERVE it (payload → window.__canary === 1).
 * L3  exfiltration — data crosses an origin boundary and is proven to arrive.
 * "Conclusive" means L2 or L3. Everything below is a signal, not a proof.
 */

export type ProofLevel = "L0" | "L1" | "L2" | "L3";
export type Severity = "info" | "low" | "medium" | "high";

/** A detector's verdict over raw facts. Never asserts more than the facts support. */
export interface Detection {
  /** The specialty this evidence belongs to (must match a Specialty.key). */
  key: string;
  /** True when the evidence is hard (a tool proved it), false when it merely suggests. */
  conclusive: boolean;
  proof: ProofLevel;
  severity: Severity;
  /** Short label used by the reporter and by SIGNAL_OWNER. */
  label: string;
  /** What specifically was seen — the raw fact, not a paraphrase. */
  detail: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fact shapes produced by agent-browser-lite (documented here so both sides agree)
// ─────────────────────────────────────────────────────────────────────────────

export interface FormFact {
  index: number; action: string; method: string;
  hasPassword: boolean; hasHiddenFields: boolean; hasCsrfToken: boolean;
  autocompleteOn: boolean; submitText?: string;
}
export interface LinkFact { href: string; target: string | null; rel: string | null; text?: string; }
export interface IframeFact { index: number; src: string; sandbox: string | null; allow: string | null; hasSrcdoc: boolean; }
export interface ClobberFact { tag: string; name: string; collidesWith: string; }
export interface CookieFact { name: string; value: string; }
export interface StorageFact { store: "localStorage" | "sessionStorage" | "cookie"; name: string; value: string; }
export interface SinkFact { kind: string; snippet: string; line?: number; }
export interface CspFact { source: "meta" | "header" | "none"; policy: string; reportOnly: boolean; directive?: string; }
export interface PathFact { path: string; where: string; }
export interface ErrorFact { kind: string; evidence: string; credential?: string; }
export interface ThirdPartyFact { tag: string; url: string; integrity: string | null; };
export interface InfoFact { kind: string; detail: string; }
export interface NetworkFact { type: string; via: string; url: string; carriesCookie?: boolean; isCrossOrigin?: boolean; }
export interface MessageFact { origin: string; data: string; }

// ─────────────────────────────────────────────────────────────────────────────
// L0/L2 detectors — deterministic judgments over facts
// ─────────────────────────────────────────────────────────────────────────────

const SECRET_NAME = /(api[_-]?key|secret|token|passw|credential|session|auth|jwt|private[_-]?key|access[_-]?key)/i;
const SECRET_VALUE = /^(sk_live_|pk_live_|AKIA|ghp_|xox[baprs]-|eyJ[A-Za-z0-9_-]{10,}|[A-Fa-f0-9]{32,}|[A-Za-z0-9+/=_-]{24,})/;
const PLACEHOLDER = /^(null|undefined|true|false|0|1|""|''|example|test|dummy|changeme|xxxxxxxx|<.*>|\$\{.*\})$/i;
const SENSITIVE_PATH = /(\/\.env|\/\.git\/|\/\.svn|\/\.DS_Store|backup|dump|\.sql(?:$|\?)|\/\.bak|\/\.old|config\.(?:json|ya?ml|php)|id_rsa|\.htpasswd|phpinfo|actuator|\/swagger|\/openapi|\/graphql|\/debug|\/\.well-known\/credentials)/i;
const COLLECT_ENDPOINT = /(collect|track|beacon|exfil|pixel|\/px|t\.gif|telemetry|analytics|metrics)/i;

const looksSecret = (v: string): boolean => {
  const s = (v || "").trim();
  if (s.length < 12 || PLACEHOLDER.test(s)) return false;
  return SECRET_VALUE.test(s) || (s.length >= 20 && /[A-Z]/.test(s) && /[a-z]/.test(s) && /[0-9]/.test(s));
};

/** Client-side secret: a name that looks like a credential OR a value that looks like one. */
export function detectSecret(facts: StorageFact[]): Detection[] {
  const hits = facts.filter(f => SECRET_NAME.test(f.name) || looksSecret(f.value));
  if (!hits.length) return [];
  return [{
    key: "secret-scan", conclusive: true, proof: "L2", severity: "high",
    label: "client-side secret exposed",
    detail: hits.map(h => `${h.store}[${h.name}]=${h.value.slice(0, 40)}`).join("; "),
  }];
}

/** Cookie readable from JS (no HttpOnly) — only a finding when it carries something worth stealing. */
export function detectJsReadableCookie(cookies: CookieFact[]): Detection[] {
  const real = cookies.filter(c => !SECRET_NAME.test(c.name) && looksSecret(c.value) ? true : looksSecret(c.value) || SECRET_NAME.test(c.name));
  if (!real.length) return [];
  return [{
    key: "secret-scan", conclusive: true, proof: "L2", severity: "high",
    label: "client-side secret exposed",
    detail: `JS-readable cookie(s): ${real.map(c => c.name).join(", ")} (no HttpOnly)`,
  }];
}

/** Insecure form: needs a REAL weakness. A GET search form with no token is not one. */
export function detectInsecureForm(forms: FormFact[]): Detection[] {
  const out: Detection[] = [];
  for (const f of forms) {
    const reasons: string[] = [];
    if (/^http:\/\//i.test(f.action)) reasons.push("action over HTTP");
    if (f.hasPassword && f.autocompleteOn) reasons.push("password field with autocomplete on");
    const stateful = f.hasPassword || f.method === "POST" || /\/login|\/signin|\/account|\/checkout|\/password/i.test(f.action);
    if (stateful && !f.hasCsrfToken) reasons.push("stateful form without CSRF token");
    if (!reasons.length) continue;
    out.push({
      key: "insecure-forms",
      conclusive: true, proof: "L2",
      severity: /^http:\/\//i.test(f.action) ? "high" : "medium",
      label: "insecure form issue",
      detail: `form#${f.index} ${f.action.slice(0, 60)} → ${reasons.join(", ")}`,
    });
  }
  return out;
}

/** Tabnabbing / javascript: URL. */
export function detectDangerousLink(links: LinkFact[]): Detection[] {
  const hits = links.filter(l =>
    (l.target === "_blank" && !/noopener/.test(l.rel || "")) || /^javascript:/i.test(l.href));
  if (!hits.length) return [];
  return [{
    key: "dom-xss", conclusive: true, proof: "L0", severity: "low",
    label: "dangerous link",
    detail: hits.map(h => (/^javascript:/i.test(h.href) ? `javascript: ${h.href.slice(0, 50)}` : `target=_blank without noopener: ${h.href.slice(0, 50)}`)).join("; "),
  }];
}

/** Iframe without sandbox — L0: the attribute is absent, we did not prove an exploit. */
export function detectUnsandboxedIframe(frames: IframeFact[]): Detection[] {
  const hits = frames.filter(f => f.sandbox === null);
  if (!hits.length) return [];
  return [{
    key: "postmessage", conclusive: false, proof: "L0", severity: "low",
    label: "cross-frame messaging surface present",
    detail: `iframe(s) without sandbox: ${hits.map(h => h.src.slice(0, 40)).join(", ")}`,
  }];
}

/** DOM clobbering — a real collision with a window global. */
export function detectClobber(facts: ClobberFact[]): Detection[] {
  if (!facts.length) return [];
  return [{
    key: "dom-xss", conclusive: true, proof: "L2", severity: "low",
    label: "exploitable surface found",
    detail: facts.map(f => `<${f.tag} name="${f.name}"> clobbers window.${f.collidesWith}`).join("; "),
  }];
}

/** XSS sink present — L0: a sink exists, we did not prove tainted data reaches it. */
export function detectSink(sinks: SinkFact[]): Detection[] {
  if (!sinks.length) return [];
  return [{
    key: "dom-xss", conclusive: false, proof: "L0", severity: "medium",
    label: "DOM-XSS sink present",
    detail: sinks.map(s => `${s.kind} — ${s.snippet.slice(0, 60)}`).join(" | "),
  }];
}

/** Weak or missing CSP. `source:"none"` is INDETERMINATE (header may exist) — explicitly not a finding. */
export function detectCspWeakness(facts: CspFact[]): Detection[] {
  const out: Detection[] = [];
  for (const f of facts) {
    if (f.source === "none") continue;                       // cannot see the header → not a finding
    const weak: string[] = [];
    if (/unsafe-inline/i.test(f.policy)) weak.push("'unsafe-inline'");
    if (/unsafe-eval/i.test(f.policy)) weak.push("'unsafe-eval'");
    if (/(^|\s)\*(\s|;|$)/.test(f.policy)) weak.push("wildcard source");
    if (!/script-src/i.test(f.policy)) weak.push("no explicit script-src");
    if (!weak.length) continue;
    out.push({
      key: "csp-misconfig", conclusive: true, proof: "L2", severity: "medium",
      label: "CSP weakness",
      detail: `${f.source} policy: ${weak.join(", ")}`,
    });
  }
  return out;
}

/** A sensitive path the page itself references. L0 until someone requests it. */
export function detectSensitivePath(facts: PathFact[]): Detection[] {
  const hits = facts.filter(f => SENSITIVE_PATH.test(f.path));
  if (!hits.length) return [];
  return [{
    key: "sensitive-path", conclusive: false, proof: "L0", severity: "high",
    label: "sensitive path referenced",
    detail: hits.map(h => `${h.path} (${h.where})`).join("; "),
  }];
}

/** Stack trace / detailed error shown to the user. */
export function detectErrorLeak(facts: ErrorFact[]): Detection[] {
  if (!facts.length) return [];
  const withCred = facts.filter(f => f.credential);
  return [{
    key: "error-disclosure", conclusive: true, proof: "L2",
    severity: withCred.length ? "high" : "medium",
    label: "error detail exposed",
    detail: facts.map(f => `${f.kind}${f.credential ? ` + credential: ${f.credential}` : ""}`).join("; "),
  }];
}

/** Third-party script without SRI — supply chain risk (L0: we cannot prove the CDN is compromised). */
export function detectNoSri(facts: ThirdPartyFact[]): Detection[] {
  const bad = facts.filter(f => !f.integrity);
  if (!bad.length) return [];
  return [{
    key: "supply-chain", conclusive: false, proof: "L0", severity: "medium",
    label: "third-party without SRI",
    detail: bad.map(b => `${b.tag}: ${b.url.slice(0, 60)}`).join("; "),
  }];
}

/** Version/sourcemap/debug disclosure. */
export function detectInfoDisclosure(facts: InfoFact[]): Detection[] {
  if (!facts.length) return [];
  return [{
    key: "info-disclosure", conclusive: true, proof: "L0", severity: "low",
    label: "information disclosure",
    detail: facts.map(f => `${f.kind}: ${f.detail.slice(0, 60)}`).join("; "),
  }];
}

/**
 * Exfiltration — L3, and only when it is REAL egress.
 *
 * A `<script src="https://cdn/x.js">` is cross-origin but it is the page loading its own
 * dependency, not data leaving. Egress is exfiltration only when the request carries a payload
 * to a collection endpoint, or provably carries the page's own cookie.
 */
export function detectExfil(facts: NetworkFact[]): Detection[] {
  const proven = facts.filter(f => f.carriesCookie);
  if (proven.length) {
    return [{
      key: "exfil-egress", conclusive: true, proof: "L3", severity: "high",
      label: "execution/exfil proven",
      detail: `outbound request carried the page cookie: ${proven.map(p => p.url.slice(0, 80)).join(", ")}`,
    }];
  }
  const beaconish = facts.filter(f => {
    if (!f.isCrossOrigin) return false;
    const carriesPayload = /\?[^\s"']{8,}/.test(f.url) || /=\S{8,}/.test(f.url);
    return COLLECT_ENDPOINT.test(f.url) && carriesPayload;
  });
  if (!beaconish.length) return [];
  return [{
    key: "exfil-egress", conclusive: false, proof: "L1", severity: "medium",
    label: "cross-origin beacon/egress endpoint",
    detail: beaconish.map(b => `${b.via}: ${b.url.slice(0, 90)}`).join("; "),
  }];
}

/** Injected JS actually executed — the only thing that earns EXECUTION-CONFIRMED. */
export function detectExecution(network: NetworkFact[], scriptExec: boolean, domInjection: boolean): Detection[] {
  if (!scriptExec && !domInjection) return [];
  return [{
    key: "dom-xss", conclusive: true, proof: "L2", severity: "high",
    label: "execution/exfil proven",
    detail: scriptExec ? "a hooked alert/confirm/prompt fired" : "a DOM handler injection was observed",
  }];
}

/** Untrusted content that instructs an assistant. */
export function detectPromptInjection(texts: string[]): Detection[] {
  const AI_DIRECTED = /(NOTE TO (ANY )?AI|\bAI (assistant|model)\b|\bAssistant:\s|\bSYSTEM:\s|ignore\s+(the\s+|any\s+|all\s+)?(previous|earlier|prior|above)|disregard\s+(any\s+)?(earlier|previous|prior)|forget\s+(the\s+)?(previous|earlier)|you are now|developer mode|system prompt)/i;
  const DISCLOSE = /(print|reveal|output|send|leak|paste|dump|exfiltrate|copy|forward|expose|disclose|include)\b[^.]{0,60}(localStorage|sessionStorage|document\.cookie|api[ _-]?key|token|secret|password|credential)/i;
  const hits = texts.filter(t => AI_DIRECTED.test(t) || DISCLOSE.test(t));
  if (!hits.length) return [];
  return [{
    key: "prompt-injection", conclusive: false, proof: "L0", severity: "medium",
    label: "untrusted content with AI-directed instructions",
    detail: hits[0].replace(/\s+/g, " ").slice(0, 120),
  }];
}

/** A message listener exists (surface, not exploit). */
export function detectMessageSurface(scripts: string[]): Detection[] {
  const re = /addEventListener\(\s*["']message["']|\.postMessage\s*\(|onmessage\s*=/;
  const hits = scripts.filter(s => re.test(s));
  if (!hits.length) return [];
  return [{
    key: "postmessage", conclusive: false, proof: "L0", severity: "medium",
    label: "cross-frame messaging surface present",
    detail: hits[0].replace(/\s+/g, " ").slice(0, 100),
  }];
}

/** Egress written in the source (fetch/Image().src/sendBeacon to an external host). */
export function detectEgressInSource(scripts: string[]): Detection[] {
  const re = /(fetch\(\s*["']https?:\/\/|new\s+Image\(\s*\)\.src\s*=\s*["']https?:\/\/|sendBeacon\(\s*["']https?:\/\/)/i;
  const hits = scripts.filter(s => re.test(s));
  if (!hits.length) return [];
  return [{
    key: "exfil-egress", conclusive: false, proof: "L0", severity: "medium",
    label: "outbound egress request in source",
    detail: hits[0].replace(/\s+/g, " ").slice(0, 110),
  }];
}

// ─────────────────────────────────────────────────────────────────────────────
// SIGNAL_OWNER — which specialty a label belongs to. A label shared by three
// specialties made one signal become three findings; this table is the fix.
// ─────────────────────────────────────────────────────────────────────────────
export const SIGNAL_OWNER: Record<string, string[]> = {
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

export function signalFits(label: string, specialtyKey: string): boolean {
  const owners = SIGNAL_OWNER[label];
  return !owners || owners.includes(specialtyKey);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bridge: run a neutral tool's facts through its detectors.
//
// The browser library returns facts; this function turns facts into evidence. It is the ONLY
// place that maps "tool name" → "which detector judged it", which keeps the policy in one file.
// ─────────────────────────────────────────────────────────────────────────────

/** Detect over one tool's raw output. `toolName` selects the detector; `facts` is the tool's return. */
export function detectFromTool(toolName: string, facts: any): Detection[] {
  switch (toolName) {
    case "storage":       return detectSecret(Array.isArray(facts) ? facts : []);
    case "cookies":       return detectJsReadableCookie(Array.isArray(facts) ? facts : []);
    case "forms":         return detectInsecureForm(Array.isArray(facts) ? facts : []);
    case "links":         return detectDangerousLink(Array.isArray(facts) ? facts : []);
    case "iframes":       return detectUnsandboxedIframe(Array.isArray(facts) ? facts : []);
    case "clobber":       return detectClobber(Array.isArray(facts) ? facts : []);
    case "sinks":         return detectSink(Array.isArray(facts) ? facts : []);
    case "csp":           return detectCspWeakness(facts ? [facts] : []);
    case "sensitivePaths":return detectSensitivePath(Array.isArray(facts) ? facts : []);
    case "errorLeak":     return detectErrorLeak(Array.isArray(facts) ? facts : []);
    case "thirdPartyNoSri": return detectNoSri(Array.isArray(facts) ? facts : []);
    case "infoDisclosure":return detectInfoDisclosure(Array.isArray(facts) ? facts : []);
    case "inspect": {
      // inspect() devolve a estrutura crua: derive fatos e rode os detectores de conteúdo
      let d: any = facts;
      if (typeof d === "string") { try { d = JSON.parse(d); } catch { d = {}; } }
      d = d || {};
      const out: Detection[] = [];
      const scripts: string[] = [...(d.inlineScripts || []), ...(d.hiddenTexts || []), ...(d.texts || []), ...(d.comments || [])];
      out.push(...detectMessageSurface(scripts));
      out.push(...detectEgressInSource(scripts));
      out.push(...detectPromptInjection(scripts));
      // segredos inline nos scripts/textos — inclui LEITURA de cookie/localStorage, que expõe
      // material de sessão ao JS (é o que define "segredo acessível client-side").
      const inlineHits = scripts.filter(s =>
        /(?:api[_-]?key|client_secret|secret_key|access_token|api_token|auth_token)\s*["']?\s*[:=]\s*["'][A-Za-z0-9_\-]{12,}["']/i.test(s) ||
        /\b(sk_live|pk_live|AKIA|ghp_|xox[baprs]-)[A-Za-z0-9_\-]{8,}/.test(s) ||
        /\bADM-[0-9a-f]{8,}/.test(s) ||
        /document\.cookie\s*(?:=|\+)|localStorage\.(?:getItem|setItem)\s*\(\s*["'][^"']*(?:token|key|secret|session|auth)/i.test(s));
      if (inlineHits.length) out.push({ key: "secret-scan", conclusive: true, proof: "L2", severity: "high",
        label: "client-side secret exposed", detail: inlineHits[0].replace(/\s+/g, " ").slice(0, 110) });
      // cookies/localStorage que o inspect expõe
      const ls = Object.entries(d.localStorage || {}).map(([name, value]) => ({ store: "localStorage" as const, name, value: String(value) }));
      if (ls.length) out.push(...detectSecret(ls));
      // cookies que a página definiu (nome+valor crus) vêm em d.cookies quando disponível
      const cks = Array.isArray(d.cookies) ? d.cookies : [];
      if (cks.length) out.push(...detectJsReadableCookie(cks));
      return out;
    }
    case "resources": {
      const list = Array.isArray(facts) ? facts : [];
      return detectExfil(list.map((r: any) => ({
        type: "resource", via: String(r.initiatorType || "resource"), url: String(r.name || r.url || ""),
        isCrossOrigin: !!r.crossOrigin,
      })));
    }
    default: return [];
  }
}
