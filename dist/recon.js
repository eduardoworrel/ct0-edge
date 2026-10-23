// src/detectors.ts
var SECRET_NAME = /(api[_-]?key|secret|token|passw|credential|session|auth|jwt|private[_-]?key|access[_-]?key)/i;
var SECRET_VALUE = /^(sk_live_|pk_live_|AKIA|ghp_|xox[baprs]-|eyJ[A-Za-z0-9_-]{10,}|[A-Fa-f0-9]{32,}|[A-Za-z0-9+/=_-]{24,})/;
var PLACEHOLDER = /^(null|undefined|true|false|0|1|""|''|example|test|dummy|changeme|xxxxxxxx|<.*>|\$\{.*\})$/i;
var SENSITIVE_PATH = /(\/\.env|\/\.git\/|\/\.svn|\/\.DS_Store|backup|dump|\.sql(?:$|\?)|\/\.bak|\/\.old|config\.(?:json|ya?ml|php)|id_rsa|\.htpasswd|phpinfo|actuator|\/swagger|\/openapi|\/graphql|\/debug|\/\.well-known\/credentials)/i;
var COLLECT_ENDPOINT = /(collect|track|beacon|exfil|pixel|\/px|t\.gif|telemetry|analytics|metrics)/i;
var looksSecret = (v) => {
  const s = (v || "").trim();
  if (s.length < 12 || PLACEHOLDER.test(s)) return false;
  return SECRET_VALUE.test(s) || s.length >= 20 && /[A-Z]/.test(s) && /[a-z]/.test(s) && /[0-9]/.test(s);
};
function detectSecret(facts) {
  const hits = facts.filter((f) => SECRET_NAME.test(f.name) || looksSecret(f.value));
  if (!hits.length) return [];
  return [{
    key: "secret-scan",
    conclusive: true,
    proof: "L2",
    severity: "high",
    label: "client-side secret exposed",
    detail: hits.map((h) => `${h.store}[${h.name}]=${h.value.slice(0, 40)}`).join("; ")
  }];
}
function detectJsReadableCookie(cookies) {
  const real = cookies.filter((c) => !SECRET_NAME.test(c.name) && looksSecret(c.value) ? true : looksSecret(c.value) || SECRET_NAME.test(c.name));
  if (!real.length) return [];
  return [{
    key: "secret-scan",
    conclusive: true,
    proof: "L2",
    severity: "high",
    label: "client-side secret exposed",
    detail: `JS-readable cookie(s): ${real.map((c) => c.name).join(", ")} (no HttpOnly)`
  }];
}
function detectInsecureForm(forms) {
  const out = [];
  for (const f of forms) {
    const reasons = [];
    if (/^http:\/\//i.test(f.action)) reasons.push("action over HTTP");
    if (f.hasPassword && f.autocompleteOn) reasons.push("password field with autocomplete on");
    const stateful = f.hasPassword || f.method === "POST" || /\/login|\/signin|\/account|\/checkout|\/password/i.test(f.action);
    if (stateful && !f.hasCsrfToken) reasons.push("stateful form without CSRF token");
    if (!reasons.length) continue;
    out.push({
      key: "insecure-forms",
      conclusive: true,
      proof: "L2",
      severity: /^http:\/\//i.test(f.action) ? "high" : "medium",
      label: "insecure form issue",
      detail: `form#${f.index} ${f.action.slice(0, 60)} \u2192 ${reasons.join(", ")}`
    });
  }
  return out;
}
function detectDangerousLink(links) {
  const hits = links.filter((l) => l.target === "_blank" && !/noopener/.test(l.rel || "") || /^javascript:/i.test(l.href));
  if (!hits.length) return [];
  return [{
    key: "dom-xss",
    conclusive: true,
    proof: "L0",
    severity: "low",
    label: "dangerous link",
    detail: hits.map((h) => /^javascript:/i.test(h.href) ? `javascript: ${h.href.slice(0, 50)}` : `target=_blank without noopener: ${h.href.slice(0, 50)}`).join("; ")
  }];
}
function detectUnsandboxedIframe(frames) {
  const hits = frames.filter((f) => f.sandbox === null);
  if (!hits.length) return [];
  return [{
    key: "postmessage",
    conclusive: false,
    proof: "L0",
    severity: "low",
    label: "cross-frame messaging surface present",
    detail: `iframe(s) without sandbox: ${hits.map((h) => h.src.slice(0, 40)).join(", ")}`
  }];
}
function detectClobber(facts) {
  if (!facts.length) return [];
  return [{
    key: "dom-xss",
    conclusive: true,
    proof: "L2",
    severity: "low",
    label: "exploitable surface found",
    detail: facts.map((f) => `<${f.tag} name="${f.name}"> clobbers window.${f.collidesWith}`).join("; ")
  }];
}
function detectSink(sinks) {
  if (!sinks.length) return [];
  return [{
    key: "dom-xss",
    conclusive: false,
    proof: "L0",
    severity: "medium",
    label: "DOM-XSS sink present",
    detail: sinks.map((s) => `${s.kind} \u2014 ${s.snippet.slice(0, 60)}`).join(" | ")
  }];
}
function detectCspWeakness(facts) {
  const out = [];
  for (const f of facts) {
    if (f.source === "none") continue;
    const weak = [];
    if (/unsafe-inline/i.test(f.policy)) weak.push("'unsafe-inline'");
    if (/unsafe-eval/i.test(f.policy)) weak.push("'unsafe-eval'");
    if (/(^|\s)\*(\s|;|$)/.test(f.policy)) weak.push("wildcard source");
    if (!/script-src/i.test(f.policy)) weak.push("no explicit script-src");
    if (!weak.length) continue;
    out.push({
      key: "csp-misconfig",
      conclusive: true,
      proof: "L2",
      severity: "medium",
      label: "CSP weakness",
      detail: `${f.source} policy: ${weak.join(", ")}`
    });
  }
  return out;
}
function detectSensitivePath(facts) {
  const hits = facts.filter((f) => SENSITIVE_PATH.test(f.path));
  if (!hits.length) return [];
  return [{
    key: "sensitive-path",
    conclusive: false,
    proof: "L0",
    severity: "high",
    label: "sensitive path referenced",
    detail: hits.map((h) => `${h.path} (${h.where})`).join("; ")
  }];
}
function detectErrorLeak(facts) {
  if (!facts.length) return [];
  const withCred = facts.filter((f) => f.credential);
  return [{
    key: "error-disclosure",
    conclusive: true,
    proof: "L2",
    severity: withCred.length ? "high" : "medium",
    label: "error detail exposed",
    detail: facts.map((f) => `${f.kind}${f.credential ? ` + credential: ${f.credential}` : ""}`).join("; ")
  }];
}
function detectNoSri(facts) {
  const bad = facts.filter((f) => !f.integrity);
  if (!bad.length) return [];
  return [{
    key: "supply-chain",
    conclusive: false,
    proof: "L0",
    severity: "medium",
    label: "third-party without SRI",
    detail: bad.map((b) => `${b.tag}: ${b.url.slice(0, 60)}`).join("; ")
  }];
}
function detectInfoDisclosure(facts) {
  if (!facts.length) return [];
  return [{
    key: "info-disclosure",
    conclusive: true,
    proof: "L0",
    severity: "low",
    label: "information disclosure",
    detail: facts.map((f) => `${f.kind}: ${f.detail.slice(0, 60)}`).join("; ")
  }];
}
function detectExfil(facts) {
  const proven = facts.filter((f) => f.carriesCookie);
  if (proven.length) {
    return [{
      key: "exfil-egress",
      conclusive: true,
      proof: "L3",
      severity: "high",
      label: "execution/exfil proven",
      detail: `outbound request carried the page cookie: ${proven.map((p) => p.url.slice(0, 80)).join(", ")}`
    }];
  }
  const beaconish = facts.filter((f) => {
    if (!f.isCrossOrigin) return false;
    const carriesPayload = /\?[^\s"']{8,}/.test(f.url) || /=\S{8,}/.test(f.url);
    return COLLECT_ENDPOINT.test(f.url) && carriesPayload;
  });
  if (!beaconish.length) return [];
  return [{
    key: "exfil-egress",
    conclusive: false,
    proof: "L1",
    severity: "medium",
    label: "cross-origin beacon/egress endpoint",
    detail: beaconish.map((b) => `${b.via}: ${b.url.slice(0, 90)}`).join("; ")
  }];
}
function detectPromptInjection(texts) {
  const AI_DIRECTED = /(NOTE TO (ANY )?AI|\bAI (assistant|model)\b|\bAssistant:\s|\bSYSTEM:\s|ignore\s+(the\s+|any\s+|all\s+)?(previous|earlier|prior|above)|disregard\s+(any\s+)?(earlier|previous|prior)|forget\s+(the\s+)?(previous|earlier)|you are now|developer mode|system prompt)/i;
  const DISCLOSE = /(print|reveal|output|send|leak|paste|dump|exfiltrate|copy|forward|expose|disclose|include)\b[^.]{0,60}(localStorage|sessionStorage|document\.cookie|api[ _-]?key|token|secret|password|credential)/i;
  const hits = texts.filter((t) => AI_DIRECTED.test(t) || DISCLOSE.test(t));
  if (!hits.length) return [];
  return [{
    key: "prompt-injection",
    conclusive: false,
    proof: "L0",
    severity: "medium",
    label: "untrusted content with AI-directed instructions",
    detail: hits[0].replace(/\s+/g, " ").slice(0, 120)
  }];
}
function detectMessageSurface(scripts) {
  const re = /addEventListener\(\s*["']message["']|\.postMessage\s*\(|onmessage\s*=/;
  const hits = scripts.filter((s) => re.test(s));
  if (!hits.length) return [];
  return [{
    key: "postmessage",
    conclusive: false,
    proof: "L0",
    severity: "medium",
    label: "cross-frame messaging surface present",
    detail: hits[0].replace(/\s+/g, " ").slice(0, 100)
  }];
}
function detectEgressInSource(scripts) {
  const re = /(fetch\(\s*["']https?:\/\/|new\s+Image\(\s*\)\.src\s*=\s*["']https?:\/\/|sendBeacon\(\s*["']https?:\/\/)/i;
  const hits = scripts.filter((s) => re.test(s));
  if (!hits.length) return [];
  return [{
    key: "exfil-egress",
    conclusive: false,
    proof: "L0",
    severity: "medium",
    label: "outbound egress request in source",
    detail: hits[0].replace(/\s+/g, " ").slice(0, 110)
  }];
}
function detectFromTool(toolName, facts) {
  switch (toolName) {
    case "storage":
      return detectSecret(Array.isArray(facts) ? facts : []);
    case "cookies":
      return detectJsReadableCookie(Array.isArray(facts) ? facts : []);
    case "forms":
      return detectInsecureForm(Array.isArray(facts) ? facts : []);
    case "links":
      return detectDangerousLink(Array.isArray(facts) ? facts : []);
    case "iframes":
      return detectUnsandboxedIframe(Array.isArray(facts) ? facts : []);
    case "clobber":
      return detectClobber(Array.isArray(facts) ? facts : []);
    case "sinks":
      return detectSink(Array.isArray(facts) ? facts : []);
    case "csp":
      return detectCspWeakness(facts ? [facts] : []);
    case "sensitivePaths":
      return detectSensitivePath(Array.isArray(facts) ? facts : []);
    case "errorLeak":
      return detectErrorLeak(Array.isArray(facts) ? facts : []);
    case "thirdPartyNoSri":
      return detectNoSri(Array.isArray(facts) ? facts : []);
    case "infoDisclosure":
      return detectInfoDisclosure(Array.isArray(facts) ? facts : []);
    case "inspect": {
      let d = facts;
      if (typeof d === "string") {
        try {
          d = JSON.parse(d);
        } catch {
          d = {};
        }
      }
      d = d || {};
      const out = [];
      const scripts = [...d.inlineScripts || [], ...d.hiddenTexts || [], ...d.texts || [], ...d.comments || []];
      out.push(...detectMessageSurface(scripts));
      out.push(...detectEgressInSource(scripts));
      out.push(...detectPromptInjection(scripts));
      const inlineHits = scripts.filter((s) => /(?:api[_-]?key|client_secret|secret_key|access_token|api_token|auth_token)\s*["']?\s*[:=]\s*["'][A-Za-z0-9_\-]{12,}["']/i.test(s) || /\b(sk_live|pk_live|AKIA|ghp_|xox[baprs]-)[A-Za-z0-9_\-]{8,}/.test(s) || /\bADM-[0-9a-f]{8,}/.test(s) || /document\.cookie\s*(?:=|\+)|localStorage\.(?:getItem|setItem)\s*\(\s*["'][^"']*(?:token|key|secret|session|auth)/i.test(s));
      if (inlineHits.length) out.push({
        key: "secret-scan",
        conclusive: true,
        proof: "L2",
        severity: "high",
        label: "client-side secret exposed",
        detail: inlineHits[0].replace(/\s+/g, " ").slice(0, 110)
      });
      const ls = Object.entries(d.localStorage || {}).map(([name, value]) => ({ store: "localStorage", name, value: String(value) }));
      if (ls.length) out.push(...detectSecret(ls));
      const cks = Array.isArray(d.cookies) ? d.cookies : [];
      if (cks.length) out.push(...detectJsReadableCookie(cks));
      return out;
    }
    case "resources": {
      const list = Array.isArray(facts) ? facts : [];
      return detectExfil(list.map((r) => ({
        type: "resource",
        via: String(r.initiatorType || "resource"),
        url: String(r.name || r.url || ""),
        isCrossOrigin: !!r.crossOrigin
      })));
    }
    default:
      return [];
  }
}

// src/recon.ts
var BROWSER_SPECIALTIES = [
  {
    key: "dom-xss",
    probe: "Does this page render user-controllable input into the DOM (a preview, comment, search echo, template) such that a stored/reflected/DOM XSS is plausible?",
    presence: "Does this excerpt itself contain markup, code or text that turns user-controllable input into DOM content \u2014 a preview/sink (innerHTML, insertAdjacentHTML, document.write), a template, a comment box, a search echo, or user input rendered as HTML?",
    tools: ["sinks", "source", "snapshot", "scan", "type", "readHTML", "events"],
    nextStep: "inject a benign marker through the input and confirm it reaches a sink (readHTML) and executes (events).",
    severity: "high"
  },
  {
    key: "prompt-injection",
    probe: "Does this page contain untrusted document/content that an AI assistant might read and be manipulated by (hidden instructions, fake system notes)?",
    presence: "Does this excerpt itself contain text that reads like instructions aimed at an AI (ignore/disregard previous instructions, fake system notes, hidden text directing an assistant), or untrusted content an assistant would ingest?",
    tools: ["source", "inspect", "scan"],
    nextStep: "quote the exact instruction text and show the path by which untrusted content reaches an assistant prompt.",
    severity: "medium"
  },
  {
    key: "secret-scan",
    probe: "Might this page expose secrets or tokens client-side (in localStorage, cookies, inline scripts, or comments)?",
    presence: "Does this excerpt itself contain a secret or token material \u2014 a hardcoded key/credential, a value read from localStorage/cookies, an inline token, or a comment carrying one?",
    tools: ["storage", "inspect", "cookies", "scan"],
    nextStep: "read the value with JS (document.cookie / localStorage) and confirm it is not HttpOnly-only.",
    severity: "high"
  },
  {
    key: "postmessage",
    probe: "Does this page use cross-frame messaging (iframes, window.postMessage, message listeners) that could be exploited across origins?",
    presence: "Does this excerpt itself contain cross-frame messaging mechanics \u2014 postMessage() calls, addEventListener('message') / onmessage listeners, or embedded iframes that exchange messages?",
    tools: ["messages", "inspect", "sinks", "source", "scan"],
    nextStep: "send a cross-origin message to the listener and observe whether origin is validated before use.",
    severity: "medium"
  },
  {
    key: "csp-misconfig",
    probe: "Is this page likely missing a Content-Security-Policy or using a weak one (unsafe-inline/eval, wildcards)?",
    presence: "Does this excerpt itself indicate a CSP problem \u2014 a meta Content-Security-Policy, or inline scripts/handlers that a strict policy would have to allow (unsafe-inline/unsafe-eval, wildcard)?",
    tools: ["csp", "source"],
    nextStep: "show the response policy and which directive is weak; demonstrate an inline payload it fails to block.",
    severity: "low"
  },
  {
    key: "exfil-egress",
    probe: "Does this page make outbound/cross-origin network requests or beacons that could carry data off-origin?",
    presence: "Does this excerpt itself contain outbound network code \u2014 fetch/XHR/sendBeacon/Image().src to an external URL, or a tracking/beacon endpoint?",
    tools: ["resources", "events", "cookies", "storage"],
    nextStep: "capture the outbound request and decode its payload to show what data leaves the origin.",
    severity: "medium"
  },
  {
    key: "sensitive-path",
    probe: "Does this page reference sensitive paths that look publicly reachable \u2014 environment files, VCS metadata, database dumps, backups, admin or debug endpoints?",
    presence: "Does this excerpt itself reference a sensitive path (.env, .git, .svn, backup, dump, .sql, .bak, config.json, id_rsa, admin, debug, swagger, graphql)?",
    tools: ["sensitivePaths", "links", "source"],
    nextStep: "request the referenced path and confirm whether it is publicly served (do not download user data).",
    severity: "high"
  },
  {
    key: "error-disclosure",
    probe: "Does this page display a stack trace or detailed internal error to the user, exposing paths, queries or credentials?",
    presence: "Does this excerpt itself contain a stack trace, framework exception, SQL error or absolute internal path?",
    tools: ["errorLeak", "inspect", "source"],
    nextStep: "capture the exact error text and any credential it leaks, and show which request triggers it.",
    severity: "high"
  },
  {
    key: "supply-chain",
    probe: "Does this page load third-party scripts or stylesheets without Subresource Integrity, allowing a compromised CDN to run code here?",
    presence: "Does this excerpt itself reference an external (cross-origin) script or stylesheet, and does it lack an integrity attribute?",
    tools: ["thirdPartyNoSri", "resources"],
    nextStep: "list the third-party origins without integrity and confirm which one can execute in this origin.",
    severity: "medium"
  },
  {
    key: "info-disclosure",
    probe: "Does this page disclose information useful to an attacker \u2014 published source maps, debug mode, framework version, or hidden debug/admin fields?",
    presence: "Does this excerpt itself contain a sourceMappingURL, a DEBUG/verbose flag, a version string, or a hidden admin/debug field?",
    tools: ["infoDisclosure", "inspect", "source"],
    nextStep: "report the specific disclosure and what an attacker learns from it (version \u2192 known CVEs, debug \u2192 internals).",
    severity: "low"
  },
  {
    key: "insecure-forms",
    probe: "Does this page have forms handling credentials or sensitive data insecurely (non-HTTPS action, autocomplete on, no CSRF token)?",
    presence: "Does this excerpt itself contain a form or credential field \u2014 a <form> (any action), a password/email input, or auth-related markup?",
    tools: ["forms", "inspect", "cookies"],
    nextStep: "submit with a non-secret probe and confirm the action URL/scheme and missing token.",
    severity: "medium"
  }
];
async function runTool(tools, toolName, specialtyKey) {
  const fn = tools[toolName];
  if (typeof fn !== "function") return [];
  try {
    const facts = await fn({});
    return detectFromTool(toolName, facts).filter((d) => d.key === specialtyKey);
  } catch {
    return [];
  }
}
var VOID_TAGS = /* @__PURE__ */ new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"]);
var INTERESTING = /* @__PURE__ */ new Set(["form", "script", "textarea", "iframe", "object", "embed", "template", "table", "main", "section", "article", "nav", "aside", "header", "footer"]);
var BLOCK_TAGS = /* @__PURE__ */ new Set(["div", "p", "li", "td", "label", "button", "a", "span", "pre", "code"]);
var MAX_REGION_TEXT = 1500;
var MAX_REGIONS = 400;
function tokenize(html) {
  const re = /<!--[\s\S]*?-->|<\/([a-zA-Z][\w:-]*)\s*>|<([a-zA-Z][\w:-]*)((?:"[^"]*"|'[^']*'|[^>"'])*)>/g;
  const out = [];
  let m;
  while (m = re.exec(html)) {
    const start = m.index, end = start + m[0].length;
    if (m[0].startsWith("<!--")) out.push({ kind: "comment", start, end });
    else if (m[1]) out.push({ kind: "close", tag: m[1].toLowerCase(), start, end });
    else out.push({ kind: "open", tag: (m[2] || "").toLowerCase(), attrs: m[3] || "", start, end });
    if (out.length > MAX_REGIONS * 8) break;
  }
  return out;
}
function locatorFor(tok, ordinal) {
  if (tok.kind === "comment" || !tok.tag) return `comment[${ordinal}]`;
  const a = tok.attrs || "";
  const idm = a.match(/\sid\s*=\s*["']?([\w:.-]+)/i);
  if (idm) return `${tok.tag}#${idm[1]}`;
  const cl = a.match(/\sclass\s*=\s*["']?([\w:.-]+)/i);
  if (cl) return `${tok.tag}.${cl[1]}`;
  return `${tok.tag}[${ordinal}]`;
}
function segment(html, opts = {}) {
  const h = html || "";
  const toks = tokenize(h);
  const cap = opts.maxRegions ?? MAX_REGIONS;
  const regions = [];
  const seen = /* @__PURE__ */ new Map();
  const stack = [];
  let depth = 0;
  const push = (kind, tag, start, end, tok, d) => {
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
      if (text.length > 12) push("comment", void 0, t.start, t.end, t, depth);
      continue;
    }
    if (t.kind === "open") {
      const tag2 = t.tag;
      const inlineHandler = /\son[a-z]+\s*=/i.test(t.attrs || "");
      const isPassword = /type\s*=\s*["']?password/i.test(t.attrs || "");
      if (inlineHandler) {
        push("handler", tag2, t.start, t.end, t, depth);
      }
      if (isPassword) push("form", tag2, t.start, t.end, t, depth);
      if (VOID_TAGS.has(tag2)) continue;
      stack.push({ tag: tag2, start: t.start, tok: t });
      depth++;
      continue;
    }
    const tag = t.tag;
    let idx = -1;
    for (let k = stack.length - 1; k >= 0; k--) if (stack[k].tag === tag) {
      idx = k;
      break;
    }
    if (idx < 0) {
      depth = Math.max(0, depth - 1);
      continue;
    }
    const opened = stack[idx];
    stack.length = idx;
    depth = stack.length;
    const span = { start: opened.start, end: t.end };
    if (INTERESTING.has(tag)) {
      const kind = tag === "form" ? "form" : tag === "script" ? "script" : tag === "textarea" ? "textarea" : tag === "iframe" ? "iframe" : "container";
      push(kind, tag, span.start, span.end, opened.tok, opened.tok === opened.tok ? Math.max(0, depth) : 0);
    } else if (BLOCK_TAGS.has(tag) && t.end - opened.start >= 40) {
      push("block", tag, span.start, span.end, opened.tok, depth);
    }
  }
  return regions;
}
var CHAIN_RULES = [
  {
    name: "unvalidated postMessage \u2192 DOM sink",
    from: "postmessage",
    to: "dom-xss",
    rationale: "A message listener that uses e.data without validating e.origin, feeding a DOM sink, lets any framing origin drive script execution in this page.",
    severity: "high",
    nextStep: "send a cross-origin message carrying a benign marker and confirm it reaches the sink and executes.",
    require: /sink|postmessage|surface/i
  },
  {
    name: "client-side secret reachable by XSS",
    from: "dom-xss",
    to: "secret-scan",
    rationale: "An XSS path plus a secret readable from JS means the secret is exfiltratable by an attacker who lands the XSS.",
    severity: "high",
    nextStep: "confirm the secret is readable (not HttpOnly) and that the sink executes attacker-controlled input.",
    require: /secret|sink|surface/i
  },
  {
    name: "exfil endpoint + client-side secret",
    from: "secret-scan",
    to: "exfil-egress",
    rationale: "A secret readable from JS plus a cross-origin beacon/egress endpoint means data can leave the origin \u2014 the skimmer pattern.",
    severity: "high",
    nextStep: "decode the outbound request payload and show which secret it carries.",
    require: /beacon|egress|secret/i
  },
  {
    name: "credential form over HTTP + XSS",
    from: "insecure-forms",
    to: "dom-xss",
    rationale: "Credentials submitted over plaintext plus script execution gives two independent ways to capture them.",
    severity: "high",
    nextStep: "submit with a non-secret probe and confirm the scheme; confirm the sink executes.",
    require: /over http|no csrf/i
  },
  {
    name: "prompt-injection targeting client-side secrets",
    from: "prompt-injection",
    to: "secret-scan",
    rationale: "Untrusted content instructing an assistant to reveal secrets, in a page that actually holds a JS-readable secret, is a viable exfiltration path through the assistant.",
    severity: "medium",
    nextStep: "show the instruction text and the secret it targets, and the path by which content reaches the assistant.",
    require: /secret/i
  }
];
async function correlate(findings, judge) {
  const byKey = {};
  for (const f of findings) if (!byKey[f.specialty]) byKey[f.specialty] = f;
  const chains = [];
  for (const rule of CHAIN_RULES) {
    const a = byKey[rule.from], b = byKey[rule.to];
    if (!a || !b) continue;
    if (rule.require && !rule.require.test(`${a.signal} ${b.signal}`)) continue;
    const chain = {
      id: `C${chains.length + 1}`,
      parts: [a.id, b.id],
      name: rule.name,
      rationale: rule.rationale,
      severity: rule.severity,
      nextStep: rule.nextStep
    };
    if (judge) {
      try {
        chain.confidence = await judge(chain, [a, b]);
      } catch {
      }
    }
    chains.push(chain);
  }
  return chains;
}
var SNIPPET = 220;
function snippetOf(text) {
  return text.replace(/\s+/g, " ").trim().slice(0, SNIPPET);
}
function severityFor(sp, corroborated, verdict) {
  const base = sp.severity || "medium";
  if (!corroborated) return base === "high" ? "medium" : "low";
  if (verdict >= 0.8) return base;
  return base === "high" ? "medium" : "low";
}
async function recon(o) {
  const specialties = o.specialties === "all" || !o.specialties ? BROWSER_SPECIALTIES : o.specialties;
  const threshold = o.threshold ?? 0.5;
  const coarseThreshold = o.coarseThreshold ?? 0.35;
  const zoomDepth = o.zoomDepth ?? 1;
  const maxDecides = o.maxDecides ?? 120;
  const usePresence = o.usePresence ?? true;
  const verdictGate = o.verdictGate ?? true;
  const strategy = o.strategy ?? "decide-first";
  const notes = [];
  let decides = 0;
  let budgetExhausted = false;
  const ask = async (state, question, fallback = 0.5) => {
    if (decides >= maxDecides) {
      budgetExhausted = true;
      return fallback;
    }
    decides++;
    try {
      const p = await o.decide(state, question);
      return Number.isFinite(p) ? Math.max(0, Math.min(1, p)) : fallback;
    } catch {
      return fallback;
    }
  };
  const SIGNAL_OWNER2 = {
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
    "information disclosure": ["info-disclosure"]
  };
  const signalFits2 = (ev, key) => {
    const owners = SIGNAL_OWNER2[ev];
    return !owners || owners.includes(key);
  };
  const keys = specialties.map((s) => s.key).join(", ");
  const COARSE_Q = `Ignoring the page's overall purpose: does THIS specific region contain anything that could matter for any of these security concerns \u2014 ${keys}? Answer yes only if this region itself holds relevant markup, code, data or text.`;
  const regions = segment(o.html);
  let regionsTotal = regions.length;
  let regionsScanned = 0;
  const hot = [];
  if (strategy === "tools-first" && o.tools) {
    for (const r of regions) {
      const sig = {};
      for (const sp of specialties) {
        for (const t of (sp.tools || []).filter((t2) => typeof o.tools[t2] === "function")) {
          const ds = await runTool(o.tools, t, sp.key);
          if (ds.length) {
            sig[sp.key] = ds[0].label;
            break;
          }
        }
      }
      regionsScanned++;
      if (Object.keys(sig).length) {
        r.p = 1;
        hot.push(r);
        r._sig = sig;
      }
    }
    notes.push(`tools-first: tools located ${hot.length}/${regions.length} region(s); System-1 used only to judge.`);
  } else if (o.exhaustive) {
    for (const r of regions) {
      r.p = 1;
      hot.push(r);
      regionsScanned++;
    }
    notes.push("exhaustive mode: coarse funnel skipped, every region crossed with every specialty.");
  } else {
    for (const r of regions) {
      const p = await ask(r.text, COARSE_Q, coarseThreshold);
      r.p = p;
      regionsScanned++;
      if (p >= coarseThreshold) hot.push(r);
      if (budgetExhausted) break;
    }
  }
  let pool = hot;
  for (let d = 0; d < zoomDepth && !budgetExhausted && pool.length; d++) {
    const next = [];
    for (const r of pool) {
      const children = segment(r.text).filter((c) => c.id !== "r1" || c.text !== r.text);
      if (children.length < 2) {
        next.push(r);
        continue;
      }
      const kept2 = [];
      for (const c of children) {
        const p = await ask(c.text, COARSE_Q, coarseThreshold);
        c.p = p;
        c.locator = `${r.locator} >> ${c.locator}`;
        c.start += r.start;
        c.end += r.start;
        c.depth = r.depth + 1;
        regionsScanned++;
        if (p >= coarseThreshold) kept2.push(c);
        if (budgetExhausted) break;
      }
      next.push(...kept2.length ? kept2 : [r]);
      if (budgetExhausted) break;
    }
    pool = next;
    regionsTotal = regionsScanned;
    o.onProgress?.(`zoom ${d + 1}: ${pool.length} hot region(s), ${decides} decides`);
  }
  const findings = [];
  const locatedOnly = [];
  for (const r of pool) {
    const cells = [];
    for (const sp of specialties) {
      const p = await ask(r.text, usePresence && sp.presence || sp.probe, 0);
      cells.push({ sp, p });
      if (budgetExhausted) break;
    }
    const crossed = cells.filter((c) => c.p >= threshold).sort((a, b) => b.p - a.p);
    if (!crossed.length) {
      const best = cells.reduce((a, b) => b.p > a.p ? b : a, cells[0]);
      locatedOnly.push({ locator: r.locator, relevance: r.p || 0, snippet: snippetOf(r.text) });
      if (best) notes.push(`located ${r.locator} (coarse ${(r.p || 0).toFixed(2)}) but no specialty crossed ${threshold} (best ${best.sp.key} ${best.p.toFixed(2)}).`);
      if (budgetExhausted) break;
      continue;
    }
    const toolSignals = {};
    const toolDetections = {};
    if (o.tools) {
      for (const sp of specialties) {
        const runnable = (sp.tools || []).filter((t) => typeof o.tools[t] === "function");
        for (const t of runnable) {
          const ds = await runTool(o.tools, t, sp.key);
          if (ds.length) {
            toolSignals[sp.key] = ds[0].label;
            toolDetections[sp.key] = ds[0];
            break;
          }
        }
      }
    }
    const hasToolSignal = Object.keys(toolSignals).length > 0;
    if (o.tools && !hasToolSignal) {
      locatedOnly.push({ locator: `${r.locator} [hot by relevance, no tool evidence]`, relevance: r.p || 0, snippet: snippetOf(r.text) });
      continue;
    }
    const accepted = strategy === "tools-first" && o.tools ? crossed.filter((c) => toolSignals[c.sp.key]) : crossed;
    if (!accepted.length) {
      locatedOnly.push({ locator: `${r.locator} [model crossed but tools found no evidence]`, relevance: r.p || 0, snippet: snippetOf(r.text) });
      continue;
    }
    for (const c of accepted) {
      let signal = toolSignals[c.sp.key] || "";
      let corroborated = !!signal;
      if (!corroborated && !o.tools) {
        corroborated = false;
      }
      const evidenceState = corroborated ? `This is region ${r.locator} of a larger page, with deterministic tool evidence.
MARKUP:
${r.text}
TOOL EVIDENCE: ${signal}` : `This is region ${r.locator} of a larger page.
MARKUP:
${r.text}`;
      const verdict = await ask(
        evidenceState,
        `Compared to the rest of the page, is THIS region specifically the place where ${c.sp.key} lives${corroborated ? " (the tool evidence above confirms it)" : ""}? Answer yes only if this region \u2014 not merely the page as a whole \u2014 is the locus.`,
        c.p
      );
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
        nextStep: c.sp.nextStep || "verify manually."
      });
    }
    if (budgetExhausted) break;
  }
  findings.sort((a, b) => Number(b.corroborated) - Number(a.corroborated) || b.verdict - a.verdict || b.confidence - a.confidence);
  const kept = [];
  for (const f of findings) {
    const dup = kept.findIndex(
      (k) => k.specialty === f.specialty && (k.offsets[0] <= f.offsets[0] && f.offsets[1] <= k.offsets[1] || f.offsets[0] <= k.offsets[0] && k.offsets[1] <= f.offsets[1])
    );
    if (dup === -1) {
      kept.push(f);
      continue;
    }
    const a = kept[dup], aLen = a.offsets[1] - a.offsets[0], fLen = f.offsets[1] - f.offsets[0];
    if (fLen < aLen || fLen === aLen && Number(f.corroborated) > Number(a.corroborated)) kept[dup] = f;
  }
  findings.length = 0;
  findings.push(...kept.map((f, i) => ({ ...f, id: `R${i + 1}` })));
  const chains = await correlate(findings);
  if (budgetExhausted) notes.push(`System-1 budget of ${maxDecides} decides exhausted \u2014 coverage is INCOMPLETE; raise maxDecides or narrow specialties.`);
  if (!findings.length) notes.push("No specialty crossed the threshold on any located span. Absence of a signal is not proof of safety.");
  return {
    target: o.target,
    specialties: specialties.map((s) => s.key),
    mode: "programmed",
    regionsTotal,
    regionsScanned,
    regionsHot: pool.length,
    decides,
    budgetExhausted,
    findings,
    chains,
    locatedOnly,
    notes
  };
}
function reconToolset(html, decide, specialties = "all") {
  const sps = specialties === "all" ? BROWSER_SPECIALTIES : specialties;
  let regions = segment(html);
  const located = [];
  return {
    regions: () => regions.slice(0, 60).map((r) => `${r.id} ${r.locator} (${r.kind}, ${r.end - r.start}b)`).join("\n") || "no regions",
    zoom: (a) => {
      const r = regions.find((x) => x.id === String(a.id || "").trim());
      if (!r) return "unknown region id \u2014 call regions()";
      const kids = segment(r.text);
      if (kids.length < 2) return `region ${r.id} has no sub-structure (${r.text.length}b)`;
      kids.forEach((k, i) => {
        k.id = `${r.id}.${i + 1}`;
        k.locator = `${r.locator} >> ${k.locator}`;
      });
      regions = [...regions.filter((x) => x.id !== r.id), ...kids];
      located.push(r);
      return kids.map((k) => `${k.id} ${k.locator} (${k.kind}, ${k.end - k.start}b)`).join("\n");
    },
    cross: async (a) => {
      const r = regions.find((x) => x.id === String(a.id || "").trim());
      if (!r) return "unknown region id \u2014 call regions()";
      const out = [];
      for (const sp of sps) {
        if (a.specialty && sp.key !== a.specialty) continue;
        const p = await decide(r.text, sp.probe);
        if (p >= 0.5) out.push(`${sp.key} P=${p.toFixed(2)}`);
      }
      located.push(r);
      return out.length ? out.join("\n") : "no specialty reached P>=0.50 in this region";
    },
    report: async (a) => `recon report drafted: ${String(a.summary || "")} | located regions: ${located.map((r) => r.locator).join(", ") || "none"} | findings: ${String(a.findings || "")}`
  };
}
function formatRecon(r) {
  const L = [];
  L.push(`# Recon${r.target ? ` \u2014 ${r.target}` : ""}`);
  L.push(`mode: ${r.mode} \xB7 specialties: ${r.specialties.join(", ")}`);
  L.push(`regions: ${r.regionsTotal} \xB7 scanned: ${r.regionsScanned} \xB7 hot: ${r.regionsHot} \xB7 System-1 decides: ${r.decides}${r.budgetExhausted ? " (budget exhausted)" : ""}`);
  L.push("");
  if (!r.findings.length) L.push("_No specialty crossed the threshold._");
  for (const f of r.findings) {
    L.push(`## ${f.id} \xB7 ${f.specialty} \u2014 ${f.severity}${f.corroborated ? " \xB7 corroborated" : ""}`);
    L.push(`- locator: \`${f.locator}\` (${f.kind}, ${f.offsets[0]}\u2013${f.offsets[1]})`);
    L.push(`- confidence: P=${f.confidence.toFixed(2)} \xB7 verdict: P=${f.verdict.toFixed(2)} \xB7 relevance: P=${f.relevance.toFixed(2)}`);
    L.push(`- signal: ${f.signal}`);
    L.push(`- snippet: \`${f.snippet.slice(0, 120)}\``);
    L.push(`- next step: ${f.nextStep}`);
    L.push("");
  }
  if (r.chains && r.chains.length) {
    L.push("## Chains (composed findings)");
    for (const c of r.chains) {
      L.push(`### ${c.id} \xB7 ${c.name} \u2014 ${c.severity}${c.confidence !== void 0 ? ` (P=${c.confidence.toFixed(2)})` : ""}`);
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
  if (r.notes.length) {
    L.push("## Notes");
    for (const n of r.notes) L.push(`- ${n}`);
  }
  return L.join("\n");
}
export {
  BROWSER_SPECIALTIES,
  correlate,
  formatRecon,
  recon,
  reconToolset,
  segment
};
