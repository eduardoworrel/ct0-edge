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
function detectExecution(network, scriptExec, domInjection) {
  if (!scriptExec && !domInjection) return [];
  return [{
    key: "dom-xss",
    conclusive: true,
    proof: "L2",
    severity: "high",
    label: "execution/exfil proven",
    detail: scriptExec ? "a hooked alert/confirm/prompt fired" : "a DOM handler injection was observed"
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
var SIGNAL_OWNER = {
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
function signalFits(label, specialtyKey) {
  const owners = SIGNAL_OWNER[label];
  return !owners || owners.includes(specialtyKey);
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
function strongEvidence(obs) {
  const o = String(obs || "");
  if (/✅ EXFIL-CONFIRMED|✅ EXECUTION-CONFIRMED/.test(o)) return "execution/exfil proven";
  if (/<-SECRET\?|readable from JS \(NOT HttpOnly|acme_session|S3CR3T-/.test(o)) return "client-side secret exposed";
  if (/⚠ action over HTTP|⚠ no CSRF token|⚠ autocomplete on/.test(o)) return "insecure form issue";
  if (/⚠ tabnabbing|⚠ javascript: URL/.test(o)) return "dangerous link";
  if (/weaknesses:[^\n]*(unsafe-inline|wildcard|no explicit script-src)/.test(o)) return "CSP weakness";
  if (/DOM-XSS sinks found/.test(o)) return "DOM-XSS sink present";
  if (/clobberar window\.|^postmessage\/|\npostmessage\//.test(o)) return "exploitable surface found";
  if (/^resources=\d+, cross-origin=[1-9]/m.test(o)) return "cross-origin beacon/egress endpoint";
  if (/"hiddenTexts":\s*\["/.test(o) && /(NOTE TO (ANY )?AI|disregard|ignore)/i.test(o))
    return "untrusted content with AI-directed instructions";
  return "";
}
export {
  SIGNAL_OWNER,
  detectClobber,
  detectCspWeakness,
  detectDangerousLink,
  detectEgressInSource,
  detectErrorLeak,
  detectExecution,
  detectExfil,
  detectFromTool,
  detectInfoDisclosure,
  detectInsecureForm,
  detectJsReadableCookie,
  detectMessageSurface,
  detectNoSri,
  detectPromptInjection,
  detectSecret,
  detectSensitivePath,
  detectSink,
  detectUnsandboxedIframe,
  signalFits,
  strongEvidence
};
