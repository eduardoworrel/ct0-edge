var __defProp = Object.defineProperty;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __esm = (fn, res, err) => function __init() {
  if (err) throw err[0];
  try {
    return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
  } catch (e) {
    throw err = [e], e;
  }
};
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// src/recon.ts
var recon_exports = {};
__export(recon_exports, {
  BROWSER_SPECIALTIES: () => BROWSER_SPECIALTIES,
  correlate: () => correlate,
  formatRecon: () => formatRecon,
  recon: () => recon,
  reconToolset: () => reconToolset,
  segment: () => segment,
  strongEvidence: () => strongEvidence
});
function strongEvidence(obs) {
  const o = obs || "";
  if (/✅ EXFIL-CONFIRMED|✅ EXECUTION-CONFIRMED/.test(o)) return "execution/exfil proven";
  if (/<-SECRET\?|readable from JS \(NOT HttpOnly|acme_session|S3CR3T-/.test(o)) return "client-side secret exposed";
  if (/⚠ action over HTTP|⚠ no CSRF token|⚠ autocomplete on/.test(o)) return "insecure form issue";
  if (/⚠ tabnabbing|⚠ javascript: URL/.test(o)) return "dangerous link";
  if (/weaknesses: [^\n]*(unsafe-inline|wildcard|no explicit script-src)|Absence of a CSP/.test(o)) return "CSP weakness";
  if (/DOM-XSS sinks found/.test(o)) return "DOM-XSS sink present";
  if (/clobberar window\.|^postmessage\/|\npostmessage\//.test(o)) return "exploitable surface found";
  if (/^resources=\d+, cross-origin=[1-9]/m.test(o) && /(collect|track|beacon|exfil|pixel|px|t\.gif|metrics|telemetry|analytics)[.?\/&]/i.test(o)) return "cross-origin beacon/egress endpoint";
  return "";
  return "";
}
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
async function correlate(findings, judge) {
  const byKey = {};
  for (const f of findings) if (!byKey[f.specialty]) byKey[f.specialty] = f;
  const chains = [];
  for (const rule of CHAIN_RULES) {
    const a = byKey[rule.from], b = byKey[rule.to];
    if (!a || !b) continue;
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
          try {
            const ev = strongEvidence(String(await o.tools[t]({ region: r.locator })));
            if (ev) {
              sig[sp.key] = ev;
              break;
            }
          } catch {
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
    if (o.tools) {
      for (const sp of specialties) {
        const runnable = (sp.tools || []).filter((t) => typeof o.tools[t] === "function");
        for (const t of runnable) {
          try {
            const obs = String(await o.tools[t]({ region: r.locator }));
            const ev = strongEvidence(obs);
            if (ev) {
              toolSignals[sp.key] = ev;
              break;
            }
          } catch {
          }
        }
      }
    }
    const hasToolSignal = Object.keys(toolSignals).length > 0;
    if (o.tools && !hasToolSignal) {
      locatedOnly.push({ locator: `${r.locator} [hot by relevance, no tool evidence]`, relevance: r.p || 0, snippet: snippetOf(r.text) });
      continue;
    }
    for (const c of crossed) {
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
var BROWSER_SPECIALTIES, VOID_TAGS, INTERESTING, BLOCK_TAGS, MAX_REGION_TEXT, MAX_REGIONS, CHAIN_RULES, SNIPPET;
var init_recon = __esm({
  "src/recon.ts"() {
    BROWSER_SPECIALTIES = [
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
        tools: ["messages", "sinks", "source", "scan"],
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
        key: "insecure-forms",
        probe: "Does this page have forms handling credentials or sensitive data insecurely (non-HTTPS action, autocomplete on, no CSRF token)?",
        presence: "Does this excerpt itself contain a form or credential field \u2014 a <form> (any action), a password/email input, or auth-related markup?",
        tools: ["forms", "inspect", "cookies"],
        nextStep: "submit with a non-secret probe and confirm the action URL/scheme and missing token.",
        severity: "medium"
      }
    ];
    VOID_TAGS = /* @__PURE__ */ new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"]);
    INTERESTING = /* @__PURE__ */ new Set(["form", "script", "textarea", "iframe", "object", "embed", "template", "table", "main", "section", "article", "nav", "aside", "header", "footer"]);
    BLOCK_TAGS = /* @__PURE__ */ new Set(["div", "p", "li", "td", "label", "button", "a", "span", "pre", "code"]);
    MAX_REGION_TEXT = 1500;
    MAX_REGIONS = 400;
    CHAIN_RULES = [
      {
        name: "unvalidated postMessage \u2192 DOM sink",
        from: "postmessage",
        to: "dom-xss",
        rationale: "A message listener that uses e.data without validating e.origin, feeding a DOM sink, lets any framing origin drive script execution in this page.",
        severity: "high",
        nextStep: "send a cross-origin message carrying a benign marker and confirm it reaches the sink and executes."
      },
      {
        name: "client-side secret reachable by XSS",
        from: "dom-xss",
        to: "secret-scan",
        rationale: "An XSS path plus a secret readable from JS means the secret is exfiltratable by an attacker who lands the XSS.",
        severity: "high",
        nextStep: "confirm the secret is readable (not HttpOnly) and that the sink executes attacker-controlled input."
      },
      {
        name: "exfil endpoint + client-side secret",
        from: "secret-scan",
        to: "exfil-egress",
        rationale: "A secret readable from JS plus a cross-origin beacon/egress endpoint means data can leave the origin \u2014 the skimmer pattern.",
        severity: "high",
        nextStep: "decode the outbound request payload and show which secret it carries."
      },
      {
        name: "credential form over HTTP + XSS",
        from: "insecure-forms",
        to: "dom-xss",
        rationale: "Credentials submitted over plaintext plus script execution gives two independent ways to capture them.",
        severity: "high",
        nextStep: "submit with a non-secret probe and confirm the scheme; confirm the sink executes."
      },
      {
        name: "prompt-injection targeting client-side secrets",
        from: "prompt-injection",
        to: "secret-scan",
        rationale: "Untrusted content instructing an assistant to reveal secrets, in a page that actually holds a JS-readable secret, is a viable exfiltration path through the assistant.",
        severity: "medium",
        nextStep: "show the instruction text and the secret it targets, and the path by which content reaches the assistant."
      }
    ];
    SNIPPET = 220;
  }
});

// src/judges.ts
function judgePrompt(instruction, state) {
  return `<|im_start|>system
You are a precise web-security reviewer. Answer only in the requested format. Do not explain.<|im_end|>
<|im_start|>user
${instruction}

STATE:
${state}<|im_end|>
<|im_start|>assistant
`;
}
async function generate(model, prompt, maxTokens) {
  let out = "";
  for await (const p of model.stream(prompt, { temperature: 0, maxNewTokens: maxTokens })) out += p;
  return out;
}
function parseLabel(reply, allowed) {
  const r = (reply || "").toLowerCase();
  for (const a of allowed) if (r.includes(a.toLowerCase())) return a;
  return void 0;
}
function parseYes(reply) {
  const r = (reply || "").toLowerCase();
  const hits = [...r.matchAll(/\b(yes|true|no|false)\b/g)].map((m) => m[1]);
  if (!hits.length) return void 0;
  const last = hits[hits.length - 1];
  return last === "yes" || last === "true";
}
async function attribute(model, finding, candidates, opts = {}) {
  const missing = candidates.filter((c) => c.key !== finding.specialty);
  if (!missing.length) return void 0;
  const allowed = missing.map((c) => c.key);
  const instruction = `The excerpt below was flagged as "${finding.specialty}". Two concerns can apply to the SAME code.
Answer only whether a DIFFERENT concern from this list is ALSO clearly present: ${allowed.join(", ")}.
If none is clearly present, answer NONE.
Reply with exactly one line: ALSO: <name or NONE>`;
  const state = `EXCERPT:
${finding.snippet}

ALREADY FLAGGED AS: ${finding.specialty} (${finding.signal})`;
  const reply = await generate(model, judgePrompt(instruction, state), opts.maxTokens ?? 12);
  const got = parseLabel(reply, allowed);
  return got ? [got] : void 0;
}
async function refute(model, finding, opts = {}) {
  const instruction = `Below is a claimed finding. Try to REFUTE it. It is refuted if the excerpt does not actually support the claim (e.g. the sink receives only constant data, the value is not a real secret, the endpoint is same-origin, the form action is HTTPS with a token, the message listener does validate origin).
Reply with exactly one line: REFUTED: yes  or  REFUTED: no`;
  const state = `CLAIM: ${finding.specialty} \u2014 ${finding.signal}
EXCERPT:
${finding.snippet}`;
  const reply = await generate(model, judgePrompt(instruction, state), opts.maxTokens ?? 8);
  return parseYes(reply);
}
async function writeFinding(model, finding, opts = {}) {
  const instruction = `Write ONE bug-bounty finding in at most 3 short lines. Format exactly:
IMPACT: <one line>
REPRO: <one line>
FIX: <one line>`;
  const state = `TYPE: ${finding.specialty}
SEVERITY: ${finding.severity}
SIGNAL: ${finding.signal}
LOCATOR: ${finding.locator}
EXCERPT:
${finding.snippet}`;
  const out = (await generate(model, judgePrompt(instruction, state), opts.maxTokens ?? 90)).trim();
  return out || void 0;
}
async function planNextStep(model, finding, opts = {}) {
  const instruction = `Name the SINGLE cheapest concrete verification that would prove or disprove this finding in a browser sandbox. One line, at most 15 words, starting with a verb.`;
  const state = `TYPE: ${finding.specialty}
SIGNAL: ${finding.signal}
LOCATOR: ${finding.locator}
EXCERPT:
${finding.snippet}`;
  const out = (await generate(model, judgePrompt(instruction, state), opts.maxTokens ?? 30)).trim();
  return out || void 0;
}
async function judgeChain(model, chain, parts, opts = {}) {
  const instruction = `Two findings were made on the same page. Decide whether they COMPOSE into a single exploitable path ("${chain.name}"), or are unrelated.
Reply with exactly one line: COMPOSES: yes  or  COMPOSES: no`;
  const state = parts.map((p) => `[${p.id}] ${p.specialty} @ ${p.locator}
     ${p.snippet.slice(0, 240)}`).join("\n\n") + `

PROPOSED CHAIN: ${chain.rationale}`;
  const reply = await generate(model, judgePrompt(instruction, state), opts.maxTokens ?? 8);
  const y = parseYes(reply);
  return y === void 0 ? void 0 : y ? 0.9 : 0.1;
}
async function judgePass(model, findings, opts = {}) {
  const log = { attributed: 0, refuted: 0, written: 0, planned: 0, chains: 0 };
  let out = findings.map((f) => ({ ...f }));
  const cap = opts.maxFindings ?? out.length;
  if (opts.attribute && opts.candidates) {
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
    const kept = [];
    for (const f of out) {
      const refuted = await refute(model, f);
      if (refuted === true && !f.corroborated) {
        log.refuted++;
        continue;
      }
      kept.push(f);
    }
    out = kept;
  }
  if (opts.plan) {
    for (const f of out.slice(0, cap)) {
      const n = await planNextStep(model, f);
      if (n) {
        f.nextStep = n;
        log.planned++;
      }
    }
  }
  if (opts.write) {
    for (const f of out.slice(0, cap)) {
      const w = await writeFinding(model, f);
      if (w) {
        f.written = w;
        log.written++;
      }
    }
  }
  let chains = [];
  if (opts.chains) {
    chains = await correlateWith(model, out);
    log.chains = chains.length;
  }
  const seen = /* @__PURE__ */ new Set();
  out = out.filter((f) => seen.has(f.specialty) ? false : (seen.add(f.specialty), true));
  return { findings: out, chains, log };
}
async function correlateWith(model, findings) {
  const { correlate: correlate2 } = await Promise.resolve().then(() => (init_recon(), recon_exports));
  return correlate2(findings, async (chain, parts) => {
    const p = await judgeChain(model, chain, parts);
    return p ?? 0.5;
  });
}
export {
  attribute,
  judgeChain,
  judgePass,
  judgePrompt,
  parseLabel,
  parseYes,
  planNextStep,
  refute,
  writeFinding
};
