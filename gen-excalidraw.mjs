#!/usr/bin/env node
// gen-excalidraw.mjs — gera o diagrama da estrutura atual do cai-lite em formato .excalidraw
import { writeFileSync } from "node:fs";

let seed = 1;
const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
const rid = () => Array.from({ length: 20 }, () => "0123456789abcdef"[Math.floor(rnd() * 16)]).join("");
const els = [];
const SANS = 2;

function base(type, x, y, w, h, extra = {}) {
  return {
    id: rid(), type, x, y, width: w, height: h, angle: 0,
    strokeColor: extra.strokeColor || "#1e1e1e",
    backgroundColor: extra.backgroundColor || "transparent",
    fillStyle: "solid", strokeWidth: extra.strokeWidth ?? 2, strokeStyle: extra.strokeStyle || "solid",
    roughness: 0, opacity: 100, groupIds: [], frameId: null, roundness: extra.roundness ?? { type: 3 },
    seed: Math.floor(rnd() * 1e9), version: 1, versionNonce: Math.floor(rnd() * 1e9),
    isDeleted: false, boundElements: [], updated: Date.now(), link: null, locked: false,
    ...extra.rest,
  };
}
function rect(x, y, w, h, o = {}) {
  const e = base("rectangle", x, y, w, h, o); els.push(e); return e;
}
function text(x, y, str, o = {}) {
  const size = o.fontSize || 14;
  const lines = String(str).split("\n");
  const w = o.width || Math.max(...lines.map(l => l.length)) * size * 0.58;
  const h = lines.length * size * 1.28;
  const e = base("text", x, y, w, h, {
    strokeColor: o.color || "#1e1e1e", rest: {
      text: str, originalText: str, fontSize: size, fontFamily: o.fontFamily ?? SANS,
      textAlign: o.align || "left", verticalAlign: "top", containerId: null,
      lineHeight: 1.28, autoResize: true, baseline: Math.round(size * 0.9),
    },
  });
  els.push(e); return e;
}
const arrow = (x1, y1, x2, y2, o = {}) => {
  const e = base("arrow", x1, y1, Math.abs(x2 - x1) || 1, Math.abs(y2 - y1) || 1, {
    strokeColor: o.strokeColor || "#495057", strokeWidth: o.strokeWidth ?? 1.5,
    strokeStyle: o.strokeStyle || "solid",
    rest: {
      points: [[0, 0], [x2 - x1, y2 - y1]], lastCommittedPoint: null,
      startBinding: null, endBinding: null, startArrowhead: null, endArrowhead: "arrow",
    },
  });
  els.push(e);
};

// cores
const C = {
  layer1: { stroke: "#1971c2", bg: "#e7f5ff" },   // cai-lite orquestração
  layer2: { stroke: "#2f9e44", bg: "#ebfbee" },   // System-1 juiz
  layer3: { stroke: "#e8590c", bg: "#fff4e6" },   // tools browser
  layer4: { stroke: "#7048e8", bg: "#f3f0ff" },   // system-2 judges
  layer5: { stroke: "#c2255c", bg: "#fff0f6" },   // output
  neutral: { stroke: "#495057", bg: "#f8f9fa" },
};
const card = (x, y, w, h, title, color) => {
  const r = rect(x, y, w, h, { strokeColor: color.stroke, backgroundColor: color.bg, rest: { boundElements: [] } });
  return r;
};

// ===================== TÍTULO =====================
text(60, 30, "cai-lite — estrutura atual", { fontSize: 32, color: "#0b7285" });
text(60, 74, "recon de bug bounty driven by System-1 · 11 especialidades · 19 tools de browser · 100% on-device", { fontSize: 16, color: "#495057" });

// ===================== CAMADA 0: ENTRADA =====================
const y0 = 130;
card(60, y0, 300, 96, "input", C.neutral);
text(78, y0 + 14, "ENTRADA", { fontSize: 17, color: C.neutral.stroke });
text(78, y0 + 40, [
  "• html (outerHTML da página-alvo)",
  "• herramientas: tools (opcional, corrobora)",
  "• specialties: \"all\" | subconjunto",
  "• strategy: tools-first | decide-first",
  "• maxDecides: orçamento de decisões",
].join("\n"), { fontSize: 13 });

// ===================== CAMADA 1: PIPELINE =====================
const px = 430, py = 130, pw = 430;
card(px, py, pw, 430, "pipeline", C.layer1);
text(px + 18, py + 12, "recon.ts — recon()  ·  PIPELINE", { fontSize: 18, color: C.layer1.stroke });

const steps = [
  ["PASS 1 — LOCALIZAR", [
    "segment(html) → regiões com locator",
    "  · form · script · textarea · iframe",
    "  · handler · comment · container · block",
    "tools-first: as tools acham a região",
    "decide-first: System-1 pontua (satura)",
    "zoom: subdivide regiões quentes (bounded)",
  ]],
  ["PASS 2 — CRUZAR", [
    "região × especialidade",
    "probe de PRESENÇA (não presença+exploit)",
    "SIGNAL_OWNER: rótulo só vale p/ dona",
    "tools-first: aceita só o que a tool provou",
  ]],
  ["PASS 3 — CORROBORAR", [
    "tools determinísticas rodam por região",
    "strongEvidenceAll → TODOS os rótulos",
    "(nunca só o primeiro: página tem 2 sinais)",
  ]],
  ["PASS 4 — VEREDITO (System-1)", [
    "decide() contrastivo: 'É O SÍTIO de X?'",
    "verdictGate: rejeitado + sem prova → descarta",
    "1 forward, ~0.5 s, determinístico",
  ]],
  ["DEDUP + CADEIAS", [
    "spans aninhados: mantém o mais específico",
    "correlate() → H6: 2 achados = exploit?",
  ]],
];
let sy = py + 52;
for (const [t, lines] of steps) {
  const h = 30 + lines.length * 15;
  rect(px + 18, sy, pw - 36, h, { strokeColor: C.layer1.stroke, backgroundColor: "#ffffff", strokeWidth: 1.5 });
  text(px + 32, sy + 8, t, { fontSize: 14, color: C.layer1.stroke });
  text(px + 32, sy + 28, lines.join("\n"), { fontSize: 11 });
  sy += h + 8;
}

// ===================== SYSTEM-1 (centro) =====================
const sx = 60, syy = 260;
card(sx, syy, 300, 200, "system1", C.layer2);
text(sx + 18, syy + 12, "SYSTEM-1  ·  o juiz", { fontSize: 18, color: C.layer2.stroke });
text(sx + 18, syy + 40, [
  "decide(state, question) → P(yes)",
  "",
  "• 1 forward, lê o LOGIT (ONNX)",
  "  ou logprobs (WebLLM / MLC)",
  "• ~522 ms medido (M2 Pro, WebGPU)",
  "• determinístico: spread 0.000",
  "• 20× seguidas sem degradar",
  "",
  "PAPEL MEDIDO:",
  "• como SCANNER (HTML cru) .... ✗ satura",
  "• como LOCALIZADOR .......... ✗ rec 0.50",
  "• como JUIZ (saída da tool) .. ✓ 0.348→0.919",
].join("\n"), { fontSize: 12 });

// ===================== TOOLS =====================
const tx = 920, ty = 130, tw = 470;
card(tx, ty, tw, 700, "tools", C.layer3);
text(tx + 18, ty + 12, "agent-browser-lite · 19 tools (neutral)", { fontSize: 17, color: C.layer3.stroke });

const toolGroups = [
  ["NAVEGAÇÃO / AÇÃO", ["snapshot — DOM limpo, refs @eN", "source · get · readHTML", "type · fill (dispara input+change)", "click · scan (regex no source)"]],
  ["RUNTIME (precisa de browser real)", ["events — fetch/xhr/beacon/alert hooks", "executionVerdict — provou execução?", "resources — Resource Timing, egress", "messages — postMessage capturado"]],
  ["ESTRUTURA / ESTADO", ["storage — localStorage/session/client", "cookies — HTTPOnly / JS-legível", "csp — meta CSP + fraquezas", "sinks — innerHTML/insertAdjacentHTML", "forms — action/scheme/CSRF/autocomplete", "links · iframes · clobber"]],
  ["NOVAS (bug bounty)", ["sensitivePaths — /.env /.git backup", "errorLeak — stacktrace + DSN com senha", "thirdPartyNoSri — supply-chain", "infoDisclosure — sourcemap/debug/versão"]],
  ["SEGURANÇA DO PRÓPRIO HOOK", ["hookInitSource — instala ANTES do load", "  (sem isso, beacon já disparou)", "instrument() adota o log pré-instalado"]],
];
let gy = ty + 48;
for (const [g, items] of toolGroups) {
  const h = 26 + items.length * 16;
  rect(tx + 16, gy, tw - 32, h, { strokeColor: C.layer3.stroke, backgroundColor: "#ffffff", strokeWidth: 1.5 });
  text(tx + 30, gy + 6, g, { fontSize: 13, color: C.layer3.stroke });
  text(tx + 30, gy + 24, items.join("\n"), { fontSize: 11 });
  gy += h + 9;
}

// ===================== ESPECIALIDADES =====================
const ex = 60, ey = 760, ew = 430;
card(ex, ey, ew, 300, "specs", C.neutral);
text(ex + 18, ey + 12, "11 ESPECIALIDADES", { fontSize: 17, color: C.neutral.stroke });
text(ex + 18, ey + 38, [
  "ALTA",
  "• dom-xss .............. sink + fonte",
  "• sensitive-path ....... /.env /.git backup",
  "• error-disclosure ..... stacktrace + credencial",
  "• secret-scan .......... localStorage/cookie/inline",
  "",
  "MÉDIA",
  "• postmessage .......... listener sem origin check",
  "• exfil-egress ......... beacon c/ payload p/ fora",
  "• supply-chain ......... 3rd-party sem SRI",
  "• insecure-forms ....... HTTP / sem CSRF / autocomplete",
  "• prompt-injection ..... conteúdo dirigido a assistente",
  "",
  "BAIXA",
  "• csp-misconfig ........ meta CSP fraca",
  "• info-disclosure ...... sourcemap / debug / versão",
].join("\n"), { fontSize: 11.5 });

// ===================== SAÍDA =====================
const ox = 520, oy = 760, ow = 380;
card(ox, oy, ow, 300, "out", C.layer5);
text(ox + 18, oy + 12, "SAÍDA — relatório de recon", { fontSize: 17, color: C.layer5.stroke });
text(ox + 18, oy + 38, [
  "formatRecon(report) → markdown",
  "",
  "por achado:",
  "• locator (script[1], form#0, div.md)",
  "• offsets (início–fim no HTML)",
  "• snippet",
  "• confidence P · verdict P · relevance P",
  "• signal (o que acendeu)",
  "• severity (alta/média/baixa)",
  "• nextStep (como fechar a prova)",
  "• corroborated (prova dura da tool?)",
  "",
  "cadeias (H6):",
  "• parts R1 + R2 · por que compõe",
  "• 5 regras: msg→sink, xss+secret,",
  "  secret+exfil, httpform+xss, inject+secret",
  "",
  "'signals, not proofs' — recon, não exploit",
].join("\n"), { fontSize: 11.5 });

// ===================== SYSTEM-2 =====================
const jx = 920, jy = 860, jw = 470;
card(jx, jy, jw, 240, "judges", C.layer4);
text(jx + 18, jy + 12, "judges.ts — System-2 (opcional)", { fontSize: 17, color: C.layer4.stroke });
text(jx + 18, jy + 38, [
  "Step CURTO medido: ~40 tok = 2.2 s (não 17 s)",
  "",
  "• H6 judgeChain ...... 2 achados compõem exploit?  ✓ ÚNICO ganho",
  "• H1 attribute ....... pergunta ADITIVA ('também é X?')  ➖ neutro",
  "• H4 refute .......... não descarta prova dura  ✗ inócuo",
  "• H2 writeFinding .... IMPACT / REPRO / FIX",
  "• H5 planNextStep .... a verificação mais barata",
  "",
  "judgePass() roda os 5, cada um 1 geração curta",
  "regra: nunca remove evidência corroborada",
].join("\n"), { fontSize: 11.5 });

// ===================== MEDIÇÃO =====================
const mx = 1420, my = 130, mw = 330;
card(mx, my, mw, 300, "metrics", C.layer2);
text(mx + 18, my + 12, "QUALIDADE MEDIDA", { fontSize: 17, color: C.layer2.stroke });
text(mx + 18, my + 38, [
  "32 páginas × 11 especialidades",
  "",
  "           TP  FP  FN   prec   rec",
  "antes      32  60   5   0.348  0.865",
  "depois     40   4   2   0.909  0.952",
  "",
  "FP: 60 → 4  (−93%)",
  "",
  "6 páginas limpas:",
  "0 falso-positivo real",
  "",
  "GANHO VEIO DE:",
  "oráculo auditado · isolamento por",
  "browser context · runtime real ·",
  "ownership de sinal · hooks pré-load",
  "NÃO de trocar modelo.",
].join("\n"), { fontSize: 11.5 });

// ===================== PRINCÍPIOS =====================
const px2 = 1420, py2 = 460, pw2 = 330;
card(px2, py2, pw2, 370, "principles", C.neutral);
text(px2 + 18, py2 + 12, "PRINCÍPIOS (medidos, não assumidos)", { fontSize: 15, color: C.neutral.stroke });
text(px2 + 18, py2 + 38, [
  "1. tools ACHAM (recall alto, grátis)",
  "2. System-1 JULGA (precisão)",
  "3. System-2 só em step CURTO",
  "4. sinal tem dono (não vaza)",
  "5. toda evidência, não a 1ª",
  "6. egress ≠ execução",
  "7. medição isolada por página",
  "8. oráculo auditado, não intenção",
  "9. prova dura nunca cai por opinião",
  "10. 'signals, not proofs'",
].join("\n"), { fontSize: 11.5 });

// ===================== SETAS =====================
arrow(360, 178, 428, 178);          // entrada -> pipeline
arrow(360, 360, 428, 300);          // entrada/system1 -> pipeline
arrow(362, 300, 428, 250);
arrow(860, 250, 918, 250);          // pipeline -> tools
arrow(860, 400, 918, 400);
arrow(1155, 830, 900, 830);         // tools -> saida
arrow(902, 830, 902, 760);
arrow(490, 830, 518, 830);          // specs -> saida
arrow(860, 500, 918, 880, { strokeStyle: "dashed" }); // pipeline -> judges
arrow(1155, 700, 1155, 858, { strokeStyle: "dashed" });

// legendas de camada
text(60, 1110, "azul = orquestração (recon.ts)   ·   verde = System-1 (juiz)   ·   laranja = tools de browser   ·   roxo = System-2 (opcional)   ·   rosa = relatório", { fontSize: 13, color: "#868e96" });

const doc = {
  type: "excalidraw", version: 2, source: "https://excalidraw.com",
  elements: els,
  appState: { gridSize: null, viewBackgroundColor: "#ffffff", scrollX: 0, scrollY: 0, zoom: { value: 0.55 } },
  files: {},
};
const out = process.argv[2] || `${process.env.HOME}/Downloads/cai-lite-estrutura.excalidraw`;
writeFileSync(out, JSON.stringify(doc, null, 2));
console.log("escrito:", out, "·", els.length, "elementos");
