// test_lab_bundles.mjs — valida os bundles que o LAB consome (alias3-mlc/*.js).
//
// Por que existe: o scanner (alias3-mlc/ct0-scanner.html) importa ct0-browser.js, ct0-cai.js e
// ct0-webllm.js. Nenhum teste cobria esses arquivos — so os dist/ dos pacotes. Uma regressao em
// `strongEvidence` (movido de recon para detectors) quebrou o scanner com
//   "does not provide an export named 'strongEvidence'"
// e passou por 14 testes verdes. Este teste fecha esse elo.
import { readFileSync, existsSync } from "node:fs";
import { pathToFileURL } from "node:url";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };

const LAB = "/Users/pc/cai/alias3-mlc";
const BUNDLES = {
  "ct0-browser.js": ["BrowserTool"],
  "ct0-cai.js": ["Agent", "BROWSER_SPECIALTIES"],
  "ct0-webllm.js": ["webllmModel"],
};

console.log("bundles do lab existem e exportam o esperado:");
for (const [file, expected] of Object.entries(BUNDLES)) {
  const p = `${LAB}/${file}`;
  if (!existsSync(p)) { ok(false, `${file} ausente (rode ./deploy.sh)`); continue; }
  let mod;
  try { mod = await import(pathToFileURL(p).href); }
  catch (e) { ok(false, `${file} nao importa: ${String(e.message).slice(0, 70)}`); continue; }
  const missing = expected.filter(x => !(x in mod));
  ok(missing.length === 0, `${file} exporta tudo (${expected.length})${missing.length ? " — falta " + missing.join(",") : ""}`);
}

console.log("\no scanner importa so o que os bundles exportam:");
{
  const html = readFileSync(`${LAB}/ct0-scanner.html`, "utf8");
  // pega os `import { a, b } from "ct0-x"` do scanner
  const re = /import\s*\{([^}]+)\}\s*from\s*["'](ct0-(?:browser|cai|webllm))["']/g;
  let m, checked = 0, bad = [];
  while ((m = re.exec(html))) {
    const names = m[1].split(",").map(s => s.trim().split(/\s+as\s+/)[0].trim()).filter(Boolean);
    const file = m[2] + ".js";
    const p = `${LAB}/${file}`;
    if (!existsSync(p)) { bad.push(`${file} ausente`); continue; }
    const mod = await import(pathToFileURL(p).href);
    for (const n of names) { checked++; if (!(n in mod)) bad.push(`${m[2]}:${n}`); }
  }
  ok(bad.length === 0, `scanner: ${checked} imports resolvem${bad.length ? " — quebram: " + bad.join(", ") : ""}`);
}

console.log("\nos bundles estao atualizados em relacao ao fonte?");
{
  const pairs = [["ct0-browser.js", "/Users/pc/cai/agent-browser-lite/dist/browser.js"],
                 ["ct0-cai.js", "/Users/pc/cai/ct0-lib/dist/cai.js"],
                 ["ct0-webllm.js", "/Users/pc/cai/ct0-lib/dist/webllm.js"]];
  for (const [lab, dist] of pairs) {
    if (!existsSync(`${LAB}/${lab}`) || !existsSync(dist)) { ok(false, `${lab}: arquivo ausente`); continue; }
    const a = readFileSync(`${LAB}/${lab}`, "utf8");
    const b = readFileSync(dist, "utf8");
    // os bundles sao gerados do mesmo fonte: o tamanho pode diferir (esbuild flags), mas os
    // simbolos exportados precisam coincidir
    const exp = /^export\s*\{([^}]+)\}/m.exec(b);
    const names = exp ? exp[1].split(",").map(s => s.trim().split(/\s+as\s+/).pop().trim()).filter(Boolean) : [];
    const absent = names.filter(n => !new RegExp(`\\b${n}\\b`).test(a));
    ok(absent.length === 0, `${lab} tem os ${names.length} simbolos do dist${absent.length ? " — falta " + absent.slice(0, 4).join(",") : ""}`);
  }
}


console.log("\narquivos duplicados entre repos nao divergiram:");
{
  // systemone.ts e headdecide.ts existem em ct0-edge E em cai-lite. O cai-lite precisa deles
  // (headdecide importa systemone); o ct0-edge tambem (sao o SDK). Para nao divergirem em
  // silencio, comparamos o conteudo.
  const pairs = ["systemone.ts", "headdecide.ts"];
  for (const f of pairs) {
    const a = `/Users/pc/cai/ct0-lib/src/${f}`;
    const b = `/Users/pc/cai/cai-lite/src/${f}`;
    if (!existsSync(a) || !existsSync(b)) { ok(false, `${f}: ausente num dos repos`); continue; }
    ok(readFileSync(a, "utf8") === readFileSync(b, "utf8"), `${f} identico em ct0-edge e cai-lite`);
  }
}
console.log(`\n=== ${pass} pass · ${fail} fail ===`);
process.exit(fail ? 1 : 0);
