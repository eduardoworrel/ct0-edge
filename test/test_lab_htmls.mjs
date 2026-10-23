// test_lab_htmls.mjs — todo HTML do lab que importa um bundle precisa:
//   (1) o bundle existir e carregar;  (2) o simbolo importado existir nele.
// O tick anterior so olhou `onnxruntime-web@` e deixou passar um HTML que importava
// `ct0-edge` sem declarar as deps — o metodo de busca estava errado, nao a checagem.
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { pathToFileURL } from "node:url";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };

const LAB = "/Users/pc/cai/alias3-mlc";
// nome do import -> arquivo do bundle no lab
const BUNDLE = {
  "ct0-browser": "ct0-browser.js",
  "ct0-cai": "ct0-cai.js",
  "ct0-webllm": "ct0-webllm.js",
  "ct0-recon": "ct0-recon.js",
  "ct0-systemone": "ct0-systemone.js",
  "ct0-judges": "ct0-judges.js",
  "ct0-edge": "ct0-onnx-bundle.js",
};

const mods = {};
for (const [name, file] of Object.entries(BUNDLE)) {
  const p = `${LAB}/${file}`;
  if (!existsSync(p)) { mods[name] = null; continue; }
  try { mods[name] = await import(pathToFileURL(p).href); } catch { mods[name] = null; }
}

console.log("bundles carregam:");
for (const [name, file] of Object.entries(BUNDLE)) {
  ok(mods[name] !== null, `${file} carrega${mods[name] ? " (" + Object.keys(mods[name]).length + " exports)" : ""}`);
}

console.log("\nimports estaticos de todos os HTMLs:");
{
  const files = readdirSync(LAB).filter(f => f.endsWith(".html"));
  const broken = [];
  let checked = 0;
  for (const f of files) {
    const h = readFileSync(`${LAB}/${f}`, "utf8");
    const re = /import\s*\{([^}]+)\}\s*from\s*["'](ct0-[a-z]+)["']/g;
    let m;
    while ((m = re.exec(h))) {
      const names = m[1].split(",").map(s => s.trim().split(/\s+as\s+/)[0].trim()).filter(Boolean);
      const mod = mods[m[2]];
      if (!mod) { broken.push(`${f}: bundle ${m[2]} nao carrega`); continue; }
      for (const n of names) { checked++; if (!(n in mod)) broken.push(`${f}: ${m[2]}.${n}`); }
    }
  }
  ok(broken.length === 0, `${checked} imports em ${files.length} HTMLs${broken.length ? " — quebrados: " + broken.slice(0, 5).join(", ") : ""}`);
}

console.log("\nHTMLs que usam ct0-edge declaram as deps do ORT:");
{
  const files = readdirSync(LAB).filter(f => f.endsWith(".html"));
  const missing = [];
  for (const f of files) {
    const h = readFileSync(`${LAB}/${f}`, "utf8");
    // so conta se IMPORTA o bundle (o scanner menciona "ct0-edge" num comentario)
    if (!/from\s*["']ct0-edge["']|import\(\s*["']\.\/ct0-onnx-bundle/.test(h)) continue;
    if (!/importmap/.test(h)) continue;                    // import dinamico nao precisa
    if (!/onnxruntime-web\/webgpu/.test(h)) missing.push(f);
  }
  ok(missing.length === 0, `todos declaram onnxruntime-web/webgpu${missing.length ? " — falta em: " + missing.join(", ") : ""}`);
}

console.log(`\n=== ${pass} pass · ${fail} fail ===`);
process.exit(fail ? 1 : 0);
