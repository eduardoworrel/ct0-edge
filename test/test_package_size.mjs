// test_package_size.mjs — docs/bench (README figures + runtime logos) stays under 200 KB, and every figure inlines the
// logos it uses (a README shows figures through <img>, which loads no external files).
import { readdirSync, statSync, readFileSync } from "node:fs";
import { join } from "node:path";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ✓", m); } else { fail++; console.log("  ✗ FAIL:", m); } };
const dir = new URL("../docs/bench/", import.meta.url).pathname;
const walk = (d) => readdirSync(d).flatMap((n) => { const p = join(d, n); return statSync(p).isDirectory() ? walk(p) : [p]; });
const total = walk(dir).reduce((a, p) => a + statSync(p).size, 0);
ok(total < 200 * 1024, `docs/bench is ${(total / 1024).toFixed(1)} KB (< 200 KB)`);
for (const f of ["pick-a-model.svg", "accuracy.svg", "size-vs-time.svg"]) {
  const s = readFileSync(join(dir, f), "utf8");
  const used = [...new Set([...s.matchAll(/href="#(lg-\w+)"/g)].map((m) => m[1]))];
  ok(used.length === 4 && used.every((id) => s.includes(`<symbol id="${id}"`)) && !/href="(?!#|data:)/.test(s), `${f}: ${used.length} runtime logos, all inlined`);
}
console.log(`\n${pass} ok, ${fail} failures`);
process.exit(fail ? 1 : 0);
