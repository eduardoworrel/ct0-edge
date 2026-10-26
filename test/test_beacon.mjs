// Guard against exfil false positives: a beacon keyword must be a path SEGMENT,
// not a word prefix ("logo.png"/"login"/"catalog" are not tracking endpoints).
// Mirrors the regex in resources() in src/browser.ts; keep the two in sync.
const re = /\/(collect|track|beacon|log|exfil|pixel|telemetry|analytics)s?([/?&#]|$)/i;
const cases = [
  ["/logo.png", false], ["/login", false], ["/catalog", false], ["/analytics.js", false], ["/api/ping", false],
  ["/log?x=1", true], ["/logs", true], ["/track?u=", true], ["/collect", true], ["/analytics?e=1", true],
  ["https://collect.evil.example/track?u=c", true],
];
let fail = 0;
for (const [u, exp] of cases) { const got = re.test(u); if (got !== exp) { fail++; console.log(`✗ ${u} -> ${got} (exp ${exp})`); } }
if (fail) { console.log(`test_beacon FAIL (${fail})`); process.exit(1); }
console.log(`test_beacon OK (${cases.length})`);
