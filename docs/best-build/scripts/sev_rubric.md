# Severity rubric (1–5) for `head_score`

Written BEFORE generating any case (2026-11-02), to record the criteria per level before
consulting CT0 — exactly what was missing from the documentation of `sev_probes.json` (EVIDENCE §51).

For each case, the generator picks the level **first**, then writes the content to fit that level's
criterion. CT0 never takes part in choosing the label.

| Level | Criterion | Anchor |
|---|---|---|
| **1** | No security risk. Static, informational content, with no script, form or dynamic data. | institutional page, text, product list with no interaction |
| **2** | Theoretical/minimal risk. A non-ideal practice exists, but with no plausible practical exploitation or with negligible impact. | missing security header on a 100% static page; external link without `rel=noopener` but with no sensitive data nearby |
| **3** | Moderate risk. Needs a specific precondition (victim interacts, insecure network, unusual scenario) to matter. | login form over HTTP but with no evidence of real use; permissive CSP but not `*`; unvalidated redirect only to internal paths |
| **4** | Serious risk. Directly exploitable or already leaking sensitive data/credentials, with real impact for whoever finds it. | hardcoded API key; DOM-XSS sink actually fed by untrusted input; sensitive endpoint with no authorization check |
| **5** | Critical. Active compromise or a fully exploitable critical vulnerability, with severe and immediate consequences. | active exfiltration to an external attacker-controlled destination; full authentication bypass with admin access; evidence of remote code execution |

## Coverage

Each level covers several vulnerability classes (the same 16 as in the cai-lite checklist, when
applicable), not just one — this avoids the problem of the old set, which had only 1 scenario per level.

## Process (what was missing from the documentation last time)

1. This rubric is written and frozen before any case.
2. The generator (`generate_sev_cases.py`) decides level + vulnerability class first, and writes the
   content afterwards, to fit the criterion.
3. No case is shown to CT0 before its label exists.
4. Duplicate check against `sev_probes.json` (30 old cases) and against the 50 severity
   questions of our decision test set (172 questions) — they must never overlap.
5. The generator script stays in this same file/folder, versioned, and is explicitly cited in the
   resulting head.bin (`source` field of the header).

