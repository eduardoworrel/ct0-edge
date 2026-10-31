"""Generate positions.json from the Python reference implementation (three_modes), the source of truth for the JS tests.

Required env: CT0_REF_SCRIPTS (directory containing three_modes.py), CT0_REF_DIR (directory with
the probes JSON and the tokenizer in tok/), CT0_PROBES_JSON (probes file name inside CT0_REF_DIR).
"""
import json, os, sys
sys.path.insert(0, os.environ["CT0_REF_SCRIPTS"])
import three_modes as tm
from transformers import AutoTokenizer

ART = os.environ["CT0_REF_DIR"]
HERE = os.path.dirname(os.path.abspath(__file__))
probes = json.load(open(os.path.join(ART, os.environ["CT0_PROBES_JSON"])))
tok = AutoTokenizer.from_pretrained(os.path.join(ART, "tok"))
out = []
for p in probes:
    prompt = tm.prompt_for(p)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = [tuple(x) for x in enc["offset_mapping"]]
    row = dict(probe={k: p[k] for k in ("kind", "state", "q", "options", "sev", "want") if k in p},
               prompt=prompt, ids=ids, pos_decide=tm.decide_pos(ids, tok), pos_last=tm.last_pos(ids))
    if p["kind"] == "choice":
        row["opt_pos"] = tm.opt_positions(prompt, offs)
    out.append(row)
json.dump(out, open(os.path.join(HERE, "positions.json"), "w"), ensure_ascii=False)
print("%d probes" % len(out))


# ------------------------------------------------------------------ ask_expected.json
# Synthetic arrays (small head, seeded random hiddens and logits) + the result of
# three_modes.eval_probe on the SAME arrays. The JS (ask.ts) must reproduce it.
import numpy as np
rng = np.random.default_rng(20261025)
D, HS = 4, 16

def mk_head(offset):
    return dict(q=rng.normal(size=(D, HS)).astype(np.float32), qb=rng.normal(size=D).astype(np.float32),
                k=rng.normal(size=(D, HS)).astype(np.float32), kb=rng.normal(size=D).astype(np.float32),
                headDim=D, hiddenSize=HS, temperature=2.0, scoreOffset=offset)

def head_json(h):
    return {k: (v.ravel().tolist() if isinstance(v, np.ndarray) else v) for k, v in h.items()}

hd_nc, hd_sc = mk_head(0.3), mk_head(-0.7)
VOCAB_IDX = [tm.YES, tm.NO] + list(tm.DIGITS)

def first_tok(o):
    return int(tok.encode(o, add_special_tokens=False)[0])

def mk_case(name, p, hd, boost=None):
    prompt = tm.prompt_for(p)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = [tuple(x) for x in enc["offset_mapping"]]
    Hseq = (0.4 * rng.normal(size=(len(ids), HS))).astype(np.float32)
    idx = sorted(set(VOCAB_IDX + [first_tok(o) for o in p.get("options", [])]))
    lg = np.zeros(152000, dtype=np.float32)
    for i in idx:
        lg[i] = np.float32(rng.normal() * 3)
    if boost is not None:
        lg[boost] = np.float32(9.0)
    p = dict(p, i=0)
    row = tm.eval_probe(hd, p, ids, offs, Hseq, lg, prompt=prompt, tok=tok)
    return dict(name=name, probe={k: p[k] for k in ("kind", "state", "q", "options", "sev", "want") if k in p},
                ids=ids, hseq=Hseq.tolist(), logits={str(i): float(lg[i]) for i in idx}, expected=row)

by = {}
for r in out:
    by.setdefault(r["probe"]["kind"], r["probe"])
choice_probe = by["choice"]
# tie: two labels sharing the same first token
tie = dict(kind="choice", state="x = 1", options=["dom-xss", "dom-based", "other"], want="dom-xss")
assert first_tok("dom-xss") == first_tok("dom-based"), "expected the same first token"
cases = [
    mk_case("noul", by["noul"], hd_nc),
    mk_case("choice", choice_probe, hd_nc),
    mk_case("choice_tie", tie, hd_nc, boost=first_tok("dom-xss")),
    dict(mk_case("score", dict(by["score"], want_min=1), hd_sc), head="score"),
]
json.dump(dict(headNoulChoice=head_json(hd_nc), headScore=head_json(hd_sc), cases=cases),
          open(os.path.join(HERE, "ask_expected.json"), "w"))
print("ask_expected: %d cases" % len(cases))
