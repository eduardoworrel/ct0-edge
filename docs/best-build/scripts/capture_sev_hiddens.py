"""Runs the 60 cases of sev_cases_v2_raw.json through the local decode-only ONNX graph (token by token,
same as three_modes.run_prompt), saves the hidden at <decide> (dec) and at the last position (last), plus the
logits of digits 1..5 (lg_digits, only for checking/plotting) -- same layout as sev_probes.json.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import three_modes as T

GRAPH = os.path.join(os.path.dirname(HERE), "decode-only", "model_q4_decode.onnx")
IN = os.path.join(os.path.dirname(HERE), "sev_cases_v2_raw.json")
OUT = os.path.join(os.path.dirname(HERE), "sev_cases_v2.json")

cases = json.load(open(IN))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(T.TOK_DIR)
# CoreMLExecutionProvider measured 2.84 s/token on this graph (GatherBlockQuantized / GatedDeltaNet
# state passing forces per-call CPU fallback dispatch); plain CPUExecutionProvider measured 0.45
# s/token on the same prompt -- 6.4x faster. Force CPU only.
sess = T._session(GRAPH, providers=["CPUExecutionProvider"])

digit_ids = {str(d): tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)}
print("digit token ids:", digit_ids)

PARTIAL = OUT + ".partial"
out = []
if os.path.exists(PARTIAL):
    out = json.load(open(PARTIAL))
    print("resuming: %d cases already done" % len(out), flush=True)
done_i = {c["i"] for c in out}

for c in cases:
    if c["i"] in done_i:
        continue
    prompt = T.render(c["state"], T.SEV_Q)
    ids, offs, H, lg = T.run_prompt(sess, tok, prompt, want_hidden=True)
    dec_i = T.decide_pos(ids, tok)
    last_i = T.last_pos(ids)
    lg_digits = {d: float(lg[tid]) for d, tid in digit_ids.items()}
    out.append({
        "i": c["i"], "sev": c["sev"], "class": c["class"], "state": c["state"],
        "dec": H[dec_i].tolist(), "last": H[last_i].tolist(), "lg_digits": lg_digits,
    })
    json.dump(out, open(PARTIAL, "w"))
    pred = max(lg_digits, key=lg_digits.get)
    print("case %2d/%d  sev=%d  class=%-16s  logits-argmax=%s" % (c["i"], len(cases), c["sev"], c["class"], pred), flush=True)

json.dump(out, open(OUT, "w"))
os.remove(PARTIAL)
print("wrote", len(out), "cases with hiddens to", OUT, flush=True)

