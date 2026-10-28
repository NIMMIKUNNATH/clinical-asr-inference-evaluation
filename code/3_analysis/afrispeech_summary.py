"""
afrispeech_summary.py - AfriSpeech-Dialog results by subset: the 20 clinical consultations (held-out) and the 29
general conversations, for every configuration scored so far. Rates pooled over recordings.

USAGE (eval venv):  python afrispeech_summary.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

SC = ROOT / "final_afrispeech" / "scores"
ids = pd.read_csv(ROOT / "cache" / "afrispeech" / "ids.csv").set_index("id")


def rates(d):
    s = d.sum(numeric_only=True)
    return dict(n=len(d), WER=100 * (s["sub"] + s["dele"] + s["ins"]) / s["ref_words"],
                sub=100 * s["sub"] / s["ref_words"], dele=100 * s["dele"] / s["ref_words"],
                ins=100 * s["ins"] / s["ref_words"], WDER=100 * s["wder_wrong"] / s["wder_aligned"],
                cpWER=100 * s["cp_err"] / s["cp_n"], halluc_per_1k=1000 * s["halluc_words"] / s["ref_words"],
                neg_incl=100 * (s["neg_err"] + s["neg_ins"]) / max(s["neg_ref"], 1))


rows = []
for f in sorted(SC.glob("*.csv")):
    d = pd.read_csv(f)
    d["domain"] = d.recording.map(ids.domain)
    for dom in ("medical", "general"):
        rows.append(dict(config=f.stem, subset="clinical (held-out)" if dom == "medical" else "general",
                         **rates(d[d.domain == dom])))
t = pd.DataFrame(rows).round(2)
pd.set_option("display.width", 200)
for sub in ("clinical (held-out)", "general"):
    print(f"\n== AfriSpeech-Dialog, {sub}")
    print(t[t.subset == sub].drop(columns="subset").to_string(index=False))
