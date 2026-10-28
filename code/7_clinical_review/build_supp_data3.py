"""Supplementary Data 3: de-identified rating matrix of the clinical-meaning review (312 events).

Rows: 180 panel events (five reviewers) + 132 physician-only events (two physicians).
R1/R2 = the two physicians (final ratings from physician_1_312 / physician_2_312), R3-R5 = the three nursing
students (unchanged, from the previous Supplementary Data 3). Reviewers are coded and labelled only by
professional background; comments and transcript text are omitted.
Decision columns: majority_* (>=3 of 5, panel events only; 'Tie' otherwise) and
physicians_agreed_meaning (common physician rating, or 'Discordant').
Usage: python build_supp_data3.py <uploads dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

U = Path(sys.argv[1])
OLD = Path("/mnt/d/fareez_optuna/paper/submission/04_supplementary_data/Supplementary_Data_3.csv")
OUT = Path(__file__).parent / "Supplementary_Data_3.csv"
ITEMS = {"flag_correct": "Automated flag correct?", "meaning_changed": "Clinical meaning changed?", "confidence": "Confidence"}


def norm(v, conf=False):
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    if conf:
        return {"low": "Low", "moderate": "Moderate", "medium": "Moderate", "high": "High"}.get(s, "")
    return "Partially" if s.startswith("partial") else {"yes": "Yes", "no": "No"}.get(s, "")


def load(f):
    d = pd.read_excel(U / f, sheet_name="Reviews")
    d = d[d["Matched Event ID"].notna()].copy()
    d["set"] = np.where(d["Event set"].astype(str).str.startswith("180"), "panel_180", "physician_132")
    return d.set_index(["set", "Matched Event ID"])


A, Z = load("527a7308-physician_1_312.xlsx"), load("007b2db9-physician_2_312.xlsx")
old = pd.read_csv(OLD).set_index("event_id")

# which old physician column is which doctor: compare with the physicians' final ratings (most rows unchanged)
a180 = A.loc["panel_180"]["Clinical meaning changed?"].map(norm)
match_r1 = (old["R1_meaning_changed"].reindex(a180.index) == a180).mean()
match_r2 = (old["R2_meaning_changed"].reindex(a180.index) == a180).mean()
first, second = (A, Z) if match_r1 >= match_r2 else (Z, A)
print(f"old R1 matches Physician 1 {match_r1:.2f}, old R2 matches {match_r2:.2f} -> R1 = {'Physician 1' if first is A else 'Physician 2'}")

rows = []
for (s, eid), a in first.iterrows():
    z = second.loc[(s, eid)]
    r = {"event_set": s, "event_id": eid, "corpus": a["Corpus"] if s == "physician_132" else "Fareez",
         "speaker_role": a["Speaker"], "cue_type": a["Cue type"]}
    for i, src in ((1, a), (2, z)):
        r[f"R{i}_role"] = "physician"
        for k, c in ITEMS.items():
            r[f"R{i}_{k}"] = norm(src[c], k == "confidence")
    for i in (3, 4, 5):
        r[f"R{i}_role"] = "nursing student" if s == "panel_180" else ""
        for k in ITEMS:
            r[f"R{i}_{k}"] = old.at[eid, f"R{i}_{k}"] if s == "panel_180" and pd.notna(old.at[eid, f"R{i}_{k}"]) else ""
    for k in ("flag_correct", "meaning_changed"):
        if s == "panel_180":
            vals = pd.Series([r[f"R{i}_{k}"] for i in range(1, 6)]).replace("", np.nan).dropna().map(lambda v: norm(v))
            vc = vals.value_counts()
            r[f"majority_{k}"] = vc.idxmax() if len(vc) and vc.max() >= 3 else "Tie"
        else:
            r[f"majority_{k}"] = ""
    r["physicians_agreed_meaning"] = r["R1_meaning_changed"] if r["R1_meaning_changed"] == r["R2_meaning_changed"] else "Discordant"
    rows.append(r)

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
p = out[out.event_set == "panel_180"]
print("rows", len(out), out.event_set.value_counts().to_dict())
print("panel majority meaning:", p.majority_meaning_changed.value_counts().to_dict(), "| majority flag:", p.majority_flag_correct.value_counts().to_dict())
print("physicians agreed meaning (312):", out.physicians_agreed_meaning.value_counts().to_dict())
print("132 by corpus:", out[out.event_set == "physician_132"].corpus.value_counts().to_dict())
