"""Two-physician agreement on the 312 ASR-flagged events (180-event set + 132-event set).

Inputs: physician_1_312.xlsx and physician_2_312.xlsx (sheet 'Reviews'), matched on Event set + Matched Event ID.
Outcomes: Automated flag correct? / Clinical meaning changed? (No, Partially, Yes) and Confidence
(Low, Moderate, High). Values normalised (strip, lower, 'partial*' -> Partially).
Statistics, as in the earlier five-reviewer analysis: exact agreement, Cohen's kappa, linear-weighted
kappa (ordinal items), Gwet's AC1 (kappa paradox for near-constant items), and the binary meaning
outcome (Yes/Partially vs No). 95% CIs: percentile bootstrap, 2,000 resamples, seed 20260915.
Subsets: all 312; 180-event set; 132-event set; 132 by corpus; by review stage (the 41 'After confirmed
calibration' events were discussed jointly, so agreement on them is not independent).
Writes Clinical_Validation_Two_Physician_312.xlsx next to this script.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

U = Path(sys.argv[1])
OUT = Path(__file__).parent / "Clinical_Validation_Two_Physician_312.xlsx"
rng = np.random.default_rng(20260915)
CATS = {"flag": ["No", "Partially", "Yes"], "meaning": ["No", "Partially", "Yes"], "confidence": ["Low", "Moderate", "High"]}
COLS = {"flag": "Automated flag correct?", "meaning": "Clinical meaning changed?", "confidence": "Confidence"}


def norm(v, item):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if item == "confidence":
        return {"low": "Low", "moderate": "Moderate", "medium": "Moderate", "high": "High"}.get(s, np.nan)
    if s.startswith("partial"):
        return "Partially"
    return {"yes": "Yes", "no": "No"}.get(s, np.nan)


def load(f, who):
    d = pd.read_excel(f, sheet_name="Reviews")
    d = d[d["Matched Event ID"].notna()].copy()
    d["key"] = d["Event set"].astype(str) + "|" + d["Matched Event ID"].astype(str)
    assert not d.key.duplicated().any(), f"duplicate event ids in {who}"
    for k, c in COLS.items():
        d[f"{k}_{who}"] = d[c].map(lambda v: norm(v, k))
    return d


def cohen(a, b, cats, weights=None):
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    if n == 0:
        return np.nan
    w = np.ones((k, k)) - np.eye(k) if weights is None else np.abs(np.subtract.outer(range(k), range(k))) / (k - 1)
    po = 1 - (w * m).sum() / n
    pe = 1 - (w * np.outer(m.sum(1), m.sum(0))).sum() / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def ac1(a, b, cats):
    n = len(a)
    if n == 0:
        return np.nan
    pa = np.mean([x == y for x, y in zip(a, b)])
    pi = np.array([(np.sum(np.array(a) == c) + np.sum(np.array(b) == c)) / (2 * n) for c in cats])
    pe = (pi * (1 - pi)).sum() / (len(cats) - 1)
    return (pa - pe) / (1 - pe)


def stats(a, b, cats, ordinal):
    a, b = np.asarray(a), np.asarray(b)
    fns = {"agreement": lambda x, y: np.mean(x == y), "kappa": lambda x, y: cohen(x, y, cats),
           "ac1": lambda x, y: ac1(x, y, cats)}
    if ordinal:
        fns["weighted_kappa"] = lambda x, y: cohen(x, y, cats, "linear")
    r = {"n": len(a)}
    boots = {k: [] for k in fns}
    for _ in range(2000):
        i = rng.integers(0, len(a), len(a))
        for k, f in fns.items():
            boots[k].append(f(a[i], b[i]))
    for k, f in fns.items():
        r[k] = f(a, b)
        lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
        r[f"{k}_ci"] = f"[{lo:.2f}, {hi:.2f}]"
    return r


A = load(U / "527a7308-physician_1_312.xlsx", "A")
Z = load(U / "007b2db9-physician_2_312.xlsx", "Z")
keep = ["key", "Event set", "Matched Event ID", "Corpus", "Speaker", "Cue type", "Review stage"]
d = A[keep + [f"{k}_A" for k in COLS]].merge(Z[["key", "Review stage"] + [f"{k}_Z" for k in COLS]], on="key", suffixes=("", "_Z"))
print("events matched:", len(d), "| review stage identical in both files:", (d["Review stage"] == d["Review stage_Z"]).all())
d["meaning_bin_A"] = d.meaning_A.map(lambda v: np.nan if pd.isna(v) else ("Changed" if v in ("Yes", "Partially") else "Not changed"))
d["meaning_bin_Z"] = d.meaning_Z.map(lambda v: np.nan if pd.isna(v) else ("Changed" if v in ("Yes", "Partially") else "Not changed"))

subsets = {"All 312 events": d,
           "180-event set": d[d["Event set"].str.startswith("180")],
           "132-event set": d[d["Event set"].str.startswith("132")],
           "132-event set excl. 41 calibration events": d[d["Event set"].str.startswith("132") & (d["Review stage"] != "After confirmed calibration")]}
for c in ["PriMock57", "Fareez", "AfriSpeech-Dialog"]:
    subsets[f"132-event set: {c}"] = d[d.Corpus == c]
for s in d["Review stage"].unique():
    subsets[f"Stage: {s}"] = d[d["Review stage"] == s]

rows = []
for name, sub in subsets.items():
    for item, cats, ordinal in [("meaning", CATS["meaning"], True), ("meaning_bin", ["Not changed", "Changed"], False),
                                ("flag", CATS["flag"], True), ("confidence", CATS["confidence"], True)]:
        x = sub[[f"{item}_A", f"{item}_Z"]].dropna()
        if len(x) < 5:
            continue
        r = stats(x[f"{item}_A"].values, x[f"{item}_Z"].values, cats, ordinal)
        rows.append({"Subset": name, "Item": {"meaning": "Clinical meaning changed (No/Partially/Yes)",
                     "meaning_bin": "Clinical meaning changed (binary)", "flag": "Automated flag correct",
                     "confidence": "Confidence"}[item], **r})
res = pd.DataFrame(rows)

dist = []
for item in COLS:
    for who, name in [("A", "Physician 1"), ("Z", "Physician 2")]:
        vc = d[f"{item}_{who}"].value_counts()
        dist.append({"Item": item, "Rater": name, **{c: int(vc.get(c, 0)) for c in CATS[item]}, "missing": int(d[f"{item}_{who}"].isna().sum())})
dist = pd.DataFrame(dist)
mat = pd.crosstab(d.meaning_A, d.meaning_Z, margins=True).reindex(index=CATS["meaning"] + ["All"], columns=CATS["meaning"] + ["All"])
cons = d[d.meaning_A == d.meaning_Z]
consensus = cons.groupby(["Event set", "Corpus"], dropna=False).meaning_A.value_counts().unstack(fill_value=0)
ev = d[["Event set", "Matched Event ID", "Corpus", "Speaker", "Cue type", "Review stage", "meaning_A", "meaning_Z", "flag_A", "flag_Z", "confidence_A", "confidence_Z"]]
ev = ev.assign(meaning_agree=(d.meaning_A == d.meaning_Z).map({True: "Agree", False: "Disagree"}))

with pd.ExcelWriter(OUT) as w:
    res.round(3).to_excel(w, sheet_name="Agreement", index=False)
    dist.to_excel(w, sheet_name="Rating distribution", index=False)
    mat.to_excel(w, sheet_name="Meaning matrix (P1 x P2)")
    consensus.to_excel(w, sheet_name="Agreed meaning by set")
    ev.to_excel(w, sheet_name="Event-level", index=False)
pd.set_option("display.width", 250)
show = res[res.Item.str.startswith("Clinical meaning changed (No")][["Subset", "n", "agreement", "kappa", "kappa_ci", "weighted_kappa", "weighted_kappa_ci", "ac1", "ac1_ci"]]
print(show.round(3).to_string(index=False))
print("\nBinary meaning:\n" + res[res.Item.str.contains("binary")][["Subset", "n", "agreement", "kappa", "kappa_ci", "ac1"]].round(3).to_string(index=False))
print("\nFlag / confidence (all 312):\n" + res[(res.Subset == "All 312 events") & res.Item.isin(["Automated flag correct", "Confidence"])][["Item", "n", "agreement", "kappa", "kappa_ci", "ac1", "ac1_ci"]].round(3).to_string(index=False))
print("\n" + dist.to_string(index=False))
print("\nMeaning matrix (rows Physician 1, cols Physician 2):\n" + mat.to_string())
