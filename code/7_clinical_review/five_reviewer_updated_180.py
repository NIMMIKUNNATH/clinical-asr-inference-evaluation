"""Five-reviewer agreement on the 180-event set, with the physicians' UPDATED ratings.

Physicians (Physician 1 = C1, Physician 2 = C2): 180-event rows of physician_1_312.xlsx / physician_2_312.xlsx (updated values).
Nursing students (N1-N3): unchanged, from the event-level
sheet of Clinical_Validation_Five_Reviewer_Kappa.xlsx (independent ratings returned 2026-10-02).
Statistics as before: Fleiss' kappa (complete case and variable-number-of-raters), mean pairwise agreement,
Gwet's AC1, pairwise Cohen kappa, students vs physician consensus; bootstrap 2,000 resamples, seed 20260915.
Also prints the same statistics with the physicians' EARLIER independent ratings for comparison.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

U = Path(sys.argv[1])
HERE = Path(__file__).parent
rng = np.random.default_rng(20260915)
CATS = {"meaning": ["No", "Partially", "Yes"], "flag": ["No", "Partially", "Yes"], "confidence": ["Low", "Moderate", "High"]}


def norm(v, item):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if item == "confidence":
        return {"low": "Low", "moderate": "Moderate", "medium": "Moderate", "high": "High"}.get(s, np.nan)
    return "Partially" if s.startswith("partial") else {"yes": "Yes", "no": "No"}.get(s, np.nan)


ev = pd.read_excel(HERE / "Clinical_Validation_Five_Reviewer_Kappa.xlsx", sheet_name="Event-level ratings", header=2)
ev = ev[ev["Event ID"].astype(str).str.match(r"E\d+")].set_index("Event ID")
new = {}
for who, f in [("C1", "527a7308-physician_1_312.xlsx"), ("C2", "007b2db9-physician_2_312.xlsx")]:
    d = pd.read_excel(U / f, sheet_name="Reviews")
    d = d[d["Event set"].astype(str).str.startswith("180")].set_index("Matched Event ID")
    new[who] = {"meaning": d["Clinical meaning changed?"], "flag": d["Automated flag correct?"], "confidence": d["Confidence"]}
col = {"meaning": "Meaning", "flag": "Flag", "confidence": "Confidence"}


def table(item, updated):
    t = pd.DataFrame(index=ev.index)
    for r in ["C1", "C2", "N1", "N2", "N3"]:
        src = new[r][item] if (updated and r in new) else ev[f"{col[item]} {r}"]
        t[r] = src.reindex(ev.index).map(lambda v: norm(v, item))
    return t


def fleiss(t, cats, variable=False):
    rows = t if variable else t.dropna()
    P, tot, n_items = [], np.zeros(len(cats)), 0
    for _, r in rows.iterrows():
        v = r.dropna().values
        n = len(v)
        if n < 2:
            continue
        c = np.array([np.sum(v == k) for k in cats])
        P.append((np.sum(c * (c - 1))) / (n * (n - 1)))
        tot += c / n
        n_items += 1
    p = tot / n_items
    pe = np.sum(p ** 2)
    return (np.mean(P) - pe) / (1 - pe)


def ac1_multi(t, cats):
    rows = t.dropna()
    P, pi = [], np.zeros(len(cats))
    for _, r in rows.iterrows():
        v = r.values
        n = len(v)
        c = np.array([np.sum(v == k) for k in cats])
        P.append(np.sum(c * (c - 1)) / (n * (n - 1)))
        pi += c / n
    pi /= len(rows)
    pe = np.sum(pi * (1 - pi)) / (len(cats) - 1)
    return (np.mean(P) - pe) / (1 - pe)


def pair_agree(t):
    rows = t.dropna()
    return np.mean([np.mean(rows[a] == rows[b]) for a, b in combinations(rows.columns, 2)])


def cohen(a, b, cats):
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    po = np.mean(a.values == b.values)
    pe = sum(np.mean(a == k) * np.mean(b == k) for k in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan, int(ok.sum()), po


def boot(fn, t, B=2000):
    vals = []
    for _ in range(B):
        vals.append(fn(t.iloc[rng.integers(0, len(t), len(t))]))
    return np.nanpercentile(vals, [2.5, 97.5])


out = []
for updated in (True, False):
    tag = "UPDATED physician values" if updated else "EARLIER independent physician values"
    for item in ["meaning", "flag", "confidence"]:
        t = table(item, updated)
        cats = CATS[item]
        fk = fleiss(t, cats)
        lo, hi = boot(lambda x: fleiss(x, cats), t)
        k_ph, n_ph, a_ph = cohen(t.C1, t.C2, cats)
        lo2, hi2 = boot(lambda x: cohen(x.C1, x.C2, cats)[0], t[["C1", "C2"]])
        stud = [cohen(t[s], t.C1.where(t.C1 == t.C2), cats)[0] for s in ["N1", "N2", "N3"]]
        out.append({"Physician values": tag, "Item": item, "Complete events (all 5)": int(t.dropna().shape[0]),
                    "Fleiss kappa (5 raters)": round(fk, 3), "95% CI": f"[{lo:.2f}, {hi:.2f}]",
                    "Fleiss kappa, variable raters (180)": round(fleiss(t, cats, True), 3),
                    "Mean pairwise agreement": round(pair_agree(t), 3), "Gwet AC1 (5 raters)": round(ac1_multi(t, cats), 3),
                    "Physician Cohen kappa": round(k_ph, 3), "Physician 95% CI": f"[{lo2:.2f}, {hi2:.2f}]",
                    "Physician agreement": f"{a_ph:.3f} ({int(round(a_ph * n_ph))}/{n_ph})",
                    "Students vs physician consensus kappa (N1/N2/N3)": " / ".join(f"{k:.2f}" for k in stud)})
        if updated and item == "meaning":
            maj = t.apply(lambda r: r.dropna().value_counts().idxmax() if r.dropna().value_counts().max() >= 3 else "Tie", axis=1)
            majority = maj.value_counts().to_dict()
res = pd.DataFrame(out)
res.to_excel(HERE / "Clinical_Validation_Five_Reviewer_180_updated.xlsx", index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))
print("\nMajority meaning decision (>=3 of 5), updated physician values:", majority)
