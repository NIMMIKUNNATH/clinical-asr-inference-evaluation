"""Consultation-level cluster bootstrap for the clinical-meaning agreement statistics.

Recording of each event: 180-event set -> matched by text of the reference context against the Fareez reference
transcripts (paper2 truth JSON); 132-event set -> the sampling tracking file (_TRACKING_not_for_reviewers.csv,
Source Event ID = event_id). Resamples recordings with replacement (2,000, seed 20260915) and recomputes
(a) five-reviewer Fleiss kappa on the 180 panel events (complete case, physicians' final ratings),
(b) two-physician Cohen kappa on all 312 events. Event-level bootstrap printed alongside for comparison.
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

U = Path(sys.argv[1]); TRUTH = Path("/mnt/e/OSCE_Videos/analysis/paper2_diarization/data/truth/fareez")
TRACK = Path("clinical_review_inputs/_TRACKING_not_for_reviewers.csv")
HERE = Path(__file__).parent
rng = np.random.default_rng(20260915)
CATS = ["No", "Partially", "Yes"]
def norm(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    return "Partially" if s.startswith("partial") else {"yes": "Yes", "no": "No"}.get(s, np.nan)
def clean(s): return re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\[\[|\]\]", "", str(s).lower())).split()

def flat(o):
    if isinstance(o, str): return [o]
    if isinstance(o, dict): return sum((flat(v) for k, v in o.items() if k not in ("start", "end", "speaker", "role")), [])
    if isinstance(o, list): return sum((flat(v) for v in o), [])
    return []
docs = {p.stem: " " + " ".join(clean(" ".join(flat(json.load(open(p)))))) + " " for p in TRUTH.glob("*.json")}

def load(f):
    d = pd.read_excel(U / f, sheet_name="Reviews"); d = d[d["Matched Event ID"].notna()].copy()
    d["key"] = d["Event set"].astype(str) + "|" + d["Matched Event ID"].astype(str); return d.set_index("key")
A, Z = load("527a7308-physician_1_312.xlsx"), load("007b2db9-physician_2_312.xlsx")
track = pd.read_csv(TRACK).set_index("event_id")

def find_rec_lines(ctx):
    lines = [l for l in str(ctx).split("\n") if l.strip()]
    best, score = None, 0
    for l in lines:
        w = clean(re.sub(r"^(>>> )?(DOCTOR|PATIENT):", "", l.strip()))
        if len(w) < 4: continue
        for n in (12, 8, 6, 4):
            if len(w) < n: continue
            frag = " " + " ".join(w[:n]) + " "
            hits = [k for k, t in docs.items() if frag in t]
            if len(hits) == 1:
                return hits[0]
            if hits: break
    return None


def find_rec(ctx):
    words = []
    for l in str(ctx).splitlines():
        words += clean(re.sub(r"^\s*(>>> )?(DOCTOR|PATIENT):", "", l.strip()))
    for n in (len(words), 40, 30, 20):
        if len(words) < 6 or n > len(words): continue
        for s in range(0, len(words) - n + 1, max(1, (len(words) - n) // 4 or 1)):
            frag = " " + " ".join(words[s:s + n]) + " "
            hits = [k for k, t in docs.items() if frag in t]
            if len(hits) == 1: return hits[0]
            if len(hits) > 1: return "AMBIG(" + ",".join(sorted(hits)[:3]) + ")"
    return find_rec_lines(ctx)

KEY = pd.read_csv("/mnt/e/New_dataset_annotation/code/tpe40/adjudication_key.csv").set_index("event_id").recording_id
rec = {}
for k, r in A.iterrows():
    if k.startswith("180"):
        rec[k] = "Fareez:" + str(KEY[r["Matched Event ID"]])   # sampler key (exact); text matching kept as a cross-check only
    else:
        t = track.loc[r["Source Event ID"]]; rec[k] = f"{t.corpus}:{t.recording}"
rec = pd.Series(rec)
unk = rec.str.endswith("None").sum()
print("events:", len(rec), "| unmatched 180 events:", unk)
for s in ("180", "132"):
    r = rec[rec.index.str.startswith(s)]
    print(s, "set: recordings", r.nunique(), "| events per recording max", r.value_counts().max(), "| distribution", r.value_counts().value_counts().sort_index().to_dict())
r132 = rec[rec.index.str.startswith("132")]
print("132 recordings by corpus:", r132.str.split(":").str[0].groupby(r132.str.split(":").str[0]).size().to_dict(),
      {c: r132[r132.str.startswith(c)].nunique() for c in ["PriMock57", "Fareez", "AfriSpeech-Dialog"]})

# five-reviewer table on 180
ev = pd.read_excel(HERE / "Clinical_Validation_Five_Reviewer_Kappa.xlsx", sheet_name="Event-level ratings", header=2)
ev = ev[ev["Event ID"].astype(str).str.match(r"E\d+")].set_index("Event ID")
P = pd.DataFrame(index=ev.index)
a180 = A[A.index.str.startswith("180")].set_index("Matched Event ID"); z180 = Z[Z.index.str.startswith("180")].set_index("Matched Event ID")
P["C1"] = a180["Clinical meaning changed?"].reindex(ev.index).map(norm); P["C2"] = z180["Clinical meaning changed?"].reindex(ev.index).map(norm)
for n in ["N1", "N2", "N3"]: P[n] = ev[f"Meaning {n}"].map(norm)
P["rec"] = ["" if pd.isna(i) else rec.get("180-event set|" + i, rec.get(next((k for k in rec.index if k.startswith("180") and k.endswith("|" + i)), ""), "")) for i in P.index]
def fleiss(t):
    t = t.dropna(); X = t.values; n = X.shape[1]
    c = np.stack([(X == k).sum(1) for k in CATS], 1)
    Pi = (c * (c - 1)).sum(1) / (n * (n - 1)); p = c.sum(0) / c.sum(); pe = (p ** 2).sum()
    return (Pi.mean() - pe) / (1 - pe)
def cohen(a, b):
    po = np.mean(a == b); pe = sum(np.mean(a == k) * np.mean(b == k) for k in CATS); return (po - pe) / (1 - pe)
R = ["C1", "C2", "N1", "N2", "N3"]
def cluster_ci(df, stat, B=2000):
    g = {k: v.index.values for k, v in df.groupby("rec")}; keys = list(g); vals = []
    for _ in range(B):
        idx = np.concatenate([g[keys[j]] for j in rng.integers(0, len(keys), len(keys))]); vals.append(stat(df.loc[idx]))
    return np.nanpercentile(vals, [2.5, 97.5])
def event_ci(df, stat, B=2000):
    return np.nanpercentile([stat(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(B)], [2.5, 97.5])
f = lambda d: fleiss(d[R])
print(f"\nPanel Fleiss kappa (180): {f(P):.3f} | event CI {event_ci(P, f).round(2)} | cluster CI {cluster_ci(P, f).round(2)} | recordings {P.rec.nunique()}")
D = pd.DataFrame({"a": A["Clinical meaning changed?"].map(norm), "b": Z["Clinical meaning changed?"].reindex(A.index).map(norm), "rec": rec}).dropna()
D = D.reset_index(drop=True)
c = lambda d: cohen(d.a.values, d.b.values)
print(f"Physicians Cohen kappa (312): {c(D):.3f} | event CI {event_ci(D, c).round(2)} | cluster CI {cluster_ci(D, c).round(2)} | recordings {D.rec.nunique()}")
rec.rename("recording").to_csv(HERE / "event_recordings_312.csv")
