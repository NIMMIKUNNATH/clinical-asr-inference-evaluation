"""Sensitivity of the hallucinated-text measure to the run-length threshold (checklist item 6).

Recomputes, from the stored transcripts, the number of inserted words that fall in runs of at least
k consecutive insertions, for k = 2..10, on the 231 held-out Fareez recordings under the released
defaults and the tuned configuration. No decoding is repeated.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/d/fareez_optuna/code")
sys.path.insert(0, "/mnt/d/fareez_optuna/code/final")

from common import REF_DIR, parse_reference          # noqa: E402
from metrics import align, norm_main                 # noqa: E402

ROOT = Path("/mnt/d/fareez_optuna/final")
KS = [2, 3, 4, 5, 6, 8, 10]
CONFIGS = ["defaults", "tuned"]

held = [r["recording"] for r in csv.DictReader(open(ROOT / "scores" / "defaults.csv"))]
print("held-out recordings: {}".format(len(held)))

res = {}
for cfg in CONFIGS:
    runs_all, nref = [], 0
    for rec in held:
        p = ROOT / cfg / "{}.json".format(rec)
        out = json.loads(p.read_text(encoding="utf-8"))
        turns = parse_reference(REF_DIR / "{}.txt".format(rec))
        ref = [t for _, x in turns for t in norm_main(x)]
        hyp = [t for s in out["segments"] for t in norm_main(s["text"])]
        w, ops = align(ref, hyp)
        nref += w["n"]
        runs_all += [len(h) for op, r, h in ops if op == "insert"]
    res[cfg] = (runs_all, nref)
    print("  {:<9} {} insertion runs, {} reference words".format(cfg, len(runs_all), nref))

print("\nthreshold   defaults/1k   tuned/1k   reduction   runs(def)  runs(tuned)")
rows = []
for k in KS:
    d_runs, d_n = res["defaults"]
    t_runs, t_n = res["tuned"]
    dw = sum(x for x in d_runs if x >= k)
    tw = sum(x for x in t_runs if x >= k)
    dr = sum(1 for x in d_runs if x >= k)
    tr = sum(1 for x in t_runs if x >= k)
    d1k, t1k = 1000 * dw / d_n, 1000 * tw / t_n
    red = 100 * (1 - t1k / d1k) if d1k else float("nan")
    rows.append((k, d1k, t1k, red, dr, tr))
    print("   >= {:<3}   {:9.2f}   {:8.2f}   {:7.1f}%   {:7d}  {:9d}".format(k, d1k, t1k, red, dr, tr))

out = Path("/mnt/d/fareez_optuna/final/scores/halluc_threshold_sensitivity.csv")
with out.open("w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(["threshold", "defaults_per_1k", "tuned_per_1k", "reduction_pct", "runs_defaults", "runs_tuned"])
    for r in rows:
        wtr.writerow([r[0], round(r[1], 2), round(r[2], 2), round(r[3], 1), r[4], r[5]])
print("\nwrote", out)
