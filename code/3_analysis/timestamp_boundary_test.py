"""Test the Discussion's hypothesis that the checkpoint gap in speaker attribution comes from the
word timestamps each checkpoint produces (checklist item 8).

Diarization is cached per recording and identical for both checkpoints, so any difference in how
cleanly words fall inside speaker turns is a property of the recogniser's timestamps alone.

For every word of every held-out Fareez recording, under the reference decoding vector on large-v2
and on large-v3, we compute its overlap with each diarization turn and summarise:
  * straddling words - overlap more than one speaker turn
  * orphan words     - overlap no turn at all (these are counted as misattributed by WDER)
  * decision margin  - (winning overlap - runner-up overlap) / word duration
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/d/fareez_optuna")
DIAR = ROOT / "cache" / "diarization"
RUNS = {"large-v2": "oldref_v2_fp16", "large-v3": "v3_oldref"}

held = [r["recording"] for r in csv.DictReader(open(ROOT / "final" / "scores" / "defaults.csv"))]
print("recordings: {}".format(len(held)))


def turns_of(rec):
    return json.loads((DIAR / "{}.json".format(rec)).read_text())


def analyse(cfg):
    straddle = orphan = total = 0
    margins, durs = [], []
    for rec in held:
        p = ROOT / "final" / cfg / "{}.json".format(rec)
        if not p.exists():
            continue
        segs = json.loads(p.read_text(encoding="utf-8"))["segments"]
        turns = turns_of(rec)
        ts = np.array([[t["start"], t["end"]] for t in turns])
        spk = [t["speaker"] for t in turns]
        for s in segs:
            for w in s.get("words", []):
                a, b = w.get("start"), w.get("end")
                if a is None or b is None or b <= a:
                    continue
                total += 1
                durs.append(b - a)
                ov = np.minimum(b, ts[:, 1]) - np.maximum(a, ts[:, 0])
                ov = np.clip(ov, 0, None)
                per = {}
                for o, sp in zip(ov, spk):
                    if o > 0:
                        per[sp] = per.get(sp, 0.0) + o
                if not per:
                    orphan += 1
                    continue
                vals = sorted(per.values(), reverse=True)
                if len(vals) > 1:
                    straddle += 1
                second = vals[1] if len(vals) > 1 else 0.0
                margins.append((vals[0] - second) / (b - a))
    return dict(total=total, straddle=straddle, orphan=orphan,
                margin_mean=float(np.mean(margins)), margin_p10=float(np.percentile(margins, 10)),
                dur_mean=float(np.mean(durs)))


rows = {}
for name, cfg in RUNS.items():
    rows[name] = analyse(cfg)
    r = rows[name]
    print("\n{}  ({} words)".format(name, r["total"]))
    print("  straddling two speakers : {:6d}  ({:.2f}%)".format(r["straddle"], 100 * r["straddle"] / r["total"]))
    print("  overlapping no turn     : {:6d}  ({:.2f}%)".format(r["orphan"], 100 * r["orphan"] / r["total"]))
    print("  mean decision margin    : {:.3f}".format(r["margin_mean"]))
    print("  10th pct margin         : {:.3f}".format(r["margin_p10"]))
    print("  mean word duration (s)  : {:.3f}".format(r["dur_mean"]))

a, b = rows["large-v2"], rows["large-v3"]
print("\n=== large-v3 minus large-v2 ===")
print("  straddling words : {:+.2f} pp".format(100 * b["straddle"] / b["total"] - 100 * a["straddle"] / a["total"]))
print("  orphan words     : {:+.2f} pp".format(100 * b["orphan"] / b["total"] - 100 * a["orphan"] / a["total"]))
print("  mean margin      : {:+.3f}".format(b["margin_mean"] - a["margin_mean"]))
print("  word duration    : {:+.3f} s".format(b["dur_mean"] - a["dur_mean"]))
