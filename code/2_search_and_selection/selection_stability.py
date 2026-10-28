"""
selection_stability.py - how often would the selection rule have chosen the same configuration?

The rule in SELECTION_RULE.md was applied once, to the 40 development recordings we happen to have. With ten
candidates separated by fractions of a percentage point, the winner may reflect which recordings were sampled
rather than which configuration is better. This resamples those 40 recordings with replacement and reapplies
the complete rule - guardrails against the resampled defaults, the fallback when none is admissible, then the
tie-break - within each resample, counting how often each candidate is chosen.

A low recovery frequency does not invalidate the selection. It quantifies how much of it is sample-specific,
which is what should be reported alongside a single chosen configuration.

Specified in ANALYSIS_PLAN.md section 6. Candidates are resampled jointly, so every configuration is evaluated
on the same drawn recordings in each replicate.

USAGE:  ~/eval_venv/bin/python selection_stability.py [--draws 2000]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

SCORES = ROOT / "final" / "scores"
SEED = 20260902
MARGIN = dict(wder=0.5, halluc=0.0, negation=1.0)
TRIALS = [70, 15, 48, 66, 30, 37, 33, 69, 59, 61]


def load(name):
    paths = sorted(SCORES.glob(f"{name}.csv")) + sorted(SCORES.glob(f"{name}_s*.csv"))
    if not paths:
        return None
    frames = [pd.read_csv(p).sort_values("recording").reset_index(drop=True) for p in paths]
    if len(frames) == 1:
        return frames[0]
    num = [f.select_dtypes("number") for f in frames]
    mean = sum(num[1:], num[0]) / len(num)
    mean.insert(0, "recording", frames[0]["recording"])
    return mean


def rates(df, idx):
    """Pooled rates over the resampled rows."""
    d = df.iloc[idx]
    s = d.sum(numeric_only=True)
    return dict(wer=100 * (s["sub"] + s["dele"] + s["ins"]) / s["ref_words"],
                wder=100 * s["wder_wrong"] / max(s["wder_aligned"], 1),
                cpwer=100 * s["cp_err"] / max(s["cp_n"], 1),
                negation=100 * (s["neg_err"] + s["neg_ins"]) / max(s["neg_ref"], 1),
                halluc=1000 * s["halluc_words"] / s["ref_words"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()

    base = load("defaults_dev")
    if base is None:
        sys.exit("scores/defaults_dev.csv is missing")
    cands = {}
    for t in TRIALS:
        df = load(f"trial_{t:03d}")
        if df is not None:
            cands[f"trial_{t:03d}"] = df
    print(f"{len(cands)} candidates, {len(base)} development recordings, {args.draws} resamples\n")

    rng = np.random.default_rng(SEED)
    n = len(base)
    winners, admissible_counts, fallback = Counter(), [], 0

    for _ in range(args.draws):
        idx = rng.integers(0, n, n)
        b = rates(base, idx)
        ok, allc = {}, {}
        for name, df in cands.items():
            m = rates(df, idx)
            allc[name] = m
            if (m["wder"] <= b["wder"] + MARGIN["wder"]
                    and m["halluc"] <= b["halluc"] + MARGIN["halluc"]
                    and m["negation"] <= b["negation"] + MARGIN["negation"]):
                ok[name] = m
        admissible_counts.append(len(ok))
        if ok:
            pick = min(ok, key=lambda k: ok[k]["wer"])
        else:
            fallback += 1
            pick = min(allc, key=lambda k: allc[k]["cpwer"])
        winners[pick] += 1

    print(f"  the guardrails failed in {100 * fallback / args.draws:.1f}% of resamples "
          f"(observed: they failed)")
    print(f"  mean admissible candidates per resample: {np.mean(admissible_counts):.2f}\n")
    print(f"  {'candidate':<12} {'selected':>9}")
    for name, c in winners.most_common():
        print(f"  {name:<12} {100 * c / args.draws:8.1f}%")

    out = ROOT / "results" / "selection_stability.json"
    out.write_text(json.dumps(dict(
        draws=args.draws, seed=SEED, margins=MARGIN,
        fallback_rate=fallback / args.draws,
        mean_admissible=float(np.mean(admissible_counts)),
        frequencies={k: v / args.draws for k, v in winners.items()}), indent=1))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
