"""
build_supplementary_data.py - Supplementary Data 1: every paired comparison, with raw and adjusted p-values.

One row per (comparison, outcome). Multiplicity handling follows the prespecified plan exactly:

  confirmatory  tuned vs defaults on the Fareez held-out set, WER and WDER only. Holm correction across those
                two tests. These are the only adjusted p-values in the study.
  secondary     every other outcome and every other comparison. Raw p-values, reported for completeness and
                explicitly unadjusted. They are not used to claim significance.

compare_heldout.py prints an 'H' beside WER and WDER for every comparison it runs; that marker reflects the
script's own two-outcome family, not the study's. The family assignment below overrides it, so that only the
prespecified comparison carries an adjusted value.

USAGE:  ~/eval_venv/bin/python build_supplementary_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

SCORES = ROOT / "final" / "scores"
OUT = ROOT / "paper" / "Supplementary_Data_1.csv"

# comparison file stem -> (label, corpus, role in the paper)
COMPARISONS = {
    "compare_tuned_vs_defaults":         ("tuned vs defaults (large-v3)",      "Fareez held-out", "confirmatory"),
    "compare_oldref_v2_vs_defaults":     ("large-v2 reference vs defaults",    "Fareez held-out", "secondary"),
    "compare_trial32_v2_vs_defaults":    ("large-v2 trial32 vs defaults",      "Fareez held-out", "secondary"),
    "compare_primock_tuned_vs_defaults": ("PriMock57-tuned vs defaults",       "Fareez held-out", "secondary"),
    "compare_v3_oldref_vs_oldref_v2":    ("checkpoint: reference vector",      "Fareez held-out", "secondary"),
    "compare_v3_trial32_vs_trial32_v2":  ("checkpoint: trial32 vector",        "Fareez held-out", "secondary"),
}
CONFIRMATORY_OUTCOMES = ["WER", "WDER"]


def holm(p):
    """Holm step-down adjustment."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    frames = []
    for stem, (label, corpus, role) in COMPARISONS.items():
        path = SCORES / f"{stem}.csv"
        if not path.exists():
            print(f"  missing {path.name} - skipped")
            continue
        d = pd.read_csv(path)
        d.insert(0, "comparison", label)
        d.insert(1, "corpus", corpus)
        d["outcome"] = d["outcome"].str.strip()
        d["family"] = np.where((role == "confirmatory") & d["outcome"].isin(CONFIRMATORY_OUTCOMES),
                               "confirmatory", "secondary")
        frames.append(d)

    sd = pd.concat(frames, ignore_index=True)
    sd["p_holm"] = np.nan
    conf = sd["family"] == "confirmatory"
    if conf.sum() != len(CONFIRMATORY_OUTCOMES):
        sys.exit(f"expected {len(CONFIRMATORY_OUTCOMES)} confirmatory tests, found {int(conf.sum())}")
    sd.loc[conf, "p_holm"] = holm(sd.loc[conf, "p"])

    sd = sd.rename(columns={"a": "reference_rate", "b": "comparator_rate", "diff": "mean_paired_difference",
                            "lo": "ci95_lower", "hi": "ci95_upper", "better": "recordings_better",
                            "worse": "recordings_worse", "tied": "recordings_tied", "rb": "rank_biserial",
                            "p": "p_raw"})
    cols = ["comparison", "corpus", "family", "outcome", "reference_rate", "comparator_rate",
            "mean_paired_difference", "ci95_lower", "ci95_upper", "recordings_better", "recordings_worse",
            "recordings_tied", "rank_biserial", "p_raw", "p_holm"]
    # the three-corpus grid comparisons, all secondary and unadjusted, appended in the same columns
    grid = ROOT / "results_grid" / "comparisons.csv"
    if grid.exists():
        g = pd.read_csv(grid)
        g = g.rename(columns={"reference_rate": "reference_rate", "comparator_rate": "comparator_rate"})
        g["comparison"] = g.kind + ": " + g.comparator + " vs " + g.reference
        g["corpus"] = g.corpus + " held-out"
        g["family"] = "secondary"
        g["p_holm"] = float("nan")
        g = g.rename(columns={"p_raw": "p_raw"})
        sd = pd.concat([sd, g[[c for c in cols if c in g.columns]]], ignore_index=True)
    sd = sd[cols]
    for c in ["reference_rate", "comparator_rate", "mean_paired_difference", "ci95_lower", "ci95_upper",
              "rank_biserial"]:
        sd[c] = sd[c].round(4)
    sd.to_csv(OUT, index=False)

    print(f"wrote {OUT}")
    print(f"  {len(sd)} rows across {sd.comparison.nunique()} comparisons")
    print(f"  confirmatory (Holm-adjusted): {int(conf.sum())}   secondary (raw, unadjusted): {int((~conf).sum())}")
    print("\n  confirmatory family:")
    print(sd.loc[conf, ["outcome", "mean_paired_difference", "ci95_lower", "ci95_upper", "p_raw", "p_holm"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
