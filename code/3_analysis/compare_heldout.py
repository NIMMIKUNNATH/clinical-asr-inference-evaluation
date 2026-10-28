"""
compare_heldout.py - the confirmatory comparison: the selected configuration against the released defaults on
the held-out recordings.

Per ANALYSIS_PLAN.md: WER primary, WDER co-primary, Holm-corrected as a family of two. Recording-level pairing
throughout, with a 10,000-resample paired percentile bootstrap (seed 20260915), Wilcoxon signed-rank, the
matched-pairs rank-biserial effect size (comparator minus reference, so negative favours the
comparator), and better/worse/tied counts. Every secondary outcome is reported with the same
paired interval but is labelled secondary and is not multiplicity-corrected.

USAGE:  ~/eval_venv/bin/python compare_heldout.py [--a defaults] [--b tuned]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

SCORES = ROOT / "final" / "scores"
SEED, B = 20260915, 10000

# name -> (numerator columns, denominator column, scale)
OUTCOMES = {
    "WER": (["sub", "dele", "ins"], "ref_words", 100),
    "  substitutions": (["sub"], "ref_words", 100),
    "  deletions": (["dele"], "ref_words", 100),
    "  insertions": (["ins"], "ref_words", 100),
    "WDER": (["wder_wrong"], "wder_aligned", 100),
    "cpWER": (["cp_err"], "cp_n", 100),
    "SA-WER": (["sa_err"], "sa_n", 100),
    "CER": (["cer_err"], "cer_n", 100),
    "doctor WER": (["doc_err"], "doc_n", 100),
    "patient WER": (["pat_err"], "pat_n", 100),
    "negation (incl. insertions)": (["neg_err", "neg_ins"], "neg_ref", 100),
    "negation (reference-anchored)": (["neg_err"], "neg_ref", 100),
    "hallucinated words / 1k": (["halluc_words"], "ref_words", 1000),
}
PRIMARY = ["WER", "WDER"]


def rate(df, num, den, scale):
    return scale * df[num].sum(axis=1) / df[den].replace(0, np.nan)


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation, (W+ - W-) / (W+ + W-).

    d is comparator minus reference per recording; zero differences are dropped and tied absolute differences get
    average ranks. Negative values mean the comparator has the lower (better) rate on most of the rank mass.
    """
    nz = d[d != 0]
    if not len(nz):
        return 0.0
    r = rankdata(np.abs(nz))
    return float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())

def holm(pvals):
    order = np.argsort(pvals)
    adj, run = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="defaults", help="reference configuration")
    ap.add_argument("--b", default="tuned", help="configuration under test")
    args = ap.parse_args()

    a = pd.read_csv(SCORES / f"{args.a}.csv").sort_values("recording").reset_index(drop=True)
    b = pd.read_csv(SCORES / f"{args.b}.csv").sort_values("recording").reset_index(drop=True)
    common = sorted(set(a.recording) & set(b.recording))
    a, b = (d[d.recording.isin(common)].reset_index(drop=True) for d in (a, b))
    print(f"\n{args.b} vs {args.a}: {len(common)} paired recordings, "
          f"{int(a.ref_words.sum()):,} reference words\n")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(common), size=(B, len(common)))
    rows, praw = [], []

    for name, (num, den, scale) in OUTCOMES.items():
        ra, rb = rate(a, num, den, scale), rate(b, num, den, scale)
        pooled_a = scale * a[num].sum().sum() / a[den].sum()
        pooled_b = scale * b[num].sum().sum() / b[den].sum()
        d = (rb - ra).to_numpy()
        ok = ~np.isnan(d)
        d = d[ok]
        boot = d[idx[:, :len(d)] % len(d)].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        try:
            w = wilcoxon(d, zero_method="wilcox")
            p = w.pvalue
            rb_eff = rank_biserial(d)
        except ValueError:
            p, rb_eff = 1.0, 0.0
        rows.append(dict(outcome=name, a=pooled_a, b=pooled_b, diff=d.mean(), lo=lo, hi=hi,
                         better=int((d < 0).sum()), worse=int((d > 0).sum()), tied=int((d == 0).sum()),
                         rb=rb_eff, p=p))
        if name in PRIMARY:
            praw.append(p)

    adj = dict(zip(PRIMARY, holm(np.array(praw))))
    w = max(len(r["outcome"]) for r in rows)
    print(f"  {'outcome':<{w}} {args.a:>9} {args.b:>9} {'diff':>8} {'95% CI':>18} "
          f"{'b/w/t':>13} {'p':>9}")
    for r in rows:
        ci = f"[{r['lo']:+.2f}, {r['hi']:+.2f}]"
        tag = f"{adj[r['outcome']]:.2e} H" if r["outcome"] in adj else f"{r['p']:.2e}  "
        print(f"  {r['outcome']:<{w}} {r['a']:9.2f} {r['b']:9.2f} {r['diff']:+8.2f} {ci:>18} "
              f"{r['better']:>4}/{r['worse']}/{r['tied']:<5} {tag:>9}")
    print("\n  diff = b minus a, negative favours the configuration under test. b/w/t = recordings better, "
          "worse, tied.\n  'H' marks a Holm-adjusted p value within the two-member primary family "
          "(WER, WDER); all others are unadjusted and secondary.")
    pd.DataFrame(rows).to_csv(SCORES / f"compare_{args.b}_vs_{args.a}.csv", index=False)
    print(f"\nsaved -> {SCORES / f'compare_{args.b}_vs_{args.a}.csv'}")


if __name__ == "__main__":
    main()
