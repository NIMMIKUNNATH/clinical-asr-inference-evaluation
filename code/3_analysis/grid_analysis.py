"""
grid_analysis.py - the three-corpus analysis: every configuration on every corpus's held-out set.

Held-out sets: Fareez 231, PriMock57 37, AfriSpeech-Dialog 20 clinical consultations. Configurations, all large-v3
unless marked: released defaults; tuned on Fareez (40-recording primary search) and on Fareez 20; tuned on
PriMock57 (20); tuned on AfriSpeech-Dialog (20); joint; and the matched checkpoint pairs (reference and trial32
decoding vectors on large-v2 and large-v3).

Writes results_grid/:
  grid_rates.csv     pooled rates for every (corpus, configuration): WER and components, WDER, cpWER,
                     SA-WER, insertion-inclusive negation, hallucinated words per 1,000, recordings scored
  comparisons.csv    paired recording-level comparisons, same method as compare_heldout.py (mean paired
                     difference, 10,000-resample percentile bootstrap, seed 20260915, Wilcoxon signed-rank,
                     rank-biserial, better/worse/tied): every configuration vs defaults; joint vs each corpus's local
                     configuration (the cost of globality); large-v3 vs large-v2 under each decoding vector.
                     All secondary and unadjusted (the confirmatory family is the Fareez primary comparison).
  grid_summary.md    readable tables

USAGE (eval venv):  python grid_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

SEED, B = 20260915, 10000
OUT = ROOT / "results_grid"
CORPORA = {
    "Fareez": ROOT / "final" / "scores",
    "PriMock57": ROOT / "final_primock" / "scores",
    "AfriSpeech-Dialog": ROOT / "final_afrispeech" / "scores",
}
LOCAL = {"Fareez": "local20_fareez", "PriMock57": "primock_tuned", "AfriSpeech-Dialog": "local20_afrispeech"}
# the joint search selected exactly the PriMock57 configuration, so "joint" reads that configuration's scores
ALIAS = {"joint": "primock_tuned"}
LABEL = {"defaults": "released defaults", "tuned": "Fareez-tuned (primary, 40)", "local20_fareez": "Fareez-tuned (20)",
         "joint": "joint (20+20+20)", "primock_tuned": "PriMock57-tuned (20) = joint", "local20_afrispeech": "AfriSpeech-tuned (20)", "joint": "joint (20+20+20)",
         "oldref_v2_fp16": "reference vector, large-v2", "v3_oldref": "reference vector, large-v3",
         "trial32_v2_fp16": "trial32 vector, large-v2", "v3_trial32": "trial32 vector, large-v3"}
OUTCOMES = {
    "WER": (["sub", "dele", "ins"], "ref_words", 100),
    "substitutions": (["sub"], "ref_words", 100),
    "deletions": (["dele"], "ref_words", 100),
    "insertions": (["ins"], "ref_words", 100),
    "WDER": (["wder_wrong"], "wder_aligned", 100),
    "cpWER": (["cp_err"], "cp_n", 100),
    "SA-WER": (["sa_err"], "sa_n", 100),
    "negation (incl. insertions)": (["neg_err", "neg_ins"], "neg_ref", 100),
    "hallucinated words / 1k": (["halluc_words"], "ref_words", 1000),
}
COMPARE = ["WER", "WDER", "cpWER", "insertions", "deletions", "hallucinated words / 1k",
           "negation (incl. insertions)"]


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

def heldout(corpus):
    if corpus == "Fareez":
        return set(pd.read_csv(CORPORA["Fareez"] / "tuned.csv").recording)
    if corpus == "PriMock57":
        return set(json.loads((ROOT / "results_primock" / "dev_split.json").read_text())["heldout"])
    ids = pd.read_csv(ROOT / "cache" / "afrispeech" / "ids.csv")
    return set(ids[ids.domain == "medical"].id)


def load(corpus, config):
    f = CORPORA[corpus] / f"{ALIAS.get(config, config)}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d[d.recording.isin(heldout(corpus))].sort_values("recording").reset_index(drop=True)


def md(t):
    """A DataFrame as a Markdown table, without the optional tabulate dependency."""
    cols = list(t.columns)
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) and abs(v) < 1e-3 and v != 0 else str(v)   # noqa: E731
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in t.itertuples(index=False)]
    return "\n".join(rows)


def pooled(d, num, den, scale):
    return scale * d[num].sum().sum() / d[den].sum()


def compare(corpus, a_name, b_name, kind):
    a, b = load(corpus, a_name), load(corpus, b_name)
    if a is None or b is None:
        return []
    common = sorted(set(a.recording) & set(b.recording))
    a, b = (x[x.recording.isin(common)].reset_index(drop=True) for x in (a, b))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(common), size=(B, len(common)))
    rows = []
    for name in COMPARE:
        num, den, scale = OUTCOMES[name]
        ra = scale * a[num].sum(axis=1) / a[den].replace(0, np.nan)
        rb = scale * b[num].sum(axis=1) / b[den].replace(0, np.nan)
        d = (rb - ra).to_numpy()
        d = d[~np.isnan(d)]
        boot = d[idx[:, :len(d)] % len(d)].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        try:
            p = wilcoxon(d, zero_method="wilcox").pvalue
        except ValueError:
            p = 1.0
        rows.append(dict(corpus=corpus, kind=kind, reference=a_name, comparator=b_name, outcome=name, n=len(common),
                         reference_rate=pooled(a, num, den, scale), comparator_rate=pooled(b, num, den, scale),
                         mean_paired_difference=d.mean(), ci95_lower=lo, ci95_upper=hi,
                         recordings_better=int((d < 0).sum()), recordings_worse=int((d > 0).sum()),
                         recordings_tied=int((d == 0).sum()), rank_biserial=rank_biserial(d), p_raw=p))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    rates = []
    for corpus in CORPORA:
        for config in LABEL:
            d = load(corpus, config)
            if d is None or not len(d):
                continue
            r = dict(corpus=corpus, config=config, label=LABEL[config], n=len(d), n_heldout=len(heldout(corpus)))
            r.update({k: pooled(d, *v) for k, v in OUTCOMES.items()})
            rates.append(r)
    rates = pd.DataFrame(rates)
    rates.round(4).to_csv(OUT / "grid_rates.csv", index=False)

    comps = []
    for corpus in CORPORA:
        for config in LABEL:
            if config != "defaults":
                comps += compare(corpus, "defaults", config, "vs defaults")
        comps += compare(corpus, LOCAL[corpus], "joint", "cost of globality (joint minus local)")
        comps += compare(corpus, "oldref_v2_fp16", "v3_oldref", "checkpoint, reference vector (v3 minus v2)")
        comps += compare(corpus, "trial32_v2_fp16", "v3_trial32", "checkpoint, trial32 vector (v3 minus v2)")
    comps = pd.DataFrame(comps)
    comps.round(6).to_csv(OUT / "comparisons.csv", index=False)

    L = ["# Three-corpus grid", "", "Pooled rates on each corpus's held-out set (Fareez 231, PriMock57 37, "
         "AfriSpeech-Dialog 20 clinical). n = recordings scored.", ""]
    for corpus in CORPORA:
        t = rates[rates.corpus == corpus][["label", "n", "WER", "deletions", "insertions", "WDER", "cpWER",
                                           "hallucinated words / 1k"]].round(2)
        L += [f"## {corpus}", "", md(t), ""]
    L += ["## Paired comparisons (WER and WDER)", "",
          "Mean paired recording-level difference, comparator minus reference; negative favours the comparator.", ""]
    c = comps[comps.outcome.isin(["WER", "WDER"])].copy()
    c["95% CI"] = c.apply(lambda r: f"{r.ci95_lower:+.2f} to {r.ci95_upper:+.2f}", axis=1)
    c["b/w"] = c.recordings_better.astype(str) + "/" + c.recordings_worse.astype(str)
    L += [c[["corpus", "kind", "reference", "comparator", "outcome", "n", "mean_paired_difference", "95% CI", "b/w",
             "p_raw"]].round(3).pipe(md), ""]
    (OUT / "grid_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(rates)} cells, {len(comps)} comparison rows")
    print((OUT / "grid_summary.md").read_text(encoding="utf-8")[:6000])


if __name__ == "__main__":
    main()
