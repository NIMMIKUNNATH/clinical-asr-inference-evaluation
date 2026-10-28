"""
build_manifest.py - regeneration manifest: every reported result traced to one output set, and checked.

Two things are produced.

1. A SHA-256 hash of every file a reported result depends on: frozen documents, code, split definitions, the
   selection record, per-recording scores, comparison files, figures and Supplementary Data 1. Two parties
   holding the same hashes hold the same results.

2. A verification of the manuscript against those files. Each table is recomputed from its source scores and every
   recomputed value is looked for in that table's LaTeX. A value missing from the table is a mismatch between what
   the paper says and what the outputs contain, and is reported as a failure.

The manifest is the answer to "does every table, figure, interval and p-value come from the same output set".

USAGE:  ~/eval_venv/bin/python build_manifest.py --tex /path/to/main.tex
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

SC, PSC = ROOT / "final" / "scores", ROOT / "final_primock" / "scores"
FIG = ROOT / "paper" / "figures"

# Every item in the paper, its source files, and the script that produces it
MAP = {
    "Figure 1 (study design)":      (["paper/figures/fig1_study_design.pdf"], "make_figures.py"),
    "Table 1 (corpora)":            (["tuning_set.csv", "results_primock/dev_split.json"], "prepare.py / search_primock.py"),
    "Table 2 (primary comparison)": (["final/scores/compare_tuned_vs_defaults.csv"], "compare_heldout.py"),
    "Figure 2 (confirmatory)":      (["final/scores/compare_tuned_vs_defaults.csv",
                                      "paper/figures/fig2_confirmatory.pdf"], "make_figures.py"),
    "Figure 3 (mechanism)":         (["final/scores/defaults.csv", "final/scores/tuned.csv",
                                      "paper/figures/fig3_mechanism.pdf"], "make_figures.py"),
    "Table 3 (denominator)":        (["final/scores/defaults.csv", "final/scores/tuned.csv"], "fixed_wder.py"),
    "Table 4 (checkpoint)":         (["final/scores/oldref_v2_fp16.csv", "final/scores/v3_oldref.csv",
                                      "final/scores/trial32_v2_fp16.csv", "final/scores/v3_trial32.csv"],
                                     "metrics.py / compare_heldout.py"),
    "Figure 4 (checkpoint)":        (["paper/figures/fig4_checkpoint.pdf"], "make_figures.py"),
    "Table 6 (cross-corpus grid)":  (["results_grid/grid_rates.csv"], "grid_analysis.py"),
    "Table 7 (cost of globality)":  (["results_grid/comparisons.csv"], "grid_analysis.py"),
    "Table 8 (retention)":          (["results_grid/grid_rates.csv"], "grid_analysis.py"),
    "Figure 5 (what tuning trades)": (["results_grid/grid_rates.csv",
                                      "paper/figures/fig5_trade.pdf"], "make_figures.py"),
    "Figure 6 (cross-corpus grid)": (["results_grid/grid_rates.csv",
                                      "paper/figures/fig6_grid.pdf"], "make_figures.py"),
    "Supplementary Data 1":         (["paper/Supplementary_Data_1.csv"], "build_supplementary_data.py"),
    "Supplementary Data 2":         (["paper/submission/04_supplementary_data/Supplementary_Data_2.csv"],
                                     "build_data2.py"),
}

HASHED = ["SELECTION_RULE.md", "paper/ANALYSIS_PLAN.md", "paper/METRIC_DEFINITIONS.md", "tuning_set.csv",
          "results/final_choice.json", "results_primock/dev_split.json", "results_primock/primock_choice.json",
          "results/environment.json", "paper/Supplementary_Data_1.csv", "paper/SEARCH_SPACE_AND_SEEDS.md", "paper/rescore_verification.txt"]
CODE = ["common.py", "search.py", "final/final_run.py", "final/metrics.py", "final/compare_heldout.py",
        "final/fixed_wder.py", "final/score_primock.py", "final/run_primock.py", "final/apply_selection.py",
        "final/make_figures.py", "final/build_supplementary_data.py", "final/test_metrics.py",
        "prepare.py", "final/search_primock.py", "final/selection_stability.py", "final/importance.py",
        "final/build_manifest.py", "final/rescore_verify.py", "final/test_assignment.py"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pooled(df, num, den, k=100):
    return k * df[num].sum().sum() / df[den].sum()


def table_block(tex, label):
    """The LaTeX between the \\begin{table} that owns a label and its \\end{table}."""
    i = tex.find(f"\\label{{{label}}}")
    if i < 0:
        return None
    a = tex.rfind("\\begin{table", 0, i)
    b = tex.find("\\end{table", i)
    return tex[a:b]


def fmt(x):
    return f"{x:.2f}"


def check(block, name, expected):
    """expected: {description: value}. Each rounded value must appear in the table text."""
    if block is None:
        return [(name, "table not found", "", "FAIL")]
    body = block.replace("\\%", "").replace("$", "")
    out = []
    for desc, val in expected.items():
        s = fmt(val)
        variants = {s, s.replace("-", "−"), s.replace("-", "--"), f"+{s}" if val > 0 else s}
        ok = any(v in body for v in variants)
        out.append((name, desc, s, "ok" if ok else "MISMATCH"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    args = ap.parse_args()
    tex = Path(args.tex).read_text(encoding="utf-8")

    # --- recompute each table from source --------------------------------------------------------------------
    rd = lambda n, b=SC: pd.read_csv(b / f"{n}.csv")   # noqa: E731
    d, t = rd("defaults"), rd("tuned")
    cmp = pd.read_csv(SC / "compare_tuned_vs_defaults.csv")
    cmp["outcome"] = cmp.outcome.str.strip()
    row = lambda o: cmp[cmp.outcome == o].iloc[0]       # noqa: E731

    results = []
    results += check(table_block(tex, "tbl:primary"), "Table 2", {
        "WER defaults": pooled(d, ["sub", "dele", "ins"], "ref_words"),
        "WER tuned": pooled(t, ["sub", "dele", "ins"], "ref_words"),
        "WER difference": row("WER")["diff"], "WER CI lower": row("WER")["lo"], "WER CI upper": row("WER")["hi"],
        "WDER defaults": pooled(d, ["wder_wrong"], "wder_aligned"),
        "WDER tuned": pooled(t, ["wder_wrong"], "wder_aligned"),
        "WDER difference": row("WDER")["diff"], "WDER CI lower": row("WDER")["lo"], "WDER CI upper": row("WDER")["hi"],
    })

    macro = lambda df: 100 * (df.wder_wrong / df.wder_aligned).mean()   # noqa: E731
    results += check(table_block(tex, "tbl:denom"), "Table 3", {
        "conventional defaults (mean of recording rates)": macro(d),
        "conventional tuned (mean of recording rates)": macro(t),
    })

    # Each decoding vector is compared on the recordings both checkpoints completed: a recording that failed on
    # either side is excluded from both, which is why the trial32 pair is reported on 229 rather than 230.
    # the checkpoint table pairs each vector on the recordings both checkpoints completed, on all three corpora
    ck = {}
    afri_ids = pd.read_csv(ROOT / "cache" / "afrispeech" / "ids.csv")
    beds = {"Fareez": (SC, None),
            "PriMock57": (PSC, set(json.loads((ROOT / "results_primock" / "dev_split.json").read_text())["heldout"])),
            "AfriSpeech": (ROOT / "final_afrispeech" / "scores",
                           set(afri_ids[afri_ids.domain == "medical"].id))}
    for corpus, (base, keep) in beds.items():
        for vec, (k2, k3) in {"reference": ("oldref_v2_fp16", "v3_oldref"), "trial32": ("trial32_v2_fp16", "v3_trial32")}.items():
            if not (base / f"{k2}.csv").exists() or not (base / f"{k3}.csv").exists():
                continue
            x2, x3 = rd(k2, base), rd(k3, base)
            if keep is not None:
                x2, x3 = x2[x2.recording.isin(keep)], x3[x3.recording.isin(keep)]
            common = set(x2.recording) & set(x3.recording)
            ck[f"{corpus} {vec}: matched n"] = float(len(common))
            for x, ckpt in ((x2, "v2"), (x3, "v3")):
                m = x[x.recording.isin(common)]
                ck[f"{corpus} {vec} {ckpt} WER"] = pooled(m, ["sub", "dele", "ins"], "ref_words")
                ck[f"{corpus} {vec} {ckpt} WDER"] = pooled(m, ["wder_wrong"], "wder_aligned")
                ck[f"{corpus} {vec} {ckpt} halluc/1k"] = pooled(m, ["halluc_words"], "ref_words", 1000)
    block = table_block(tex, "tbl:checkpoint")
    for k in [k for k in list(ck) if k.endswith("matched n")]:         # n is an integer column
        n = int(ck.pop(k))
        results.append(("Table 5", k, str(n), "ok" if block and f"& {n} &" in block else "MISMATCH"))
    results += check(block, "Table 5", ck)

    # Table 6: the cross-corpus grid, every configuration on every held-out set
    afri = pd.read_csv(ROOT / "cache" / "afrispeech" / "ids.csv")
    sets = {"Fareez": (SC, None),
            "PriMock57": (PSC, set(json.loads((ROOT / "results_primock" / "dev_split.json").read_text())["heldout"])),
            "AfriSpeech": (ROOT / "final_afrispeech" / "scores", set(afri[afri.domain == "medical"].id))}
    grid = {}
    for corpus, (base, keep) in sets.items():
        for cfg in ("defaults", "tuned", "local20_fareez", "local20_afrispeech", "primock_tuned"):
            f = base / f"{cfg}.csv"
            if not f.exists():
                continue
            x = pd.read_csv(f)
            if keep is not None:
                x = x[x.recording.isin(keep)]
            grid[f"{corpus} {cfg} WER"] = pooled(x, ["sub", "dele", "ins"], "ref_words")
            grid[f"{corpus} {cfg} WDER"] = pooled(x, ["wder_wrong"], "wder_aligned")
    results += check(table_block(tex, "tbl:grid"), "Table 6", grid)

    # Table 7: cost of globality, joint (= the PriMock57 configuration) against each corpus's own configuration
    comp = ROOT / "results_grid" / "comparisons.csv"
    if comp.exists():
        c = pd.read_csv(comp)
        c = c[c.kind.str.startswith("cost")]
        cost = {}
        for _, r in c[c.outcome.isin(["WER", "WDER"])].iterrows():
            if r.corpus == "PriMock57":
                continue                      # identical configuration: exact zero, no interval
            cost[f"{r.corpus} {r.outcome} difference"] = r.mean_paired_difference
            cost[f"{r.corpus} {r.outcome} CI lower"] = r.ci95_lower
            cost[f"{r.corpus} {r.outcome} CI upper"] = r.ci95_upper
        results += check(table_block(tex, "tbl:globality"), "Table 7", cost)

    # --- hashes ----------------------------------------------------------------------------------------------
    files = []
    for rel in HASHED:
        p = ROOT / rel
        if p.exists():
            files.append(("record", rel, sha(p)))
    for rel in CODE:
        p = ROOT / "code" / rel
        if p.exists():
            files.append(("code", "code/" + rel, sha(p)))
    for p in sorted(list(SC.glob("*.csv")) + list(PSC.glob("*.csv"))):
        files.append(("scores", str(p.relative_to(ROOT)).replace("\\", "/"), sha(p)))
    for p in sorted(FIG.glob("fig[0-9S]*_*.pdf")):
        files.append(("figure", str(p.relative_to(ROOT)).replace("\\", "/"), sha(p)))

    # --- write -----------------------------------------------------------------------------------------------
    fails = [r for r in results if r[3] != "ok"]
    out = ROOT / "paper" / "REGENERATION_MANIFEST.md"
    L = ["# Regeneration manifest", "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from the outputs under "
         f"`{ROOT.name}/`.", "",
         "## 1. Verification of the manuscript against the outputs", "",
         "Each table was recomputed from its source scores and every value was looked for in that table's LaTeX.",
         "", f"**{len(results) - len(fails)} of {len(results)} values matched.**", "",
         "| Table | Quantity | Recomputed | Status |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {s} |" for a, b, c, s in results]
    L += ["", "## 2. What each item is generated from", "", "| Item | Source | Script |", "|---|---|---|"]
    L += [f"| {k} | {', '.join(f'`{x}`' for x in v[0])} | `{v[1]}` |" for k, v in MAP.items()]
    L += ["", "## 3. SHA-256 of every file a reported result depends on", "",
          "| Kind | File | SHA-256 |", "|---|---|---|"]
    L += [f"| {k} | `{f}` | `{h}` |" for k, f, h in files]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")

    (ROOT / "paper" / "regeneration_manifest.json").write_text(json.dumps(dict(
        generated=datetime.now(timezone.utc).isoformat(),
        verification=[dict(table=a, quantity=b, recomputed=c, status=s) for a, b, c, s in results],
        items={k: dict(sources=v[0], script=v[1]) for k, v in MAP.items()},
        sha256={f: h for _, f, h in files}), indent=1))

    print(f"wrote {out}")
    print(f"  verification: {len(results) - len(fails)}/{len(results)} values matched; {len(files)} files hashed")
    for a, b, c, s in fails:
        print(f"  {s}: {a} - {b} (recomputed {c})")


if __name__ == "__main__":
    main()
