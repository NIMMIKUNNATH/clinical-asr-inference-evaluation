"""
apply_selection.py - apply the frozen rule in SELECTION_RULE.md and write results/final_choice.json.

Implements, in order:
  1. candidates = the ten lowest development-WER complete trials, re-transcribed on the 40 development
     recordings; stochastic candidates (temperature_fallback = short) averaged over three seeds
  2. primary metric = pooled development WER
  3. admissibility guardrails against the default configuration on the same recordings:
       WDER      <= default + 0.5 points
       MC-WER    <= default + 0.5 points        (skipped when concept files are absent; logged as a deviation)
       halluc/1k <= default                      (no margin)
       negation  <= default + 1.0 points
     if no candidate is admissible, the guardrails are reported as failed and the lowest-cpWER candidate in the
     pool is chosen, recorded as a deviation
  4. ties = candidates within one paired-bootstrap standard error of the best WER (10,000 resamples, seed 20260915)
  5. tie-break = lower cpWER, then lower hallucination rate, then fewer settings differing from the defaults

Negation is counted insertion-inclusive, per METRIC_DEFINITIONS.md section 3.1.

USAGE:  ~/eval_venv/bin/python apply_selection.py [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

SCORES = ROOT / "final" / "scores"
OUT = ROOT / "results" / "final_choice.json"
SEED = 20260915
B = 10000
MARGIN = dict(wder=0.5, mc_wer=0.5, halluc=0.0, negation=1.0)


def pooled(df):
    """Pooled (micro) rates from summed counts, per METRIC_DEFINITIONS.md."""
    s = df.sum(numeric_only=True)
    neg_err_incl = s["neg_err"] + s["neg_ins"]
    return dict(
        wer=100 * (s["sub"] + s["dele"] + s["ins"]) / s["ref_words"],
        wder=100 * s["wder_wrong"] / s["wder_aligned"],
        cpwer=100 * s["cp_err"] / s["cp_n"],
        sawer=100 * s["sa_err"] / s["sa_n"],
        negation=100 * neg_err_incl / s["neg_ref"],
        negation_ref_anchored=100 * s["neg_err"] / s["neg_ref"],
        halluc=1000 * s["halluc_words"] / s["ref_words"],
        rtf=s["decode_s"] / s["duration_s"],
        ref_words=int(s["ref_words"]), recordings=len(df))


def per_recording_wer(df):
    return (df["sub"] + df["dele"] + df["ins"]) / df["ref_words"]


def load(config):
    """One candidate: a single scores file, or the mean over seed variants when several exist."""
    paths = sorted(SCORES.glob(f"{config}.csv")) + sorted(SCORES.glob(f"{config}_s*.csv"))
    if not paths:
        return None, []
    frames = [pd.read_csv(p).sort_values("recording").reset_index(drop=True) for p in paths]
    if len(frames) == 1:
        return frames[0], [p.stem for p in paths]
    num = [f.select_dtypes("number") for f in frames]
    mean = sum(num[1:], num[0]) / len(num)
    mean.insert(0, "recording", frames[0]["recording"])
    return mean, [p.stem for p in paths]


def one_se(cand_wer, best_wer):
    """Paired-bootstrap SE of the WER difference from the best candidate (Hastie et al. one-SE rule)."""
    rng = np.random.default_rng(SEED)
    d = np.asarray(cand_wer) - np.asarray(best_wer)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    return float(d[idx].mean(axis=1).std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_df, _ = load("defaults_dev")
    if base_df is None:
        sys.exit("defaults on the development set has not been scored: run final_run.py --config defaults "
                 "--set dev, then metrics.py, and save it as scores/defaults_dev.csv")
    base = pooled(base_df)

    trials = [70, 15, 48, 66, 30, 37, 33, 69, 59, 61]
    cands = {}
    for t in trials:
        name = f"trial_{t:03d}"
        df, sources = load(name)
        if df is None:
            print(f"  {name}: NOT SCORED - skipping")
            continue
        cands[name] = dict(metrics=pooled(df), df=df, seeds=len(sources), sources=sources)
    if not cands:
        sys.exit("no candidates scored yet")

    # --- guardrails -------------------------------------------------------------------------------------------
    for name, c in cands.items():
        m, fails = c["metrics"], []
        if m["wder"] > base["wder"] + MARGIN["wder"]:
            fails.append(f"WDER {m['wder']:.2f} > {base['wder'] + MARGIN['wder']:.2f}")
        if m["halluc"] > base["halluc"] + MARGIN["halluc"]:
            fails.append(f"halluc/1k {m['halluc']:.2f} > {base['halluc']:.2f}")
        if m["negation"] > base["negation"] + MARGIN["negation"]:
            fails.append(f"negation {m['negation']:.2f} > {base['negation'] + MARGIN['negation']:.2f}")
        c["fails"] = fails
        c["admissible"] = not fails

    admissible = {k: v for k, v in cands.items() if v["admissible"]}
    deviations = ["MC-WER guardrail not applied: Google Healthcare NL concept files are absent."]

    if admissible:
        pool, basis = admissible, "admissible candidates, minimum pooled WER"
    else:
        pool, basis = cands, "NO candidate admissible; rule step 3 fallback: minimum pooled cpWER in the pool"
        deviations.append("All ten candidates failed at least one guardrail; selection fell back to lowest "
                          "cpWER as specified in SELECTION_RULE.md step 3.")

    key = "wer" if admissible else "cpwer"
    best = min(pool, key=lambda k: pool[k]["metrics"][key])

    # --- one-standard-error tie set ---------------------------------------------------------------------------
    best_wer_rec = per_recording_wer(pool[best]["df"]).to_numpy()
    se = {k: one_se(per_recording_wer(v["df"]).to_numpy(), best_wer_rec) for k, v in pool.items()}
    thresh = pool[best]["metrics"][key] + 100 * se[best] if se[best] else pool[best]["metrics"][key]
    tied = [k for k, v in pool.items() if v["metrics"][key] <= thresh]

    # --- tie-break: cpWER, then hallucination, then parsimony -------------------------------------------------
    chosen = sorted(tied, key=lambda k: (cands[k]["metrics"]["cpwer"], cands[k]["metrics"]["halluc"]))[0]

    report = dict(
        decided="2026-09-19", rule="SELECTION_RULE.md", basis=basis, seed=SEED, bootstrap=B,
        default_reference={k: v for k, v in base.items()},
        margins=MARGIN,
        one_se_threshold=round(thresh, 4), tie_set=tied, chosen=chosen,
        deviations=deviations,
        candidates={k: dict(metrics={kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                     for kk, vv in v["metrics"].items()},
                            seeds=v["seeds"], sources=v["sources"],
                            admissible=v["admissible"], guardrail_failures=v["fails"])
                    for k, v in cands.items()})

    w = max(len(k) for k in cands)
    print(f"\ndefaults (dev): WER {base['wer']:.2f}  WDER {base['wder']:.2f}  cpWER {base['cpwer']:.2f}  "
          f"neg {base['negation']:.2f}  halluc/1k {base['halluc']:.2f}\n")
    print(f"  {'candidate':<{w}} {'WER':>6} {'WDER':>6} {'cpWER':>6} {'neg':>6} {'hal/1k':>7}  status")
    for k in sorted(cands, key=lambda k: cands[k]["metrics"]["wer"]):
        m = cands[k]["metrics"]
        status = "admissible" if cands[k]["admissible"] else "; ".join(cands[k]["fails"])
        print(f"  {k:<{w}} {m['wer']:6.2f} {m['wder']:6.2f} {m['cpwer']:6.2f} {m['negation']:6.2f} "
              f"{m['halluc']:7.2f}  {status}")
    print(f"\nbasis: {basis}")
    print(f"tie set ({len(tied)}): {', '.join(tied)}")
    print(f"CHOSEN: {chosen}")
    for d in deviations:
        print(f"deviation: {d}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    trial_no = int(chosen.split("_")[1])

    # Resolve the winner's decoder settings from the study so final_run.py --config tuned is runnable.
    # Optuna lives in the transcription environment, not the scoring one, so this is optional here: without it
    # the file is still written and resolve_choice.py fills in the settings afterwards.
    report["trial"] = trial_no
    try:
        import optuna
        from common import STORAGE_FILE, STUDY_NAME
        study = optuna.load_study(study_name=STUDY_NAME, storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(STORAGE_FILE))))
        t = study.trials[trial_no]
        report["params"] = t.params
        report["transcribe_kwargs"] = json.loads(t.user_attrs["kwargs"])
        report["dev_wer_from_search"] = t.value
        resolved = True
    except ImportError:
        report["transcribe_kwargs"] = None
        resolved = False

    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwritten -> {OUT}")
    print(f"chosen trial {trial_no}" + (", settings resolved; --config tuned is runnable" if resolved else
                                        "; run resolve_choice.py in the transcription environment to add its "
                                        "settings before the held-out run"))


if __name__ == "__main__":
    main()
