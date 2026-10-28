"""
select_configs.py - apply JOINT_SELECTION_RULE.md once the searches have finished.

  --locals          each 20-recording local search: the completed trial with the lowest pooled development WER
                    -> results_<corpus>_local20/choice.json
  --joint-top5      print the five completed joint trials with the lowest search objective (for re-evaluation)
  --joint-select    after joint_reevaluate.py has scored those five on the FULL development sets: lowest mean of
                    the three full-development WERs; within 0.10 points, the lowest maximum per-corpus WER
                    -> results_joint/joint_choice.json

USAGE (osce venv):  python select_configs.py --locals | --joint-top5 | --joint-select
"""
import argparse
import json
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

HOME = Path.home() / "fareez_optuna"


def load(study, journal):
    return optuna.load_study(study_name=study, storage=optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(HOME / journal))))


def complete(study):
    return sorted((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                   and "duplicate_of" not in t.user_attrs), key=lambda t: t.value)


def write_choice(path, study_name, t, basis, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = dict(study=study_name, trial=t.number, dev_value=t.value, basis=basis, params=t.params,
             transcribe_kwargs=json.loads(t.user_attrs["kwargs"]),
             complete_trials=None, **(extra or {}))
    path.write_text(json.dumps(d, indent=1))
    print(f"wrote {path}: trial {t.number}, value {t.value:.4f}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--locals", action="store_true")
    g.add_argument("--joint-top5", action="store_true")
    g.add_argument("--joint-select", action="store_true")
    args = ap.parse_args()

    if args.locals:
        for corpus in ("fareez", "afrispeech"):
            name = f"{corpus}_local20_large_v3"
            s = load(name, f"{corpus}_local20.journal")
            c = complete(s)
            write_choice(ROOT / f"results_{corpus}_local20" / "choice.json", name, c[0],
                         "lowest pooled development WER on the 20-recording local search set",
                         dict(n_complete=len(c), n_trials=len(s.trials)))
        return

    s = load("joint_three_corpus_large_v3", "joint.journal")
    top5 = complete(s)[:5]
    if args.joint_top5:
        print(" ".join(str(t.number) for t in top5))
        return

    rows = []
    for t in top5:
        r = json.loads((ROOT / "results_joint" / "reeval" / f"trial_{t.number:03d}.json").read_text())
        rows.append((t, r["mean_wer"], max(r["wer"].values()), r))
    best = min(rows, key=lambda x: x[1])
    near = [x for x in rows if x[1] - best[1] <= 0.0010]
    chosen = min(near, key=lambda x: x[2])
    for t, m, mx, r in rows:
        print(f"trial {t.number}: search {t.value:.4f}  full-dev mean {m:.4f}  max {mx:.4f}  "
              f"{ {k: round(v, 4) for k, v in r['wer'].items()} }" + ("  <- chosen" if t is chosen[0] else ""))
    write_choice(ROOT / "results_joint" / "joint_choice.json", "joint_three_corpus_large_v3", chosen[0],
                 "top five by search objective re-scored on full development sets; lowest mean WER, ties within "
                 "0.10 points to the lowest maximum per-corpus WER (JOINT_SELECTION_RULE.md)",
                 dict(full_dev=chosen[3], candidates={t.number: r for t, _, _, r in rows},
                      n_complete=len(complete(s)), n_trials=len(s.trials)))


if __name__ == "__main__":
    main()
