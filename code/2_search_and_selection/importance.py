"""
importance.py - which parameters actually moved the objective?

Reports fANOVA and MDI importances over the completed trials of a study. Importance is a property of the
sampled region, not of the parameter in general: TPE concentrates trials where the objective is good, so a
parameter abandoned early (for example previous-text conditioning) can show low importance simply because it
was rarely varied. The trial counts per level are printed alongside for that reason.

For the paper this supports a claim about which parameters are worth a controlled one-factor test, and -- once
the same study is run on a second corpus -- whether the importance ranking is stable across corpora.

USAGE (WSL, osce venv):
    python importance.py [--target wer|wder]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import STORAGE_FILE, STUDY_NAME  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="wer", choices=["wer", "wder"])
    args = ap.parse_args()

    study = optuna.load_study(study_name=STUDY_NAME, storage=optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(STORAGE_FILE))))
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if args.target == "wder":
        complete = [t for t in complete if t.user_attrs.get("wder") is not None]
    print(f"study {STUDY_NAME}: {len(complete)} complete trials, target {args.target}\n")

    target = None if args.target == "wer" else (lambda t: t.user_attrs["wder"])
    for name, ev in (("fANOVA", optuna.importance.FanovaImportanceEvaluator(seed=20260915)),
                     ("MDI", optuna.importance.MeanDecreaseImpurityImportanceEvaluator(seed=20260915))):
        imp = optuna.importance.get_param_importances(study, evaluator=ev, target=target)
        print(f"--- {name}")
        for k, v in imp.items():
            levels = Counter(str(t.params.get(k)) for t in complete if k in t.params)
            spread = ", ".join(f"{lv}:{n}" for lv, n in levels.most_common(4))
            print(f"  {v:6.3f}  {k:32s} {len(levels):3d} distinct  [{spread}]")
        print()


if __name__ == "__main__":
    main()
