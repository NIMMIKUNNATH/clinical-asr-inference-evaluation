"""
resolve_choice.py - add the selected trial's decoder settings to results/final_choice.json.

apply_selection.py runs in the evaluation environment, which has no Optuna, so it records the chosen trial
number but cannot look up that trial's resolved arguments. This script runs in the transcription environment
and fills them in, leaving every other field untouched.

USAGE (WSL, osce venv):  python resolve_choice.py
"""
import json
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, STORAGE_FILE, STUDY_NAME  # noqa: E402

OUT = ROOT / "results" / "final_choice.json"


def main():
    if not OUT.exists():
        sys.exit("results/final_choice.json does not exist - run apply_selection.py first")
    report = json.loads(OUT.read_text())
    if report.get("transcribe_kwargs"):
        print(f"already resolved: trial {report['trial']}")
        return

    study = optuna.load_study(study_name=STUDY_NAME, storage=optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(STORAGE_FILE))))
    t = study.trials[report["trial"]]
    report["params"] = t.params
    report["transcribe_kwargs"] = json.loads(t.user_attrs["kwargs"])
    report["dev_wer_from_search"] = t.value
    OUT.write_text(json.dumps(report, indent=1))

    print(f"resolved trial {t.number}: search development WER {t.value:.4f}")
    for k, v in t.params.items():
        print(f"  {k:32s} {v}")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
