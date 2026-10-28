"""
resolve_primock_choice.py - pick PriMock57's own winner and write it where the runners can use it.

The PriMock57 search optimises pooled word error rate on that corpus's 20 development consultations. Its lowest
trial is that corpus's local ceiling estimate. This resolves the winning trial's settings and records them, so
they can be evaluated on PriMock57's own 37 held-out consultations and transferred to Fareez in the reverse
direction.

Unlike the Fareez selection, no guardrails are applied here: the PriMock57 search exists to establish what
tuning on that corpus can achieve, not to choose a configuration for deployment. That difference is stated
rather than hidden, because it makes the local ceiling an optimistic bound - the strongest possible comparator
for the shared configuration, which is what the cost-of-globality argument needs.

USAGE (WSL, osce venv):  python resolve_primock_choice.py
"""
import json
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

STUDY = "primock57_large_v3"
STORAGE = Path.home() / "fareez_optuna" / "primock.journal"
OUT = ROOT / "results_primock" / "primock_choice.json"


def main():
    study = optuna.load_study(study_name=STUDY, storage=optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(STORAGE))))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        sys.exit("no completed trials yet")
    best = min(done, key=lambda t: t.value)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        study=STUDY, trial=best.number, dev_wer=best.value, complete_trials=len(done),
        basis="minimum pooled development WER on PriMock57's own 20-consultation split; no guardrails applied",
        params=best.params, transcribe_kwargs=json.loads(best.user_attrs["kwargs"])), indent=1))

    print(f"PriMock57 winner: trial {best.number}, development WER {best.value:.4f} "
          f"(of {len(done)} complete trials)")
    for k, v in best.params.items():
        print(f"  {k:32s} {v}")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
