"""
search_primock.py - the same parameter search, run independently on PriMock57.

This establishes PriMock57's *local ceiling*: the accuracy attainable by tuning on that corpus itself. Without
it, a configuration transferred from Fareez cannot be distinguished from a corpus that is simply harder, and no
statement about generality is testable. The difference between the local ceiling and a single shared
configuration is the cost of globality.

Everything is held identical to the Fareez search: the same 17-parameter space (search.suggest), the same
checkpoint and precision, the same sampler and seed policy, the same pooled-WER objective, the same duplicate
reuse. Only the corpus differs.

The development split is drawn once, stratified by recording day, and frozen before the first trial; the
remaining consultations are never seen by the search.

USAGE (WSL, osce venv, HF_TOKEN set):
    python search_primock.py --trials 50 [--worker 0]
"""
import argparse
import csv
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, SR, normalize, wer_counts  # noqa: E402
import search  # noqa: E402

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

CACHE = ROOT / "cache" / "primock"
RESULTS = ROOT / "results_primock"
SPLIT = RESULTS / "dev_split.json"
STUDY = "primock57_large_v3"
STORAGE = Path.home() / "fareez_optuna" / "primock.journal"
SEED = 20260919
N_DEV = 20


def dev_split():
    """Twenty consultations, stratified by day, drawn once and frozen."""
    if SPLIT.exists():
        return json.loads(SPLIT.read_text())["dev"]
    recs = sorted(p.stem for p in (CACHE / "audio").glob("*.npy"))
    by_day = {}
    for r in recs:
        by_day.setdefault(r.split("_")[0], []).append(r)
    rng = random.Random(SEED)
    dev, pool = [], []
    for day in sorted(by_day):
        items = sorted(by_day[day])
        rng.shuffle(items)
        take = max(1, round(N_DEV * len(items) / len(recs)))
        dev += items[:take]
        pool += items[take:]
    rng.shuffle(pool)
    while len(dev) < N_DEV:
        dev.append(pool.pop())
    dev = sorted(dev[:N_DEV])
    RESULTS.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps(dict(seed=SEED, n_dev=len(dev), dev=dev,
                                     heldout=sorted(set(recs) - set(dev))), indent=1))
    print(f"development split frozen: {len(dev)} of {len(recs)} consultations -> {SPLIT}")
    return dev


def reference_tokens(rec):
    data = json.loads((CACHE / "ref" / f"{rec}.json").read_text(encoding="utf-8"))
    turns = data["turns"] if isinstance(data, dict) else data
    turns = sorted(turns, key=lambda t: float(t["start"]))
    return [t for turn in turns for t in normalize(turn["text"])]


class Evaluator:
    def __init__(self, worker):
        from faster_whisper import WhisperModel
        self.worker = worker
        RESULTS.mkdir(parents=True, exist_ok=True)
        self.recs = dev_split()
        self.audio = {r: np.load(CACHE / "audio" / f"{r}.npy") for r in self.recs}
        self.ref = {r: reference_tokens(r) for r in self.recs}
        total = sum(len(v) for v in self.ref.values())
        mins = sum(len(a) for a in self.audio.values()) / SR / 60
        print(f"[p{worker}] {len(self.recs)} consultations, {mins:.0f} min, {total} reference words", flush=True)
        print(f"[p{worker}] loading {search.MODEL_NAME} {search.COMPUTE_TYPE}", flush=True)
        self.model = WhisperModel(search.MODEL_NAME, device="cuda", compute_type=search.COMPUTE_TYPE,
                                  cpu_threads=4)
        self.detail = RESULTS / f"details_w{worker}.csv"
        self.fields = ["trial", "recording", "wer", "sub", "dele", "ins", "n", "seconds"]
        if not self.detail.exists():
            with open(self.detail, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def evaluate(self, kw, label):
        tot = dict(sub=0, dele=0, ins=0, n=0)
        for rec in self.recs:
            audio = self.audio[rec].astype(np.float32) / 32768.0
            t0 = time.time()
            segments, _ = self.model.transcribe(audio, language="en", task="transcribe",
                                                word_timestamps=True, **kw)
            hyp = [t for s in segments for t in normalize(s.text)]
            secs = time.time() - t0
            c = wer_counts(self.ref[rec], hyp)
            for k in ("sub", "dele", "ins", "n"):
                tot[k] += c[k]
            with open(self.detail, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writerow(
                    dict(trial=label, recording=rec,
                         wer=round((c["sub"] + c["dele"] + c["ins"]) / c["n"], 5),
                         sub=c["sub"], dele=c["dele"], ins=c["ins"], n=c["n"], seconds=round(secs, 1)))
        return (tot["sub"] + tot["dele"] + tot["ins"]) / tot["n"], tot

    def objective(self, trial):
        kw = search.suggest(trial)
        trial.set_user_attr("kwargs", json.dumps(kw))
        for t in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if t.params == trial.params:
                trial.set_user_attr("duplicate_of", t.number)
                return t.value
        wer, tot = self.evaluate(kw, f"trial_{trial.number:03d}")
        for k, v in tot.items():
            trial.set_user_attr(k, v)
        return wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--worker", type=int, default=0)
    args = ap.parse_args()
    if "HF_TOKEN" not in os.environ:
        sys.exit("HF_TOKEN is not set")

    ev = Evaluator(args.worker)
    STORAGE.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(STORAGE)))
    study = optuna.create_study(study_name=STUDY, storage=storage, direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED + args.worker, n_startup_trials=10,
                                                                   multivariate=True, group=True,
                                                                   constant_liar=True))
    if not study.get_trials(deepcopy=False):
        # anchor the search with the two configurations whose PriMock57 performance is already known
        study.enqueue_trial(json.loads((ROOT / "results" / "final_choice.json").read_text())["params"],
                            user_attrs={"seeded": "fareez_tuned"}, skip_if_exists=True)
        print("[p%d] queued the Fareez-selected configuration as a seed" % args.worker, flush=True)

    stop = optuna.study.MaxTrialsCallback(args.trials, states=(optuna.trial.TrialState.COMPLETE,))

    def log(study, t):
        print(f"[p{args.worker}] trial {t.number} {t.state.name} value={t.value} best={study.best_value}",
              flush=True)

    study.optimize(ev.objective, callbacks=[stop, log], catch=(Exception,), gc_after_trial=True)


if __name__ == "__main__":
    main()
