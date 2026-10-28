"""
search_joint.py - one parameter search over all three corpora at once, to select a single general configuration.

The Fareez and PriMock57 searches each found the best settings for one corpus. A configuration offered for use
across sites should instead be chosen for all of them. This search optimises the MEAN of the three corpora's
pooled word error rates, each corpus weighted equally, so no corpus dominates because it is longer.

Held identical to the earlier searches: the 17-parameter space (search.suggest), large-v3 float16, TPE with the
same settings, duplicate reuse. Differences, all fixed before the first trial (JOINT_SELECTION_RULE.md):

  development recordings  20 per corpus (the size of the smallest development split, PriMock57), drawn only from
                          each corpus's EXISTING development split, so no held-out
                          recording of Fareez or PriMock57 is ever seen:
                            Fareez     20 of the 40 development recordings (stratified by category)
                            PriMock57  all 20 development consultations
                            AfriSpeech 20 of the 29 GENERAL conversations; the 20 clinical consultations are the
                                        AfriSpeech held-out set and are never seen
  order                   recordings interleaved across corpora, so every prefix covers all three corpora and
                          median pruning compares like with like
  runaway guard           a recording whose decode exceeds max(600 s, its own duration) is abandoned and the trial
                          is pruned: such settings fabricate text without end (seen with trial32 on large-v3)
  warm start              the Fareez-selected and PriMock57-selected configurations are the first two trials

USAGE (WSL, osce venv):  python search_joint.py --trials 60 --worker 0
"""
import argparse
import csv
import json
import os
import random
import sys
import threading
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import REF_DIR, ROOT, SR, normalize, reference_tokens as fareez_reference_tokens, wer_counts  # noqa: E402
import search  # noqa: E402

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

RESULTS = ROOT / "results_joint"
SPLIT = RESULTS / "joint_dev.json"
STUDY = "joint_three_corpus_large_v3"
STORAGE = Path.home() / "fareez_optuna" / "joint.journal"
SEED = 20260922
N = {"fareez": 20, "primock": 20, "afrispeech": 20}
SCOPE = list(N)   # corpora searched; --scope afrispeech restricts to one corpus (local search)
CACHE = {"fareez": ROOT / "cache", "primock": ROOT / "cache" / "primock", "afrispeech": ROOT / "cache" / "afrispeech"}


def stratified(groups, n, rng):
    """Proportional allocation across groups (at least one each where possible), then fill at random."""
    total = sum(len(v) for v in groups.values())
    pick, rest = [], []
    for g in sorted(groups):
        items = sorted(groups[g])
        rng.shuffle(items)
        k = min(len(items), max(1, round(n * len(items) / total)))
        pick += items[:k]
        rest += items[k:]
    rng.shuffle(rest)
    pick = pick[:n] + rest[:max(0, n - len(pick))]
    return sorted(pick[:n])


def joint_split():
    if SPLIT.exists():
        return json.loads(SPLIT.read_text())
    rng = random.Random(SEED)
    ts = pd.read_csv(ROOT / "tuning_set.csv")
    fz = stratified(ts.groupby("category").recording.apply(list).to_dict(), N["fareez"], rng)
    pm_dev = json.loads((ROOT / "results_primock" / "dev_split.json").read_text())["dev"]
    by_day = {}
    for r in pm_dev:
        by_day.setdefault(r.split("_")[0], []).append(r)
    pm = stratified(by_day, N["primock"], rng)
    ids = pd.read_csv(CACHE["afrispeech"] / "ids.csv")
    gen = sorted(ids[ids.domain == "general"].id)
    af = stratified({"general": gen}, N["afrispeech"], rng)
    split = dict(seed=SEED, fareez=fz, primock=pm, afrispeech=af,
                 afrispeech_heldout=sorted(ids[ids.domain == "medical"].id),
                 note="Fareez and PriMock57 recordings are drawn only from their existing development splits.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps(split, indent=1))
    return split


def ref_tokens(corpus, rec):
    if corpus == "fareez":
        return fareez_reference_tokens(REF_DIR / f"{rec}.txt")[0]
    data = json.loads((CACHE[corpus] / "ref" / f"{rec}.json").read_text(encoding="utf-8"))
    if all("start" in t for t in data):
        data = sorted(data, key=lambda t: float(t["start"]))
    return [t for turn in data for t in normalize(turn["text"])]


class Evaluator:
    def __init__(self, worker):
        from faster_whisper import WhisperModel
        self.worker = worker
        split = joint_split()
        # interleave: fareez, primock, afrispeech, fareez, ... so every prefix spans the three corpora
        lists = [[("fareez", r) for r in split["fareez"]], [("primock", r) for r in split["primock"]],
                 [("afrispeech", r) for r in split["afrispeech"]]]
        lists = [lst for lst in lists if lst and lst[0][0] in SCOPE]
        self.items = [x for i in range(max(map(len, lists))) for lst in lists if i < len(lst) for x in [lst[i]]]
        self.audio = {x: np.load(CACHE[x[0]] / "audio" / f"{x[1]}.npy") for x in self.items}
        self.ref = {x: ref_tokens(*x) for x in self.items}
        for c in SCOPE:
            sel = [x for x in self.items if x[0] == c]
            print(f"[j{worker}] {c}: {len(sel)} recordings, "
                  f"{sum(len(self.audio[x]) for x in sel) / SR / 60:.0f} min, "
                  f"{sum(len(self.ref[x]) for x in sel)} reference words", flush=True)
        self.model = WhisperModel(search.MODEL_NAME, device="cuda", compute_type=search.COMPUTE_TYPE,
                                  cpu_threads=4)
        self.detail = RESULTS / f"details_w{worker}.csv"
        self.fields = ["trial", "corpus", "recording", "wer", "sub", "dele", "ins", "n", "seconds", "abandoned"]
        if not self.detail.exists():
            with open(self.detail, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    @staticmethod
    def mean_wer(tot):
        rates = [(t["sub"] + t["dele"] + t["ins"]) / t["n"] for t in tot.values() if t["n"]]
        return sum(rates) / len(rates)

    def objective(self, trial):
        kw = search.suggest(trial)
        trial.set_user_attr("kwargs", json.dumps(kw))
        for t in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if t.params == trial.params:
                trial.set_user_attr("duplicate_of", t.number)
                return t.value
        tot = {c: dict(sub=0, dele=0, ins=0, n=0) for c in SCOPE}
        for step, (corpus, rec) in enumerate(self.items):
            audio = self.audio[(corpus, rec)].astype(np.float32) / 32768.0
            limit = max(600.0, len(audio) / SR)
            t0, hyp, abandoned = time.time(), [], False
            # the between-segment check below cannot fire while the model is stuck inside one enormous segment
            # (seen on AfriSpeech: ~9.7 h on a 16-minute recording), so a watchdog ends the worker; the pool
            # restarts it, and the interrupted trial is marked failed at start-up
            watchdog = threading.Timer(limit * 2, lambda: (print(f"[j{self.worker}] watchdog: {corpus}/{rec} "
                                                                 f"exceeded {limit * 2:.0f}s", flush=True),
                                                           os._exit(4)))
            watchdog.daemon = True
            watchdog.start()
            segments, _ = self.model.transcribe(audio, language="en", task="transcribe", word_timestamps=True, **kw)
            for s in segments:
                hyp += normalize(s.text)
                if time.time() - t0 > limit:
                    abandoned = True
                    break
            watchdog.cancel()       # the decode returned: it must not fire during the next recording
            secs = time.time() - t0
            c = wer_counts(self.ref[(corpus, rec)], hyp)
            with open(self.detail, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writerow(dict(
                    trial=f"trial_{trial.number:03d}", corpus=corpus, recording=rec,
                    wer=round((c["sub"] + c["dele"] + c["ins"]) / c["n"], 5), sub=c["sub"], dele=c["dele"],
                    ins=c["ins"], n=c["n"], seconds=round(secs, 1), abandoned=abandoned))
            if abandoned:
                trial.set_user_attr("abandoned", f"{corpus}/{rec} exceeded {limit:.0f} s")
                raise optuna.TrialPruned()
            for k in ("sub", "dele", "ins", "n"):
                tot[corpus][k] += c[k]
            trial.report(self.mean_wer(tot), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        for corpus, t in tot.items():
            trial.set_user_attr(f"wer_{corpus}", (t["sub"] + t["dele"] + t["ins"]) / t["n"])
        return self.mean_wer(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--scope", default="joint", choices=["joint", "afrispeech", "fareez"],
                    help="joint: all three corpora; afrispeech / fareez: that corpus alone (its local best) on the "
                         "same 20 recordings the joint search uses, so every local search has 20 recordings")
    args = ap.parse_args()
    global RESULTS, STUDY, STORAGE, SCOPE
    if args.scope != "joint":
        RESULTS = ROOT / f"results_{args.scope}_local20"
        STUDY = f"{args.scope}_local20_large_v3"
        STORAGE = Path.home() / "fareez_optuna" / f"{args.scope}_local20.journal"
        SCOPE = [args.scope]
        RESULTS.mkdir(parents=True, exist_ok=True)
    warmup = 15 if args.scope == "joint" else 10

    ev = Evaluator(args.worker)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(STORAGE)))
    study = optuna.create_study(
        study_name=STUDY, storage=storage, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED + args.worker, n_startup_trials=10, multivariate=True,
                                           group=True, constant_liar=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=warmup))
    # a trial left RUNNING for more than six hours belongs to a worker that was killed (watchdog, power loss);
    # mark it failed so the sampler and the trial count are not held up by it
    cutoff = datetime.now() - timedelta(hours=6)
    for t in study.get_trials(deepcopy=False):
        if t.state == optuna.trial.TrialState.RUNNING and t.datetime_start and t.datetime_start < cutoff:
            storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
            print(f"[j{args.worker}] marked interrupted trial {t.number} as failed", flush=True)

    if not study.get_trials(deepcopy=False):
        for name, path in (("fareez_tuned", ROOT / "results" / "final_choice.json"),
                           ("primock_tuned", ROOT / "results_primock" / "primock_choice.json")):
            study.enqueue_trial(json.loads(path.read_text())["params"], user_attrs={"seeded": name},
                                skip_if_exists=True)
        print(f"[j{args.worker}] queued the Fareez- and PriMock57-selected configurations", flush=True)

    stop = optuna.study.MaxTrialsCallback(args.trials, states=(optuna.trial.TrialState.COMPLETE,
                                                               optuna.trial.TrialState.PRUNED))

    def log(study, t):
        best = study.best_value if any(x.state.name == "COMPLETE" for x in study.trials) else None
        extra = {k: round(v, 4) for k, v in t.user_attrs.items() if k.startswith("wer_")}
        print(f"[j{args.worker}] trial {t.number} {t.state.name} value={t.value} best={best} {extra}", flush=True)

    study.optimize(ev.objective, callbacks=[stop, log], catch=(Exception,), gc_after_trial=True)


if __name__ == "__main__":
    main()
