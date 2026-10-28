"""
search.py - one Optuna worker tuning faster-whisper large-v3 (float16, sequential decoding) on the 40 tuning
recordings. Several workers share one study (run_parallel.sh). Objective: corpus WER over the 40 recordings.

Per recording, every trial:
  cached 16 kHz audio -> faster-whisper transcribe(**trial settings, word_timestamps=True)
  -> WER against the normalised reference
  -> WhisperX assign_word_speakers with the cached pyannote diarization -> WDER (recorded, not optimised:
     diarization is fixed, so WDER barely depends on decoding settings)

Pruning: the running corpus WER is reported after each recording (fixed shuffled order, identical for every
trial) and a trial is stopped if it is worse than the median of earlier trials at the same point.

USAGE (WSL, venv active)
    python search.py --worker 0 --trials 60 [--baseline-first]
"""
import argparse
import csv
import json
import time
import traceback
import warnings

import numpy as np
import optuna
import pandas as pd
from faster_whisper import WhisperModel
from whisperx.diarize import assign_word_speakers

from common import (CACHE, REF_DIR, RESULTS, STORAGE_FILE, STUDY_NAME, hypothesis_tokens, load_json,
                    load_tuning_set, normalize, reference_tokens, wder, wer_counts)

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

MODEL_NAME = "large-v3"
COMPUTE_TYPE = "float16"
SEED = 20260915
MAX_TRIAL_HOURS = 4.0       # a pathological setting (runaway fallback/hallucination loops) is cut off here
CLINICAL_PROMPT = "A medical history-taking consultation between a doctor and a patient."

# faster-whisper's own defaults with VAD on - the untuned pipeline. Scored separately (not part of the space,
# because its full 6-step temperature fallback is deliberately excluded from the search: with larger beams it
# re-decodes failing windows up to six times and makes a trial many times slower).
BASELINE = dict(beam_size=5, patience=1.0, length_penalty=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], best_of=5, compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0, no_speech_threshold=0.6, condition_on_previous_text=True,
                initial_prompt=None, hallucination_silence_threshold=None, vad_filter=True,
                vad_parameters=dict(threshold=0.5, min_speech_duration_ms=0, min_silence_duration_ms=2000,
                                    speech_pad_ms=400))


def suggest(trial):
    kw = {"vad_filter": True}
    kw["beam_size"] = trial.suggest_int("beam_size", 1, 8)
    if kw["beam_size"] > 1:                                   # only meaningful for beam search
        kw["patience"] = trial.suggest_float("patience", 0.5, 2.0)
        kw["length_penalty"] = trial.suggest_float("length_penalty", 0.8, 1.2)
    kw["repetition_penalty"] = trial.suggest_float("repetition_penalty", 1.0, 1.3)
    kw["no_repeat_ngram_size"] = trial.suggest_categorical("no_repeat_ngram_size", [0, 3, 4, 5])

    if trial.suggest_categorical("temperature_fallback", ["off", "short"]) == "off":
        kw["temperature"] = [0.0]
    else:                                                     # these only act when a fallback can happen
        kw["temperature"] = [0.0, 0.2, 0.4]
        kw["best_of"] = trial.suggest_int("best_of", 1, 5)
        kw["compression_ratio_threshold"] = trial.suggest_float("compression_ratio_threshold", 1.8, 3.0)
    kw["log_prob_threshold"] = trial.suggest_float("log_prob_threshold", -2.0, -0.4)   # also gates no-speech skips
    kw["no_speech_threshold"] = trial.suggest_float("no_speech_threshold", 0.3, 0.9)
    kw["condition_on_previous_text"] = trial.suggest_categorical("condition_on_previous_text", [True, False])
    kw["initial_prompt"] = {"none": None, "clinical": CLINICAL_PROMPT}[
        trial.suggest_categorical("initial_prompt", ["none", "clinical"])]
    if trial.suggest_categorical("hallucination_silence", ["off", "on"]) == "on":
        kw["hallucination_silence_threshold"] = trial.suggest_float("hallucination_silence_threshold", 0.5, 4.0)

    kw["vad_parameters"] = dict(
        threshold=trial.suggest_float("vad_threshold", 0.2, 0.7),
        min_speech_duration_ms=trial.suggest_int("vad_min_speech_ms", 0, 250),
        min_silence_duration_ms=trial.suggest_int("vad_min_silence_ms", 100, 2000, log=True),
        speech_pad_ms=trial.suggest_int("vad_speech_pad_ms", 50, 800),
    )
    return kw


class Evaluator:
    def __init__(self, worker):
        self.worker = worker
        self.recs = [r["recording"] for r in sorted(load_tuning_set(), key=lambda r: int(r["order"]))]
        self.audio = {r: np.load(CACHE / "audio" / f"{r}.npy") for r in self.recs}
        self.ref = {r: reference_tokens(REF_DIR / f"{r}.txt") for r in self.recs}
        self.diar = {r: pd.DataFrame(load_json(CACHE / "diarization" / f"{r}.json")) for r in self.recs}
        print(f"[w{worker}] loading {MODEL_NAME} {COMPUTE_TYPE}", flush=True)
        self.model = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE_TYPE, cpu_threads=4)
        RESULTS.mkdir(parents=True, exist_ok=True)
        self.detail_path = RESULTS / f"details_w{worker}.csv"
        self.fields = ["trial", "step", "recording", "wer", "sub", "dele", "ins", "n", "hyp_words", "wder",
                       "aligned_words", "unassigned_words", "seconds", "rtf"]
        if not self.detail_path.exists():
            with open(self.detail_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def run_one(self, rec, kw):
        audio = self.audio[rec].astype(np.float32) / 32768.0
        t0 = time.time()
        segments, _ = self.model.transcribe(audio, language="en", task="transcribe", word_timestamps=True, **kw)
        segments = list(segments)
        secs = time.time() - t0
        hyp = hypothesis_tokens(s.text for s in segments)
        c = wer_counts(self.ref[rec][0], hyp)

        tr = {"segments": [{"start": s.start, "end": s.end, "text": s.text,
                            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in (s.words or [])]}
                           for s in segments]}
        tr = assign_word_speakers(self.diar[rec], tr)
        words = [w for s in tr["segments"] for w in s.get("words", [])]
        runs, unassigned = [], 0
        for w in words:
            spk = w.get("speaker", "UNKNOWN")
            unassigned += spk == "UNKNOWN"
            if runs and runs[-1][0] == spk:
                runs[-1][1].append(w["word"])
            else:
                runs.append([spk, [w["word"]]])
        hyp_tagged = [(tok, spk) for spk, ws in runs for tok in normalize("".join(ws))]
        wd, aligned, mapping = wder(self.ref[rec][1], hyp_tagged)

        row = dict(recording=rec, sub=c["sub"], dele=c["dele"], ins=c["ins"], n=c["n"],
                   wer=round((c["sub"] + c["dele"] + c["ins"]) / c["n"], 5), hyp_words=len(hyp),
                   wder=None if wd is None else round(wd, 5), aligned_words=aligned, unassigned_words=unassigned,
                   seconds=round(secs, 1), rtf=round(secs / (len(audio) / 16000), 4))
        transcript = "\n".join(f"[{s.get('start', 0):8.2f}] {s.get('speaker', 'UNKNOWN')}: {s['text'].strip()}"
                               for s in tr["segments"])
        return row, transcript, mapping

    def evaluate(self, kw, label, trial=None):
        hyp_dir = RESULTS / "hyp" / label
        hyp_dir.mkdir(parents=True, exist_ok=True)
        tot = dict(sub=0, dele=0, ins=0, n=0, wrong=0.0, aligned=0, seconds=0.0)
        start = time.time()
        for step, rec in enumerate(self.recs):
            row, transcript, mapping = self.run_one(rec, kw)
            (hyp_dir / f"{rec}.txt").write_text(f"# speaker mapping for WDER: {mapping}\n{transcript}\n",
                                                encoding="utf-8")
            with open(self.detail_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writerow({"trial": label, "step": step, **row})
            for k in ("sub", "dele", "ins", "n"):
                tot[k] += row[k]
            if row["wder"] is not None:
                tot["wrong"] += row["wder"] * row["aligned_words"]
                tot["aligned"] += row["aligned_words"]
            tot["seconds"] += row["seconds"]
            running = (tot["sub"] + tot["dele"] + tot["ins"]) / tot["n"]
            print(f"[w{self.worker}] {label} step {step + 1:2d}/{len(self.recs)} {rec}  WER {row['wer']:.4f}  "
                  f"running {running:.4f}  {row['seconds']:.0f}s", flush=True)
            if trial is not None:
                trial.report(running, step)
                if trial.should_prune():
                    trial.set_user_attr("pruned_at_step", step)
                    raise optuna.TrialPruned()
                if time.time() - start > MAX_TRIAL_HOURS * 3600:
                    trial.set_user_attr("pruned_reason", f"over {MAX_TRIAL_HOURS} h")
                    raise optuna.TrialPruned()
        n = tot["n"]
        return dict(wer=(tot["sub"] + tot["dele"] + tot["ins"]) / n, sub_rate=tot["sub"] / n,
                    del_rate=tot["dele"] / n, ins_rate=tot["ins"] / n,
                    wder=tot["wrong"] / tot["aligned"] if tot["aligned"] else None,
                    decode_seconds=tot["seconds"], ref_words=n)

    def objective(self, trial):
        kw = suggest(trial)
        trial.set_user_attr("kwargs", json.dumps(kw))
        trial.set_user_attr("worker", self.worker)
        # identical settings already scored -> reuse, do not spend 40 recordings of GPU time again
        for t in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if t.params == trial.params:
                trial.set_user_attr("duplicate_of", t.number)
                return t.value
        res = self.evaluate(kw, f"trial_{trial.number:03d}", trial)
        for k, v in res.items():
            if k != "wer":
                trial.set_user_attr(k, v)
        return res["wer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--trials", type=int, default=60, help="total finished trials (all workers) to stop at")
    ap.add_argument("--baseline-first", action="store_true")
    args = ap.parse_args()

    ev = Evaluator(args.worker)
    if args.baseline_first and not (RESULTS / "baseline.json").exists():
        res = ev.evaluate(BASELINE, "baseline")
        (RESULTS / "baseline.json").write_text(json.dumps(dict(settings=BASELINE, **res), indent=1))
        print(f"[w{args.worker}] BASELINE corpus WER {res['wer']:.4f}  WDER {res['wder']:.4f}", flush=True)

    STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(STORAGE_FILE)))
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=storage, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED + args.worker, n_startup_trials=12, multivariate=True,
                                           group=True, constant_liar=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=10))
    stop = optuna.study.MaxTrialsCallback(args.trials, states=(optuna.trial.TrialState.COMPLETE,
                                                               optuna.trial.TrialState.PRUNED))

    fails = [0]

    def log_trial(study, t):
        best = study.best_value if any(x.state.name == "COMPLETE" for x in study.trials) else None
        print(f"[w{args.worker}] trial {t.number} {t.state.name} value={t.value} best={best}", flush=True)
        fails[0] = fails[0] + 1 if t.state == optuna.trial.TrialState.FAIL else 0
        if fails[0] >= 3:   # something systematic is broken - stop rather than burn through trials
            print(f"[w{args.worker}] 3 failed trials in a row - stopping this worker", flush=True)
            study.stop()

    study.optimize(ev.objective, callbacks=[stop, log_trial], catch=(Exception,), gc_after_trial=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
