"""
final_run.py - transcribe a set of recordings with one fixed configuration and save the RAW output
(segments, words with timestamps, WhisperX speakers, timings). Scoring is done separately by metrics.py, so
metrics can be added or changed later without re-running the GPU.

Configurations
  tuned        large-v3 float16, settings from results/final_choice.json (written when the winner is picked)
  defaults     large-v3 float16, faster-whisper defaults with VAD on (the search baseline)
  oldref_v2    large-v2 int8_float16, the reference configuration the earlier pipeline used
  trial32_v2   large-v2 int8_float16, the earlier TPE winner (trial 32)

Sets
  heldout      the 231 recordings not used for tuning (duplicate RES0202 excluded)
  dev          the 40 tuning recordings

Audio decoding and pyannote diarization (community-1, 2 speakers) are cached per recording in cache/ and shared
by every configuration. Resumable: recordings that already have an output file are skipped.

USAGE (WSL, osce venv, HF_TOKEN set):
    python final_run.py --config tuned --set heldout [--shard 0 --nshards 3]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import AUDIO_DIR, CACHE, DUPLICATES, ROOT, SR, load_tuning_set  # noqa: E402
import search  # noqa: E402  (BASELINE settings)

OUT = ROOT / "final"

REFERENCE_KW = dict(beam_size=5, patience=1.0, length_penalty=1.0, best_of=5, repetition_penalty=1.0,
                    no_repeat_ngram_size=0, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], no_speech_threshold=0.60,
                    log_prob_threshold=-1.0, compression_ratio_threshold=3.0, condition_on_previous_text=True,
                    initial_prompt=None, vad_filter=True,
                    vad_parameters=dict(threshold=0.30, min_speech_duration_ms=250, min_silence_duration_ms=2000,
                                        speech_pad_ms=400))
TRIAL32_KW = dict(beam_size=5, patience=2.0, length_penalty=1.2, best_of=8, repetition_penalty=1.0,
                  no_repeat_ngram_size=0, temperature=[0.0, 0.2, 0.4], no_speech_threshold=0.75,
                  log_prob_threshold=-0.5, compression_ratio_threshold=2.4, condition_on_previous_text=True,
                  initial_prompt=None, vad_filter=True,
                  vad_parameters=dict(threshold=0.30, min_speech_duration_ms=100, min_silence_duration_ms=1000,
                                      speech_pad_ms=400))


def config(name):
    if name.startswith("trial_"):          # a search trial, e.g. trial_002 - candidate re-scoring on dev
        import optuna
        from common import STORAGE_FILE, STUDY_NAME
        study = optuna.load_study(study_name=STUDY_NAME, storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(STORAGE_FILE))))
        t = study.trials[int(name.split("_")[1])]
        return "large-v3", "float16", json.loads(t.user_attrs["kwargs"]), {"trial": t.number, "dev_wer": t.value}
    if name == "tuned":
        choice = json.loads((ROOT / "results" / "final_choice.json").read_text())
        return "large-v3", "float16", choice["transcribe_kwargs"], choice
    if name == "primock_tuned":     # PriMock57's own search winner, used for the reverse transfer onto Fareez
        choice = json.loads((ROOT / "results_primock" / "primock_choice.json").read_text())
        return "large-v3", "float16", choice["transcribe_kwargs"], choice
    if name in ("joint", "local20_fareez", "local20_afrispeech"):   # the cross-corpus configurations
        path = {"joint": ROOT / "results_joint" / "joint_choice.json",
                "local20_fareez": ROOT / "results_fareez_local20" / "choice.json",
                "local20_afrispeech": ROOT / "results_afrispeech_local20" / "choice.json"}[name]
        choice = json.loads(path.read_text())
        return "large-v3", "float16", choice["transcribe_kwargs"], choice
    if name == "joint_beam1_pad400":
        return joint_variant(1, 400)
    if name in ONE_FACTOR:
        return one_factor(name)
    if name == "defaults":
        return "large-v3", "float16", search.BASELINE, {}
    if name == "v3_oldref":     # oldref_v2's decoding settings on large-v3: isolates the checkpoint from
        # previous-text conditioning, which varied together in every earlier comparison
        return "large-v3", "float16", REFERENCE_KW, {"note": "oldref settings, large-v3"}
    if name == "v3_trial32":    # likewise for the trial-32 vector
        return "large-v3", "float16", TRIAL32_KW, {"note": "trial32 settings, large-v3"}
    if name == "oldref_v2":
        return "large-v2", "int8_float16", REFERENCE_KW, {}
    if name == "trial32_v2":
        return "large-v2", "int8_float16", TRIAL32_KW, {}
    # large-v2 in float16, so the checkpoint comparison holds numerical precision constant as well as
    # decoding parameters; the int8_float16 runs above are kept for the precision sensitivity analysis
    if name == "oldref_v2_fp16":
        return "large-v2", "float16", REFERENCE_KW, {"note": "oldref settings, large-v2, float16"}
    if name == "trial32_v2_fp16":
        return "large-v2", "float16", TRIAL32_KW, {"note": "trial32 settings, large-v2, float16"}
    sys.exit(f"unknown config {name}")


# ---------------------------------------------------------------- controlled one-factor family
# Every condition differs from the joint configuration in exactly one setting, so the difference
# it produces is an effect of that setting rather than a marginal of where the search sampled.
ONE_FACTOR = {
    "of_reference":        (None, None),
    "of_condprev_on":      ("condition_on_previous_text", True),
    "of_temp_full":        ("temperature", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    "of_vadsilence_1000":  ("vad.min_silence_duration_ms", 1000),
    "of_vadsilence_2000":  ("vad.min_silence_duration_ms", 2000),
    "of_vadthresh_020":    ("vad.threshold", 0.20),
    "of_vadthresh_050":    ("vad.threshold", 0.50),
    "of_reppen_110":       ("repetition_penalty", 1.10),
    "of_reppen_125":       ("repetition_penalty", 1.25),
    "of_logprob_100":      ("log_prob_threshold", -1.0),
    "of_logprob_030":      ("log_prob_threshold", -0.3),
    "of_nospeech_030":     ("no_speech_threshold", 0.30),
    "of_nospeech_070":     ("no_speech_threshold", 0.70),
    "of_beam_1":           ("beam_size", 1),
    "of_beam_8":           ("beam_size", 8),
    "of_patience_100":     ("patience", 1.0),
    "of_lenpen_080":       ("length_penalty", 0.80),
    "of_norepeat_3":       ("no_repeat_ngram_size", 3),
    "of_bestof_5":         ("best_of", 5),
    "of_compression_300":  ("compression_ratio_threshold", 3.0),
    "of_prompt_clinical":  ("initial_prompt", "A medical history-taking consultation between a doctor and a patient."),
    "of_hallsil_off":      ("hallucination_silence_threshold", None),
    "of_vadminspeech_0":   ("vad.min_speech_duration_ms", 0),
    "of_vadpad_400":       ("vad.speech_pad_ms", 400),
    "of_vadfilter_off":    ("vad_filter", False),
}


def one_factor(name):
    """The joint vector with exactly one setting replaced."""
    import copy
    choice = json.loads((ROOT / "results_joint" / "joint_choice.json").read_text())
    kw = copy.deepcopy(choice["transcribe_kwargs"])
    key, value = ONE_FACTOR[name]
    if key is not None:
        if key.startswith("vad."):
            kw.setdefault("vad_parameters", {})[key.split(".", 1)[1]] = value
        else:
            kw[key] = value
    return "large-v3", "float16", kw, {"family": "one-factor", "changed": key, "to": value}


def joint_variant(beam, pad):
    """The joint configuration with the two settings the one-factor family found to improve both
    word error and speaker attribution. Post hoc: the selection rule was frozen long before this."""
    import copy
    choice = json.loads((ROOT / "results_joint" / "joint_choice.json").read_text())
    kw = copy.deepcopy(choice["transcribe_kwargs"])
    kw["beam_size"] = beam
    kw.setdefault("vad_parameters", {})["speech_pad_ms"] = pad
    return "large-v3", "float16", kw, {"family": "post hoc candidate", "beam": beam, "speech_pad_ms": pad}


def recordings(which):
    dev = [r["recording"] for r in load_tuning_set()]
    if which == "dev":
        return dev
    every = sorted(p.stem for p in AUDIO_DIR.glob("*.mp3") if p.stem not in DUPLICATES)
    return [r for r in every if r not in set(dev)]


def audio_for(rec):
    npy = CACHE / "audio" / f"{rec}.npy"
    if not npy.exists():
        import subprocess
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(AUDIO_DIR / f"{rec}.mp3"), "-ac", "1", "-ar",
                              str(SR), "-f", "s16le", "-"], capture_output=True, check=True).stdout
        tmp = npy.with_suffix(".tmp.npy")
        np.save(tmp, np.frombuffer(raw, dtype=np.int16))
        os.replace(tmp, npy)
    return np.load(npy)


_pipeline = None


def diarization_for(rec, audio):
    dj = CACHE / "diarization" / f"{rec}.json"
    if not dj.exists():
        global _pipeline
        import torch
        from pyannote.audio import Pipeline
        if _pipeline is None:
            _pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                                 token=os.environ["HF_TOKEN"])
            _pipeline.to(torch.device("cuda"))
        wav = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)
        out = _pipeline({"waveform": wav, "sample_rate": SR}, num_speakers=2)
        diar = [{"start": round(float(t.start), 3), "end": round(float(t.end), 3), "speaker": spk}
                for t, _, spk in out.exclusive_speaker_diarization.itertracks(yield_label=True)]
        tmp = dj.with_suffix(".tmp")
        tmp.write_text(json.dumps(diar, indent=1))
        os.replace(tmp, dj)
    return json.loads(dj.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", default="heldout", choices=["heldout", "dev"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260915,
                    help="CTranslate2 random seed (temperature-fallback sampling); non-default seeds get their own "
                         "output folder <config>_s<seed>")
    ap.add_argument("--require-cache", action="store_true",
                    help="skip recordings whose diarization is not cached, so this process only ever loads Whisper. "
                         "Lets a transcription shard run alongside cache_diarization.py without holding two large "
                         "models at once. Re-run without the flag afterwards to pick up the skipped recordings.")
    args = ap.parse_args()
    if "HF_TOKEN" not in os.environ:
        sys.exit("HF_TOKEN is not set")

    model_name, compute, kw, meta = config(args.config)
    label = args.config if args.seed == 20260915 else f"{args.config}_s{args.seed}"
    out_dir = OUT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    (CACHE / "audio").mkdir(parents=True, exist_ok=True)
    (CACHE / "diarization").mkdir(parents=True, exist_ok=True)
    recs = recordings(args.set)[args.shard::args.nshards]
    todo = [r for r in recs if not (out_dir / f"{r}.json").exists()]
    if args.require_cache:
        cached = [r for r in todo if (CACHE / "diarization" / f"{r}.json").exists()]
        print(f"[{args.config} {args.shard}/{args.nshards}] --require-cache: {len(cached)} of {len(todo)} "
              f"have cached diarization", flush=True)
        todo = cached
    print(f"[{args.config} {args.shard}/{args.nshards}] {len(recs)} recordings, {len(todo)} to do", flush=True)
    if not todo:
        return

    import ctranslate2
    from faster_whisper import WhisperModel
    from whisperx.diarize import assign_word_speakers
    model = WhisperModel(model_name, device="cuda", compute_type=compute, cpu_threads=4)

    for n, rec in enumerate(todo, 1):
        audio = audio_for(rec)
        diar = diarization_for(rec, audio)
        ctranslate2.set_random_seed(args.seed)   # per recording, so results do not depend on processing order
        t0 = time.time()
        try:
            segments, _ = model.transcribe(audio.astype(np.float32) / 32768.0, language="en", task="transcribe",
                                           word_timestamps=True, **kw)
            segments = list(segments)
        except Exception as e:  # noqa: BLE001
            # faster-whisper 1.2.1 raises IndexError in word-timestamp alignment when a decoded segment
            # contains no tokens. It is deterministic for a given configuration and recording. Record the
            # failure and continue: ANALYSIS_PLAN.md treats such a recording as a transcription failure and
            # does not replace it, so the shard must not die with it.
            (out_dir / f"{rec}.failed.json").write_text(json.dumps(
                dict(recording=rec, config=label, error=type(e).__name__, message=str(e)[:500]), indent=1))
            print(f"[{args.config} {args.shard}/{args.nshards}] {n}/{len(todo)} {rec} FAILED "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        secs = time.time() - t0
        tr = {"segments": [{"start": s.start, "end": s.end, "text": s.text,
                            "words": [{"word": w.word, "start": w.start, "end": w.end,
                                       "probability": w.probability} for w in (s.words or [])]}
                           for s in segments]}
        tr = assign_word_speakers(pd.DataFrame(diar), tr)
        result = dict(recording=rec, config=label, model=model_name, compute_type=compute, settings=kw,
                      selection=meta, seed=args.seed, duration_s=len(audio) / SR, decode_seconds=secs,
                      segments=tr["segments"])
        tmp = out_dir / f"{rec}.tmp"
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, out_dir / f"{rec}.json")
        print(f"[{label} {args.shard}/{args.nshards}] {n}/{len(todo)} {rec}  {len(audio) / SR / 60:.1f} min  "
              f"decode {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
