"""
run_primock.py - transcribe the prepared PriMock57 consultations with one fixed configuration.

PriMock57 is the second corpus. Its reference turns carry timestamps, so it also supports time-aligned
diarization scoring, which Fareez cannot. Audio was mixed to one mono 16 kHz channel by primock_prep.py, so the
pipeline hears what a single-microphone consultation would give it.

Configurations resolve exactly as in final_run.py, so `defaults` and `tuned` mean the same settings on both
corpora and the transfer comparison is like for like.

Resumable: recordings that already have an output file are skipped.

USAGE (WSL, osce venv, HF_TOKEN set):
    python run_primock.py --config defaults [--shard 0 --nshards 3]
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_run import config as resolve_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, SR  # noqa: E402

# the external corpora share this runner, so every configuration runs by identical code on each
CORPORA = {"primock": (ROOT / "cache" / "primock", ROOT / "final_primock", "primock57"),
           "afrispeech": (ROOT / "cache" / "afrispeech", ROOT / "final_afrispeech", "afrispeech-dialog")}
CACHE, OUT, CORPUS = CORPORA["primock"]

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
        dj.parent.mkdir(parents=True, exist_ok=True)
        tmp = dj.with_suffix(".tmp")
        tmp.write_text(json.dumps(diar, indent=1))
        os.replace(tmp, dj)
    return json.loads(dj.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--corpus", default="primock", choices=sorted(CORPORA))
    args = ap.parse_args()
    global CACHE, OUT, CORPUS
    CACHE, OUT, CORPUS = CORPORA[args.corpus]
    if "HF_TOKEN" not in os.environ:
        sys.exit("HF_TOKEN is not set")

    model_name, compute, kw, meta = resolve_config(args.config)
    out_dir = OUT / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    (CACHE / "diarization").mkdir(parents=True, exist_ok=True)

    recs = sorted(p.stem for p in (CACHE / "audio").glob("*.npy"))[args.shard::args.nshards]
    # a recording already transcribed, or already recorded as a failure, is not attempted again
    todo = [r for r in recs if not (out_dir / f"{r}.json").exists() and not (out_dir / f"{r}.failed.json").exists()]
    print(f"[{args.corpus} {args.config} {args.shard}/{args.nshards}] {len(recs)} consultations, {len(todo)} to do",
          flush=True)
    if not todo:
        return

    import ctranslate2
    from faster_whisper import WhisperModel
    from whisperx.diarize import assign_word_speakers
    model = WhisperModel(model_name, device="cuda", compute_type=compute, cpu_threads=4)

    for n, rec in enumerate(todo, 1):
        audio = np.load(CACHE / "audio" / f"{rec}.npy")
        diar = diarization_for(rec, audio)
        ctranslate2.set_random_seed(args.seed)
        t0 = time.time()
        # runaway guard: some settings fabricate text without end (one Fareez recording ran >13 h, and two
        # AfriSpeech recordings ran ~9.7 h each). A decode that exceeds five times the recording's duration (at
        # least 30 min) is recorded as a failure. Two layers, because the between-segment check below cannot fire
        # while the model is stuck inside a single enormous segment: a watchdog thread records the failure and
        # ends the process, and the runner is restarted by the pool and carries on with the next recording.
        limit = max(1800.0, 5 * len(audio) / SR)

        def watchdog():
            (out_dir / f"{rec}.failed.json").write_text(json.dumps(
                dict(recording=rec, config=args.config, error="Runaway",
                     message=f"decode exceeded {limit:.0f} s inside one segment; process ended by watchdog"),
                indent=1))
            print(f"[{args.corpus} {args.config}] {rec} FAILED runaway (watchdog at {limit:.0f}s)", flush=True)
            sys.stdout.flush()
            os._exit(3)

        timer = threading.Timer(limit * 1.2, watchdog)
        timer.daemon = True
        timer.start()
        gen, _ = model.transcribe(audio.astype(np.float32) / 32768.0, language="en", task="transcribe",
                                  word_timestamps=True, **kw)
        segments, runaway = [], False
        for s in gen:
            segments.append(s)
            if time.time() - t0 > limit:
                runaway = True
                break
        timer.cancel()          # the decode returned: the watchdog must not fire during the next recording
        secs = time.time() - t0
        if runaway:
            (out_dir / f"{rec}.failed.json").write_text(json.dumps(
                dict(recording=rec, config=args.config, error="Runaway",
                     message=f"decode exceeded {limit:.0f} s ({len(segments)} segments so far)"), indent=1))
            print(f"[{args.corpus} {args.config} {args.shard}/{args.nshards}] {n}/{len(todo)} {rec} FAILED runaway "
                  f"after {secs:.0f}s", flush=True)
            continue
        tr = {"segments": [{"start": s.start, "end": s.end, "text": s.text,
                            "words": [{"word": w.word, "start": w.start, "end": w.end,
                                       "probability": w.probability} for w in (s.words or [])]}
                           for s in segments]}
        tr = assign_word_speakers(pd.DataFrame(diar), tr)
        result = dict(recording=rec, config=args.config, corpus=CORPUS, model=model_name,
                      compute_type=compute, settings=kw, selection=meta, seed=args.seed,
                      duration_s=len(audio) / SR, decode_seconds=secs,
                      segments=tr["segments"], diarization=diar)
        tmp = out_dir / f"{rec}.json.tmp"
        tmp.write_text(json.dumps(result), encoding="utf-8")
        os.replace(tmp, out_dir / f"{rec}.json")
        print(f"[{args.corpus} {args.config} {args.shard}/{args.nshards}] {n}/{len(todo)} {rec}  "
              f"{len(audio) / SR / 60:.1f} min  decode {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
