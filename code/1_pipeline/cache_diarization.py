"""
cache_diarization.py - decode audio and run pyannote once per recording, into cache/.

Every configuration (defaults, tuned, the one-factor family) shares the same diarization, so it is computed
once here rather than inside each transcription run. Running pyannote and Whisper in the same process, in
several processes at once, exhausted the GPU and produced `CUDA failed with error unknown error` on
2026-09-16 and again on 2026-09-18; separating the two stages avoids holding both models at once.

Single process by design. Resumable: recordings already cached are skipped.

USAGE (WSL, osce venv, HF_TOKEN set):
    python cache_diarization.py --set heldout
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_run import audio_for, diarization_for, recordings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CACHE, SR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="heldout", choices=["heldout", "dev"])
    args = ap.parse_args()
    if "HF_TOKEN" not in os.environ:
        sys.exit("HF_TOKEN is not set")

    (CACHE / "audio").mkdir(parents=True, exist_ok=True)
    (CACHE / "diarization").mkdir(parents=True, exist_ok=True)
    recs = recordings(args.set)
    todo = [r for r in recs if not (CACHE / "diarization" / f"{r}.json").exists()]
    print(f"[diar] {len(recs)} recordings, {len(todo)} to do", flush=True)

    t_start = time.time()
    for n, rec in enumerate(todo, 1):
        t0 = time.time()
        audio = audio_for(rec)
        turns = diarization_for(rec, audio)
        mins = len(audio) / SR / 60
        secs = time.time() - t0
        done = time.time() - t_start
        eta = done / n * (len(todo) - n) / 60
        print(f"[diar] {n}/{len(todo)} {rec}  {mins:.1f} min  {len(turns)} turns  {secs:.0f}s  "
              f"eta {eta:.0f} min", flush=True)


if __name__ == "__main__":
    main()
