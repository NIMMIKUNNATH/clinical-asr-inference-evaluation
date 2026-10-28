"""
prepare.py - pick the 40 tuning recordings, check them, and cache what does not depend on Whisper settings.

  1. Stratified random pick of 40 recordings (fixed seed), duplicate RES0202 excluded.
  2. Decode each mp3 once to 16 kHz mono int16 -> cache/audio/<rec>.npy
  3. Parse + normalise each reference and check it -> cache/ref_clean/<rec>.txt (for eyeballing)
  4. Diarize once with pyannote community-1 (2 speakers, exclusive) -> cache/diarization/<rec>.json
     Diarization does not depend on any Whisper setting, so every trial reuses it.
  5. tuning_set.csv + prepare_report.txt

USAGE (WSL, venv active, HF_TOKEN set):  python prepare.py
"""
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter

import numpy as np

from common import (AUDIO_DIR, CACHE, DUPLICATES, REF_DIR, ROOT, SR, TUNING_SET, normalize, parse_reference,
                    reference_tokens)

SEED = 20260915
QUOTA = {"RES": 29, "MSK": 7, "GAS": 1, "CAR": 1, "DER": 1, "GEN": 1}  # 40, every specialty present


def select():
    stems = sorted(p.stem for p in AUDIO_DIR.glob("*.mp3") if p.stem not in DUPLICATES)
    by_cat = {}
    for s in stems:
        by_cat.setdefault(re.match(r"[A-Z]+", s).group(), []).append(s)
    rng = random.Random(SEED)
    picked = []
    for cat, k in QUOTA.items():
        picked += rng.sample(by_cat[cat], k)
    rng.shuffle(picked)  # this is also the evaluation order inside each trial (pruning steps)
    return picked, {c: len(v) for c, v in by_cat.items()}


def decode(stem):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(AUDIO_DIR / f"{stem}.mp3"), "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def main():
    if "HF_TOKEN" not in os.environ:
        sys.exit("HF_TOKEN is not set")
    for d in ("audio", "diarization", "ref_clean"):
        (CACHE / d).mkdir(parents=True, exist_ok=True)

    picked, available = select()
    report = [f"Tuning set: {len(picked)} recordings, seed {SEED}, stratified quotas {QUOTA}",
              f"Available after removing duplicates {DUPLICATES}: {available}", ""]
    problems = []

    import torch
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=os.environ["HF_TOKEN"])
    pipeline.to(torch.device("cuda"))

    rows = []
    for order, stem in enumerate(picked, 1):
        t0 = time.time()
        # ---- audio
        npy = CACHE / "audio" / f"{stem}.npy"
        audio = np.load(npy) if npy.exists() else decode(stem)
        if not npy.exists():
            np.save(npy, audio)
        dur = len(audio) / SR
        peak = int(np.abs(audio).max())
        if dur < 60:
            problems.append(f"{stem}: audio only {dur:.1f} s")
        if peak < 1000:
            problems.append(f"{stem}: audio nearly silent (peak {peak})")

        # ---- reference
        ref_path = REF_DIR / f"{stem}.txt"
        if not ref_path.exists():
            problems.append(f"{stem}: no reference transcript")
            continue
        turns = parse_reference(ref_path)
        whole, tagged = reference_tokens(ref_path)
        roles = Counter(r for r, _ in turns)
        if set(roles) != {"Doctor", "Patient"}:
            problems.append(f"{stem}: reference roles {dict(roles)}")
        stray = [w for i, w in enumerate(whole) if w in ("d", "p") and not (i and whole[i - 1] == "vitamin")]
        if stray:
            problems.append(f"{stem}: {len(stray)} stray 'd'/'p' tokens in normalised reference - prefix leak?")
        with open(CACHE / "ref_clean" / f"{stem}.txt", "w", encoding="utf-8") as f:
            for role, text in turns:
                f.write(f"{role}: {' '.join(normalize(text))}\n")

        # ---- diarization (once)
        dj = CACHE / "diarization" / f"{stem}.json"
        if dj.exists():
            diar = json.loads(dj.read_text())
        else:
            wav = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)
            out = pipeline({"waveform": wav, "sample_rate": SR}, num_speakers=2)
            diar = [{"start": round(float(t.start), 3), "end": round(float(t.end), 3), "speaker": spk}
                    for t, _, spk in out.exclusive_speaker_diarization.itertracks(yield_label=True)]
            dj.write_text(json.dumps(diar, indent=1))
        speakers = sorted({d["speaker"] for d in diar})
        speech = sum(d["end"] - d["start"] for d in diar)
        if len(speakers) != 2:
            problems.append(f"{stem}: diarization found {len(speakers)} speakers")

        rows.append(dict(order=order, recording=stem, category=re.match(r"[A-Z]+", stem).group(),
                         duration_s=round(dur, 1), ref_words=len(whole), ref_turns=len(turns),
                         doctor_turns=roles["Doctor"], patient_turns=roles["Patient"],
                         diar_turns=len(diar), diar_speech_share=round(speech / dur, 3), peak=peak))
        print(f"[{order:2d}/40] {stem}  {dur / 60:5.1f} min  ref {len(whole):5d} words / {len(turns):3d} turns  "
              f"diar {len(diar):4d} turns, {speech / dur:.0%} speech  ({time.time() - t0:.0f}s)", flush=True)

    with open(TUNING_SET, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    total_min = sum(r["duration_s"] for r in rows) / 60
    report += ["", f"Recordings cached: {len(rows)}   audio {total_min:.0f} min ({total_min / 60:.1f} h)",
               f"Reference words: {sum(r['ref_words'] for r in rows)}",
               f"By category: {dict(Counter(r['category'] for r in rows))}", "",
               "PROBLEMS:" if problems else "No problems found."] + [f"  {p}" for p in problems]
    (ROOT / "prepare_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
