"""
primock_prep.py - prepare PriMock57 as an external test set.

For each of the 57 consultations:
  * mix the separate doctor and patient channels into one mono 16 kHz track (what the pipeline would hear in a
    single-microphone consultation) -> cache/primock/audio/<id>.npy
  * parse both TextGrids into timed reference turns -> cache/primock/ref/<id>.json
    [{start, end, speaker: Doctor|Patient, text}] with the annotation tags of the corpus kept in `text_raw`
    and stripped in `text` (the corpus's own convention: <UNSURE>, </UNSURE>, <UNIN/>, <INAUDIBLE_SPEECH/>)

Reference turns carry timestamps, so PriMock57 also supports DER and tcpWER, which the Fareez corpus cannot.

USAGE:  python primock_prep.py [--src /mnt/d/primock57]
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

SR = 16000
OUT = Path("/mnt/d/fareez_optuna/cache/primock")
TAGS = ["<UNSURE>", "</UNSURE>", "<UNIN/>", "<INAUDIBLE_SPEECH/>"]

_NUM = r"[-+]?\d*\.?\d+"


def parse_textgrid(path):
    """Minimal ooTextFile IntervalTier parser -> (tier_name, [(xmin, xmax, text)])."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    name = re.search(r'name\s*=\s*"([^"]*)"', txt).group(1)
    items = re.findall(rf'xmin\s*=\s*({_NUM})\s*\n\s*xmax\s*=\s*({_NUM})\s*\n\s*text\s*=\s*"((?:[^"]|"")*)"', txt)
    return name, [(float(a), float(b), t.replace('""', '"')) for a, b, t in items]


def strip_tags(text):
    for t in TAGS:
        text = text.replace(t, "")
    return re.sub(r"\s+", " ", text).strip()


def mix_audio(doctor_wav, patient_wav):
    """Mix the two channels to mono 16 kHz, as the corpus's own mix_audio.sh does with sox -m."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(doctor_wav), "-i", str(patient_wav), "-filter_complex",
           "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0[a]", "-map", "[a]", "-ac", "1", "-ar", str(SR),
           "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/d/primock57")
    args = ap.parse_args()
    src = Path(args.src)
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    (OUT / "ref").mkdir(parents=True, exist_ok=True)

    doctors = sorted(src.glob("audio/*_doctor.wav"))
    print(f"{len(doctors)} consultations")
    rows = []
    for d in doctors:
        cid = d.name.replace("_doctor.wav", "")
        p = d.with_name(f"{cid}_patient.wav")
        tg_d = src / "transcripts" / f"{cid}_doctor.TextGrid"
        tg_p = src / "transcripts" / f"{cid}_patient.TextGrid"
        if not (p.exists() and tg_d.exists() and tg_p.exists()):
            print(f"  {cid}: missing channel or transcript - skipped")
            continue

        npy = OUT / "audio" / f"{cid}.npy"
        if not npy.exists():
            np.save(npy, mix_audio(d, p))
        audio = np.load(npy, mmap_mode="r")

        turns = []
        for tg, role in ((tg_d, "Doctor"), (tg_p, "Patient")):
            tier, intervals = parse_textgrid(tg)
            for a, b, t in intervals:
                if strip_tags(t):
                    turns.append(dict(start=round(a, 3), end=round(b, 3), speaker=role,
                                      text=strip_tags(t), text_raw=t.strip(), tier=tier))
        turns.sort(key=lambda x: x["start"])
        (OUT / "ref" / f"{cid}.json").write_text(json.dumps(turns, ensure_ascii=False, indent=1), encoding="utf-8")

        words = sum(len(t["text"].split()) for t in turns)
        rows.append((cid, len(audio) / SR, len(turns), words,
                     sum(1 for t in turns if t["speaker"] == "Doctor")))
        print(f"  {cid}: {len(audio) / SR / 60:5.1f} min, {len(turns):3d} turns, {words:5d} words", flush=True)

    total_min = sum(r[1] for r in rows) / 60
    print(f"\n{len(rows)} consultations, {total_min:.0f} min ({total_min / 60:.1f} h), "
          f"{sum(r[3] for r in rows)} reference words")


if __name__ == "__main__":
    main()
