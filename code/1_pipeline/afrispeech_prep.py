"""
afrispeech_prep.py - prepare AfriSpeech-Dialog (Intron Health, CC BY-NC-SA 4.0) as the third corpus.

AfriSpeech-Dialog has 49 two-speaker English conversations with African accents: 20 simulated doctor-patient
consultations and 29 general conversations. Nothing is tuned on it. It is a held-out transfer test: every
configuration already fixed on Fareez and PriMock57 is run on it once, unchanged.

  * audio (one mixed recording per conversation) -> mono 16 kHz int16 -> cache/afrispeech/audio/<id>.npy
  * transcript -> ordered reference turns [{speaker: Doctor|Patient, text}] -> cache/afrispeech/ref/<id>.json

The transcripts label speakers "Speaker 1"/"Speaker 2" in three layouts (with or without brackets, with or
without MM:SS:cc timestamp lines). Timestamps are inconsistent across files and are not needed: every metric in
this study is computed from ordered speaker turns, as for Fareez. In the medical conversations the clinician is
the speaker who asks more questions (checked by hand on every file, see the printed table); in the general
conversations the roles are arbitrary, so role-specific outcomes are reported for the medical subset only.

Recording identifiers are MED01..MED20 and GEN01..GEN29 in metadata order; ids.csv maps them to source files.

USAGE (WSL):  python afrispeech_prep.py [--src /mnt/e/corpora/afrispeech_dialog]
"""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np

SR = 16000
OUT = Path("/mnt/d/fareez_optuna/cache/afrispeech")

LABEL = re.compile(r"^\s*\[?\s*speaker\s*([12])\s*\]?\s*:\s*(.*)$", re.I)
STAMP = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{1,3})?\s*$")


def parse(transcript):
    """-> [(speaker_number, text)], continuation lines folded into the current turn, timestamp lines dropped."""
    turns = []
    for line in transcript.splitlines():
        if not line.strip() or STAMP.match(line):
            continue
        m = LABEL.match(line)
        if m:
            turns.append([m.group(1), m.group(2).strip()])
        elif turns:
            turns[-1][1] = (turns[-1][1] + " " + line.strip()).strip()
    return [(s, t) for s, t in turns if t]


def load_audio(path):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/e/corpora/afrispeech_dialog")
    args = ap.parse_args()
    src = Path(args.src)
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    (OUT / "ref").mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(src / "metadata.csv", encoding="utf-8")))
    counters, ids = {"medical": 0, "general": 0}, []
    print(f"{'id':6s} {'min':>5s} {'turns':>5s} {'words':>6s}  clinician  q1/q2   accent")
    for r in rows:
        dom = r["domain"]
        counters[dom] += 1
        cid = f"{'MED' if dom == 'medical' else 'GEN'}{counters[dom]:02d}"
        turns = parse(r["transcript"])
        speakers = {s for s, _ in turns}
        if speakers != {"1", "2"}:
            print(f"  {cid}: speakers {sorted(speakers)} - skipped")
            continue
        q = {s: sum(t.count("?") for x, t in turns if x == s) for s in ("1", "2")}
        clinician = max(q, key=q.get) if dom == "medical" else "1"
        ref = [dict(speaker="Doctor" if s == clinician else "Patient", text=t) for s, t in turns]
        (OUT / "ref" / f"{cid}.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")

        npy = OUT / "audio" / f"{cid}.npy"
        if not npy.exists():
            np.save(npy, load_audio(src / r["file_name"]))
        audio = np.load(npy, mmap_mode="r")
        words = sum(len(t.split()) for _, t in turns)
        ids.append(dict(id=cid, domain=dom, file_name=r["file_name"], accent=r["accent"], country=r["country"],
                        duration_s=round(len(audio) / SR, 1), turns=len(turns), words=words,
                        clinician_speaker=clinician if dom == "medical" else ""))
        print(f"{cid:6s} {len(audio) / SR / 60:5.1f} {len(turns):5d} {words:6d}  "
              f"{'Speaker ' + clinician if dom == 'medical' else '-':9s}  {q['1']:>2d}/{q['2']:<3d} {r['accent']}",
              flush=True)

    with open(OUT / "ids.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ids[0]))
        w.writeheader()
        w.writerows(ids)
    for dom in ("medical", "general"):
        sel = [x for x in ids if x["domain"] == dom]
        print(f"{dom}: {len(sel)} conversations, {sum(x['duration_s'] for x in sel) / 3600:.2f} h, "
              f"{sum(x['words'] for x in sel):,} reference words")


if __name__ == "__main__":
    main()
