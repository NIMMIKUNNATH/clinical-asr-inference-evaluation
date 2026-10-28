"""
joint_reevaluate.py - score one joint-search trial on the FULL development sets (JOINT_SELECTION_RULE.md step 2).

Full development sets: Fareez 40, PriMock57 20, AfriSpeech-Dialog 29 general. No held-out recording is used.
Writes results_joint/reeval/trial_NNN.json with per-corpus pooled WER counts and their mean.

USAGE (osce venv):  python joint_reevaluate.py --trial 17
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_joint as sj  # noqa: E402

OUT = sj.ROOT / "results_joint" / "reeval"


def full_dev():
    return {"fareez": list(pd.read_csv(sj.ROOT / "tuning_set.csv").recording),
            "primock": json.loads((sj.ROOT / "results_primock" / "dev_split.json").read_text())["dev"],
            "afrispeech": sorted(pd.read_csv(sj.CACHE["afrispeech"] / "ids.csv").query("domain == 'general'").id)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, required=True)
    args = ap.parse_args()
    out = OUT / f"trial_{args.trial:03d}.json"
    if out.exists():
        print(f"{out} exists")
        return
    study = optuna.load_study(study_name=sj.STUDY, storage=optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(sj.STORAGE))))
    kw = json.loads(study.trials[args.trial].user_attrs["kwargs"])

    from faster_whisper import WhisperModel
    model = WhisperModel(sj.search.MODEL_NAME, device="cuda", compute_type=sj.search.COMPUTE_TYPE, cpu_threads=4)
    tot, per = {}, []
    for corpus, recs in full_dev().items():
        t = dict(sub=0, dele=0, ins=0, n=0)
        for rec in recs:
            audio = np.load(sj.CACHE[corpus] / "audio" / f"{rec}.npy").astype(np.float32) / 32768.0
            t0 = time.time()
            segments, _ = model.transcribe(audio, language="en", task="transcribe", word_timestamps=True, **kw)
            hyp = [w for s in segments for w in sj.normalize(s.text)]
            c = sj.wer_counts(sj.ref_tokens(corpus, rec), hyp)
            for k in t:
                t[k] += c[k]
            per.append(dict(corpus=corpus, recording=rec, seconds=round(time.time() - t0, 1), **c))
            print(f"trial {args.trial} {corpus} {rec} {(c['sub'] + c['dele'] + c['ins']) / c['n']:.4f}", flush=True)
        tot[corpus] = t
    wer = {c: (t["sub"] + t["dele"] + t["ins"]) / t["n"] for c, t in tot.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(trial=args.trial, wer=wer, mean_wer=sum(wer.values()) / len(wer), counts=tot,
                                   n={c: len(r) for c, r in full_dev().items()}, per_recording=per), indent=1))
    print(f"wrote {out}: mean {sum(wer.values()) / len(wer):.4f} {wer}")


if __name__ == "__main__":
    main()
