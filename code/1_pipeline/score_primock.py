"""
score_primock.py - score PriMock57 runs with the same metrics, and the same code, used for Fareez.

The only difference between the two corpora is where the reference comes from: Fareez supplies speaker-prefixed
text files, PriMock57 supplies timed turns parsed from its TextGrids by primock_prep.py. Everything downstream -
normalisation, alignment, WER, WDER, cpWER, SA-WER, per-role rates, cue errors, hallucination runs - is the
identical code path, so the cross-corpus comparison is like for like.

PriMock57 references carry timestamps, so unlike Fareez this corpus could also support time-aligned diarization
error rate. That is not computed here; it needs the diarization scored against reference speech regions rather
than against words, and is reported separately if used.

USAGE:  ~/eval_venv/bin/python score_primock.py --config defaults
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

# defined here rather than imported from run_primock, so scoring needs only the evaluation environment
# the external corpora share one runner and one scorer, so they are scored by identical code
CORPORA = {"primock": (ROOT / "cache" / "primock", ROOT / "final_primock", "PriMock57"),
           "afrispeech": (ROOT / "cache" / "afrispeech", ROOT / "final_afrispeech", "AfriSpeech-Dialog")}
CACHE, OUT, _ = CORPORA["primock"]


def reference_turns(rec, cache=None):
    """[(speaker, text)] in spoken order. PriMock57 turns carry start times and are sorted by them;
    AfriSpeech-Dialog turns are already stored in spoken order."""
    data = json.loads(((cache or CACHE) / "ref" / f"{rec}.json").read_text(encoding="utf-8"))
    turns = data["turns"] if isinstance(data, dict) else data
    if all("start" in t for t in turns):
        turns = sorted(turns, key=lambda t: float(t["start"]))
    return [(t["speaker"], t["text"]) for t in turns if t.get("text", "").strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", default="primock", choices=sorted(CORPORA))
    args = ap.parse_args()
    cache, OUT, name = CORPORA[args.corpus]

    files = sorted(p for p in (OUT / args.config).glob("*.json") if not p.name.endswith(".failed.json"))
    if not files:
        sys.exit(f"no {name} outputs for '{args.config}' - run run_primock.py --corpus {args.corpus} first")

    rows = [score(f, turns=reference_turns(f.stem, cache)) for f in files]
    df = pd.DataFrame(rows)
    dest = OUT / "scores"
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / f"{args.config}.csv", index=False)

    s = df.sum(numeric_only=True)
    print(f"{args.config}: {len(df)} consultations, {int(s['ref_words']):,} reference words")
    print(f"  WER    {100 * (s['sub'] + s['dele'] + s['ins']) / s['ref_words']:6.2f}%   "
          f"(sub {100 * s['sub'] / s['ref_words']:.2f}  del {100 * s['dele'] / s['ref_words']:.2f}  "
          f"ins {100 * s['ins'] / s['ref_words']:.2f})")
    print(f"  WDER   {100 * s['wder_wrong'] / s['wder_aligned']:6.2f}%")
    print(f"  cpWER  {100 * s['cp_err'] / s['cp_n']:6.2f}%")
    print(f"  SA-WER {100 * s['sa_err'] / s['sa_n']:6.2f}%")
    print(f"  negation, insertion-inclusive {100 * (s['neg_err'] + s['neg_ins']) / s['neg_ref']:6.2f}%   "
          f"reference-anchored {100 * s['neg_err'] / s['neg_ref']:.2f}%")
    print(f"  hallucinated words / 1k       {1000 * s['halluc_words'] / s['ref_words']:6.2f}")
    print(f"\nsaved -> {dest / f'{args.config}.csv'}")


if __name__ == "__main__":
    main()
