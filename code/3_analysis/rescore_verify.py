"""
rescore_verify.py - re-score every saved transcript behind a reported result with the final scorer, and check
that the per-recording scores the paper was built from are exactly what the final scorer produces.

No transcription is repeated: the transcripts (and their speaker labels) are the saved run outputs. What is
re-executed is everything downstream of them - normalisation, alignment, WDER, cpWER, SA-WER, per-role error,
cue counting and hallucination detection - using the current metrics.py.

For each configuration it writes <corpus>/scores_verify/<config>.csv and compares it, column by column and
recording by recording, with <corpus>/scores/<config>.csv. Any difference is reported. With --replace, the
verified files are copied over the originals (only needed if a difference was found).

USAGE:  ~/eval_venv/bin/python rescore_verify.py [--replace]
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import score  # noqa: E402
from score_primock import reference_turns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

FAREEZ, PRIMOCK = ROOT / "final", ROOT / "final_primock"
HELDOUT_CONFIGS = ["defaults", "tuned", "oldref_v2", "trial32_v2", "v3_oldref", "v3_trial32", "primock_tuned"]
PRIMOCK_CONFIGS = ["defaults", "tuned", "oldref_v2", "trial32_v2", "primock_tuned"]


def outputs(folder):
    return sorted(p for p in folder.glob("*.json") if not p.name.endswith(".failed.json"))


def compare(new, old_path):
    if not old_path.exists():
        return "no previous file"
    old = pd.read_csv(old_path)
    if set(new.recording) != set(old.recording):
        return f"recording sets differ ({len(new)} vs {len(old)})"
    a = new.set_index("recording").sort_index()
    b = old.set_index("recording").sort_index()
    diffs = []
    for c in sorted(set(a.columns) & set(b.columns)):
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if c in ("decode_seconds", "rtf"):          # timing is measured, not recomputed
                continue
            d = np.nanmax(np.abs(x.astype(float) - y.astype(float))) if len(x) else 0.0
            if d > 1e-9:
                diffs.append(f"{c} (max |diff| {d:.4g})")
        elif not x.astype(str).equals(y.astype(str)):
            diffs.append(f"{c} (text)")
    extra = sorted(set(a.columns) ^ set(b.columns))
    msg = "identical" if not diffs else "DIFFERS: " + ", ".join(diffs)
    return msg + (f"; columns only in one file: {extra}" if extra else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args()

    heldout = set(pd.read_csv(FAREEZ / "scores" / "tuned.csv").recording)
    dev = set(pd.read_csv(ROOT / "tuning_set.csv").recording)
    assert len(heldout) == 231 and len(dev) == 40 and not heldout & dev, "split check failed"
    print(f"split check: 231 held-out, 40 development, no overlap")

    jobs = [(FAREEZ, c, lambda f: f.stem in heldout, None) for c in HELDOUT_CONFIGS]
    jobs += [(FAREEZ, "defaults_dev", lambda f: f.stem in dev, "defaults")]
    jobs += [(PRIMOCK, c, lambda f: True, None) for c in PRIMOCK_CONFIGS]

    report = []
    for corpus, name, keep, src in jobs:
        files = [f for f in outputs(corpus / (src or name)) if keep(f)]
        turns = (lambda f: reference_turns(f.stem)) if corpus == PRIMOCK else (lambda f: None)
        df = pd.DataFrame([score(f, turns=turns(f)) for f in files])
        dest = corpus / "scores_verify"
        dest.mkdir(exist_ok=True)
        df.to_csv(dest / f"{name}.csv", index=False)
        res = compare(df, corpus / "scores" / f"{name}.csv")
        tag = "Fareez" if corpus == FAREEZ else "PriMock57"
        line = f"{tag:9s} {name:14s} n={len(df):3d}  {res}"
        print(line, flush=True)
        report.append(line)
        if args.replace and not res.startswith("identical"):
            shutil.copy(dest / f"{name}.csv", corpus / "scores" / f"{name}.csv")
            print(f"          replaced scores/{name}.csv")

    (ROOT / "paper" / "rescore_verification.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
