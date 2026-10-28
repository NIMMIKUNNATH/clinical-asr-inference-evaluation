"""Does the checkpoint result survive when large-v2 runs in float16, the precision large-v3 used?

Two questions, kept separate:
  1. int8_float16 against float16 on large-v2 itself - how much does precision alone move?
  2. the checkpoint comparison redone with both checkpoints in float16 - does the ordering hold?
Every comparison is on the recordings both runs completed.
"""
import sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/fareez_optuna/code")
sys.path.insert(0, "/mnt/d/fareez_optuna/code/final")
import json
ROOT = pathlib.Path('/mnt/d/fareez_optuna')
CORPORA = [('Fareez', ROOT/'final'/'scores', None),
           ('PriMock57', ROOT/'final_primock'/'scores', 'primock'),
           ('AfriSpeech-Dialog', ROOT/'final_afrispeech'/'scores', 'clinical')]

def heldout_scores(name, base, kind=None):
    f = base / (name + '.csv')
    if not f.exists():
        return None
    d = pd.read_csv(f).sort_values('recording').reset_index(drop=True)
    if kind == 'primock':
        ho = set(json.loads((ROOT/'results_primock'/'dev_split.json').read_text())['heldout'])
        return d[d.recording.isin(ho)]
    if kind == 'clinical':
        ids = pd.read_csv(ROOT/'cache'/'afrispeech'/'ids.csv')
        return d[d.recording.isin(set(ids[ids.domain=='medical'].id))]
    return d

RNG = np.random.default_rng(20260915)


def rates(d):
    ref = d.ref_words.sum()
    return (100 * (d["sub"] + d.dele + d.ins).sum() / ref,
            100 * d.wder_wrong.sum() / d.wder_aligned.sum(),
            1000 * d.halluc_words.sum() / ref)


def paired(a, b, col):
    """mean paired difference b - a with a 10,000-sample percentile interval, matched recordings"""
    common = sorted(set(a.recording) & set(b.recording))
    a = a[a.recording.isin(common)].sort_values("recording").reset_index(drop=True)
    b = b[b.recording.isin(common)].sort_values("recording").reset_index(drop=True)
    if col == "wer":
        va = 100 * (a["sub"] + a.dele + a.ins) / a.ref_words
        vb = 100 * (b["sub"] + b.dele + b.ins) / b.ref_words
    else:
        va = 100 * a.wder_wrong / a.wder_aligned
        vb = 100 * b.wder_wrong / b.wder_aligned
    d = (vb - va).to_numpy()
    boot = [d[RNG.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    return len(common), d.mean(), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


print("=" * 104)
print("1. PRECISION ALONE on large-v2: int8_float16 -> float16, same decoding vector")
print("=" * 104)
print("{:<20} {:<10} {:>28} {:>28}".format("corpus", "vector", "int8_float16 (WER/WDER/hall)",
                                           "float16 (WER/WDER/hall)"))
for corpus, base, kind in CORPORA:
    for vec, old, new in (("reference", "oldref_v2", "oldref_v2_fp16"),
                          ("trial32", "trial32_v2", "trial32_v2_fp16")):
        a, b = heldout_scores(old, base, kind), heldout_scores(new, base, kind)
        if a is None or b is None or not len(b):
            continue
        common = sorted(set(a.recording) & set(b.recording))
        a2, b2 = a[a.recording.isin(common)], b[b.recording.isin(common)]
        ra, rb = rates(a2), rates(b2)
        n, m, lo, hi = paired(a2, b2, "wer")
        n2, m2, lo2, hi2 = paired(a2, b2, "wder")
        print("{:<20} {:<10} {:>9.2f} {:>8.2f} {:>9.2f} {:>9.2f} {:>8.2f} {:>9.2f}   "
              "dWER {:+.2f} [{:+.2f},{:+.2f}]  dWDER {:+.2f} [{:+.2f},{:+.2f}]  n={}"
              .format(corpus, vec, ra[0], ra[1], ra[2], rb[0], rb[1], rb[2],
                      m, lo, hi, m2, lo2, hi2, n))

print()
print("=" * 104)
print("2. CHECKPOINT with precision held constant: large-v2 float16 vs large-v3 float16")
print("=" * 104)
print("{:<20} {:<10} {:>10} {:>26} {:>26}".format("corpus", "vector", "n", "large-v2 fp16 WER/WDER",
                                                  "large-v3 fp16 WER/WDER"))
for corpus, base, kind in CORPORA:
    for vec, v2, v3 in (("reference", "oldref_v2_fp16", "v3_oldref"),
                        ("trial32", "trial32_v2_fp16", "v3_trial32")):
        a, b = heldout_scores(v2, base, kind), heldout_scores(v3, base, kind)
        if a is None or b is None or not len(a) or not len(b):
            continue
        common = sorted(set(a.recording) & set(b.recording))
        a2, b2 = a[a.recording.isin(common)], b[b.recording.isin(common)]
        ra, rb = rates(a2), rates(b2)
        n, m, lo, hi = paired(a2, b2, "wer")
        _, m2, lo2, hi2 = paired(a2, b2, "wder")
        print("{:<20} {:<10} {:>10} {:>13.2f} {:>12.2f} {:>13.2f} {:>12.2f}   "
              "v3-v2 WER {:+.2f} [{:+.2f},{:+.2f}]  WDER {:+.2f} [{:+.2f},{:+.2f}]"
              .format(corpus, vec, n, ra[0], ra[1], rb[0], rb[1], m, lo, hi, m2, lo2, hi2))
