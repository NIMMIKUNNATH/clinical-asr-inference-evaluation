"""Supplementary Data 2: the per-recording scores behind every reported comparison.

Counts only - no transcript text - so it is redistributable although the corpora are not.
Anyone can recompute every pooled rate and every paired difference in the paper from this file.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, "/mnt/d/fareez_optuna/code")
sys.path.insert(0, "/mnt/d/fareez_optuna/code/final")
from make_figures import CORPORA, ORDER, heldout_scores   # noqa: E402

OUT = Path("/mnt/d/fareez_optuna/paper/submission/04_supplementary_data/Supplementary_Data_2.csv")

frames = []
for corpus, base, kind in CORPORA:
    for key, label in ORDER:
        d = heldout_scores(key, base, kind)
        if d is None or not len(d):
            continue
        d = d.copy()
        d.insert(0, "corpus", corpus)
        d.insert(1, "configuration", key)
        d.insert(2, "configuration_label", label)
        d.insert(3, "split", "held-out")
        frames.append(d)

# The exploratory variant (joint configuration with beam 1 and 400 ms speech padding) was decoded after the
# prespecified held-out evaluation, on Fareez only; it is kept apart by its split label.
SCORES = Path("/mnt/d/fareez_optuna/final/scores")
x = pd.read_csv(SCORES / "joint_beam1_pad400.csv").sort_values("recording").reset_index(drop=True)
x.insert(0, "corpus", "Fareez")
x.insert(1, "configuration", "joint_beam1_pad400")
x.insert(2, "configuration_label", "joint, beam 1, 400 ms padding (exploratory, post hoc)")
x.insert(3, "split", "held-out, exploratory post hoc")
frames.append(x)

out = pd.concat(frames, ignore_index=True)
out = out.drop(columns=[c for c in ("config",) if c in out.columns])
out.to_csv(OUT, index=False)

print("wrote", OUT)
print("rows:", len(out), " columns:", len(out.columns))
print("corpora x configurations:")
print(out.groupby(["corpus", "configuration_label"]).size().to_string())
print("\nno free text columns:", [c for c in out.columns if out[c].dtype == object])
