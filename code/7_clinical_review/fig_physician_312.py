"""Figure for the two-physician review of 312 flagged events (all three corpora), in the Figure 7 house style.

Input: Clinical_Validation_Two_Physician_312.xlsx (sheet 'Event-level', written by two_physician_312.py).
The 180-event set is all Fareez. Panel a: Physician 1 x Physician 2 meaning-change matrix.
Panel b: share of events where both physicians judged meaning changed ('Yes'), by corpus and speaker.
Writes fig8_physician_312.pdf / .png next to this script.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

HERE = Path(__file__).parent
# Shared palette (see make_figures.py): BLUE = study configurations / primary grouping, LIGHT = secondary grouping.
INK, FIRE, GREY = "#4E8FB0", "#D15E5C", "#7AB6CB"
from matplotlib.colors import LinearSegmentedColormap
SHARE = LinearSegmentedColormap.from_list("share", ["#FFFFFF", "#4E8FB0", "#3E6E88"])
LAB = ["No", "Partially", "Yes"]

d = pd.read_excel(HERE / "Clinical_Validation_Two_Physician_312.xlsx", sheet_name="Event-level")
d["Corpus"] = d.Corpus.fillna("Fareez")
d["agreed_yes"] = (d.meaning_A == "Yes") & (d.meaning_Z == "Yes")
mat = pd.crosstab(d.meaning_A, d.meaning_Z).reindex(index=LAB, columns=LAB, fill_value=0).values
agree = np.trace(mat) / mat.sum()
ag = pd.read_excel(HERE / "Clinical_Validation_Two_Physician_312.xlsx", sheet_name="Agreement")
row = ag[(ag.Subset == "All 312 events") & ag.Item.str.startswith("Clinical meaning changed (No")].iloc[0]

for face in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    p = Path("/mnt/c/Windows/Fonts") / face
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "stix", "font.size": 9,
                     "axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa5ad",
                     "axes.linewidth": 0.8, "xtick.color": "#6b7780", "ytick.color": "#6b7780",
                     "axes.grid": False, "figure.dpi": 300})

def panel(ax, letter, title, pad=1.02):
    """Bold panel letter plus a plain sentence-case title, as in make_figures.py."""
    ax.text(-0.02, pad, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom", ha="right",
            color="#1d262d")
    ax.text(0.02, pad, title, transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left", color="#1d262d")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [0.85, 1.2]})
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.85, wspace=0.45)

ax1.imshow(mat, cmap=SHARE, vmin=0, vmax=mat.max() * 1.15, aspect="auto")
ax1.set_xticks(range(3), LAB)
ax1.set_yticks(range(3), LAB)
ax1.set_xlabel("Physician 2")
ax1.set_ylabel("Physician 1")
for r in range(3):
    for c in range(3):
        v = mat[r, c]
        ax1.text(c, r, str(v), ha="center", va="center", fontsize=9,
                 color="white" if v > mat.max() * 0.6 else "#1d262d")
panel(ax1, "a", "Meaning-change ratings", pad=1.03)
for sp in ("top", "right", "left", "bottom"):
    ax1.spines[sp].set_visible(False)
ax1.tick_params(length=0)

groups = [("PriMock57", d.Corpus == "PriMock57", INK), ("Fareez", d.Corpus == "Fareez", INK),
          ("AfriSpeech-\nDialog", d.Corpus == "AfriSpeech-Dialog", INK),
          ("Patient", d.Speaker == "PATIENT", GREY), ("Clinician", d.Speaker == "DOCTOR", GREY)]
xs, pct, lab, col = [], [], [], []
for k, (nm, sel, c) in enumerate(groups):
    x = k + (0.5 if k >= 3 else 0)
    yes, n = int(d[sel].agreed_yes.sum()), int(sel.sum())
    xs.append(x); pct.append(100 * yes / n); col.append(c); lab.append("{}\n{}/{}".format(nm, yes, n))
bars = ax2.bar(xs, pct, width=0.62, color=col, edgecolor="#4A5A66", linewidth=0.6)
for b, v in zip(bars, pct):
    ax2.text(b.get_x() + b.get_width() / 2, v + 1.8, "{:.1f}%".format(v), ha="center", va="bottom", fontsize=8.5)
ax2.set_xticks(xs, lab, fontsize=8.3)
ax2.set_ylim(0, 100)
ax2.set_ylabel("Events where meaning changed (%)")
panel(ax2, "b", "Agreed decision by corpus and speaker", pad=1.03)

for ext in ("pdf", "png"):
    fig.savefig(HERE / ("fig8_physician_312." + ext), bbox_inches="tight", facecolor="white")
print(mat, agree, {g[0]: p for g, p in zip(groups, pct)})
