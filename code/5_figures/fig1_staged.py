"""Figure 1, study design and evaluation workflow.

Each stage is a panel with a tinted header band (title, optional subtitle) above a white body; every item in the
body sits in its own box. Development stages in teal, held-out stages in blue, shared stages in grey.
Times New Roman throughout; the figure title is left to the caption.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch   # noqa: E402
from matplotlib import font_manager                   # noqa: E402

OUT = Path("/mnt/d/fareez_optuna/paper/figures")

INK = "#1f2a35"
GREY = "#5b6773"
CELL_EDGE = "#cfd6dc"
#            edge       header band  metric fill
NEUTRAL = ("#4a5865", "#eef1f5", "#eef1f5")
DEV = ("#1a6b6b", "#e8f3f2", "#e8f3f2")
HELD = ("#2f5791", "#ebf0f8", "#e9eef7")

for face in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    for dd in (Path(r"C:\Windows\Fonts"), Path("/mnt/c/Windows/Fonts")):
        if (dd / face).exists():
            font_manager.fontManager.addfont(str(dd / face))
            break

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(7.2, 7.6))
ax.set_xlim(0, 100)
ax.set_ylim(13.0, 97.5)
ax.axis("off")

PAD, GAP = 1.4, 1.1
STEP = 1.95                     # line spacing inside a cell


def cell(x, y, w, h, lines, metric=None):
    """One boxed item: a bold name and grey value lines, or a single tinted metric label."""
    if metric:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=0.6, edgecolor=metric[1], facecolor=metric[1], zorder=2))
        ax.text(x + w / 2, y + h / 2, lines[0], fontsize=7.6, fontweight="bold", color=metric[0],
                ha="center", va="center", zorder=3)
        return
    ax.add_patch(Rectangle((x, y), w, h, linewidth=0.6, edgecolor=CELL_EDGE, facecolor="white", zorder=2))
    top = y + h / 2 + (len(lines) - 1) * STEP / 2
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, top - i * STEP, ln, fontsize=7.4 if i == 0 else 7.2,
                fontweight="bold" if i == 0 else "normal", color=INK if i == 0 else GREY,
                ha="center", va="center", zorder=3)


def panel(x, y, w, h, title, colours, rows, sub=None, dashed=False, metric=False, row_weights=None):
    """A stage panel: tinted header band with title (and subtitle), white body holding rows of boxed cells."""
    edge, band, mfill = colours
    head = 5.6 if sub else 3.8
    ax.add_patch(Rectangle((x, y), w, h, linewidth=0.9, edgecolor=edge, facecolor="white",
                           linestyle=(0, (3, 2)) if dashed else "-", zorder=1))
    ax.add_patch(Rectangle((x, y + h - head), w, head, linewidth=0, facecolor=band, zorder=1.5))
    ax.plot([x, x + w], [y + h - head] * 2, color=edge, lw=0.5, alpha=0.35, zorder=1.6)
    ax.text(x + PAD, y + h - 1.95, title, fontsize=8.8, fontweight="bold", color=edge, va="center", zorder=4)
    if sub:
        ax.text(x + PAD, y + h - 4.05, sub, fontsize=7.2, color=GREY, va="center", zorder=4)
    body_top, body_bot = y + h - head - PAD, y + PAD
    weights = row_weights or [1] * len(rows)
    avail = body_top - body_bot - GAP * (len(rows) - 1)
    yy = body_top
    for row, wt in zip(rows, weights):
        rh = avail * wt / sum(weights)
        cw = (w - 2 * PAD - GAP * (len(row) - 1)) / len(row)
        for j, lines in enumerate(row):
            cell(x + PAD + j * (cw + GAP), yy - rh, cw, rh, lines, (edge, mfill) if metric else None)
        yy -= rh + GAP


def arrow(x1, y1, x2, y2, colour=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, linewidth=0.9,
                                 color=colour, shrinkA=0, shrinkB=0, zorder=0))


def line(xs, ys, colour=GREY):
    ax.plot(xs, ys, color=colour, lw=0.9, zorder=0, solid_capstyle="butt")


L, R, W = 1, 51.5, 47.5

# ---------------------------------------------------------------- corpora
panel(L, 84.5, 98, 12.5, "Corpora", NEUTRAL,
      [[["Fareez", "272 interviews; 271 retained"], ["PriMock57", "57 consultations"],
        ["AfriSpeech-Dialog", "49 conversations (29 general, 20 clinical)"]]])
line([50, 50], [84.5, 82.6], NEUTRAL[0])
line([L + W / 2, R + W / 2], [82.6, 82.6], NEUTRAL[0])
ax.text(51.0, 83.6, "split by recording", fontsize=7.2, color=GREY, style="italic", ha="left", va="center")
arrow(L + W / 2, 82.6, L + W / 2, 80.6, DEV[0])
arrow(R + W / 2, 82.6, R + W / 2, 80.6, HELD[0])

# ---------------------------------------------------------------- sets
panel(L, 69, W, 11.6, "Development sets", DEV,
      [[["Fareez", "n = 40"], ["PriMock57", "n = 20"], ["AfriSpeech-Dialog", "n = 20 general"]]])
panel(R, 69, W, 11.6, "Held-out sets (n = 288)", HELD,
      [[["Fareez", "n = 231"], ["PriMock57", "n = 37"], ["AfriSpeech-Dialog", "n = 20 clinical"]]])

# ---------------------------------------------------------------- search | checkpoints
arrow(L + W / 2, 69, L + W / 2, 66.6, DEV[0])
panel(L, 51, W, 15.6, "Parameter search", DEV,
      [[["Primary (Fareez)", "40 recordings", "50 trials"],
        ["Corpus-specific", "20 recordings each", "14, 52, 14 trials"],
        ["Joint", "60 recordings", "15 trials"]]], sub="Whisper large-v3, TPE")
arrow(R + W / 2, 69, R + W / 2, 66.6, HELD[0])
panel(R, 51, W, 15.6, "Checkpoint comparison", HELD,
      [[["Checkpoints", "Whisper large-v2", "Whisper large-v3"],
        ["Identical decoding", "reference vector", "trial32 vector"]]], sub="Matched decoding on held-out sets")

# ---------------------------------------------------------------- selection | evaluation
arrow(L + W / 2, 51, L + W / 2, 48.6, DEV[0])
panel(L, 32.5, W, 16.1, "Configuration selection", DEV,
      [[["Primary", "lowest WER", "meeting criteria"], ["Fallback", "lowest cpWER"],
        ["Joint", "lowest mean WER", "across corpora"]]], sub="Development recordings only")
# held-out sets also feed the evaluation: route round the outside of the checkpoint panel
line([R + W, 99.6], [74.8, 74.8], HELD[0])
line([99.6, 99.6], [74.8, 49.6], HELD[0])
line([99.6, R + W - 4], [49.6, 49.6], HELD[0])
arrow(R + W - 4, 49.6, R + W - 4, 48.6, HELD[0])
panel(R, 32.5, W, 16.1, "Held-out evaluation", HELD,
      [[["WER"], ["WDER"], ["cpWER"]], [["SA-WER"], ["Hallucinated text"], ["Negation cues"]]],
      sub="Prespecified configurations, scored once", metric=True)
arrow(L + W, 40.5, R, 40.5, DEV[0])

# ---------------------------------------------------------------- bottom row
panel(L, 13.5, W, 16.6, "Post hoc analyses", NEUTRAL,
      [[["Single-parameter", "24 variants"], ["Sensitivity", "threshold, precision"],
        ["Exploratory", "1 held-out variant"]]], sub="Added after the held-out evaluation; not used for selection", dashed=True)
panel(R, 13.5, W, 16.6, "Clinical-meaning review", NEUTRAL,
      [[["Fareez", "218 errors"], ["PriMock57", "55 errors"], ["AfriSpeech-Dialog", "39 errors"]],
       [["Nursing students: 180 Fareez errors"]]], sub="Two physicians, 312 errors", row_weights=[1.45, 0.75])

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / ("fig1_study_design." + ext), bbox_inches="tight", facecolor="white")
print("wrote", OUT / "fig1_study_design.pdf")
