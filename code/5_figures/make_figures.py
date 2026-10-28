#!/usr/bin/env python3
"""
make_figures.py - publication figures for the held-out evaluation.

Matches the house style of the companion manuscript's figures: Times New Roman, 9 pt, the same palette, no top or right
spines, light grid, 300 dpi, PDF for the manuscript and PNG for slides.

  Figure 1  Confirmatory comparison on the 231 held-out recordings: paired differences with 95% intervals
  Figure 2  Where the gain comes from: error composition and hallucination
  Figure 3  Checkpoint against parameters, the isolation result
  Figure 4  Cross-corpus transfer and the cost of globality

Run:  python make_figures.py          (the transcription venv, which has matplotlib)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT  # noqa: E402

SCORES = ROOT / "final" / "scores"
PSCORES = ROOT / "final_primock" / "scores"
ASCORES = ROOT / "final_afrispeech" / "scores"
OUT = ROOT / "paper" / "figures"

# Lighter palette, keeping the hues of the companion manuscript so the figures still sit together.
# Print-safe: every fill stays distinguishable in greyscale and at 300 dpi.
DEEP = "#4E8FB0"    # medium blue, the large-v3 series
TEAL = "#7AB6CB"    # light teal, secondary fills
WARN = "#D15E5C"    # rose red, deterioration
GOOD = "#1F4E5F"    # dark blue, the large-v2 series (contrasts with DEEP by lightness and marker)
MUTE = "#80909D"    # grey, the released defaults
EDGE = "#4A5A66"    # outline and text

# Saturated marks on a clean field, as in the companion figure set: no grid, two spines only.
INK = "#4E8FB0"     # blue, the improving (tuned) series - same as DEEP
FIRE = "#D15E5C"    # rose red, the deteriorating series
GREY = "#80909D"

# Times New Roman, to match the manuscript body text. The faces ship with Windows; register them
# explicitly because matplotlib does not scan the Windows font directory from WSL.
for _face in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    _path = Path("/mnt/c/Windows/Fonts") / _face
    if _path.exists():
        font_manager.fontManager.addfont(str(_path))
if "Times New Roman" not in {f.name for f in font_manager.fontManager.ttflist}:
    print("  warning: Times New Roman not found; falling back to the nearest serif")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    # TrueType rather than matplotlib's default Type 3, which several publishers reject.
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9aa5ad", "axes.linewidth": 0.8,
    "xtick.color": "#6b7780", "ytick.color": "#6b7780",
    "axes.grid": False, "figure.dpi": 300,
})


def fig_title(fig, text, x=0.5, y=1.0):
    """The bold figure-level title the companion set carries above its panels."""
    fig.suptitle(text, x=x, y=y, fontsize=11, fontweight="bold", color="#1d262d")


KEEP_LOWER = ("large-v", "trial32", "cpWER", "n = ")


def sentence(t):
    """Sentence case for a figure label: capitalise the first letter unless it starts a model or metric name."""
    if not t or not t[0].isalpha() or not t[0].islower() or t.startswith(KEEP_LOWER):
        return t
    return t[0].upper() + t[1:]


def polish(fig):
    """Apply sentence case to axis labels, categorical tick labels, panel titles and legend entries."""
    from matplotlib.ticker import FixedFormatter, FuncFormatter
    for ax in fig.axes:
        ax.set_xlabel(sentence(ax.get_xlabel()))
        ax.set_ylabel(sentence(ax.get_ylabel()))
        for loc in ("left", "center", "right"):
            t = ax.get_title(loc=loc)
            if t:
                ax.set_title(sentence(t), loc=loc)
        for axis in (ax.xaxis, ax.yaxis):
            f = axis.get_major_formatter()
            if isinstance(f, FixedFormatter):
                f.seq = [sentence(x) for x in f.seq]
            elif isinstance(f, FuncFormatter) and not getattr(f, "_sentence", False):
                inner = f.func
                f.func = lambda x, pos=None, inner=inner: sentence(inner(x, pos))
                f._sentence = True
        leg = ax.get_legend()
        if leg:
            for t in leg.get_texts():
                t.set_text(sentence(t.get_text()))
    for leg in fig.legends:
        for t in leg.get_texts():
            t.set_text(sentence(t.get_text()))


def save(fig, name):
    polish(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def panel(ax, letter, title, pad=1.02):
    """Bold panel letter plus a short title, in the convention of the companion figures."""
    ax.text(-0.02, pad, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            va="bottom", ha="right", color="#1d262d")
    ax.text(0.02, pad, sentence(title), transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left",
            color="#1d262d")


def label_bars(ax, bars, fmt="{:.2f}", dy=0.012, fontsize=7.6, colour=None):
    """Print each bar's value just above it, so the figure is readable without the axis."""
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy * span, fmt.format(h), ha="center",
                va="bottom", fontsize=fontsize, color=colour or EDGE)


def reference_line(ax, value, text, horizontal=True, loc=0.99, fontsize=7.6):
    """A dashed reference line carrying its own label, as in the companion figures."""
    if horizontal:
        ax.axhline(value, color=EDGE, lw=0.9, ls="--", zorder=1)
        ax.text(loc, value, " " + text, transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=fontsize, color=EDGE)
    else:
        ax.axvline(value, color=EDGE, lw=0.9, ls="--", zorder=1)
        ax.text(value, loc, " " + text, transform=ax.get_xaxis_transform(), ha="left",
                va="top", fontsize=fontsize, color=EDGE)


def pooled(df, num, den, scale=100):
    return scale * df[num].sum().sum() / df[den].sum()


def load(name, base=SCORES):
    p = base / f"{name}.csv"
    return pd.read_csv(p).sort_values("recording").reset_index(drop=True) if p.exists() else None




def heldout_scores(name, base, kind=None):
    """Scores restricted to that corpus's held-out set: all Fareez rows, PriMock57's 37, AfriSpeech's clinical 20."""
    d = load(name, base)
    if d is None or not len(d):
        return None
    if kind == 'primock':
        ho = set(json.loads((ROOT / 'results_primock' / 'dev_split.json').read_text())['heldout'])
        return d[d.recording.isin(ho)]
    if kind == 'clinical':
        ids = pd.read_csv(ROOT / 'cache' / 'afrispeech' / 'ids.csv')
        return d[d.recording.isin(set(ids[ids.domain == 'medical'].id))]
    return d

# ---------------------------------------------------------------- Figure 1
def figure1():
    cmp = SCORES / "compare_tuned_vs_defaults.csv"
    if not cmp.exists():
        print("  skipping figure 1: comparison not found")
        return
    d = pd.read_csv(cmp)
    keep = ["WER", "  insertions", "  deletions", "  substitutions", "CER", "cpWER", "SA-WER", "WDER",
            "negation (incl. insertions)", "hallucinated words / 1k"]
    d = d[d.outcome.isin(keep)].copy()
    d["label"] = d.outcome.str.strip().str.replace("hallucinated words / 1k", "hallucinated text (words / 1,000)")
    d = d.iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(d)) * 1.0
    for i, r in enumerate(d.itertuples()):
        c = FIRE if r.diff > 0 else INK
        ax.hlines(i, r.lo, r.hi, color=c, lw=1.6, alpha=0.95, zorder=2)
        ax.plot([r.lo, r.hi], [i, i], "|", ms=7, color=c, mew=1.6, zorder=3)
        ax.plot(r.diff, i, "o", ms=7, color=c, mec="white", mew=1.0, zorder=4)
        ax.text(r.diff, i + 0.30, "{:+.2f} [{:+.2f}, {:+.2f}]".format(r.diff, r.lo, r.hi),
                ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=c)
    ax.axvline(0, color="#9aa5ad", lw=0.9, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d.label)
    ax.set_ylim(-0.8, len(d) - 0.25)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("paired difference, tuned minus defaults (percentage points; hallucinated text in words per 1,000)")
    lo, hi = float(d.lo.min()), float(d.hi.max())
    pad = 0.10 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.text(0.0, -0.13, "Favours tuned configuration", transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=INK)
    ax.text(1.0, -0.13, "Favours released defaults", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=FIRE)

    fig.tight_layout()
    save(fig, "fig2_confirmatory")


# ---------------------------------------------------------------- Figure 2
def figure2():
    a, b = load("defaults"), load("tuned")
    if a is None or b is None:
        print("  skipping figure 2")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})

    # a: word error rate as stacked components, so the total and its source are read together
    parts = [("substitutions", "sub", TEAL), ("deletions", "dele", DEEP), ("insertions", "ins", WARN)]
    x = np.array([0, 1])
    bottom = np.zeros(2)
    for name, col, colour in parts:
        vals = np.array([pooled(a, [col], "ref_words"), pooled(b, [col], "ref_words")])
        ax1.bar(x, vals, 0.55, bottom=bottom, color=colour, edgecolor=EDGE, linewidth=0.6, label=name)
        for xi, v, b0 in zip(x, vals, bottom):
            ax1.text(xi, b0 + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if colour != TEAL else "#1d262d")
        bottom += vals
    for xi, tot in zip(x, bottom):
        ax1.text(xi, tot + 0.2, f"WER {tot:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                 color="#1d262d")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["released\ndefaults", "tuned"])
    ax1.set_xlim(-0.6, 2.75)
    ax1.set_ylim(0, bottom.max() * 1.18)
    ax1.set_ylabel("percentage of reference words")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], frameon=False, fontsize=8, loc="upper right")
    ha = 1000 * a.halluc_words.sum() / a.ref_words.sum()
    hb = 1000 * b.halluc_words.sum() / b.ref_words.sum()
    ax1.text(0.6, 0.55, "Hallucinated text:\n{:.1f} → {:.1f} per 1,000 words\n\nInsertion runs:\n{} → {}".format(
        ha, hb, int(a.halluc_runs.sum()), int(b.halluc_runs.sum())), transform=ax1.transAxes, ha="left", va="top",
        fontsize=7.6, color=EDGE, linespacing=1.4)
    panel(ax1, "a", "word error composition")

    # b: negation-cue error under the two counting rules
    rules = [("cues introduced by\ninsertions counted", lambda d: 100 * (d.neg_err.sum() + d.neg_ins.sum()) / d.neg_ref.sum()),
             ("reference cues\nonly", lambda d: 100 * d.neg_err.sum() / d.neg_ref.sum())]
    xr = np.arange(len(rules))
    for off, d, colour, lab in ((-0.19, a, MUTE, "released defaults"), (0.19, b, DEEP, "tuned")):
        vals = [f(d) for _, f in rules]
        bars = ax2.bar(xr + off, vals, 0.36, color=colour, edgecolor=EDGE, linewidth=0.6, label=lab)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.25, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(xr)
    ax2.set_xticklabels([r for r, _ in rules], fontsize=8.5)
    ax2.set_ylabel("negation-cue error (%)")
    ax2.set_ylim(0, 16)
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    panel(ax2, "b", "negation-cue error")
    fig.tight_layout()
    save(fig, "fig3_mechanism")


# ---------------------------------------------------------------- Figure 3
def figure3():
    names = [("defaults", "large-v3", "released defaults"), ("tuned", "large-v3", "tuned (ours)"),
             ("v3_oldref", "large-v3", "reference parameters"), ("oldref_v2_fp16", "large-v2", "reference parameters"),
             ("trial32_v2_fp16", "large-v2", "trial32 parameters"), ("v3_trial32", "large-v3", "trial32 parameters")]
    rows = []
    for key, model, label in names:
        d = load(key)
        if d is None or len(d) < 100:
            continue
        rows.append(dict(key=key, model=model, label=label, n=len(d),
                         wer=pooled(d, ["sub", "dele", "ins"], "ref_words"),
                         wder=pooled(d, ["wder_wrong"], "wder_aligned")))
    if len(rows) < 3:
        print("  skipping figure 3: isolation runs incomplete")
        return
    r = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for model, colour, marker in (("large-v2", GOOD, "o"), ("large-v3", DEEP, "s")):
        s = r[r.model == model]
        ax.scatter(s.wer, s.wder, s=70, color=colour, marker=marker, label=model, zorder=3,
                   edgecolor=EDGE, linewidth=0.9)
    ax.set_xlabel("word error rate (%)")
    ax.set_ylabel("word diarization error rate (%)")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.margins(x=0.30, y=0.28)

    # Labels go to the right of their point, except near the right edge where there is no room. Points that
    # nearly coincide are stacked downward so their labels never overlap each other or the axis.
    lo, hi = ax.get_xlim()
    r = r.sort_values(["wer", "wder"]).reset_index(drop=True)
    placed = []
    for _, row in r.iterrows():
        rank = sum(1 for px, py in placed if abs(px - row.wer) < 0.30 and abs(py - row.wder) < 0.45)
        right_edge = row.wer > lo + 0.72 * (hi - lo)
        dx, ha = (-10, "right") if right_edge else (10, "left")
        ax.annotate(row.label, (row.wer, row.wder), textcoords="offset points",
                    xytext=(dx, -3 - 12 * rank), fontsize=8, color=EDGE, ha=ha, va="center")
        placed.append((row.wer, row.wder))

    save(fig, "fig4_checkpoint")


# ---------------------------------------------------------------- Figure 4
def figure4():
    """The cross-corpus grid: every configuration on every held-out set, one group per corpus."""
    corpora = [("Fareez (231)", SCORES, None, DEEP), ("PriMock57 (37)", PSCORES, "primock", DEEP),
               ("AfriSpeech-Dialog (20)", ASCORES, "clinical", DEEP)]
    cfgs = [("defaults", "released\ndefaults"), ("tuned", "Fareez\n(40)"), ("local20_fareez", "Fareez\n(20)"),
            ("local20_afrispeech", "AfriSpeech\n(20)"), ("primock_tuned", "joint")]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6))
    for pi, (ax, (name, base, kind, colour)) in enumerate(zip(axes, corpora)):
        vals, labels = [], []
        for key, lab in cfgs:
            d = heldout_scores(key, base, kind)
            if d is None or not len(d):
                continue
            vals.append(pooled(d, ["sub", "dele", "ins"], "ref_words"))
            labels.append(lab)
        x = np.arange(len(vals))
        bars = ax.bar(x, vals, 0.62, color=[MUTE if l.startswith("released") else colour for l in labels],
                      edgecolor=EDGE, linewidth=0.7)
        best = int(np.argmin(vals))
        bars[best].set_hatch("//")
        top = max(vals)
        for i, v in enumerate(vals):
            ax.text(i, v + top * 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, top * 1.22)
        panel(ax, "abc"[pi], name)
    axes[0].set_ylabel("word error rate (%)")
    fig.tight_layout()
    save(fig, "fig6_grid")


if __name__ == "__main__":
    print("writing figures to", OUT)
    figure1()
    figure2()
    figure3()
    figure4()


# ---------------------------------------------------------------- Figure 1 (study design)
def figure_design():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, text, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=fill, edgecolor=EDGE, linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6, color="#1d262d",
                linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=0.8, shrinkA=2, shrinkB=2))

    # three corpora, each with its own local search; one joint search over all three; one held-out grid
    box(1, 45, 19, 10, "Fareez\n271 consultations\n40 dev / 231 held-out", TEAL)
    box(1, 27, 19, 10, "PriMock57\n57 consultations\n20 dev / 37 held-out", TEAL)
    box(1, 9, 19, 10, "AfriSpeech-Dialog\n49 conversations\n20 general dev /\n20 clinical held-out", TEAL)

    box(24, 45, 20, 10, "Local search\n17 parameters\nFareez 40, then 20", MUTE)
    box(24, 27, 20, 10, "Local search\nsame space\nPriMock57 20", MUTE)
    box(24, 9, 20, 10, "Local search\nsame space\nAfriSpeech 20", MUTE)

    box(48, 27, 20, 10, "Joint search\n20 + 20 + 20\nmean WER, prespecified rule", MUTE)
    box(48, 45, 20, 10, "Checkpoint test\n2 decoding vectors\non large-v2 and v3", GOOD)

    box(72, 33, 26, 14, "Held-out grid, scored once\nFareez 231 | PriMock57 37\nAfriSpeech clinical 20\n"
                        "every configuration", DEEP)
    box(72, 12, 26, 12, "Clinician review\n180 flagged events\ncost of globality", WARN)

    for y in (50, 32, 14):
        arrow(20, y, 24, y)
    arrow(44, 50, 48, 50)
    arrow(44, 32, 48, 32)
    arrow(44, 14, 55, 27)
    arrow(58, 45, 58, 37)
    arrow(68, 32, 72, 38)
    arrow(68, 50, 72, 45)
    arrow(85, 33, 85, 24)
    ax.text(1, 58.5, "Study design", fontsize=11, ha="left", va="top")
    # Superseded by fig1_staged.py, which draws the numbered staged layout the paper uses.
    # Saving here would overwrite that figure, so this earlier draft is kept but not written.
    plt.close(fig)


# ------------------------------------------------- Figure 4 (checkpoint, one panel per corpus)
def figure_checkpoint_three():
    vectors = [("reference", "oldref_v2_fp16", "v3_oldref"), ("trial32", "trial32_v2_fp16", "v3_trial32")]
    corpora = [("Fareez", SCORES, None), ("PriMock57", PSCORES, "primock"), ("AfriSpeech-Dialog", ASCORES, "clinical")]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9))
    for pi, (ax, (name, base, kind)) in enumerate(zip(axes, corpora)):
        # colour = checkpoint, marker = decoding vector; a thin line joins the two checkpoints run on one vector
        CAP = 30.0
        for vector, v2, v3 in vectors:
            marker = "o" if vector == "reference" else "s"
            xy = []
            for key, colour in ((v2, GOOD), (v3, DEEP)):
                d = heldout_scores(key, base, kind)
                if d is None or not len(d):
                    continue
                wer = pooled(d, ["sub", "dele", "ins"], "ref_words")
                wder = pooled(d, ["wder_wrong"], "wder_aligned")
                x = min(wer, CAP)
                ax.scatter(x, wder, s=62, color=colour, marker=marker, zorder=3, edgecolor=EDGE, linewidth=0.9)
                if wer > CAP:   # off-scale point drawn at the axis edge with its true values
                    ax.annotate("{:.2f} / {:.2f}".format(wer, wder), (x, wder), xytext=(-6, 12),
                                textcoords="offset points", ha="right", fontsize=7.5, color=EDGE,
                                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=0.8))
                xy.append((x, wder))
            if len(xy) == 2:
                ax.plot(*zip(*xy), color="#9aa5ad", lw=0.8, ls=":", zorder=1)
        panel(ax, "abc"[pi], name)
        ax.set_xlabel("word error rate (%)")
        ax.margins(x=0.25, y=0.25)
    axes[0].set_ylabel("word diarization error rate (%)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=7.5, color=GOOD, mec=EDGE, label="large-v2"),
               Line2D([], [], marker="o", ls="", ms=7.5, color=DEEP, mec=EDGE, label="large-v3"),
               Line2D([], [], marker="o", ls="", ms=7.5, color="white", mec=EDGE, label="reference vector"),
               Line2D([], [], marker="s", ls="", ms=7.5, color="white", mec=EDGE, label="trial32 vector")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "fig4_checkpoint")


# ------------------------------------------------- Figure 6 (what tuning trades, per corpus)
def figure_trade():
    corpora = [("Fareez", SCORES, None), ("PriMock57", PSCORES, "primock"), ("AfriSpeech-Dialog", ASCORES, "clinical")]
    configs = [("defaults", "released\ndefaults"), ("tuned", "Fareez-tuned"), ("primock_tuned", "joint")]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.9))
    for n, (ax, (name, base, kind)) in enumerate(zip(axes, corpora)):
        labels, dele, ins = [], [], []
        for key, label in configs:
            d = heldout_scores(key, base, kind)
            if d is None or not len(d):
                continue
            labels.append(label)
            dele.append(pooled(d, ["dele"], "ref_words"))
            ins.append(pooled(d, ["ins"], "ref_words"))
        x = np.arange(len(labels))
        top = max(max(ins), max(dele))
        ax.bar(x - 0.19, ins, 0.36, color=FIRE, edgecolor="none", label="insertions", zorder=3)
        ax.bar(x + 0.19, dele, 0.36, color=INK, edgecolor="none", label="deletions", zorder=3)
        for xi, v in zip(x - 0.19, ins):
            ax.text(xi, v + 0.025 * top, "{:.2f}".format(v), ha="center", va="bottom",
                    fontsize=7.8, fontweight="bold", color=FIRE)
        for xi, v in zip(x + 0.19, dele):
            ax.text(xi, v + 0.025 * top, "{:.2f}".format(v), ha="center", va="bottom",
                    fontsize=7.8, fontweight="bold", color=INK)
        ax.axhline(ins[0], color="#9aa5ad", lw=0.9, ls="--", zorder=1)
        ax.set_ylim(0, top * 1.24)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        panel(ax, "abc"[n], name)
    axes[0].set_ylabel("percentage of reference words")
    axes[0].legend(frameon=False, loc="upper center", fontsize=8, ncol=2)

    fig.tight_layout()
    save(fig, "fig5_trade")


# ---------------------------------------------------------------- Figure S8 (runaway tail)
def figure_tail():
    a, b = load("trial32_v2_fp16"), load("v3_trial32")
    if a is None or b is None:
        print("  skipping tail figure")
        return
    wa = 100 * (a["sub"] + a["dele"] + a["ins"]) / a.ref_words
    wb = 100 * (b["sub"] + b["dele"] + b["ins"]) / b.ref_words
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bins = np.linspace(0, max(60, float(np.percentile(wb, 99))), 40)
    ax.hist(wa.clip(upper=bins[-1]), bins=bins, color=GOOD, edgecolor=EDGE, linewidth=0.5, alpha=0.9,
            label=f"large-v2  (median {wa.median():.1f}%)")
    ax.hist(wb.clip(upper=bins[-1]), bins=bins, color=DEEP, edgecolor=EDGE, linewidth=0.5, alpha=0.7,
            label=f"large-v3  (median {wb.median():.1f}%)")
    n50 = int((wb > 50).sum())
    ax.set_xlabel("recording-level word error rate (%), clipped at the right edge")
    ax.set_ylabel("recordings")
    ax.legend(frameon=False)
    ax.text(0.98, 0.55, f"{n50} large-v3 recordings\nabove 50%; worst {wb.max():.0f}%",
            transform=ax.transAxes, ha="right", fontsize=8, color=EDGE)
    save(fig, "figS1_runaway_tail")




# ---------------------------------------------------------------- supplementary figure set
GRID = ROOT / "results_grid"

CORPORA = [("Fareez", SCORES, None), ("PriMock57", PSCORES, "primock"),
           ("AfriSpeech-Dialog", ASCORES, "clinical")]

# Configuration order used by every supplementary figure. "joint" is omitted because it resolved to
# exactly the PriMock57 configuration and would draw the same row twice.
ORDER = [("defaults", "released defaults"),
         ("tuned", "Fareez-selected (40)"),
         ("local20_fareez", "Fareez-selected (20)"),
         ("primock_tuned", "joint = PriMock57 (20)"),
         ("local20_afrispeech", "AfriSpeech-selected (20)"),
         ("oldref_v2_fp16", "large-v2, reference vector"),
         ("v3_oldref", "large-v3, reference vector"),
         ("trial32_v2_fp16", "large-v2, trial32 vector"),
         ("v3_trial32", "large-v3, trial32 vector")]

STUDIES = [("Fareez (20)", "fareez_local20.journal", GOOD),
           ("PriMock57 (20), joint source", "primock.journal", DEEP),
           ("AfriSpeech (20)", "afrispeech_local20.journal", WARN),
           ("joint (20+20+20)", "joint.journal", TEAL)]

CONTINUOUS = [("vad_min_silence_ms", "VAD minimum silence (ms)", 2000),
              ("vad_threshold", "VAD threshold", 0.5),
              ("vad_speech_pad_ms", "VAD speech padding (ms)", 400),
              ("beam_size", "beam size", 5),
              ("log_prob_threshold", "log-probability threshold", -1.0),
              ("no_speech_threshold", "no-speech threshold", 0.6)]

CATEGORICAL = [("condition_on_previous_text", "previous-text conditioning"),
               ("temperature_fallback", "temperature fallback")]

CAP_S2 = 12.0   # excess word error beyond this is a runaway decode, not a parameter response


def trial_frame():
    """Every completed trial of the four searches, with its parameters and its excess over that search's best."""
    try:
        import optuna
    except ImportError:
        return None
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    home = Path.home() / "fareez_optuna"
    rows = []
    for label, journal, colour in STUDIES:
        path = home / journal
        if not path.exists():
            continue
        storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(path)))
        for name in optuna.study.get_all_study_names(storage=storage):
            done = [t for t in optuna.load_study(study_name=name, storage=storage).trials
                    if t.value is not None]
            if not done:
                continue
            best = min(t.value for t in done)
            for t in done:
                rows.append({"study": label, "colour": colour,
                             "excess": 100 * (t.value - best), **t.params})
    return pd.DataFrame(rows) if rows else None


def figure_parameter_response():
    """Supplementary Figure S2: how the objective responded to each searched parameter, in all four searches."""
    df = trial_frame()
    if df is None or df.empty:
        print("  skipping parameter-response figure (no study journals)")
        return
    offscale = int((df.excess > CAP_S2).sum())
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.6))
    for ax, (key, title, default) in zip(axes.flat, CONTINUOUS):
        for label, _journal, colour in STUDIES:
            d = df[(df.study == label) & df[key].notna()]
            if not len(d):
                continue
            lo, hi = d[d.excess <= CAP_S2], d[d.excess > CAP_S2]
            ax.scatter(lo[key], lo.excess, s=17, color=colour, edgecolor=EDGE, linewidth=0.35,
                       alpha=0.85, zorder=3)
            ax.scatter(hi[key], np.full(len(hi), CAP_S2 * 0.97), s=27, marker="^", color=colour,
                       edgecolor=EDGE, linewidth=0.35, alpha=0.85, zorder=3)
        ax.set_ylim(-0.7, CAP_S2 + 0.6)
        if default is not None and float(df[key].min()) <= default <= max(float(df[key].max()), default):
            ax.axvline(default, color=EDGE, lw=0.8, ls=":", zorder=2)
            ax.text(default, ax.get_ylim()[1], " Default", fontsize=7, color=EDGE, va="top")
        ax.set_title(title, loc="left", fontsize=9)
    for ax, (key, title) in zip(axes.flat[len(CONTINUOUS):], CATEGORICAL):
        levels = sorted(df[key].dropna().unique(), key=str)
        for i, level in enumerate(levels):
            for j, (label, _journal, colour) in enumerate(STUDIES):
                d = df[(df.study == label) & (df[key] == level)]
                if not len(d):
                    continue
                lo, hi = d[d.excess <= CAP_S2], d[d.excess > CAP_S2]
                ax.scatter(np.full(len(lo), i + (j - 1.5) * 0.11), lo.excess, s=17, color=colour,
                           edgecolor=EDGE, linewidth=0.35, alpha=0.85, zorder=3)
                ax.scatter(np.full(len(hi), i + (j - 1.5) * 0.11), np.full(len(hi), CAP_S2 * 0.97),
                           s=27, marker="^", color=colour, edgecolor=EDGE, linewidth=0.35,
                           alpha=0.85, zorder=3)
        ax.set_ylim(-0.7, CAP_S2 + 0.6)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(v) for v in levels], fontsize=8)
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_title(title, loc="left", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("excess word error over\nthe best trial (points)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec=EDGE, mew=0.4, ms=6, label=l)
               for l, _j, c in STUDIES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "figS2_parameter_response")


def figure_vs_defaults():
    """Supplementary Figure S3: every configuration against the released defaults, on all three corpora."""
    path = GRID / "comparisons.csv"
    if not path.exists():
        print("  skipping vs-defaults figure")
        return
    c = pd.read_csv(path)
    c = c[c.kind == "vs defaults"]
    names = dict(ORDER)
    wanted = [k for k, _ in ORDER if k != "defaults"]
    outcomes = [("WER", "word error"), ("WDER", "speaker attribution"),
                ("negation (incl. insertions)", "negation-cue error")]
    fig, axes = plt.subplots(len(outcomes), 3, figsize=(11.6, 9.2))
    for col, (corpus, _b, _k) in enumerate(CORPORA):
        for row, (outcome, pretty) in enumerate(outcomes):
            ax = axes[row, col]
            d = c[(c.corpus == corpus) & (c.outcome == outcome)].set_index("comparator")
            d = d.reindex([k for k in wanted if k in set(d.index)])
            y = np.arange(len(d))[::-1]
            for yi, (_, r) in zip(y, d.iterrows()):
                colour = DEEP if r.ci95_upper < 0 else (WARN if r.ci95_lower > 0 else MUTE)
                ax.hlines(yi, r.ci95_lower, r.ci95_upper, color=colour, lw=2.4, alpha=0.9, zorder=2)
                ax.plot(r.mean_paired_difference, yi, "o", ms=5.5, color=colour, mec=EDGE, mew=0.6,
                        zorder=3)
            # The runaway large-v3 vector is an order of magnitude off every other row, so the axis is
            # scaled to the rest and that row is drawn at the edge with its value printed.
            rest = d.drop(index="v3_trial32", errors="ignore")
            lo = float(min(rest.ci95_lower.min(), 0))
            hi = float(max(rest.ci95_upper.max(), 0))
            pad = 0.28 * max(hi - lo, 0.4)
            ax.set_xlim(lo - pad, hi + pad)
            for yi, (key, r) in zip(y, d.iterrows()):
                if r.mean_paired_difference > hi + pad:
                    ax.annotate("+{:.1f}".format(r.mean_paired_difference), (hi + pad, yi),
                                xytext=(-3, 6), textcoords="offset points", ha="right", va="bottom",
                                fontsize=7.5, color=EDGE, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"),
                                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.1,
                                                shrinkA=1, shrinkB=0))
            ax.axvline(0, color=EDGE, lw=0.9, ls="--", zorder=1)
            ax.set_yticks(y)
            ax.set_yticklabels([names.get(i, i) for i in d.index], fontsize=8)
            ax.set_ylim(-0.7, len(d) - 0.3)
            ax.set_title(corpus + " - " + pretty, loc="left", fontsize=9.5)
            ax.set_xlabel("difference from the released defaults (points)")
    fig.tight_layout(rect=(0, 0.01, 1, 1))
    save(fig, "figS3_vs_defaults")


def figure_composition():
    """Supplementary Figure S4: how each configuration's word errors divide into the three components."""
    path = GRID / "grid_rates.csv"
    if not path.exists():
        print("  skipping composition figure")
        return
    g = pd.read_csv(path)
    keys = [k for k, _ in ORDER]
    names = dict(ORDER)
    corpora = [c for c, _b, _k in CORPORA]
    parts = ["substitutions", "deletions", "insertions"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("share", ["#FFFFFF", DEEP, "#3E6E88"])
    fig, axes = plt.subplots(1, 6, figsize=(13.2, 4.9),
                             gridspec_kw={"width_ratios": [3, 1.15, 3, 1.15, 3, 1.15], "wspace": 0.12})
    for j, corpus in enumerate(corpora):
        heat, dots = axes[2 * j], axes[2 * j + 1]
        share = np.full((len(keys), len(parts)), np.nan)
        wer = np.full(len(keys), np.nan)
        for i, k in enumerate(keys):
            row = g[(g.corpus == corpus) & (g.config == k)]
            if not len(row):
                continue
            vals = np.array([float(row[p].iloc[0]) for p in parts])
            share[i] = 100 * vals / vals.sum()
            wer[i] = float(row["WER"].iloc[0])
        heat.imshow(share, cmap=cmap, aspect="auto", vmin=5, vmax=60)
        for i in range(len(keys)):
            for c in range(len(parts)):
                if not np.isnan(share[i, c]):
                    heat.text(c, i, "{:.0f}".format(share[i, c]), ha="center", va="center",
                              fontsize=7.6, color=EDGE)
        heat.set_xticks(range(len(parts)))
        heat.set_xticklabels(["subst.", "delet.", "insert."], fontsize=8)
        heat.set_yticks(range(len(keys)))
        heat.set_yticklabels([names[k] for k in keys] if j == 0 else [""] * len(keys), fontsize=8)
        heat.set_title(corpus, loc="left", fontsize=10)
        heat.grid(False)
        dots.plot(wer, np.arange(len(keys)), "o", ms=5, color=DEEP, mec=EDGE, mew=0.6, ls="none",
                  zorder=3)
        for i, v in enumerate(wer):
            if not np.isnan(v):
                dots.hlines(i, 0, v, color=MUTE, lw=0.7, alpha=0.6, zorder=2)
        dots.set_ylim(len(keys) - 0.5, -0.5)
        dots.set_xlim(0, float(np.nanmax(wer)) * 1.18)
        dots.set_yticks([])
        dots.set_xlabel("WER (%)", fontsize=8)
        dots.tick_params(labelsize=7.5)
    fig.tight_layout(rect=(0, 0.01, 1, 1))
    save(fig, "figS4_composition")


def figure_distributions():
    """Supplementary Figure S5: recording-level word error distributions behind the pooled rates."""
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 5.2))
    for ax, (corpus, base, kind) in zip(axes, CORPORA):
        data, labels, keys = [], [], []
        for key, label in ORDER:
            d = heldout_scores(key, base, kind)
            if d is None or not len(d):
                continue
            data.append(100 * (d["sub"] + d["dele"] + d["ins"]) / d.ref_words)
            labels.append(label)
            keys.append(key)
        if not data:
            continue
        steady = [np.asarray(v) for v, k in zip(data, keys) if k != "v3_trial32"]
        cap = float(np.percentile(np.concatenate(steady), 99.5))
        bp = ax.boxplot([np.clip(np.asarray(v), 0, cap) for v in data], vert=False, patch_artist=True,
                        widths=0.62,
                        flierprops=dict(marker="o", ms=0, mfc="none", mec="none"),
                        medianprops=dict(color=EDGE, lw=1.2),
                        whiskerprops=dict(color=EDGE, lw=0.8), capprops=dict(color=EDGE, lw=0.8))
        for patch, key in zip(bp["boxes"], keys):
            fill = MUTE if key == "defaults" else (GOOD if "_v2" in key else DEEP)
            patch.set(facecolor=fill, edgecolor=EDGE, linewidth=0.7, alpha=0.9)
        # individual recordings over the boxes, as in the companion figure of the earlier analysis
        rng = np.random.default_rng(20260915)
        for i, v in enumerate(data):
            v = np.clip(np.asarray(v), 0, cap)
            ax.scatter(v, (i + 1) + rng.uniform(-0.17, 0.17, size=len(v)), s=4.5, color=EDGE,
                       alpha=0.28 if len(v) > 100 else 0.55, linewidths=0, zorder=4)
        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("recording-level word error rate (%)")
        n = len(heldout_scores("defaults", base, kind))
        ax.set_title("{}  (n = {})".format(corpus, n), loc="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.01, 1, 1))
    save(fig, "figS5_distributions")


if __name__ == "__main__":
    figure_design()
    figure_checkpoint_three()
    figure_trade()
    figure_tail()
    figure_parameter_response()
    figure_vs_defaults()
    figure_composition()
    figure_distributions()
