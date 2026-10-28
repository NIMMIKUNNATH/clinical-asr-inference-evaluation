"""
fixed_wder.py - is a word diarization error rate difference a difference in attribution, or in coverage?

WDER is defined over substituted and correctly recognised words; deletions are excluded, because a deleted word
cannot be mapped to a reference speaker (El Shafey et al. 2019). The denominator is therefore a property of the
configuration being scored: one that transcribes less is evaluated over fewer words, and the words it misses are
disproportionately short tokens at turn boundaries - the ones hardest to attribute. Two configurations are scored
over different word sets, so part of any difference between them can be coverage rather than attribution.

This script scores both configurations a second time over the intersection of the reference words each
recognised correctly, so the denominator is identical. Reported side by side with the conventional form, per
ANALYSIS_PLAN.md section 4.

USAGE:  ~/eval_venv/bin/python fixed_wder.py --a defaults --b tuned
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import align, norm_main, word_runs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import REF_DIR, ROOT, parse_reference  # noqa: E402

FINAL = ROOT / "final"
SEED, B = 20260915, 10000


def tagged(path):
    """Hypothesis tokens with the speaker each was assigned, in transcript order."""
    out = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(t, spk) for spk, text in word_runs(out["segments"]) for t in norm_main(text)]


def attribution(ref_tagged, hyp_tagged):
    """Per reference-word index: (correct?, hypothesis speaker) for words that align to a hypothesis word."""
    _, ops = align([t for t, _ in ref_tagged], [t for t, _ in hyp_tagged])
    got = {}
    ri = hi = 0
    for op, r, h in ops:
        if op in ("equal", "substitute"):
            for k in range(len(r)):
                got[ri + k] = (op == "equal", hyp_tagged[hi + k][1])
        ri += len(r)
        hi += len(h)
    return got


def wder_over(ref_tagged, got, keys):
    """Minimum over the two speaker-to-role mappings, restricted to the given reference indices."""
    pairs = [(ref_tagged[i][1], got[i][1]) for i in keys if i in got]
    if not pairs:
        return None, 0
    spk = sorted({h for _, h in pairs if h != "UNKNOWN"})
    if not spk:
        return 100.0, len(pairs)
    maps = [{spk[0]: "Doctor", **({spk[1]: "Patient"} if len(spk) > 1 else {})},
            {spk[0]: "Patient", **({spk[1]: "Doctor"} if len(spk) > 1 else {})}]
    wrong = min(sum(m.get(h) != r for r, h in pairs) for m in maps)
    return 100 * wrong / len(pairs), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="defaults")
    ap.add_argument("--b", default="tuned")
    args = ap.parse_args()

    recs = sorted(p.stem for p in (FINAL / args.a).glob("*.json")
                  if (FINAL / args.b / f"{p.stem}.json").exists() and (REF_DIR / f"{p.stem}.txt").exists())
    print(f"{args.b} vs {args.a}: {len(recs)} paired recordings\n")

    rows = []
    for rec in recs:
        ref_tagged = [(t, role) for role, x in parse_reference(REF_DIR / f"{rec}.txt") for t in norm_main(x)]
        ga = attribution(ref_tagged, tagged(FINAL / args.a / f"{rec}.json"))
        gb = attribution(ref_tagged, tagged(FINAL / args.b / f"{rec}.json"))
        common = sorted(i for i in ga if i in gb and ga[i][0] and gb[i][0])   # correct in BOTH
        wa, na = wder_over(ref_tagged, ga, ga.keys())
        wb, nb = wder_over(ref_tagged, gb, gb.keys())
        fa, nf = wder_over(ref_tagged, ga, common)
        fb, _ = wder_over(ref_tagged, gb, common)
        if None in (wa, wb, fa, fb):
            continue
        rows.append(dict(rec=rec, std_a=wa, std_b=wb, n_a=na, n_b=nb, fix_a=fa, fix_b=fb, n_fixed=nf))

    r = {k: np.array([x[k] for x in rows], float) for k in rows[0] if k != "rec"}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), size=(B, len(rows)))

    def paired(d):
        boot = d[idx].mean(axis=1)
        return d.mean(), *np.percentile(boot, [2.5, 97.5]), int((d > 0).sum()), int((d < 0).sum())

    print(f"  denominator sizes: {args.a} {int(r['n_a'].sum()):,} words, {args.b} {int(r['n_b'].sum()):,}, "
          f"common to both {int(r['n_fixed'].sum()):,}")
    print(f"  coverage difference: {int(r['n_b'].sum() - r['n_a'].sum()):+,} words "
          f"({100 * (r['n_b'].sum() - r['n_a'].sum()) / r['n_a'].sum():+.1f}%)\n")

    print(f"  {'convention':<22} {args.a:>9} {args.b:>9} {'diff':>8} {'95% CI':>18}  {'worse/better':>13}")
    for label, a_, b_ in (("conventional WDER", r["std_a"], r["std_b"]),
                          ("fixed-denominator", r["fix_a"], r["fix_b"])):
        m, lo, hi, w, bt = paired(b_ - a_)
        print(f"  {label:<22} {a_.mean():9.2f} {b_.mean():9.2f} {m:+8.2f} "
              f"[{lo:+.2f}, {hi:+.2f}]".ljust(62) + f"{w}/{bt:>6}")

    print("\n  If the two conventions disagree in direction, or the fixed-denominator difference is much\n"
          "  smaller, the conventional comparison reflects coverage rather than attribution.")


if __name__ == "__main__":
    main()
