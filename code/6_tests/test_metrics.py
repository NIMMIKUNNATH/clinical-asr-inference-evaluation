"""
test_metrics.py - synthetic validation of the scoring functions.

Every case has a hand-computed expected value, so a change in jiwer, the normaliser or our own code that alters
a metric is caught rather than absorbed. This is the suite reported in the paper's reproducibility section.

Covered: alignment counts for substitution, deletion, insertion and combinations; empty reference and empty
hypothesis; hypothesis longer than reference; normalisation of non-lexical tokens under both conventions;
speaker-run construction including unassigned words; polarity-cue counting including insertions and flips;
hallucination-run detection at the threshold boundary; and the WDER denominator property that motivates the
fixed-denominator analysis.

USAGE:  ~/eval_venv/bin/python test_metrics.py
Exit status 0 if every case passes, 1 otherwise.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import HALLUCINATION_RUN, align, norm_main, norm_nonlex, polarity, word_runs  # noqa: E402

CASES = []


def case(name, got, want):
    CASES.append((name, got, want, got == want))


def counts(ref, hyp):
    c, _ = align(ref.split() if ref else [], hyp.split() if hyp else [])
    return (c["sub"], c["dele"], c["ins"], c["hit"], c["n"])


def run_alignment_cases():
    # (sub, del, ins, hit, n)
    case("L01 identical", counts("a b c", "a b c"), (0, 0, 0, 3, 3))
    case("L02 one substitution", counts("a b c", "a x c"), (1, 0, 0, 2, 3))
    case("L03 terminal deletion", counts("a b c", "a b"), (0, 1, 0, 2, 3))
    case("L04 internal deletion", counts("a b c", "a c"), (0, 1, 0, 2, 3))
    case("L05 one insertion", counts("a b", "a x b"), (0, 0, 1, 2, 2))
    case("L06 two subs one insertion", counts("a b c d e", "a x y d e z"), (2, 0, 1, 3, 5))
    case("L07 empty hypothesis", counts("a b c", ""), (0, 3, 0, 0, 3))
    case("L08 empty reference", counts("", "a b"), (0, 0, 2, 0, 0))
    case("L09 both empty", counts("", ""), (0, 0, 0, 0, 0))
    case("L10 hypothesis longer than reference", counts("a", "a b c d"), (0, 0, 3, 1, 1))

    # WER can exceed 100% when insertions outnumber reference words
    s, d, i, _, n = counts("a", "a b c d")
    case("L11 WER may exceed 100%", (s + d + i) / n > 1.0, True)


def run_normalisation_cases():
    # main convention removes non-lexical tokens; non-lexical convention maps them to fixed forms
    case("N01 affirmative removed (main)", "mmhm" in norm_main("mm-hmm I see"), False)
    case("N02 affirmative kept (nonlex)", "mmhm" in norm_nonlex("mm-hmm I see"), True)
    case("N03 negative removed (main)", "uhuh" in norm_main("uh-uh not really"), False)
    case("N04 negative kept (nonlex)", "uhuh" in norm_nonlex("uh-uh not really"), True)
    # the two conventions must agree on ordinary lexical content
    case("N05 lexical content unchanged",
         [t for t in norm_nonlex("mm-hmm the pain is sharp") if t not in ("mmhm", "uhuh")],
         norm_main("the pain is sharp"))
    # a negation must survive both conventions: losing it is the error the study is about
    case("N06 'no' survives main", "no" in norm_main("no chest pain"), True)
    case("N07 'not' survives nonlex", "not" in norm_nonlex("i have not taken it"), True)


def run_speaker_cases():
    segs = [{"words": [{"word": " a", "speaker": "S1"}, {"word": " b", "speaker": "S1"},
                       {"word": " c", "speaker": "S2"}]},
            {"words": [{"word": " d", "speaker": "S2"}, {"word": " e"}]}]
    runs = word_runs(segs)
    case("S01 runs merge across segments", [spk for spk, _ in runs], ["S1", "S2", "UNKNOWN"])
    case("S02 run text concatenates", [t.strip() for _, t in runs], ["a b", "c d", "e"])
    case("S03 unassigned word labelled UNKNOWN", runs[-1][0], "UNKNOWN")

    # The WDER denominator is hits + substitutions, so deleting words shrinks it. This is the property that
    # motivates reporting a fixed-denominator WDER alongside the conventional one.
    full, _ = align(["a", "b", "c", "d"], ["a", "b", "c", "d"])
    partial, _ = align(["a", "b", "c", "d"], ["a", "b"])
    case("S04 WDER denominator shrinks when words are deleted",
         (full["hit"] + full["sub"]) > (partial["hit"] + partial["sub"]), True)


def run_polarity_cases():
    _, ops = align(["i", "have", "no", "chest", "pain"], ["i", "have", "chest", "pain"])
    p = polarity(ops)
    case("P01 deleted negation counted", (p["neg_ref"], p["neg_err"]), (1, 1))

    _, ops = align(["i", "have", "no", "pain"], ["i", "have", "yes", "pain"])
    p = polarity(ops)
    case("P02 negation substituted by affirmation is a flip", (p["neg_err"], p["flips"]), (1, 1))

    _, ops = align(["i", "have", "pain"], ["i", "have", "no", "pain"])
    p = polarity(ops)
    case("P03 inserted negation counted separately", (p["neg_ins"], p["neg_err"]), (1, 0))

    _, ops = align(["no", "pain"], ["no", "pain"])
    p = polarity(ops)
    case("P04 correct negation is not an error", (p["neg_ref"], p["neg_err"]), (1, 0))


def run_hallucination_cases():
    def halluc_words(ref, hyp):
        _, ops = align(ref.split(), hyp.split())
        return sum(len(h) for op, _, h in ops if op == "insert" and len(h) >= HALLUCINATION_RUN)

    case("H01 run below threshold not counted",
         halluc_words("a b", "a " + " ".join(["x"] * (HALLUCINATION_RUN - 1)) + " b"), 0)
    case("H02 run at threshold counted",
         halluc_words("a b", "a " + " ".join(["x"] * HALLUCINATION_RUN) + " b"), HALLUCINATION_RUN)
    case("H03 run above threshold counted in full",
         halluc_words("a b", "a " + " ".join(["x"] * (HALLUCINATION_RUN + 3)) + " b"), HALLUCINATION_RUN + 3)
    case("H04 scattered insertions are not hallucination",
         halluc_words("a b c", "a x b y c"), 0)


def main():
    for fn in (run_alignment_cases, run_normalisation_cases, run_speaker_cases, run_polarity_cases,
               run_hallucination_cases):
        fn()
    width = max(len(n) for n, _, _, _ in CASES)
    failed = 0
    for name, got, want, ok in CASES:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}" + ("" if ok else f"   got {got!r}, want {want!r}"))
        failed += not ok
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
