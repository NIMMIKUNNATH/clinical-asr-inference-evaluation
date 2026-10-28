"""
test_assignment.py - synthetic validation of word-to-speaker assignment, exactly as the runs performed it.

The runs call whisperx.diarize.assign_word_speakers (WhisperX 3.8.6, fill_nearest=False) on the cached pyannote
turns. Its rule: a word receives the single speaker whose turns overlap it for the longest total time, summed over
that speaker's turns; equal totals go to the speaker whose overlapping turn starts earliest; a word overlapping no
turn (including a zero-length word) receives no speaker and is scored as misattributed. These cases pin that rule
down, so an upgrade that changed it would be caught.

USAGE (osce venv, which holds whisperx):  python test_assignment.py
Exit status 0 if every case passes, 1 otherwise.
"""
import copy
import sys

import pandas as pd
from whisperx.diarize import assign_word_speakers

CASES = []


def case(name, got, want):
    CASES.append((name, got, want, got == want))


def assign(turns, words):
    """turns: [(start, end, speaker)]; words: [(start, end)] -> list of speaker labels (None if unassigned)."""
    diar = pd.DataFrame([dict(start=s, end=e, speaker=k) for s, e, k in turns])
    seg = dict(start=min(w[0] for w in words), end=max(w[1] for w in words),
               words=[dict(word=f"w{i}", start=s, end=e) for i, (s, e) in enumerate(words)])
    out = assign_word_speakers(diar, {"segments": [seg]})
    return [w.get("speaker") for w in out["segments"][0]["words"]]


def main():
    A, B = "SPEAKER_00", "SPEAKER_01"
    two = [(0.0, 5.0, A), (5.0, 10.0, B)]

    case("A01 word inside one turn", assign(two, [(1.0, 1.5)]), [A])
    case("A02 word inside the second turn", assign(two, [(6.0, 6.4)]), [B])
    case("A03 boundary-spanning word, larger overlap wins", assign(two, [(4.8, 5.6)]), [B])
    case("A04 boundary-spanning word, other side larger", assign(two, [(4.2, 5.3)]), [A])
    case("A05 equal overlap tie -> earlier-starting turn", assign(two, [(4.5, 5.5)]), [A])
    case("A06 tie independent of diarization row order",
         assign(list(reversed(two)), [(4.5, 5.5)]), [A])
    case("A07 word in a gap between turns is unassigned",
         assign([(0.0, 2.0, A), (3.0, 5.0, B)], [(2.2, 2.8)]), [None])
    case("A08 word after the last turn is unassigned", assign(two, [(10.5, 11.0)]), [None])
    case("A09 zero-length word is unassigned", assign(two, [(3.0, 3.0)]), [None])
    case("A10 overlap summed across a speaker's turns",
         assign([(0.0, 1.0, A), (1.0, 1.5, B), (1.5, 2.2, A)], [(0.6, 2.0)]), [A])
    case("A11 overlapping turns: longer intersection wins",
         assign([(0.0, 4.0, A), (3.0, 8.0, B)], [(3.2, 4.6)]), [B])

    words = [(0.2, 0.6), (4.8, 5.6), (7.0, 7.4), (9.9, 10.4), (12.0, 12.5)]
    labels = assign(two, words)
    case("A12 exactly one label per word (no duplicates, none lost)", len(labels), len(words))
    case("A13 assignment is deterministic on repeat",
         assign(copy.deepcopy(two), copy.deepcopy(words)), labels)

    width = max(len(n) for n, _, _, _ in CASES)
    failed = 0
    for name, got, want, ok in CASES:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}" + ("" if ok else f"   got {got!r}, want {want!r}"))
        failed += not ok
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
