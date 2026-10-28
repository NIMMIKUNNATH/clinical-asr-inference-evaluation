#!/usr/bin/env python3
"""
cue_events.py - emit one row per flagged clinical-cue event, with the dialogue around it.

The adjudication sheet a clinician can actually read needs the turn the cue sits in, the turns
either side, and who said each one: a bare "no" answering "have you had chest pain?" negates
chest pain, but the governed concept is in the previous turn and spoken by the other party.

Reference turns come from the corrected transcript, so they keep their punctuation and casing.
Hypothesis turns are rebuilt from the stored word-level output by grouping consecutive words
with the same diarization label, then mapping the two labels to clinician and patient by
which one the reference says spoke first.

Emits the schema sample_adjudication.py expects.

USAGE (WSL, eval venv):
    python cue_events.py --configs defaults tuned --out /mnt/d/fareez_optuna/adjudication/events.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import HALLUCINATION_RUN, NEG_CUES, align, norm_main  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import REF_DIR, ROOT, parse_reference  # noqa: E402

FINAL = ROOT / "final"
NUMBERS = set("zero one two three four five six seven eight nine ten eleven twelve twenty thirty forty "
              "fifty sixty seventy eighty ninety hundred half".split())


def is_number(w):
    return w in NUMBERS or any(c.isdigit() for c in w)


def reference_turns(rec):
    """(role, raw_text, first_word_index, last_word_index) over the normalised word stream."""
    turns, i = [], 0
    for role, text in parse_reference(REF_DIR / f"{rec}.txt"):
        words = norm_main(text)
        if not words:
            continue
        turns.append((role, text.strip(), i, i + len(words) - 1))
        i += len(words)
    return turns


def hypothesis_turns(path):
    """Group the stored word output into speaker turns, keeping raw text."""
    data = json.loads(Path(path).read_text())
    turns, cur, spk = [], [], None
    idx = 0
    for seg in data["segments"]:
        for w in seg.get("words", []):
            s = w.get("speaker")
            if s != spk and cur:
                turns.append([spk, "".join(x[0] for x in cur).strip(), cur[0][1], cur[-1][1]])
                cur = []
            spk = s
            tok = norm_main(w["word"])
            cur.append((w["word"], idx))
            idx += len(tok)
    if cur:
        turns.append([spk, "".join(x[0] for x in cur).strip(), cur[0][1], cur[-1][1]])
    return turns


def turn_at(turns, i):
    for k, t in enumerate(turns):
        if t[2] <= i <= t[3]:
            return k
    return None


def events_for(rec, config):
    p = FINAL / config / f"{rec}.json"
    if not p.exists():
        return []
    rt = reference_turns(rec)
    ht = hypothesis_turns(p)
    ref_words = [w for _, text, _, _ in rt for w in norm_main(text)]
    hyp_words = []
    for _, text, _, _ in ht:
        hyp_words.extend(norm_main(text))
    _, ops = align(ref_words, hyp_words)

    # map the two diarization labels onto roles: whoever speaks first in the reference
    labels = [t[0] for t in ht if t[0]]
    first_ref_role = rt[0][0] if rt else "Doctor"
    order, seen = [], set()
    for l in labels:
        if l not in seen:
            seen.add(l); order.append(l)
    role_of = {}
    if order:
        role_of[order[0]] = first_ref_role
        if len(order) > 1:
            role_of[order[1]] = "Patient" if first_ref_role == "Doctor" else "Doctor"

    out, ri, hi = [], 0, 0
    for op, r, h in ops:
        for k, tok in enumerate(r):
            cue = ("negation" if tok in NEG_CUES else "number" if is_number(tok) else None)
            if cue and op in ("delete", "substitute"):
                ti = turn_at(rt, ri + k)
                if ti is None:
                    continue
                hj = turn_at(ht, hi) if op == "substitute" else None
                out.append(dict(
                    recording=rec, config=config, cue_class=cue,
                    speaker_role=rt[ti][0],
                    reference_term=tok,
                    hypothesis_term=(h[k] if op == "substitute" and k < len(h) else ""),
                    reference_utterance=rt[ti][1],
                    previous_reference_utterance_2=rt[ti - 2][1] if ti >= 2 else "",
                    previous_speaker_role_2=rt[ti - 2][0] if ti >= 2 else "",
                    previous_reference_utterance=rt[ti - 1][1] if ti >= 1 else "",
                    previous_speaker_role=rt[ti - 1][0] if ti >= 1 else "",
                    following_reference_utterance=rt[ti + 1][1] if ti + 1 < len(rt) else "",
                    following_speaker_role=rt[ti + 1][0] if ti + 1 < len(rt) else "",
                    hypothesis_utterance=(ht[hj][1] if hj is not None else
                                          (ht[turn_at(ht, hi)][1] if turn_at(ht, hi) is not None else "")),
                ))
        # fabricated passages: a run of consecutive inserted words has no reference cue, so the
        # reviewer judges the surrounding dialogue against text the system produced over silence.
        if op == "insert" and len(h) >= HALLUCINATION_RUN:
            ti = turn_at(rt, max(0, ri - 1))
            hj = turn_at(ht, hi)
            if ti is not None:
                out.append(dict(
                    recording=rec, config=config, cue_class="fabricated",
                    speaker_role=rt[ti][0],
                    reference_term="(nothing said)",
                    hypothesis_term=" ".join(h[:14]),
                    reference_utterance=rt[ti][1],
                    previous_reference_utterance_2=rt[ti - 2][1] if ti >= 2 else "",
                    previous_speaker_role_2=rt[ti - 2][0] if ti >= 2 else "",
                    previous_reference_utterance=rt[ti - 1][1] if ti >= 1 else "",
                    previous_speaker_role=rt[ti - 1][0] if ti >= 1 else "",
                    following_reference_utterance=rt[ti + 1][1] if ti + 1 < len(rt) else "",
                    following_speaker_role=rt[ti + 1][0] if ti + 1 < len(rt) else "",
                    hypothesis_utterance=ht[hj][1] if hj is not None else " ".join(h),
                ))
        ri += len(r)
        hi += len(h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["defaults", "tuned"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for cfg in args.configs:
        recs = sorted(p.stem for p in (FINAL / cfg).glob("*.json") if not p.name.endswith(".failed.json"))
        for rec in recs:
            rows.extend(events_for(rec, cfg))
        print(f"  {cfg}: {len(rows)} cumulative events")
    if not rows:
        sys.exit("no events")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out, len(rows), "events")


if __name__ == "__main__":
    main()
