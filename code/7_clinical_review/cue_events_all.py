"""Flagged clinical-cue events across all three corpora, under the deployed joint configuration.

Generalises cue_events.py, which handled Fareez only. Fareez reference turns come from the corrected
transcripts; PriMock57 and AfriSpeech-Dialog from their cached turn JSON. AfriSpeech is restricted to
the 20 clinical (MED) consultations, the held-out set the paper reports.

USAGE:  python cue_events_all.py --config primock_tuned --out events_all.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/d/fareez_optuna/code/final")
sys.path.insert(0, "/mnt/d/fareez_optuna/code")

from metrics import AFF_CUES, HALLUCINATION_RUN, NEG_CUES, align, norm_main   # noqa: E402
from med_lexicon import MED_CUES                                              # noqa: E402
from common import REF_DIR, ROOT, parse_reference                   # noqa: E402

NUMBERS = set("zero one two three four five six seven eight nine ten eleven twelve twenty thirty forty "
              "fifty sixty seventy eighty ninety hundred half".split())

PM_HELDOUT = set(json.loads((ROOT / "results_primock" / "dev_split.json").read_text())["heldout"])

CORPORA = {
    "Fareez":            dict(out=ROOT / "final", cache=None, filt=None),
    "PriMock57":         dict(out=ROOT / "final_primock", cache=ROOT / "cache" / "primock",
                              filt=lambda r: r in PM_HELDOUT),
    "AfriSpeech-Dialog": dict(out=ROOT / "final_afrispeech", cache=ROOT / "cache" / "afrispeech",
                              filt=lambda r: r.startswith("MED")),
}


def is_number(w):
    return w in NUMBERS or any(c.isdigit() for c in w)


def raw_turns(rec, cache):
    """[(role, text)] in spoken order, punctuation preserved."""
    if cache is None:
        return [(role, text) for role, text in parse_reference(REF_DIR / "{}.txt".format(rec))]
    data = json.loads((cache / "ref" / "{}.json".format(rec)).read_text(encoding="utf-8"))
    turns = data["turns"] if isinstance(data, dict) else data
    if all("start" in t for t in turns):
        turns = sorted(turns, key=lambda t: float(t["start"]))
    return [(t["speaker"], t["text"]) for t in turns if t.get("text", "").strip()]


def reference_turns(rec, cache):
    out, i = [], 0
    for role, text in raw_turns(rec, cache):
        words = norm_main(text)
        if not words:
            continue
        out.append((role, text.strip(), i, i + len(words) - 1))
        i += len(words)
    return out


def hypothesis_turns(path):
    data = json.loads(Path(path).read_text())
    turns, cur, spk, idx = [], [], None, 0
    for seg in data["segments"]:
        for w in seg.get("words", []):
            s = w.get("speaker")
            if s != spk and cur:
                turns.append([spk, "".join(x[0] for x in cur).strip(), cur[0][1], cur[-1][1]])
                cur = []
            spk = s
            cur.append((w["word"], idx))
            idx += len(norm_main(w["word"]))
    if cur:
        turns.append([spk, "".join(x[0] for x in cur).strip(), cur[0][1], cur[-1][1]])
    return turns


def turn_at(turns, i):
    for k, t in enumerate(turns):
        if t[2] <= i <= t[3]:
            return k
    return None


def events_for(corpus, rec, config):
    spec = CORPORA[corpus]
    p = spec["out"] / config / "{}.json".format(rec)
    if not p.exists():
        return []
    rt = reference_turns(rec, spec["cache"])
    ht = hypothesis_turns(p)
    if not rt:
        return []
    ref_words = [w for _, text, _, _ in rt for w in norm_main(text)]
    hyp_words = [w for _, text, _, _ in ht for w in norm_main(text)]
    _, ops = align(ref_words, hyp_words)

    out, ri, hi = [], 0, 0
    for op, r, h in ops:
        for k, tok in enumerate(r):
            cue = ("negation" if tok in NEG_CUES else "affirmation" if tok in AFF_CUES
                   else "medication" if tok in MED_CUES
                   else "number" if is_number(tok) else None)
            if cue and op in ("delete", "substitute"):
                # a negation replaced by an affirmation, or the reverse, is a polarity flip
                if op == "substitute" and k < len(h):
                    if (tok in NEG_CUES and h[k] in AFF_CUES) or (tok in AFF_CUES and h[k] in NEG_CUES):
                        cue = "polarity flip"
                ti = turn_at(rt, ri + k)
                if ti is None:
                    continue
                hj = turn_at(ht, hi) if op == "substitute" else None
                out.append(dict(
                    corpus=corpus, recording=rec, config=config, cue_class=cue,
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
        # Cues the recogniser invented. These are the i_C term of the insertion-inclusive cue rate,
        # the paper's primary cue outcome, and the part the earlier review never covered.
        if op == "insert":
            ti = turn_at(rt, max(0, ri - 1))
            hj = turn_at(ht, hi)
            for k, tok in enumerate(h):
                kind = ("inserted negation" if tok in NEG_CUES else
                        "inserted affirmation" if tok in AFF_CUES else
                        "inserted medication" if tok in MED_CUES else None)
                if not kind or ti is None:
                    continue
                out.append(dict(
                    corpus=corpus, recording=rec, config=config, cue_class=kind,
                    speaker_role=rt[ti][0],
                    reference_term="(not said)",
                    hypothesis_term=tok,
                    reference_utterance=rt[ti][1],
                    previous_reference_utterance_2=rt[ti - 2][1] if ti >= 2 else "",
                    previous_speaker_role_2=rt[ti - 2][0] if ti >= 2 else "",
                    previous_reference_utterance=rt[ti - 1][1] if ti >= 1 else "",
                    previous_speaker_role=rt[ti - 1][0] if ti >= 1 else "",
                    following_reference_utterance=rt[ti + 1][1] if ti + 1 < len(rt) else "",
                    following_speaker_role=rt[ti + 1][0] if ti + 1 < len(rt) else "",
                    hypothesis_utterance=(ht[hj][1] if hj is not None else " ".join(h)),
                ))
        if op == "insert" and len(h) >= HALLUCINATION_RUN:
            ti = turn_at(rt, max(0, ri - 1))
            hj = turn_at(ht, hi)
            if ti is not None:
                out.append(dict(
                    corpus=corpus, recording=rec, config=config, cue_class="hallucinated",
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
                    hypothesis_utterance=(ht[hj][1] if hj is not None else ""),
                ))
        if op in ("equal", "substitute", "delete"):
            ri += len(r)
        if op in ("equal", "substitute", "insert"):
            hi += len(h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="primock_tuned")
    ap.add_argument("--out", default="/mnt/d/fareez_optuna/adjudication/events_all_corpora.csv")
    a = ap.parse_args()

    rows = []
    for corpus, spec in CORPORA.items():
        d = spec["out"] / a.config
        if not d.exists():
            print("  {:<18} no output directory for {}".format(corpus, a.config))
            continue
        recs = sorted(p.stem for p in d.glob("*.json") if not p.name.endswith(".failed.json"))
        if spec["filt"]:
            recs = [r for r in recs if spec["filt"](r)]
        n0 = len(rows)
        for rec in recs:
            rows.extend(events_for(corpus, rec, a.config))
        print("  {:<18} {:3d} recordings -> {:4d} events".format(corpus, len(recs), len(rows) - n0))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\ntotal {} events -> {}".format(len(rows), out))
    from collections import Counter
    print("by corpus:", dict(Counter(r["corpus"] for r in rows)))
    print("by cue   :", dict(Counter(r["cue_class"] for r in rows)))


if __name__ == "__main__":
    main()
