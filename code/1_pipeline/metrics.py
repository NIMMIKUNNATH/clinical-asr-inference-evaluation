"""
metrics.py - score saved final_run.py outputs with every metric used in the clinical-conversation ASR literature.
Runs in the evaluation venv (~/eval_venv: meeteval, jiwer, whisper-normalizer, scipy, pandas). CPU only.

Per recording it writes one row with the COUNTS behind each metric, so report_final.py can pool them
(sum of errors / sum of reference units) and bootstrap over recordings:

  WER (main)          normalised as in the search, plus every spelling of affirmative/negative non-lexical sounds
                      (mhm, mhmm, mm-hmm, uh-huh, uh-uh, mm-mm) and fillers removed on both sides
  WER (non-lexical)   sensitivity analysis (Tran et al. 2023): affirmative sounds kept as 'mmhm', negative as 'uhuh'
  CER                 characters of the main-normalised text (supplementary; Radford et al. 2023)
  WDER                El Shafey et al. 2019: aligned words (hits + substitutions) whose speaker is wrong; words
                      WhisperX left without a speaker count as wrong
  cpWER               Watanabe et al. 2020, via MeetEval: per-speaker streams, best speaker permutation
  SA-WER              Kanda et al. 2020 style, fixed roles by a deployable rule: the first speaker to talk is the
                      doctor (true in 264/271 references)
  doctor/patient WER  Adedeji et al. 2024/2025: WER of each role's stream under the cpWER assignment
  polarity cues       exploratory: errors on negation cues (no, not, never, ..., uhuh) and affirmation cues
                      (yes, yeah, yep, yup, mmhm), on the non-lexical-kept alignment
  hallucination       runs of >= 5 consecutive inserted words (Koenecke et al. 2024 motivation)
  MC-WER              Adedeji et al. 2025: filled in when concept files exist (concepts.py, Google Healthcare NL)
  RTF                 decode seconds / audio seconds

USAGE:  ~/eval_venv/bin/python metrics.py --config tuned [--set heldout]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import jiwer
import pandas as pd
from meeteval.wer import cp_word_error_rate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import REF_DIR, ROOT, normalize, parse_reference  # noqa: E402

FINAL = ROOT / "final"
CONCEPTS = FINAL / "concepts"

_AFFIRM = re.compile(r"\b(?:m+-?h+m+|mhm+|mm+[- ]?hm+|m+h+m+|uh[- ]?huh)\b", re.I)
_NEG_NL = re.compile(r"\b(?:uh-uh|mm-mm|huh-uh|nuh-uh)\b", re.I)
_FILLER = re.compile(r"\b(?:hmm+|m{3,}|uhm+|umm+|uhh+|erm+)\b", re.I)

NEG_CUES = {"no", "not", "never", "none", "nothing", "nobody", "nowhere", "neither", "nor", "deny", "denies",
            "denied", "without", "negative", "nope", "uhuh"}
AFF_CUES = {"yes", "yeah", "yep", "yup", "mmhm"}
HALLUCINATION_RUN = 5


def norm_main(text):
    return normalize(_FILLER.sub(" ", _NEG_NL.sub(" ", _AFFIRM.sub(" ", text))))


def norm_nonlex(text):
    return normalize(_FILLER.sub(" ", _NEG_NL.sub(" uhuh ", _AFFIRM.sub(" mmhm ", text))))


def align(ref, hyp):
    """jiwer alignment -> (counts, list of (op, ref_tokens, hyp_tokens))"""
    if not ref and not hyp:
        return dict(sub=0, dele=0, ins=0, hit=0, n=0), []
    if not hyp:
        return dict(sub=0, dele=len(ref), ins=0, hit=0, n=len(ref)), [("delete", ref, [])]
    if not ref:
        return dict(sub=0, dele=0, ins=len(hyp), hit=0, n=0), [("insert", [], hyp)]
    o = jiwer.process_words(" ".join(ref), " ".join(hyp))
    ops = [(c.type, ref[c.ref_start_idx:c.ref_end_idx], hyp[c.hyp_start_idx:c.hyp_end_idx])
           for c in o.alignments[0]]
    return dict(sub=o.substitutions, dele=o.deletions, ins=o.insertions, hit=o.hits, n=len(ref)), ops


def word_runs(segments):
    """Consecutive words with the same WhisperX speaker -> [(speaker, text)]; unassigned words -> 'UNKNOWN'."""
    runs = []
    for s in segments:
        for w in s.get("words", []):
            spk = w.get("speaker", "UNKNOWN")
            if runs and runs[-1][0] == spk:
                runs[-1][1].append(w["word"])
            else:
                runs.append([spk, [w["word"]]])
    return [(spk, "".join(ws)) for spk, ws in runs]


def polarity(ops):
    c = dict(neg_ref=0, neg_err=0, neg_ins=0, aff_ref=0, aff_err=0, aff_ins=0, flips=0)
    for op, r, h in ops:
        for i, tok in enumerate(r):
            for cues, key, other in ((NEG_CUES, "neg", AFF_CUES), (AFF_CUES, "aff", NEG_CUES)):
                if tok in cues:
                    c[f"{key}_ref"] += 1
                    if op in ("delete", "substitute"):
                        c[f"{key}_err"] += 1
                        if op == "substitute" and i < len(h) and h[i] in other:
                            c["flips"] += 1
        if op == "insert":
            c["neg_ins"] += sum(t in NEG_CUES for t in h)
            c["aff_ins"] += sum(t in AFF_CUES for t in h)
    return c


def mc_counts(config, rec):
    """Medical-concept S/D/I/M, concepts aligned as whole units (Adedeji et al. 2025). None until concepts exist."""
    rp, hp = CONCEPTS / "reference" / f"{rec}.json", CONCEPTS / config / f"{rec}.json"
    if not (rp.exists() and hp.exists()):
        return dict(mc_sub=None, mc_del=None, mc_ins=None, mc_n=None)
    ref = [c["unit"] for c in json.loads(rp.read_text())["concepts"]]
    hyp = [c["unit"] for c in json.loads(hp.read_text())["concepts"]]
    # map every distinct concept string to a single placeholder token so multi-word concepts align as one unit
    vocab = {u: f"c{i}" for i, u in enumerate(sorted(set(ref) | set(hyp)))}
    counts, _ = align([vocab[u] for u in ref], [vocab[u] for u in hyp])
    return dict(mc_sub=counts["sub"], mc_del=counts["dele"], mc_ins=counts["ins"], mc_n=counts["n"])


def score(path, turns=None):
    """Score one saved run output. `turns` lets another corpus supply its own [(speaker, text)] reference;
    when omitted the Fareez transcripts are read, which is what every Fareez caller relies on."""
    out = json.loads(Path(path).read_text(encoding="utf-8"))
    rec, segs = out["recording"], out["segments"]
    if turns is None:
        turns = parse_reference(REF_DIR / f"{rec}.txt")

    # --- WER, main and non-lexical sensitivity (reference per turn, hypothesis per segment) -------------------
    ref_main = [t for _, x in turns for t in norm_main(x)]
    hyp_main = [t for s in segs for t in norm_main(s["text"])]
    wm, ops_main = align(ref_main, hyp_main)
    ref_nl = [t for _, x in turns for t in norm_nonlex(x)]
    hyp_nl = [t for s in segs for t in norm_nonlex(s["text"])]
    wn, ops_nl = align(ref_nl, hyp_nl)

    # --- CER ---------------------------------------------------------------------------------------------------
    rc, hc = " ".join(ref_main), " ".join(hyp_main)
    co = jiwer.process_characters(rc, hc) if hc else None
    cer = dict(cer_err=(co.substitutions + co.deletions + co.insertions) if co else len(rc), cer_n=len(rc))

    # --- speaker-attributed metrics ------------------------------------------------------------------------------
    ref_tagged = [(t, role) for role, x in turns for t in norm_main(x)]
    runs = word_runs(segs)
    hyp_tagged = [(t, spk) for spk, x in runs for t in norm_main(x)]

    # WDER (El Shafey et al. 2019) with the better of the two speaker->role mappings
    _, ops_w = align([t for t, _ in ref_tagged], [t for t, _ in hyp_tagged])
    pairs, ri, hi = [], 0, 0
    for op, r, h in ops_w:
        if op in ("equal", "substitute"):
            pairs += [(ref_tagged[ri + k][1], hyp_tagged[hi + k][1]) for k in range(len(r))]
        ri += len(r)
        hi += len(h)
    spk = sorted({h for _, h in pairs if h != "UNKNOWN"})
    maps = [{}] if not spk else [{spk[0]: "Doctor", **({spk[1]: "Patient"} if len(spk) > 1 else {})},
                                 {spk[0]: "Patient", **({spk[1]: "Doctor"} if len(spk) > 1 else {})}]
    wder_wrong = min(sum(m.get(h) != r for r, h in pairs) for m in maps) if pairs else 0

    # per-speaker streams
    ref_streams = {"Doctor": [], "Patient": []}
    for t, role in ref_tagged:
        ref_streams[role].append(t)
    hyp_streams = {}
    for t, s in hyp_tagged:
        hyp_streams.setdefault(s, []).append(t)

    cp = cp_word_error_rate({k: " ".join(v) for k, v in ref_streams.items()},
                            {k: " ".join(v) for k, v in hyp_streams.items()})
    assign = dict(cp.assignment)
    role_wer = {}
    for role in ("Doctor", "Patient"):
        c, _ = align(ref_streams[role], hyp_streams.get(assign.get(role), []))
        role_wer[role] = c

    # SA-WER: fixed roles, first speaker to talk = doctor
    first = next((s for s, _ in runs if s != "UNKNOWN"), None)
    others = [s for s in hyp_streams if s not in (first, "UNKNOWN")]
    sa_map = {"Doctor": first, "Patient": others[0] if others else None}
    sa_err = 0
    for role in ("Doctor", "Patient"):
        c, _ = align(ref_streams[role], hyp_streams.get(sa_map[role], []))
        sa_err += c["sub"] + c["dele"] + c["ins"]
    used = {v for v in sa_map.values() if v}
    sa_err += sum(len(v) for k, v in hyp_streams.items() if k not in used)   # unmapped speech = insertions
    first_role_correct = assign.get("Doctor") == first

    # --- polarity cues and hallucination runs ---------------------------------------------------------------
    pol = polarity(ops_nl)
    runs_ins = [len(h) for op, r, h in ops_main if op == "insert" and len(h) >= HALLUCINATION_RUN]

    # Fareez identifiers start with a specialty code (CAR0001 -> CAR); other corpora use their own scheme
    _cat = re.match(r"[A-Z]+", rec)
    return dict(recording=rec, category=_cat.group() if _cat else rec.split("_")[0], config=out["config"],
                duration_s=round(out["duration_s"], 1), decode_s=round(out["decode_seconds"], 1),
                ref_words=wm["n"], hyp_words=len(hyp_main),
                sub=wm["sub"], dele=wm["dele"], ins=wm["ins"],
                nl_sub=wn["sub"], nl_dele=wn["dele"], nl_ins=wn["ins"], nl_n=wn["n"],
                **cer,
                wder_wrong=wder_wrong, wder_aligned=len(pairs),
                unassigned_words=sum(len(norm_main(x)) for s, x in runs if s == "UNKNOWN"),
                cp_err=cp.errors, cp_n=cp.length, cp_ins=cp.insertions, cp_del=cp.deletions, cp_sub=cp.substitutions,
                sa_err=sa_err, sa_n=len(ref_tagged), first_speaker_is_doctor=first_role_correct,
                # per-role error under both conventions (METRIC_DEFINITIONS.md 2.4). The primary convention
                # EXCLUDES insertions, because an inserted word has no reference speaker and assigning it to a
                # role is an assumption; the inclusive form is the pre-specified sensitivity analysis.
                doc_err_noins=role_wer["Doctor"]["sub"] + role_wer["Doctor"]["dele"],
                pat_err_noins=role_wer["Patient"]["sub"] + role_wer["Patient"]["dele"],
                doc_err=role_wer["Doctor"]["sub"] + role_wer["Doctor"]["dele"] + role_wer["Doctor"]["ins"],
                doc_n=role_wer["Doctor"]["n"],
                pat_err=role_wer["Patient"]["sub"] + role_wer["Patient"]["dele"] + role_wer["Patient"]["ins"],
                pat_n=role_wer["Patient"]["n"],
                **pol, halluc_runs=len(runs_ins), halluc_words=sum(runs_ins),
                **mc_counts(out["config"], rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--recordings", nargs="*", help="limit to these recordings (default: all outputs)")
    args = ap.parse_args()
    # *.failed.json marks a recording the runner could not transcribe; it holds the error, not segments
    files = sorted(p for p in (FINAL / args.config).glob("*.json") if not p.name.endswith(".failed.json"))
    if args.recordings:
        files = [f for f in files if f.stem in set(args.recordings)]
    rows = [score(f) for f in files]
    df = pd.DataFrame(rows)
    (FINAL / "scores").mkdir(parents=True, exist_ok=True)
    df.to_csv(FINAL / "scores" / f"{args.config}.csv", index=False)
    if len(df):
        print(f"{args.config}: {len(df)} recordings  WER {(df['sub'] + df.dele + df.ins).sum() / df.ref_words.sum():.4f}  "
              f"cpWER {df.cp_err.sum() / df.cp_n.sum():.4f}  WDER {df.wder_wrong.sum() / df.wder_aligned.sum():.4f}")


if __name__ == "__main__":
    main()
