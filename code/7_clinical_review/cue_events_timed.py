"""cue_events_all.py events plus the audio time of each inserted word/passage, for audio-backed review."""
import csv, json, os, sys
from pathlib import Path
sys.path.insert(0, "/mnt/d/fareez_optuna/code/final"); sys.path.insert(0, "/mnt/d/fareez_optuna/code")
from metrics import AFF_CUES, HALLUCINATION_RUN, NEG_CUES, align, norm_main
from med_lexicon import MED_CUES
from cue_events_all import CORPORA, reference_turns, hypothesis_turns, turn_at

CONFIGS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["primock_tuned"]

def hyp_times(path):
    d = json.loads(Path(path).read_text()); t = []; HW.clear(); T2W.clear()
    for seg in d["segments"]:
        for w in seg.get("words", []):
            n = len(norm_main(w["word"]))
            t += [(w["start"], w["end"])] * n; T2W.extend([len(HW)] * n); HW.append(w["word"])
    return t
HW, T2W = [], []

def aligned_text(ops, s, e):
    """ASR words aligned to reference tokens s..e (insertions attach to the preceding reference token)."""
    ri = hi = 0; toks = []
    for op, r, h in ops:
        if op in ("equal", "substitute"):
            for k in range(len(h)):
                if s <= ri + min(k, len(r) - 1) <= e: toks.append(hi + k)
        elif op == "insert":
            if s <= ri - 1 <= e or (ri == s and s == 0): toks.extend(range(hi, hi + len(h)))
        if op in ("equal", "substitute", "delete"): ri += len(r)
        if op in ("equal", "substitute", "insert"): hi += len(h)
    if not toks: return ""
    w0, w1 = T2W[min(toks)], T2W[max(toks)]
    return "".join(HW[w0:w1 + 1]).strip()

def ctx(rt, ti):
    g = lambda j, k: rt[j][k] if 0 <= j < len(rt) else ""
    return dict(speaker_role=rt[ti][0], reference_utterance=rt[ti][1],
                previous_reference_utterance_2=g(ti-2,1), previous_speaker_role_2=g(ti-2,0),
                previous_reference_utterance=g(ti-1,1), previous_speaker_role=g(ti-1,0),
                following_reference_utterance=g(ti+1,1), following_speaker_role=g(ti+1,0))

rows = []
for CONFIG in CONFIGS:
 for corpus, spec in CORPORA.items():
     if os.environ.get("CORPORA") and corpus not in os.environ["CORPORA"].split(","): continue
     d = spec["out"] / CONFIG
     for p in sorted(d.glob("*.json")):
         rec = p.stem
         if ".failed" in rec: continue
         if spec["filt"] and not spec["filt"](rec): continue
         rt = reference_turns(rec, spec["cache"]); ht = hypothesis_turns(p); tm = hyp_times(p)
         if not rt: continue
         rw = [w for _, t, _, _ in rt for w in norm_main(t)]; hw = [w for _, t, _, _ in ht for w in norm_main(t)]
         _, ops = align(rw, hw); ri = hi = 0
         for op, r, h in ops:
             if op == "insert":
                 ti = turn_at(rt, max(0, ri-1)); hj = turn_at(ht, hi)
                 if ti is not None:
                     hu = aligned_text(ops, rt[ti][2], rt[ti][3]) or (ht[hj][1] if hj is not None else " ".join(h))
                     for k, tok in enumerate(h):
                         kind = ("inserted negation" if tok in NEG_CUES else "inserted affirmation" if tok in AFF_CUES
                                 else "inserted medication" if tok in MED_CUES else None)
                         if kind and hi+k < len(tm):
                             rows.append(dict(corpus=corpus, recording=rec, config=CONFIG, cue_class=kind,
                                 reference_term="(not said)", hypothesis_term=tok, hypothesis_utterance=hu,
                                 t_start=tm[hi+k][0], t_end=tm[hi+k][1], **ctx(rt, ti)))
                     if len(h) >= HALLUCINATION_RUN and hi+len(h)-1 < len(tm):
                         rows.append(dict(corpus=corpus, recording=rec, config=CONFIG, cue_class="hallucinated",
                             reference_term="(nothing said)", hypothesis_term=" ".join(h[:14]), hypothesis_utterance=hu,
                             t_start=tm[hi][0], t_end=tm[hi+len(h)-1][1], **ctx(rt, ti)))
             if op in ("delete", "substitute"):
                 for k, tok in enumerate(r):
                     cue = ("negation" if tok in NEG_CUES else "affirmation" if tok in AFF_CUES
                            else "medication" if tok in MED_CUES else None)
                     if not cue: continue
                     if op == "substitute" and k < len(h) and ((tok in NEG_CUES and h[k] in AFF_CUES) or (tok in AFF_CUES and h[k] in NEG_CUES)):
                         cue = "polarity flip"
                     ti = turn_at(rt, ri + k)
                     if ti is None or not tm: continue
                     if op == "substitute" and k < len(h) and hi + k < len(tm):
                         a, b = tm[hi + k]
                     else:
                         a = tm[hi - 1][1] if hi > 0 else 0.0
                         b = tm[hi][0] if hi < len(tm) else a + 1.0
                     rows.append(dict(corpus=corpus, recording=rec, config=CONFIG, cue_class=cue,
                         reference_term=tok, hypothesis_term=(h[k] if op == "substitute" and k < len(h) else ""),
                         hypothesis_utterance=aligned_text(ops, rt[ti][2], rt[ti][3]),
                         t_start=a, t_end=b, **ctx(rt, ti)))
             if op in ("equal","substitute","delete"): ri += len(r)
             if op in ("equal","substitute","insert"): hi += len(h)
import os
out = Path(os.environ.get("OUT", "/mnt/d/fareez_optuna/adjudication/events_inserted_timed.csv"))
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "events ->", out)
