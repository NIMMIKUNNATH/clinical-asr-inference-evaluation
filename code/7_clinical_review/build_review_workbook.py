"""Reviewer workbook for the three-corpus clinical-meaning review.

Same three-sheet layout, columns, dropdowns and typography as the original 180-event workbook, but
the events are sampled from the held-out sets of all three corpora under the deployed joint
configuration, and insertion-driven (hallucinated) events are included so the insertion-inclusive
cue rate can be reviewed as well.
"""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

import argparse  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--target", type=int, default=200)
_ap.add_argument("--out", default="/mnt/d/fareez_optuna/adjudication/Clinical_Review_Three_Corpora.xlsx")
_ap.add_argument("--classes", nargs="*", default=None,
                 help="restrict to these cue classes, e.g. 'inserted negation' 'inserted affirmation'")
_ap.add_argument("--quota", nargs=3, type=int, default=None, metavar=("FAREEZ", "PRIMOCK", "AFRI"))
_A = _ap.parse_args()

SRC = Path("/mnt/d/fareez_optuna/adjudication/events_all_corpora.csv")
OUT = Path(_A.out)
TARGET = _A.target           # whole review, across the three corpora
MAX_PER_RECORDING = 3        # spread the sample over as many consultations as possible
MAX_PER_PATTERN = 2          # at most two instances of the same error pattern
SEED = 20261012

HEAD_FILL = PatternFill("solid", fgColor="2D8098")      # matches the figure palette
HEAD_FONT = Font(name="Times New Roman", size=11, bold=True, color="FFFFFF")
BODY_FONT = Font(name="Times New Roman", size=11)
WRAP_TOP = Alignment(wrap_text=True, vertical="top")

COLS = [("Event ID", 11), ("Corpus", 16), ("Speaker", 12), ("Cue type", 15),
        ("Gold-standard event context", 46.8), ("Time-Based ASR", 45.5),
        ("Algorithm-detected change", 18.8), ("Is the automated flag correct?", 14),
        ("Did clinically relevant meaning change?", 12.8), ("Confidence", 13.8),
        ("Reviewer comments", 40)]


import re  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, "/mnt/d/fareez_optuna/code")
from common import normalize  # noqa: E402  (metrics.py pulls meeteval; only the normaliser is needed)


def norm_main(text):
    return normalize(text)


MAX_LINE = 300
# Simulated corpora still contain actor names; prefer events without them for the reviewer pack.
NAMEY = re.compile(r"\b(?:mr|mrs|ms|dr)\.?\s+[A-Z][a-z]+|\bmy name is\s+[A-Z][a-z]+", re.I)


def role(r):
    r = (r or "").strip().lower()
    return "DOCTOR" if r.startswith(("doc", "clin", "speaker 1")) else "PATIENT"


def mark_cue(text, term):
    """Bracket the spoken word that carries the cue.

    The flagged term is the normalised form, so “not” can live inside the spoken “don't”.
    Without this the reviewer is told a term changed and cannot find it in the transcript."""
    if not term or term in ("(nothing said)", "(not said)"):
        return text
    out, done = [], False
    for tok in re.split(r"(\s+)", text):
        core = tok.strip(".,;:!?…\"'()[]")
        if core and not done and term in norm_main(core):
            tok = tok.replace(core, "[[{}]]".format(core), 1)
            done = True
        out.append(tok)
    return "".join(out)


def clip(line):
    return line if len(line) <= MAX_LINE else line[:MAX_LINE - 1].rstrip() + "…"


def context(ev, side):
    """Four-line dialogue window; the flagged turn is marked with >>>."""
    pre2, pre1 = ev["previous_reference_utterance_2"], ev["previous_reference_utterance"]
    pre2r, pre1r = ev["previous_speaker_role_2"], ev["previous_speaker_role"]
    post, postr = ev["following_reference_utterance"], ev["following_speaker_role"]
    lines = []
    if pre2:
        lines.append(clip("    {}: {}".format(role(pre2r), pre2)))
    if pre1:
        lines.append(clip("    {}: {}".format(role(pre1r), pre1)))
    if side == "ref":
        lines.append(clip(">>> {}: {}".format(
            role(ev["speaker_role"]), mark_cue(ev["reference_utterance"], ev["reference_term"]))))
    else:
        hyp = ev["hypothesis_utterance"].strip()
        if ev["cue_class"].startswith("inserted "):
            shown = mark_cue(hyp, ev["hypothesis_term"]) if hyp else "[[{}]]".format(ev["hypothesis_term"])
            if "[[" not in shown:
                shown = "{}   <-- invented word: [[{}]]".format(hyp, ev["hypothesis_term"])
        elif ev["cue_class"] == "hallucinated":
            shown = ("{}   <-- includes inserted text: “{}”".format(hyp, ev["hypothesis_term"])
                     if hyp else "[inserted] {}".format(ev["hypothesis_term"]))
        elif not hyp:
            shown = "[dropped entirely]"
        else:
            shown = hyp
        lines.append(clip(">>> {}: {}".format(role(ev["speaker_role"]), shown)))
    if post:
        lines.append(clip("    {}: {}".format(role(postr), post)))
    return "\n".join(lines)


def change(ev):
    if ev["cue_class"].startswith("inserted "):
        return "(not said) → system added “{}”".format(ev["hypothesis_term"])
    if ev["cue_class"] == "hallucinated":
        return "(nothing said) → inserted: {}".format(ev["hypothesis_term"][:60])
    if ev["hypothesis_term"]:
        return "{} → {}".format(ev["reference_term"], ev["hypothesis_term"])
    return "{} → dropped entirely".format(ev["reference_term"])


rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
if _A.classes:
    rows = [r for r in rows if r["cue_class"] in set(_A.classes)]
    print("restricted to {}: {} candidate events".format(", ".join(_A.classes), len(rows)))
rng = random.Random(SEED)

def usable(ev):
    """Reject anything a clinician could not judge from what is on the page.

    There are far more candidates than slots, so a questionable event is replaced rather than
    shown with a caveat."""
    g, a = context(ev, "ref"), context(ev, "hyp")
    gf = next((l for l in g.split("\n") if l.startswith(">>>")), "")
    af = next((l for l in a.split("\n") if l.startswith(">>>")), "")
    if gf == af:                                   # nothing visibly different to judge
        return False
    if len(gf) < 12 or len(af) < 12:               # too little to go on
        return False
    if NAMEY.search(g) or NAMEY.search(a):         # actor names in the dialogue
        return False
    if ev["cue_class"].startswith("inserted "):
        if "[[" not in af:                         # the invented word must be visible in the ASR column
            return False
    elif ev["cue_class"] != "hallucinated" and "[[" not in gf:
        return False                               # the flagged term could not be located in the text
    if len(g) > 1100 or len(a) > 1100:
        return False
    return True


def pattern(r):
    """What a reviewer would recognise as 'the same error again'."""
    return (r["cue_class"], r["reference_term"].lower().strip(),
            r["hypothesis_term"].lower().strip()[:24])


# Round-robin over (corpus, cue class) so the review spans corpora, error types and speaker roles
# rather than filling up with whichever pattern happens to be commonest.
buckets = defaultdict(list)
for r in rows:
    buckets[(r["corpus"], r["cue_class"])].append(r)
for v in buckets.values():
    rng.shuffle(v)
    v.sort(key=lambda r: 0 if role(r["speaker_role"]) == "PATIENT" else 1)   # interleave roles below

# Per-corpus quotas so the review spans all three roughly evenly; the smallest corpus has fewer
# distinct patterns, so its caps are relaxed a step at a time until its quota is met.
QUOTA = (dict(zip(("Fareez", "PriMock57", "AfriSpeech-Dialog"), _A.quota)) if _A.quota
         else {"Fareez": 70, "PriMock57": 70, "AfriSpeech-Dialog": 60})
per_rec, per_pat, seen_role = Counter(), Counter(), Counter()
sample = []
for corpus, quota in QUOTA.items():
    keys = sorted([k for k in buckets if k[0] == corpus and buckets[k]], key=lambda k: k[1])
    got = 0
    for rec_cap, pat_cap in ((MAX_PER_RECORDING, MAX_PER_PATTERN), (4, 3), (6, 5), (99, 99)):
        cursor = {k: 0 for k in keys}
        progress = True
        while got < quota and progress:
            progress = False
            for k in keys:
                if got >= quota:
                    break
                pool = buckets[k]
                while cursor[k] < len(pool):
                    r = pool[cursor[k]]
                    cursor[k] += 1
                    if r in sample:
                        continue
                    if per_rec[(r["corpus"], r["recording"])] >= rec_cap:
                        continue
                    if per_pat[pattern(r)] >= pat_cap:
                        continue
                    if not usable(r):
                        continue
                    per_rec[(r["corpus"], r["recording"])] += 1
                    per_pat[pattern(r)] += 1
                    seen_role[role(r["speaker_role"])] += 1
                    sample.append(r)
                    got += 1
                    progress = True
                    break
        if got >= quota:
            break
    print("  {:<18} {} events".format(corpus, got))

print("distinct error patterns in sample: {}".format(len({pattern(r) for r in sample})))
print("distinct recordings in sample    : {}".format(len({(r['corpus'], r['recording']) for r in sample})))

rng.shuffle(sample)
print("sampled {} events".format(len(sample)))
print("  by corpus:", dict(Counter(r["corpus"] for r in sample)))
print("  by cue   :", dict(Counter(r["cue_class"] for r in sample)))
print("  by role  :", dict(Counter(role(r["speaker_role"]) for r in sample)))

wb = Workbook()

# ---- Instructions -------------------------------------------------------------------
ins = wb.active
ins.title = "Instructions"
ins.column_dimensions["A"].width = 52.8
ins.column_dimensions["B"].width = 96.0
L = [("Independent Clinical Review of ASR-Flagged Events", None),
     ("Reviewer details", None), ("Reviewer name", ""), ("Professional title/qualification", ""),
     ("Clinical specialty", ""), ("Years of relevant experience", ""), ("Review date", ""),
     ("Reviewer code/initials", ""), ("Purpose and procedure", None),
     ("Purpose", "Independently assess {} prespecified ASR-flagged events drawn from the held-out sets of three "
                 "clinician–patient corpora (simulated respiratory consultations, mock primary care, and "
                 "African-accented clinical conversations), transcribed with the study's recommended "
                 "configuration. Automated preliminary decisions are not shown.".format(len(sample))),
     ("Compare", "Use the Corpus, Speaker, reference context, ASR output and stated term change for each event."),
     ("How events are marked", "The turn under review begins with >>>; the lines above and below it are the "
                               "surrounding dialogue. In the gold-standard column the word carrying the flagged "
                               "term is wrapped in [[double brackets]]. Where the spoken form differs from the "
                               "dictionary form, for example “don't” carrying the term “not”, the brackets mark the "
                               "spoken word."),
     ("Decide", "Select whether the automated flag is correct and whether the transcription difference changes "
                "clinically relevant meaning."),
     ("Insufficient context", "Explain any missing context in Reviewer comments."),
     ("Decision definitions", None),
     ("Flag correct?", "Yes = the displayed change is a genuine ASR error; No = the flag is incorrect; "
                       "Partially = the flag is partly correct."),
     ("Meaning changed?", "Yes = clinically relevant meaning changed; No = meaning is adequately preserved; "
                          "Partially = meaning changed in part."),
     ("Confidence", "High = clear judgement; Moderate = some ambiguity; Low = difficult judgement or limited "
                    "context. This is reviewer confidence, not AI confidence."),
     ("Comments", "Briefly explain No, Partially or clinically important decisions."),
     ("Event types", "Deleted or substituted cue = a negation or numeric term in the reference that the system "
                     "lost or changed. Inserted passage = a run of consecutive words the system produced where "
                     "the reference has none."),
     ("Independent review", "Complete the workbook without consulting the other reviewers. Discuss disagreements "
                            "only after all reviewers submit their initial decisions. Do not add identifiable "
                            "information.")]
for i, (a, b) in enumerate(L, start=1):
    ins.cell(i, 1, a).font = Font(name="Times New Roman", size=11, bold=(b is None))
    if b is not None:
        c = ins.cell(i, 2, b)
        c.font = BODY_FONT
        c.alignment = WRAP_TOP

# ---- Comparison ---------------------------------------------------------------------
ws = wb.create_sheet("ASR Comparison - {}".format(len(sample)))
for j, (name, width) in enumerate(COLS, start=1):
    c = ws.cell(1, j, name)
    c.font, c.fill, c.alignment = HEAD_FONT, HEAD_FILL, Alignment(wrap_text=True, vertical="center",
                                                                  horizontal="center")
    ws.column_dimensions[get_column_letter(j)].width = width
ws.row_dimensions[1].height = 75.75
ws.freeze_panes = "E2"

CUE_LABEL = {"negation": "negation cue", "affirmation": "affirmation cue",
             "number": "numeric term", "hallucinated": "inserted passage",
             "inserted negation": "invented negation", "inserted affirmation": "invented affirmation",
             "medication": "medication term", "inserted medication": "invented medication",
             "polarity flip": "polarity flip"}
for i, ev in enumerate(sample, start=2):
    vals = ["E{:04d}".format(i - 1), ev["corpus"], role(ev["speaker_role"]),
            CUE_LABEL.get(ev["cue_class"], ev["cue_class"]),
            context(ev, "ref"), context(ev, "hyp"), change(ev), "", "", "", ""]
    for j, v in enumerate(vals, start=1):
        c = ws.cell(i, j, v)
        c.font, c.alignment = BODY_FONT, WRAP_TOP
    ws.row_dimensions[i].height = 82

n = len(sample) + 1
dv1 = DataValidation(type="list", formula1='"Yes,No,Partially"', allow_blank=True)
dv2 = DataValidation(type="list", formula1='"High,Moderate,Low"', allow_blank=True)
ws.add_data_validation(dv1); ws.add_data_validation(dv2)
dv1.add("H2:I{}".format(n))
dv2.add("J2:J{}".format(n))

# ---- Sign-off -----------------------------------------------------------------------
so = wb.create_sheet("Reviewer Sign-off")
for col, w in (("A", 76.0), ("B", 32.0), ("C", 14.0)):
    so.column_dimensions[col].width = w
rows_so = [("Reviewer Sign-off and Completion Check", None, None),
           ("Reviewer name", "", None), ("Professional title/qualification", "", None),
           ("Clinical specialty", "", None), ("Years of relevant experience", "", None),
           ("Institution (optional)", "", None), ("Completion date", "", None),
           ("Signature or initials", "", None),
           ("Completion summary", "Completed", "Required"),
           ("Flag decisions", "", len(sample)), ("Meaning-change decisions", "", len(sample)),
           ("Confidence ratings", "", len(sample)),
           ("Reviewer declaration", "Response", None),
           ("I completed this review independently.", "", None),
           ("I did not consult another reviewer before completing my initial decisions.", "", None),
           ("I noted insufficient context in Reviewer comments.", "", None),
           ("I understand that disagreements will be considered only after all independent reviews are complete.",
            "", None),
           ("General comments", "", None)]
for i, (a, b, c) in enumerate(rows_so, start=1):
    so.cell(i, 1, a).font = Font(name="Times New Roman", size=11, bold=(b is None or b in ("Completed", "Response")))
    if b not in (None,):
        so.cell(i, 2, b).font = BODY_FONT
    if c is not None:
        so.cell(i, 3, c).font = BODY_FONT

OUT.parent.mkdir(parents=True, exist_ok=True)
wb.save(OUT)
print("\nwrote", OUT)

# Manifest of the events this round used, so a later round can draw from the reserve without repeats.
KEY = ("corpus", "recording", "cue_class", "reference_term", "hypothesis_term", "reference_utterance")
man = OUT.with_name(OUT.stem + "_manifest.csv")
with man.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(("event_id",) + KEY)
    for i, e in enumerate(sample, start=1):
        w.writerow(("E{:04d}".format(i),) + tuple(e[k] for k in KEY))
print("manifest ->", man)
