"""Physician review workbook for speaker-attribution errors (PriMock57 held-out, jointly selected configuration).

Candidates: runs of >= 3 consecutive correctly recognised words that the ASR pipeline assigned to the wrong speaker
and that contain a clinical keyword (same rule as find_attribution_examples.py). Layout, fonts, dropdowns, audio-clip
links, Instructions and Reviewer Sign-off sheets follow the earlier physician workbook (Clinical_Review_ASR_Events.xlsx).
Writes, into speaker_attribution_review/:
  Speaker_Attribution_Review.xlsx   reviewer workbook (no configuration shown)
  clips/A0001.wav ...               ~6-10 s audio around each event
  _KEY_not_for_reviewers.csv        recording, configuration, word indices and times
"""
import csv
import difflib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

D = Path("/mnt/d/fareez_optuna")
REF, AUDIO = D / "cache/primock/ref", D / "cache/primock/audio"
CONFIG = "primock_tuned"
OUT = Path(__file__).resolve().parent / "speaker_attribution_review"
(OUT / "clips").mkdir(parents=True, exist_ok=True)
SR = 16000
KEY = re.compile(r"\b(no|not|never|yes|pain|chest|breath|breathing|allerg\w*|tablet\w*|mg|milligrams?|dose|"
                 r"inhaler|smoke\w*|drink\w*|blood|fever|cough|vomit\w*|dizzy|faint\w*|take|taking|medication\w*)\b")
ROLE = {"Doctor": "DOCTOR", "Patient": "PATIENT", None: "UNASSIGNED"}


def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


import random, sys as _s
_s.path.insert(0, '/mnt/d/fareez_optuna/code')
from common import parse_reference, REF_DIR
dev_fareez = {r['recording'] for r in csv.DictReader(open(D / 'tuning_set.csv'))}
CORPORA = {
    'Fareez': dict(out=D / 'final' / CONFIG, audio=D / 'cache/audio',
                   recs=sorted(p.stem for p in (D / 'final' / CONFIG).glob('*.json') if p.stem not in dev_fareez),
                   ref=lambda r: [dict(speaker=s, text=t) for s, t in parse_reference(REF_DIR / f'{r}.txt')]),
    'PriMock57': dict(out=D / 'final_primock' / CONFIG, audio=D / 'cache/primock/audio',
                      recs=json.load(open(D / 'results_primock/dev_split.json'))['heldout'],
                      ref=lambda r: sorted(json.load(open(REF / f'{r}.json')), key=lambda u: u['start'])),
    'AfriSpeech-Dialog': dict(out=D / 'final_afrispeech' / CONFIG, audio=D / 'cache/afrispeech/audio',
                              recs=[f'MED{i:02d}' for i in range(1, 21)],
                              ref=lambda r: json.load(open(D / f'cache/afrispeech/ref/{r}.json')))}
CAP_CORPUS, CAP_REC, SEED = 30, 2, 20261022
pool = {c: [] for c in CORPORA}
for corpus, cfg in CORPORA.items():
  for rec in cfg['recs']:
    utts = cfg['ref'](rec)
    rw, rr, rpos = [], [], []
    toks = [u['text'].split() for u in utts]
    for ui, tk in enumerate(toks):
        for ti, w in enumerate(tk):
            if norm(w):
                rw.append(norm(w)); rr.append(utts[ui]['speaker']); rpos.append((ui, ti))
    hyp = json.load(open(cfg['out'] / f'{rec}.json'))
    hw, hs, ht, hraw = [], [], [], []
    for s in hyp['segments']:
        for w in s.get('words', []):
            if norm(w['word']):
                hw.append(norm(w['word'])); hs.append(w.get('speaker')); ht.append((w['start'], w['end'])); hraw.append(w['word'].strip())
    sm = difflib.SequenceMatcher(None, rw, hw, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    votes = Counter((hs[b.b + k], rr[b.a + k]) for b in blocks for k in range(b.size))
    mapping = {spk: max(('Doctor', 'Patient'), key=lambda r: votes[(spk, r)]) for spk in set(hs) if spk}
    hrole = [mapping.get(s) for s in hs]
    pairs = [(b.a + k, b.b + k) for b in blocks for k in range(b.size)]
    run = []
    for i, j in pairs + [(None, None)]:
        bad = i is not None and hrole[j] != rr[i]
        if bad and (not run or i == run[-1][0] + 1):
            run.append((i, j)); continue
        if len(run) >= 3:
            a, z, ja, jz = run[0][0], run[-1][0], run[0][1], run[-1][1]
            if KEY.search(' '.join(rw[a:z + 1])):
                pool[corpus].append(dict(corpus=corpus, rec=rec, a=a, z=z, ja=ja, jz=jz, true=rr[a], given=hrole[ja],
                                         utts=utts, toks=toks, rpos=rpos, hraw=hraw, hrole=hrole, ht=ht, audio=cfg['audio']))
        run = [(i, j)] if bad else []
rng = random.Random(SEED)
events = []
for corpus in CORPORA:
    cand = pool[corpus][:]; rng.shuffle(cand); per = Counter(); chosen = []
    for e in cand:
        if per[e['rec']] < CAP_REC and len(chosen) < CAP_CORPUS:
            chosen.append(e); per[e['rec']] += 1
    chosen.sort(key=lambda e: (e['rec'], e['a']))
    print(corpus, 'candidates', len(pool[corpus]), 'consultations', len({e['rec'] for e in pool[corpus]}), '-> selected', len(chosen))
    events += chosen
rng.shuffle(events)

def ref_context(e):
    """Previous two utterances, the target utterance(s) marked >>>, next utterance; run wrapped in [[ ]]."""
    first, last = e["rpos"][e["a"]], e["rpos"][e["z"]]
    lines = []
    for ui in range(max(0, first[0] - 2), min(len(e["utts"]), last[0] + 2)):
        tk = list(e["toks"][ui])
        if first[0] <= ui <= last[0]:
            s = first[1] if ui == first[0] else 0
            t = last[1] if ui == last[0] else len(tk) - 1
            tk[s] = "[[" + tk[s]; tk[t] = tk[t] + "]]"
        mark = ">>> " if first[0] <= ui <= last[0] else "    "
        lines.append(f"{mark}{ROLE[e['utts'][ui]['speaker']]}: {' '.join(tk)}")
    return "\n".join(lines)


def asr_context(e):
    """ASR words grouped into turns by assigned speaker; turn(s) holding the run marked >>>, run in [[ ]]."""
    turns, cur = [], None
    for k, r in enumerate(e["hrole"]):
        if cur is None or r != cur["role"]:
            cur = dict(role=r, idx=[]); turns.append(cur)
        cur["idx"].append(k)
    tin = [n for n, t in enumerate(turns) if any(e["ja"] <= k <= e["jz"] for k in t["idx"])]
    lines = []
    for n in range(max(0, tin[0] - 2), min(len(turns), tin[-1] + 2)):
        words = []
        for k in turns[n]["idx"]:
            w = e["hraw"][k]
            if k == e["ja"]: w = "[[" + w
            if k == e["jz"]: w = w + "]]"
            words.append(w)
        mark = ">>> " if n in tin else "    "
        lines.append(f"{mark}{ROLE[turns[n]['role']]}: {' '.join(words)}")
    return "\n".join(lines)


audio_cache, key_rows = {}, []
wb = Workbook()
T = Font(name="Times New Roman", size=11)
TB = Font(name="Times New Roman", size=11, bold=True)
TH = Font(name="Times New Roman", size=11, bold=True, color="FFFFFF")
HEAD = PatternFill("solid", fgColor="2D8098")
thin = Side(style="thin", color="BFBFBF")
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
WRAP = Alignment(wrap_text=True, vertical="top")

ins = wb.active; ins.title = "Instructions"
N = len(events)
rows = [
    ("Independent Clinical Review of Speaker-Attribution Errors", None),
    ("Reviewer details", None), ("Reviewer name", None), ("Professional title/qualification", None),
    ("Clinical specialty", None), ("Years of relevant experience", None), ("Review date", None),
    ("Reviewer code/initials", None),
    ("Purpose and procedure", None),
    ("Purpose", f"Independently assess {N} events in which the transcription system attributed correctly recognised words to the wrong speaker, drawn from the held-out consultations of three clinical speech corpora (simulated respiratory interviews, mock primary care and African-accented clinical conversations). Automated preliminary decisions are not shown."),
    ("Compare", "Use the gold-standard context (who actually spoke), the ASR context (who the system says spoke) and the audio clip for each event."),
    ("What is being rated", "Only the words in [[double brackets]] are under review. The words themselves were recognised correctly; what differs is the speaker they are attributed to. Differences in punctuation, capitals, fillers (um, uh) and repeated words are expected and are NOT being rated. Keep the clips folder next to this workbook."),
    ("Audio clip", "Click the link to play the audio around the event. First decide who speaks the bracketed words in the audio (Doctor / Patient / Unclear), then answer the remaining questions."),
    ("How events are marked", "The turn under review begins with >>>; the lines above and below it are the surrounding dialogue. Speaker labels (DOCTOR / PATIENT) show who spoke in the gold-standard column and who the system assigned the words to in the ASR column. UNASSIGNED means the system gave the words no speaker."),
    ("Decide", "Select whether the speaker attribution in the ASR transcript is wrong and whether the misattribution changes clinically relevant meaning."),
    ("Insufficient context", "Explain any missing context in Reviewer comments."),
    ("Decision definitions", None),
    ("Who speaks in the audio?", "Doctor or Patient = the bracketed words are spoken by that person; Unclear = cannot tell."),
    ("Attribution wrong?", "Yes = the ASR transcript gives the words to the wrong speaker; No = the attribution is correct; Partially = only some of the words are misattributed."),
    ("Meaning changed?", "Yes = attributing these words to the wrong speaker changes clinically relevant meaning (for example, a patient's symptom, allergy or denial now appears as the clinician's words); No = meaning is adequately preserved; Partially = meaning changed in part."),
    ("Confidence", "High = clear judgement; Moderate = some ambiguity; Low = difficult judgement or limited context."),
    ("Comments", "Briefly explain No, Partially or clinically important decisions."),
    ("Independent review", "Complete the workbook without consulting the other reviewer. Do not add identifiable information."),
]
for r in rows:
    ins.append(r)
for row in ins.iter_rows():
    for c in row:
        c.font = T; c.alignment = WRAP
for r in (1, 2, 9, 17):
    ins.cell(row=r, column=1).font = TB
ins.column_dimensions["A"].width = 30; ins.column_dimensions["B"].width = 110

ws = wb.create_sheet(f"Speaker Attribution - {N}")
head = ["Event ID", "Corpus", "Said by (gold standard)", "Assigned by ASR", "Gold-standard event context",
        "ASR context (system-assigned speakers)", "Misattributed words", "Audio clip",
        "Who speaks these words in the audio?", "Is the ASR speaker attribution wrong?",
        "Did clinically relevant meaning change?", "Confidence", "Reviewer comments"]
ws.append(head)
for c in ws[1]:
    c.font = TH; c.fill = HEAD; c.alignment = Alignment(wrap_text=True, vertical="center"); c.border = BOX
for n, e in enumerate(events, 1):
    eid = f"A{n:04d}"
    t0 = max(0.0, e["ht"][e["ja"]][0] - 2.0); t1 = e["ht"][e["jz"]][1] + 2.0
    t1 = min(t1, t0 + 12.0)
    if e["rec"] not in audio_cache:
        audio_cache[e["rec"]] = np.load(e["audio"] / f"{e['rec']}.npy")
    x = audio_cache[e["rec"]]
    sf.write(OUT / "clips" / f"{eid}.wav", x[int(t0 * SR):int(t1 * SR)], SR)
    words = " ".join(e["hraw"][e["ja"]:e["jz"] + 1])
    ws.append([eid, e["corpus"], ROLE[e["true"]], ROLE[e["given"]], ref_context(e), asr_context(e), words,
               f"Play {eid}.wav", None, None, None, None, None])
    r = ws.max_row
    ws.cell(row=r, column=8).hyperlink = f"clips/{eid}.wav"
    for c in ws[r]:
        c.font = T; c.alignment = WRAP; c.border = BOX
    ws.cell(row=r, column=8).font = Font(name="Times New Roman", size=11, color="0563C1", underline="single")
    key_rows.append(dict(event=eid, corpus=e["corpus"], recording=e["rec"], configuration=CONFIG, said_by=e["true"],
                         assigned=e["given"] or "none", clip_start_s=round(t0, 2), clip_end_s=round(t1, 2), words=words))
last = ws.max_row
for col, opts in (("I", '"Doctor,Patient,Unclear"'), ("J", '"Yes,No,Partially"'), ("K", '"Yes,No,Partially"'),
                  ("L", '"High,Moderate,Low"')):
    dv = DataValidation(type="list", formula1=opts, allow_blank=True); ws.add_data_validation(dv)
    dv.add(f"{col}2:{col}{last}")
for col, w in zip("ABCDEFGHIJKLM", (10, 12, 13, 13, 52, 52, 26, 12, 15, 15, 14, 13, 40)):
    ws.column_dimensions[col].width = w
ws.freeze_panes = "E2"

so = wb.create_sheet("Reviewer Sign-off")
for r in [("Reviewer Sign-off and Completion Check", None, None), ("Reviewer name", None, None),
          ("Professional title/qualification", None, None), ("Clinical specialty", None, None),
          ("Years of relevant experience", None, None), ("Institution (optional)", None, None),
          ("Completion date", None, None), ("Signature or initials", None, None),
          ("Completion summary", "Completed", "Required"),
          ("Audio decisions", f"=COUNTA('{ws.title}'!I2:I{last})", N),
          ("Attribution decisions", f"=COUNTA('{ws.title}'!J2:J{last})", N),
          ("Meaning-change decisions", f"=COUNTA('{ws.title}'!K2:K{last})", N),
          ("Confidence ratings", f"=COUNTA('{ws.title}'!L2:L{last})", N),
          ("Reviewer declaration", "Response", None),
          ("I completed this review independently.", None, None),
          ("I did not consult another reviewer before completing my decisions.", None, None),
          ("I noted insufficient context in Reviewer comments.", None, None),
          ("General comments", None, None)]:
    so.append(r)
for row in so.iter_rows():
    for c in row:
        c.font = T; c.alignment = WRAP
so["A1"].font = TB; so["A9"].font = TB; so["A14"].font = TB
so.column_dimensions["A"].width = 60; so.column_dimensions["B"].width = 18; so.column_dimensions["C"].width = 12

wb.save(OUT / "Speaker_Attribution_Review.xlsx")
with open(OUT / "_KEY_not_for_reviewers.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(key_rows[0])); w.writeheader(); w.writerows(key_rows)
print("events", N, "| clips", len(list((OUT / 'clips').glob('*.wav'))))
print(ws["E2"].value); print("---"); print(ws["F2"].value)
