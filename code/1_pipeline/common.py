"""Shared paths, reference parsing, text normalisation and scoring for the Fareez Optuna study."""
import json
import re
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

AUDIO_DIR = Path("/mnt/d/fareez_all")                                   # the .mp3s - the only audio input
REF_DIR = Path("/mnt/e/New_dataset_annotation/Data/Clean Transcripts")   # used for scoring only
ROOT = Path("/mnt/d/fareez_optuna")
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
TUNING_SET = ROOT / "tuning_set.csv"
STUDY_NAME = "fareez_large_v3_whisperx"
STORAGE_FILE = Path.home() / "fareez_optuna" / "study.journal"          # ext4, safe for parallel workers
SR = 16000

# Same conversation recorded twice (audio match, 8.6 s offset). RES0043 is kept.
DUPLICATES = {"RES0202": "RES0043"}

# Speaker line: D or P, then ':' '::' ';' '"' '.' (optionally after a space), or just a space,
# or the one-off typo 'DL'. Anything else is a wrapped continuation of the previous turn.
_PREFIX = re.compile(r"""^\s*([DP])(?:\s*[:;".]+|L(?=\s)|(?=\s))\s*(.*)$""")
# A second turn typed on the same line, e.g. "...any chest pain? p: No." (RES0086/88/92/93/98).
# Requires the colon, so "vitamin D and ..." is untouched.
_INLINE = re.compile(r"\s+([DdPp])\s*:\s+")
ROLE = {"D": "Doctor", "P": "Patient"}


def read_text(path):
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def parse_reference(path):
    """Return a list of (role, text) turns, continuation lines folded into the previous turn."""
    turns = []
    for line in read_text(path).splitlines():
        if not line.strip():
            continue
        m = _PREFIX.match(line)
        if m:
            turns.append([ROLE[m.group(1)], ""])
            text = m.group(2)
        elif turns:
            text = line
        else:
            raise ValueError(f"{path}: text before the first speaker label: {line!r}")
        parts = _INLINE.split(" " + text)
        turns[-1][1] = (turns[-1][1] + " " + parts[0]).strip()
        for label, more in zip(parts[1::2], parts[2::2]):
            turns.append([ROLE[label.upper()], more.strip()])
    return [tuple(t) for t in turns]


_whisper_norm = EnglishTextNormalizer()
_ANNOTATION = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_OK = re.compile(r"\b(?:okay|okey|o\.\s?k\.?|o k)\b", re.I)
_EXTRA_FILLERS = re.compile(r"\b(?:uhm+|umm+|uhh+|erm+)\b", re.I)


def normalize(text):
    """Pre-clean, then the standard Whisper English normaliser. Applied identically to reference and hypothesis.

    Pre-cleaning fixes what the Whisper normaliser gets wrong on this data:
      curly apostrophes ("I’m" -> "i m"), "O.K." -> "0 k", "okay" vs "ok" never matching,
      fillers it does not know (uhm/umm/uhh/erm - it already drops um/uh/hmm/mm/mhm),
      and the transcriber's note "(unclear speech due to connection issues)".
    """
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = _ANNOTATION.sub(" ", text)
    text = _OK.sub("ok", text)
    text = _EXTRA_FILLERS.sub(" ", text)
    return _whisper_norm(text).split()


def reference_tokens(path):
    """(tokens for WER, [(token, role)] for WDER). Each turn is normalised on its own: normalising the whole
    text at once lets the number normaliser merge across a speaker change ("...two?" / "Three" -> "23")."""
    tagged = [(w, role) for role, t in parse_reference(path) for w in normalize(t)]
    return [w for w, _ in tagged], tagged


def hypothesis_tokens(segment_texts):
    """Same idea for Whisper output: normalise segment by segment, never across segment boundaries."""
    return [w for t in segment_texts for w in normalize(t)]


def wer_counts(ref_tokens, hyp_tokens):
    if not hyp_tokens:
        return dict(sub=0, dele=len(ref_tokens), ins=0, n=len(ref_tokens))
    o = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
    return dict(sub=o.substitutions, dele=o.deletions, ins=o.insertions, n=len(ref_tokens))


def wder(ref_tagged, hyp_tagged):
    """Word-level diarization error: share of aligned (hit/substituted) words whose speaker is wrong.

    Speaker names from diarization are arbitrary, so both mappings of the two hypothesis speakers onto
    Doctor/Patient are tried and the better one is used. Words WhisperX could not assign count as wrong.
    Returns (wder, aligned_words, mapping).
    """
    if not ref_tagged or not hyp_tagged:
        return None, 0, {}
    o = jiwer.process_words(" ".join(w for w, _ in ref_tagged), " ".join(w for w, _ in hyp_tagged))
    pairs = []
    for ch in o.alignments[0]:
        if ch.type in ("equal", "substitute"):
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                pairs.append((ref_tagged[ch.ref_start_idx + k][1], hyp_tagged[ch.hyp_start_idx + k][1]))
    if not pairs:
        return None, 0, {}
    spk = sorted({h for _, h in pairs if h != "UNKNOWN"})
    maps = [{}]
    if spk:
        a, b = spk[0], spk[1] if len(spk) > 1 else None
        maps = [{a: "Doctor", b: "Patient"}, {a: "Patient", b: "Doctor"}]
    best = min(maps, key=lambda m: sum(m.get(h) != r for r, h in pairs))
    wrong = sum(best.get(h) != r for r, h in pairs)
    return wrong / len(pairs), len(pairs), {k: v for k, v in best.items() if k}


def load_tuning_set():
    import csv
    with open(TUNING_SET, newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
