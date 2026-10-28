# Does inference tuning improve clinical conversation transcription without harming speaker attribution?

Code, configurations and aggregate outputs for the manuscript of the same title. Audio, reference transcripts and
model weights are not redistributed; obtain them from their original sources (Fareez; PriMock57; AfriSpeech-Dialog;
Whisper large-v2/large-v3 via faster-whisper; pyannote speaker-diarization-community-1).

## Study at a glance
- Corpora and held-out sets: Fareez (231 held-out, 40 development), PriMock57 (37 / 20), AfriSpeech-Dialog
  (20 clinical consultations held out, recorded in Nigeria with six accents; 20 general conversations for development).
  288 held-out consultations in total.
- Searches (Optuna TPE): Fareez primary (40 recordings, 50 completed trials); PriMock57 (20, 52 completed; started
  from the Fareez-selected configuration); Fareez and AfriSpeech-Dialog equal-count searches (20 each, 14 completed
  each) and a joint search (60 recordings, 15 completed), each started from the Fareez- and PriMock57-selected
  configurations.
- Checkpoint comparison: two decoding vectors (reference, trial32) run unchanged on large-v2 and large-v3.
- Clinical-meaning review: two physicians rated 312 flagged events from all three corpora; three nursing students also
  rated the 180-event Fareez subset (Supplementary Data 3).
- Validation: 43 synthetic cases (30 scoring, 13 word-to-speaker assignment), all passing (code/6_tests).
- Environment: Python 3.12.3, faster-whisper 1.2.1, CTranslate2 4.8.1, PyTorch 2.8.0 (CUDA 12.8), pyannote.audio 4.0.7,
  Optuna 4.9.0, MeetEval 0.4.3 (configurations/environment.json).

## Metric definitions (as implemented in code/1_pipeline/metrics.py)
- Normalisation: Whisper English normaliser (lower case, punctuation and fillers removed, numbers as digits,
  contractions expanded, so "don't" contributes the negation cue "not").
- WER = (S + D + I) / N over all hypothesis text, including words outside diarization turns.
- WDER: share of aligned (correct or substituted) words given the wrong role, under the better of the two
  speaker-to-role mappings; aligned words with no speaker count as misattributed; insertions excluded. A
  fixed-denominator variant scores two configurations on the reference words both recognised correctly.
- cpWER (MeetEval): minimum-permutation word error over speaker streams; words outside every diarization turn form
  their own stream and are counted as insertions. cpWER = role-specific errors under the optimal mapping plus
  unassigned words, divided by N.
- SA-WER: word errors with a fixed rule mapping the first hypothesis speaker to talk to the clinician, plus words of
  unmapped streams. When the patient speaks first the roles are reversed for the whole recording (19 default and 20
  tuned Fareez held-out recordings, SA-WER >= 97%); otherwise SA-WER equals cpWER.
- Clinician and patient error: WER on each role's stream under the mapping chosen by cpWER, insertions included.
- Hallucinated text: words in runs of at least five consecutive insertions (reference-defined), per 1,000 reference
  words.
- Negation-cue error: reference-anchored (deleted or substituted cues) and insertion-inclusive (plus inserted cues).
- Paired differences are comparator minus reference, per recording. The matched-pairs rank-biserial correlation is
  r_rb = (W+ - W-) / (W+ + W-), where W+ and W- are the rank sums of the positive and negative differences; zero
  differences are dropped, tied absolute differences get average ranks, and r_rb = 0 when every difference is zero.
  Negative values mean lower error under the comparator. For tuned minus defaults on the Fareez held-out set,
  r_rb = -0.9451 for WER and +0.4920 for WDER.

## Layout
- code/1_pipeline ... 7_clinical_review: transcription, scoring, searches, selection, analyses, figures, tests and the
  clinical-review analyses (see each script's docstring). Paths point to the original working folders.
- configurations/: every selected configuration at full numerical precision, and the software environment.
- protocol/: selection rules, analysis plan, search space and seeds, regeneration manifest (SHA-256 of result files).
- supplementary_data/: Supplementary Data 1 (paired comparisons), 2 (per-recording scores), 3 (de-identified rating
  matrix, 312 events), plus the cross-corpus grid.

## Notes
- one_factor_table.py prints change labels written for the Fareez-selected configuration; every one-factor condition
  was run on the joint configuration (Supplementary Table S5 gives the corrected labels).
- The beam-1 + 400 ms padding run (joint_beam1_pad400) is exploratory: it was decoded after the held-out evaluation, and its per-recording scores are in Supplementary Data 2 with split "held-out, exploratory post hoc".
- The primary selection rule was fixed and hash-stamped on 2026-09-15, before any held-out recording was transcribed (protocol/SELECTION_RULE.sha256.txt); the joint rule was fixed on 2026-09-22.

## Partitions
- partitions/fareez_development_40.csv: the 40 Fareez development recordings (seed 20260915); the other 231 eligible recordings are held out.
- partitions/primock57_dev_heldout.json: PriMock57 development (20) and held-out (37) consultations (seed 20260919).
- partitions/joint_development_20_per_corpus.json: the 20 development recordings per corpus used by the equal-count and joint searches (seed 20260922).
- partitions/afrispeech_dialog_recordings.csv: AfriSpeech-Dialog recordings with domain (general or medical), accent, country and the clinician speaker identified as the questioning speaker.
- configurations/reference_vector.json and configurations/trial32_vector.json: the two decoding vectors of the matched-checkpoint comparison.
