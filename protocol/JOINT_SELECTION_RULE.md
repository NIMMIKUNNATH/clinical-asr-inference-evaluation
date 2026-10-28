# Joint three-corpus selection rule

Frozen 2026-09-22, before the first trial of the joint search (`code/final/search_joint.py`).

## Purpose
Select one inference configuration for Whisper large-v3 intended for two-speaker clinical conversations across
corpora, rather than for one corpus.

## Development data (search only)
Twenty recordings per corpus, the size of the smallest existing development split (PriMock57), so each corpus
contributes the same number of recordings; all are drawn only from existing development material.
- Fareez: 20 of its 40 development recordings, stratified by category.
- PriMock57: all 20 development consultations.
- AfriSpeech-Dialog: 20 of the 29 general (non-clinical) conversations.
- Draw seed 20260922; the draw is written to `results_joint/joint_dev.json` at the first start.

(A first start with 10/8/8 recordings was stopped after about fifteen minutes with no trial completed, when the
set was enlarged to equal counts of 20; its files are archived in `results_joint/aborted_26rec/`.)

## Held-out data (scored once, after selection)
- Fareez held-out 231, PriMock57 held-out 37, AfriSpeech-Dialog clinical 20.
- No held-out recording enters the search.

## Objective
Mean of the three corpora's pooled word error rates on their development recordings (equal corpus weight).

## Search
17-parameter space identical to the Fareez and PriMock57 searches; TPE (seed 20260922 + worker, 10 start-up
trials, multivariate, grouped, constant liar); median pruning after 15 recordings; a recording whose decode
exceeds max(600 s, its duration) is abandoned and the trial pruned; the Fareez- and PriMock57-selected
configurations are enqueued as the first two trials. Target 60 trials (completed plus pruned), or stopped earlier
by the analyst for time, which is then reported.

## Selection
Amended 2026-09-22 before any trial had completed (0 complete, 0 pruned), to guard against selecting on a small
search subset by chance:
1. The search set (20 + 20 + 20 recordings) is used only to rank trials.
2. The five completed trials with the lowest search objective are re-transcribed on the FULL development sets:
   Fareez 40, PriMock57 20, AfriSpeech-Dialog general 29 (still no held-out recording).
3. Among those five, the configuration with the lowest mean of the three full-development pooled WERs is
   selected.
4. If another of the five is within 0.10 points of it, choose, among those, the one with the lowest maximum
   per-corpus WER (the most even across corpora).
5. No speaker-attribution guardrail is applied: the checkpoint analysis showed that decoding settings on large-v3
   do not close the attribution gap, and WDER is reported for every configuration on every held-out set.

## Reporting
On each held-out set: released defaults, each corpus's 20-recording local configuration, and the joint
configuration (see the local-search section below for the cost of globality).

## Local 20-recording searches (added 2026-09-22, before either started)
So that every corpus has the same three reference points (released defaults, its own local best, the joint
configuration) and every search in the cross-corpus comparison uses the same number of development recordings,
each corpus receives a local search on exactly the 20 recordings it contributes to the joint search.
- Code: `search_joint.py --scope fareez` and `--scope afrispeech` (same evaluator, space, sampler settings and
  runaway guard as the joint search). PriMock57's existing local search already used exactly its 20
  development consultations and the same objective, so it is not repeated.
- Development: Fareez, the 20 Fareez recordings of the joint search (drawn from its 40-recording development
  split); AfriSpeech-Dialog, the 20 general conversations of the joint search.
- Held-out: Fareez 231; AfriSpeech-Dialog the 20 clinical consultations, never seen.
- Objective: pooled development WER. Median pruning after 10 recordings. One worker each, run side by side.
- Warm start: the Fareez- and PriMock57-selected configurations. Target 50 trials (PriMock57 completed 52).
- Selection: the completed trial with the lowest pooled development WER, as for PriMock57 (no guardrails).
- The original 40-recording Fareez search remains the primary confirmatory analysis; its frozen rule and
  held-out comparison are unchanged. The 20-recording Fareez search serves only as the equal-count local
  reference in the cross-corpus comparison.
- Cost of globality per corpus: joint configuration WER minus that corpus's 20-recording local configuration
  WER, on that corpus's held-out set.
