# Search space and random seeds

Source of truth: `code/search.py` (`suggest()`, `BASELINE`, sampler set-up). This file restates it for the release.

Fixed in every trial: faster-whisper 1.2.1 (CTranslate2), checkpoint large-v3, float16, `vad_filter=True`,
language English, word timestamps on, pyannote community-1 diarization with `num_speakers=2` (cached once per recording).

## 1. The 17 searched decoding parameters

| # | Parameter | Type | Range / values | Active when |
|---|---|---|---|---|
| 1 | beam_size | int | 1 – 8 | always |
| 2 | patience | float | 0.5 – 2.0 | beam_size > 1 |
| 3 | length_penalty | float | 0.8 – 1.2 | beam_size > 1 |
| 4 | repetition_penalty | float | 1.0 – 1.3 | always |
| 5 | no_repeat_ngram_size | categorical | {0, 3, 4, 5} | always |
| 6 | temperature fallback | categorical | off = [0.0]; short = [0.0, 0.2, 0.4] | always |
| 7 | best_of | int | 1 – 5 | fallback = short |
| 8 | compression_ratio_threshold | float | 1.8 – 3.0 | fallback = short |
| 9 | log_prob_threshold | float | −2.0 – −0.4 | always |
| 10 | no_speech_threshold | float | 0.3 – 0.9 | always |
| 11 | condition_on_previous_text | categorical | {True, False} | always |
| 12 | initial_prompt | categorical | none; "A medical history-taking consultation between a doctor and a patient." | always |
| 13 | hallucination_silence_threshold | off, or float | off; 0.5 – 4.0 s | always (switch + value) |
| 14 | VAD threshold | float | 0.2 – 0.7 | always |
| 15 | VAD min_speech_duration_ms | int | 0 – 250 | always |
| 16 | VAD min_silence_duration_ms | int, log scale | 100 – 2000 | always |
| 17 | VAD speech_pad_ms | int | 50 – 800 | always |

Parameter 13 is implemented as two Optuna dimensions (an on/off switch and the threshold), so the Optuna study
records 18 dimensions for 17 parameters.

## 2. Reference (default) configuration

beam_size 5, patience 1.0, length_penalty 1.0, repetition_penalty 1.0, no_repeat_ngram_size 0,
temperature [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], best_of 5, compression_ratio_threshold 2.4, log_prob_threshold −1.0,
no_speech_threshold 0.6, condition_on_previous_text True, no initial prompt, hallucination_silence_threshold off,
VAD threshold 0.5, min_speech 0 ms, min_silence 2000 ms, speech_pad 400 ms.
The full six-step fallback is outside the search space (Section 1, parameter 6) and is scored only as the reference.

## 3. Sampler

Optuna TPE, `multivariate=True`, `group=True`, `constant_liar=True`; parallel workers seeded `SEED + worker index`.

| Study | Seed | Startup trials | Pruner | Per-trial cap |
|---|---|---|---|---|
| Fareez (large-v3) | 20260915 | 12 | MedianPruner (8 startup, 10 warm-up steps) | 4 h |
| PriMock57 | 20260919 | 10 | none | — |

## 4. All other seeds

| Purpose | Seed | Script |
|---|---|---|
| Fareez 40 / 231 development/held-out split | 20260915 | `code/prepare.py` |
| PriMock57 20 / 37 split (stratified by day) | 20260919 | `code/final/search_primock.py` |
| Selection-rule bootstrap | 20260915 | `code/final/apply_selection.py` |
| Paired bootstrap, 10,000 resamples (all held-out comparisons) | 20260915 | `code/final/compare_heldout.py` |
| Selection stability resampling | 20260902 | `code/final/selection_stability.py` |
| Parameter importance (fANOVA, MDI) | 20260915 | `code/final/importance.py` |
| Clinical-cue review sheet order | 20260919 | `code/final/adjudication_sample.py` |
