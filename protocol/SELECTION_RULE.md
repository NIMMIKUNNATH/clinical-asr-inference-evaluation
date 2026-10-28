# Pre-specified selection rule for the final decoding configuration

**Fixed on:** 2026-09-15, while the Optuna search was still running. At that point no held-out recording had been transcribed or scored, and no configuration had been chosen.
**Status:** frozen. Any later change must be recorded under "Deviations" at the end, with its date and reason.
**Basis:** literature review, Section 9 (`references/Evaluation_Metrics_Literature_Review.docx`) and `references/selection_methodology.bib`.
**Who decided:** the study authors. The margins below were set on methodological grounds, before any candidate had been scored on MC-WER, hallucination or negation metrics. The rationale for each is recorded next to it.

## Data

- **Development set:** the 40 tuning recordings listed in `tuning_set.csv` (seed 20260915, stratified by specialty). They are used for tuning and selection only.
- **Test set:** the other 231 recordings of Fareez et al. (2022). RES0202 is excluded as a duplicate of RES0043. The test set is used **once**, only after the configuration has been chosen by this rule.

## Search (already running, unchanged)

- Optuna TPE (multivariate, constant liar), seed 20260915 + worker id, MedianPruner, 3 parallel workers.
- Target: 60 finished trials (complete or pruned).
- Objective: pooled development WER.
- Model: faster-whisper large-v3, float16, sequential decoding.
- Speaker pipeline: pyannote community-1 with 2 speakers, then WhisperX `assign_word_speakers`.

## Selection rule

1. **Candidates.** Take the 10 COMPLETE trials with the lowest development WER. Re-transcribe them on the 40 development recordings with `final_run.py`, which saves word-level output.
   - A candidate that uses temperature fallback (`temperature_fallback = short`) is run with 3 random seeds, and its metrics are averaged over the seeds.
   - Score every candidate and the faster-whisper default configuration with `metrics.py`.
2. **Primary metric.** Pooled development WER.
   *Rationale:* WER is the metric every clinical-conversation ASR paper reports, so it is what readers will compare. It is also the objective the search optimised; choosing on a different metric than the search used would add optimism without adding information. cpWER, the CHiME-6 style joint metric, enters as the first tie-breaker and is reported as co-primary on the test set.
3. **Admissibility guardrails.** Each is a non-inferiority check against the default configuration, on the same 40 recordings with the same scorer, using the point estimate of the paired difference.
   - **WDER ≤ default + 0.5 percentage points (δ₁).**
     *Rationale:* diarization is fixed, so decoding changes WDER only through word timings and segment boundaries. A shift under 0.5 points is within recording-to-recording variation. More than that means a real loss in attributing speech to the right person.
   - **MC-WER ≤ default + 0.5 points (δ₂).** If MC-WER is not available yet, this guardrail is applied once it is, before the test run.
     *Rationale:* about 100–200 medical concepts per recording, so roughly 5–8k in total, is far fewer units than about 53k words. Its sampling noise is therefore a few times larger than WER's, and a zero margin would reject configurations by chance.
   - **Hallucinated words per 1,000 reference words ≤ default** (no margin).
     *Rationale:* reducing hallucinated insertions is the main mechanism of the gains so far, and hallucinations are the harm reported by Koenecke et al. (2024). A tuned configuration must not be worse here.
   - **Negation-cue error rate ≤ default + 1.0 point.**
     *Rationale:* there are about 1,000 negation cues across the 40 recordings, so this rate is the noisiest metric. A margin of 1 point (roughly 10 cues) blocks systematic damage to negations without rejecting on noise. Flips are also audited by hand on the test set.
   - If no candidate is admissible, the guardrails are reported as failed and the candidate with the lowest cpWER is chosen. This must be recorded as a deviation.
4. **Ties.** Among admissible candidates, compute the paired-bootstrap standard error of the WER difference from the best candidate, resampling the 40 recordings with 10,000 resamples and seed 20260915. Candidates within **one standard error** of the best WER are treated as tied (one-standard-error rule, Hastie et al. 2009).
   *Rationale:* with 60 trials on 40 recordings, the lowest score partly reflects luck (Cawley & Talbot 2010). The one-standard-error rule is data-driven, can be cited, and avoids choosing a fixed margin arbitrarily.
5. **Tie-break order:**
   1. lower pooled cpWER;
   2. lower hallucination rate;
   3. fewer settings that differ from the faster-whisper defaults.
6. **Output.** The chosen settings are written to `results/final_choice.json`, together with the trial number, all candidate scores, and the date.

## Test evaluation (run once)

- Configurations:
  - `tuned` (from `final_choice.json`);
  - `defaults` (faster-whisper defaults, VAD on);
  - `oldref_v2` (the reference configuration of the earlier pipeline: large-v2, int8_float16).
- Random seed fixed for every run.
- Primary result: pooled WER with a 95% bootstrap CI over recordings (10,000 resamples).
- Also reported: substitution, deletion and insertion rates; WER with non-lexical sounds kept; CER; WDER; cpWER and cpWER − WER; SA-WER; doctor WER and patient WER; MC-WER; negation-cue and affirmation-cue errors; hallucination rate; RTF. Each is pooled with a 95% CI, plus mean ± SD and median [IQR] per recording.
- Paired comparison, `tuned` against each other configuration:
  - Wilcoxon signed-rank test on per-recording values;
  - rank-biserial effect size;
  - median difference with 95% CI;
  - paired-bootstrap CI of the pooled difference;
  - better / worse / tied counts.
- Results by specialty.
- Development-set numbers are reported as optimistically biased.

## Deviations

1. **2026-09-15 ~21:55, search interrupted by the operating environment. The selection rule is unchanged.**
   - The worker processes were terminated when the parent process that had launched them closed.
   - The trials in progress (20, 21, 22) were marked FAIL. Identical parameter sets were re-queued with `enqueue_trial`, so the same configurations are evaluated in full rather than dropped.
   - The search was resumed at 22:45 from a stand-alone launcher (`code/start_search.bat`).
   - No completed or pruned trial was altered.
   - Earlier, 15:50–18:10: a laptop sleep paused the workers. They resumed without error; no trial exceeded the 4-hour limit, and nothing changed.

2. **2026-09-16, secondary study on two locally recorded consultations (does not affect the Fareez study).**
   - A separate Optuna study (`madiha_two_videos`, journal `~/fareez_optuna/madiha.journal`) tunes the same search space on two locally recorded consultations, scored against researcher-corrected references with the earlier pipeline's duplicated words removed.
   - Budget reduced from 40 to **25 finished trials**, decided at 07:50 on operational grounds (a result was needed sooner), before any trial had completed and therefore without knowledge of any outcome.
   - This study is engineering selection, not a publishable estimate: two recordings cannot support a 17-parameter fit, and the chosen configuration will be reported as "best of N tried on two annotated consultations", with a cross-recording check (choose on one, measure on the other).
