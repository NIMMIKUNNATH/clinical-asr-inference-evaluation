# Pre-specified analysis plan

**Fixed on:** 2026-09-18, while the held-out transcription was still running and **before any held-out recording
had been scored**. At this point `final/defaults/` contained raw transcription output only; no metric had been
computed on it, and no configuration had been selected.

**Status:** frozen. Any later change is recorded under "Deviations", with its date and reason.

**Relationship to `SELECTION_RULE.md`:** that document governs how the tuned configuration is chosen on the
development set. This document governs everything downstream: what is computed on the held-out data, which
comparisons are inferential, and how they are reported. Where the two overlap, `SELECTION_RULE.md` wins.

---

## 1. Question

Does a single inference configuration exist that improves clinical conversation transcription across corpora, and
if so, which parameters must still be set locally?

Three claims, separately evidenced:

- **C1 (within-corpus).** A configuration tuned on a development subset improves outcomes on held-out recordings
  of the same corpus.
- **C2 (transfer).** A configuration tuned on corpus A retains part of its advantage on corpus B.
- **C3 (generality).** A single jointly-fitted configuration lands within a stated margin of every corpus's own
  locally-tuned ceiling.

C1 is testable with Fareez alone. C2 needs two corpora. **C3 needs at least three, and is the claim the word
"general" refers to.** No claim of generality will be made from fewer.

## 2. Estimands

For each corpus, each configuration, each outcome:

- **Pooled (micro) rate** - total errors divided by total reference units. Primary for corpus-level reporting.
- **Macro rate** - mean of recording-level rates. Used for all paired comparisons, because pairing is at the
  recording level.

Both are reported wherever either is quoted, because they answer different questions and the two manuscripts in
this project have previously mixed them. Every table states which is shown.

**Local ceiling** for corpus C: the held-out outcome of the configuration selected by searching on C's own
development subset.
**Cost of globality** for corpus C: joint configuration minus local ceiling, on C's held-out recordings. Positive
values mean the shared configuration is worse than tuning locally.

## 3. Outcomes

**Primary:** WER.

**Co-primary, reported alongside WER wherever WER is reported:** WDER.

**Secondary:** substitution, deletion and insertion rates; WER with non-lexical tokens retained; CER; cpWER;
SA-WER and SA-WER minus WER; clinician and patient WER under both insertion conventions; hallucination runs
(>= 5 consecutive inserted words) per 1,000 reference words; negation-cue and affirmation-cue errors; real-time
factor; coverage gap (diarized speech seconds producing no transcribed word).

**Excluded unless credentials are obtained:** MC-WER, which requires the Google Cloud Healthcare NL API. If it
becomes available it enters as a secondary outcome only, and its absence from the selection guardrails is
recorded as a deviation in `SELECTION_RULE.md`.

## 4. Fixed-denominator WDER

WDER excludes deletions by construction, so a configuration that deletes more words removes candidates from its
own denominator. Across the 50 completed Fareez trials, deletion rate and WDER correlate at r = -0.62 overall
and r = -0.36 within the 47 trials that disabled previous-text conditioning.

Every WDER comparison between two configurations is therefore reported twice:

1. **Standard WDER**, each configuration scored on its own denominator, as defined by El Shafey et al. (2019).
2. **Fixed-denominator WDER**, both configurations scored over only those reference words correctly recognised by
   **both**, so the denominator is identical.

If the two disagree in direction or materially in magnitude, the standard comparison is reported as
denominator-dependent and no claim of degraded speaker attribution is made from it alone. This analysis is
specified here before any held-out WDER has been computed.

## 5. Comparisons and their inferential status

| Comparison | Status |
|---|---|
| Tuned vs defaults, held-out, per corpus | **Confirmatory.** Pre-specified, one comparison per outcome |
| One-factor family vs reference, development | Inferential within outcome family, Holm-corrected |
| Transfer (tuned on A, evaluated on B) | Descriptive |
| Joint configuration vs local ceiling | Descriptive, with interval |
| Model-family comparisons | Descriptive; non-Whisper models are not inference-tuned |
| Search trials, importance, selection stability | Exploratory |

The confirmatory comparison is **tuned vs defaults on held-out recordings**, with WER primary and WDER
co-primary. Everything else is secondary, descriptive or exploratory, and will be labelled as such in the text,
not only in a limitations paragraph.

## 6. Statistics

- All comparisons paired at the recording level.
- Paired percentile bootstrap, 10,000 resamples, seed 20260915, resampling recording identifiers jointly across
  the configurations being compared.
- Wilcoxon signed-rank on recording-level differences; rank-biserial effect size; better/worse/tied counts.
- Holm correction within each outcome family for the one-factor comparisons. No correction across outcome
  families; each family answers a separate pre-specified question and no joint claim is made across them.
- The two confirmatory comparisons (WER, WDER) are Holm-corrected as a family of two per corpus.
- Bootstrap intervals are not multiplicity-adjusted and are reported as descriptive uncertainty; where an
  unadjusted interval and an adjusted p value disagree, both are shown and the adjusted p value governs.
- Parameter importance: fANOVA and mean-decrease-impurity, reported together with the trial count per level,
  because adaptive sampling makes both descriptive rather than causal.

## 7. Failure handling

- A recording producing an empty hypothesis is a transcription failure and scores 100% WER. Attempted, scoreable
  and failed counts are reported for every configuration.
- Results are also reported conditional on scoreable recordings, as a sensitivity analysis.
- A recording that fails deterministically under one configuration is **not** replaced. Comparisons involving it
  use the matched subset, and both the matched-subset and failure-penalised estimates are given.

## 8. Reporting

- Every configuration reported with its complete parameter vector.
- Every number labelled pooled or macro.
- Deviations logged with date and reason in this file and in `SELECTION_RULE.md`.
- Analysis code, environment specification, split manifests and aggregate outputs released at a versioned DOI.

---

## Deviations

*(none at time of freezing)*

- **2026-09-18.** The WDER guardrail in `SELECTION_RULE.md` (WDER <= defaults + 0.5 points) is failed by all ten
  candidates in the top-10 development WER pool; the closest, trial 61, misses by 0.003 points. Per step 3 of
  that rule the guardrails are reported as failed and selection falls back to lowest cpWER within the pool. The
  first configuration that would satisfy the guardrail, trial 45 (WER 7.16%, WDER 3.66%), lies outside the
  pre-specified pool and is **not** substituted for the rule's own fallback. Recorded before the candidates were
  re-scored.
- **2026-09-18.** MC-WER guardrail not applied; Google Cloud Healthcare NL API credentials are not available.
