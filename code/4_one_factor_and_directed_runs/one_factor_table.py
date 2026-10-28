"""Controlled one-factor family: each condition changes exactly one inference parameter from the
selected (tuned) configuration, everything else held fixed. Reports the effect on the held-out
Fareez recordings relative to the of_reference baseline."""
import csv
from pathlib import Path

S = Path("/mnt/d/fareez_optuna/final/scores")

LABELS = {
    "of_reference": ("baseline (selected configuration)", ""),
    "of_condprev_on": ("previous-text conditioning", "off -> on"),
    "of_temp_full": ("temperature fallback", "short -> six steps"),
    "of_vadsilence_1000": ("VAD minimum silence", "471 -> 1000 ms"),
    "of_vadsilence_2000": ("VAD minimum silence", "471 -> 2000 ms"),
    "of_vadthresh_020": ("VAD speech threshold", "0.28 -> 0.20"),
    "of_vadthresh_050": ("VAD speech threshold", "0.28 -> 0.50"),
    "of_reppen_110": ("repetition penalty", "1.06 -> 1.10"),
    "of_reppen_125": ("repetition penalty", "1.06 -> 1.25"),
    "of_logprob_100": ("log-probability threshold", "-1.49 -> -1.0"),
    "of_logprob_030": ("log-probability threshold", "-1.49 -> -0.3"),
    "of_nospeech_030": ("no-speech threshold", "0.31 -> 0.30"),
    "of_nospeech_070": ("no-speech threshold", "0.31 -> 0.70"),
    "of_beam_1": ("beam size", "6 -> 1"),
    "of_beam_8": ("beam size", "6 -> 8"),
    "of_patience_100": ("patience", "1.23 -> 1.0"),
    "of_lenpen_080": ("length penalty", "1.06 -> 0.80"),
    "of_norepeat_3": ("no-repeat n-gram", "0 -> 3"),
    "of_bestof_5": ("best of", "5 -> 5"),
    "of_compression_300": ("compression-ratio threshold", "2.38 -> 3.0"),
    "of_prompt_clinical": ("initial prompt", "none -> clinical sentence"),
    "of_hallsil_off": ("hallucination-silence threshold", "off -> off"),
    "of_vadminspeech_0": ("VAD minimum speech", "231 -> 0 ms"),
    "of_vadpad_400": ("VAD speech padding", "404 -> 400 ms"),
    "of_vadfilter_off": ("voice-activity filter", "on -> off"),
}


def pooled(cfg):
    f = S / "{}.csv".format(cfg)
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    g = lambda k: sum(int(r[k]) for r in rows)        # noqa: E731
    N = g("ref_words")
    return dict(n=len(rows), wer=100 * (g("sub") + g("dele") + g("ins")) / N,
                wder=100 * g("wder_wrong") / g("wder_aligned"),
                hall=1000 * g("halluc_words") / N,
                words=g("hyp_words"))


base = pooled("of_reference")
print("baseline of_reference: n={n}  WER={wer:.2f}  WDER={wder:.2f}  halluc={hall:.2f}".format(**base))
print()
print("{:<34} {:<26} {:>4} {:>8} {:>8} {:>9}".format("parameter changed", "value", "n", "dWER", "dWDER", "dHalluc"))
rows = []
for cfg, (name, chg) in LABELS.items():
    if cfg == "of_reference":
        continue
    p = pooled(cfg)
    if p is None:
        print("{:<34} {:<26}  MISSING".format(name, chg))
        continue
    identical = p["words"] == base["words"] and abs(p["wer"] - base["wer"]) < 1e-9
    rows.append((name, chg, p, p["wer"] - base["wer"], p["wder"] - base["wder"], p["hall"] - base["hall"], identical))
    print("{:<34} {:<26} {:>4} {:>+8.2f} {:>+8.2f} {:>+9.2f}{}".format(
        name, chg, p["n"], p["wer"] - base["wer"], p["wder"] - base["wder"], p["hall"] - base["hall"],
        "   (no change)" if identical else ""))

print("\nconditions producing a byte-identical transcript: {}".format(sum(1 for r in rows if r[6])))
act = [r for r in rows if not r[6]]
print("conditions that changed the output: {}".format(len(act)))
print("\nlargest WER effects:")
for r in sorted(act, key=lambda x: -abs(x[3]))[:6]:
    print("   {:<34} {:<24} dWER {:+.2f}  dWDER {:+.2f}".format(r[0], r[1], r[3], r[4]))
print("\nlargest WDER effects:")
for r in sorted(act, key=lambda x: -abs(x[4]))[:6]:
    print("   {:<34} {:<24} dWDER {:+.2f}  dWER {:+.2f}".format(r[0], r[1], r[4], r[3]))
