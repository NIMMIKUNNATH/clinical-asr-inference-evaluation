"""Recompute every pooled rate the manuscript reports, straight from the per-recording score files,
and compare against the values printed in the paper's tables."""
import csv
from pathlib import Path

ROOT = Path("/mnt/d/fareez_optuna")
CORPORA = {
    "Fareez": ROOT / "final" / "scores",
    "PriMock57": ROOT / "final_primock" / "scores",
    "AfriSpeech": ROOT / "final_afrispeech" / "scores",
}


import json
_PM = json.load(open(ROOT / "results_primock" / "dev_split.json"))
PM_HELDOUT = set(_PM["heldout"])


def pooled(path, corpus=None):
    rows = list(csv.DictReader(open(path)))
    # the score files hold every recording; the manuscript reports the held-out split only
    if corpus == "PriMock57":
        rows = [r for r in rows if r["recording"] in PM_HELDOUT]
    elif corpus == "AfriSpeech":
        rows = [r for r in rows if r["category"] == "MED"]
    if not rows:
        return None
    g = lambda k: sum(int(r[k]) for r in rows)          # noqa: E731
    N = g("ref_words")
    out = {
        "n": len(rows),
        "wer": 100 * (g("sub") + g("dele") + g("ins")) / N,
        "sub": 100 * g("sub") / N,
        "del": 100 * g("dele") / N,
        "ins": 100 * g("ins") / N,
        "wder": 100 * g("wder_wrong") / g("wder_aligned"),
        "hall": 1000 * g("halluc_words") / N,
    }
    if "cer_err" in rows[0]:
        out["cer"] = 100 * g("cer_err") / g("cer_n")
    if "cp_err" in rows[0]:
        out["cpwer"] = 100 * g("cp_err") / g("cp_n")
    if "sa_err" in rows[0]:
        out["sawer"] = 100 * g("sa_err") / g("sa_n")
    if "doc_err" in rows[0]:
        out["doc"] = 100 * g("doc_err") / g("doc_n")
        out["pat"] = 100 * g("pat_err") / g("pat_n")
    if "neg_ref" in rows[0]:
        nr = g("neg_ref")
        out["neg_incl"] = 100 * (g("neg_err") + g("neg_ins")) / nr
        out["neg_anch"] = 100 * g("neg_err") / nr
    return out


# what the paper prints: corpus -> config -> (wer, wder, hall)
PAPER = {
    "Fareez": {
        "defaults": (9.75, 3.43, 22.0), "tuned": (6.95, 4.05, 1.2),
        "local20_fareez": (7.15, 4.09, 1.0), "local20_afrispeech": (6.98, 4.75, 1.2),
        "primock_tuned": (7.04, 4.51, 1.1),
        "oldref_v2_fp16": (7.00, 1.96, 1.60), "v3_oldref": (9.61, 3.28, 20.97),
        "trial32_v2_fp16": (6.90, 1.89, 1.20), "v3_trial32": (23.20, 7.92, 92.21),
    },
    "PriMock57": {
        "defaults": (13.58, 5.99, 16.3), "tuned": (12.26, 5.78, 1.5),
        "local20_fareez": (12.12, 6.03, 1.6), "local20_afrispeech": (11.60, 6.43, 1.3),
        "primock_tuned": (11.75, 6.24, 1.3),
        "oldref_v2_fp16": (12.24, 4.38, 1.76), "v3_oldref": (12.48, 5.61, 11.14),
        "trial32_v2_fp16": (12.03, 4.36, 1.74), "v3_trial32": (17.87, 8.63, 23.94),
    },
    "AfriSpeech": {
        "defaults": (14.36, 11.32, 13.2), "tuned": (14.49, 11.83, 0.7),
        "local20_fareez": (13.57, 11.92, 0.6), "local20_afrispeech": (13.70, 12.51, 0.6),
        "primock_tuned": (13.35, 12.41, 0.6),
        "oldref_v2_fp16": (10.95, 8.76, 0.64), "v3_oldref": (16.66, 11.97, 34.45),
        "trial32_v2_fp16": (11.11, 8.64, 0.64), "v3_trial32": (69.60, 22.63, 448.51),
    },
}

print("{:<11} {:<20} {:>5} {:>18} {:>18} {:>18}".format("corpus", "config", "n", "WER", "WDER", "halluc/1k"))
bad = []
for corpus, sdir in CORPORA.items():
    for cfg, (pw, pd_, ph) in PAPER[corpus].items():
        f = sdir / "{}.csv".format(cfg)
        if not f.exists():
            print("{:<11} {:<20} {:>5}  FILE MISSING".format(corpus, cfg, "-"))
            continue
        r = pooled(f, corpus)
        def cmp(got, want, tol):
            flag = "" if abs(got - want) <= tol else "  <-- MISMATCH"
            if flag:
                bad.append("{} / {}: printed {} vs computed {:.2f}".format(corpus, cfg, want, got))
            return "{:>7.2f} vs {:<6}{}".format(got, want, "" if not flag else "*")
        print("{:<11} {:<20} {:>5} {} {} {}".format(
            corpus, cfg, r["n"], cmp(r["wer"], pw, 0.015), cmp(r["wder"], pd_, 0.015),
            cmp(r["hall"], ph, 0.06 if ph > 2 else 0.051)))

print("\n--- Table 2 secondary outcomes, Fareez held-out ---")
sec = {"defaults": dict(sub=2.19, dele=2.29, ins=5.27, cer=6.86, cpwer=14.51, sawer=22.47, doc=10.89, pat=12.69,
                        neg_incl=12.73, neg_anch=5.36),
       "tuned": dict(sub=1.90, dele=2.29, ins=2.77, cer=4.57, cpwer=12.68, sawer=20.93, doc=11.41, pat=13.69,
                     neg_incl=8.13, neg_anch=5.19)}
KEY = {"sub": "sub", "dele": "del", "ins": "ins", "cer": "cer", "cpwer": "cpwer", "sawer": "sawer",
       "doc": "doc", "pat": "pat", "neg_incl": "neg_incl", "neg_anch": "neg_anch"}
for cfg, want in sec.items():
    r = pooled(CORPORA["Fareez"] / "{}.csv".format(cfg), "Fareez")
    for k, w in want.items():
        got = r.get(KEY[k])
        if got is None:
            print("  {:<9} {:<9} not in score file".format(cfg, k))
            continue
        ok = abs(got - w) <= 0.015
        if not ok:
            bad.append("Table 2 {} {}: printed {} vs computed {:.2f}".format(cfg, k, w, got))
        print("  {:<9} {:<9} {:>7.2f} vs {:<6} {}".format(cfg, k, got, w, "" if ok else "<-- MISMATCH"))

print("\n==================== SUMMARY ====================")
if bad:
    print("{} discrepancies:".format(len(bad)))
    for b in bad:
        print("  *", b)
else:
    print("no discrepancies: every printed value reproduces from the score files")
