"""Week 5: calibrate the three scores the API reports.

    python src/calibrate.py

The model's raw probabilities are not frequencies: a raw 0.9 need not mean "right 90% of
the time". docs/API_Specification.md therefore labelled every score
`uncalibrated_probability` until this step. Here each score gets an isotonic mapping
(a monotone lookup table from raw score to observed success rate):

    aspect_score    P(aspect present)          event: the aspect is in the gold labels
    polarity_score  P(polarity | present)      event: the polarity is right, on aspects
                                               the model found correctly
    doc score       P(document label)          event: the document label is right

The mappings are FITTED on the validation split and only MEASURED on the test split,
with expected calibration error (ECE, 10 bins) and the Brier score, before and after.

Calibration changes the numbers the API reports, never the decisions: which aspects are
reported, and with which polarity, still comes from the frozen v1.1 settings applied to
the raw probabilities. Being monotone, it also keeps every ranking by score unchanged.
"""

import json
from datetime import datetime

import numpy as np
from sklearn.isotonic import IsotonicRegression

from config import ASPECTS, DATASET_VERSION, DOC_SENTIMENTS, PROCESSED, RAW, RESULTS
from train_transformer import load_split
from transformer_absa import SLOT_LABELS, decide_slots
from week4_common import SEED_MODELS, transformer_outputs

CAL_VERSION = "transformer-absa-0.2.2"   # 0.2.1 decisions + calibrated scores
N_BINS = 10


def score_events(out, tuples, docs, thresholds):
    """Raw scores and their 0/1 outcomes for the three reported scores."""
    asp_s, asp_y, pol_s, pol_y, doc_s, doc_y = [], [], [], [], [], []
    for P, D, gold, gdoc in zip(out["slot_probs"], out["doc_probs"], tuples, docs):
        g = dict(gold)
        slots = decide_slots(P, thresholds)
        for a_i, aspect in enumerate(ASPECTS):
            present = 1.0 - P[a_i, 0]
            asp_s.append(present)
            asp_y.append(int(aspect in g))
            j = int(slots[a_i])
            if j and aspect in g:                      # found correctly: is polarity right?
                pol_s.append(P[a_i, j] / max(1e-9, present))
                pol_y.append(int(SLOT_LABELS[j] == g[aspect]))
        k = int(D.argmax())
        doc_s.append(D[k])
        doc_y.append(int(DOC_SENTIMENTS[k] == gdoc))
    return {"aspect": (np.array(asp_s), np.array(asp_y)),
            "polarity": (np.array(pol_s), np.array(pol_y)),
            "doc": (np.array(doc_s), np.array(doc_y))}


def ece(p, y, n_bins=N_BINS):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability(p, y, edges=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            rows.append({"bin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": int(m.sum()),
                         "mean_score": float(p[m].mean()), "observed": float(y[m].mean())})
    return rows


def main():
    settings = json.loads((PROCESSED / "decision_settings.json").read_text())
    th = settings["thresholds"]
    data = load_split(str(RAW / "annotations.csv"), names=("validation", "test"))
    va, te = data["validation"], data["test"]
    with open(RESULTS / "test_openings.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}  test split opened  "
                f"calibrate.py (measuring calibration fitted on validation; no decisions)\n")

    out_va = transformer_outputs(SEED_MODELS[42], va["texts"])
    out_te = transformer_outputs(SEED_MODELS[42], te["texts"])
    ev_va = score_events(out_va, va["tuples"], va["doc"], th)
    ev_te = score_events(out_te, te["tuples"], te["doc"], th)

    cal, report = {}, {}
    for name in ("aspect", "polarity", "doc"):
        s_va, y_va = ev_va[name]
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s_va, y_va)
        cal[name] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
        s_te, y_te = ev_te[name]
        c_te = np.interp(s_te, iso.X_thresholds_, iso.y_thresholds_)
        report[name] = {
            "fitted_on": f"validation ({len(s_va)} scores)",
            "measured_on": f"test ({len(s_te)} scores)",
            "ece_before": ece(s_te, y_te), "ece_after": ece(c_te, y_te),
            "brier_before": float(np.mean((s_te - y_te) ** 2)),
            "brier_after": float(np.mean((c_te - y_te) ** 2)),
            "reliability_before": reliability(s_te, y_te),
            "reliability_after": reliability(c_te, y_te)}
        r = report[name]
        print(f"{name:<9} ECE {r['ece_before']:.3f} -> {r['ece_after']:.3f}   "
              f"Brier {r['brier_before']:.3f} -> {r['brier_after']:.3f}   (test, n={len(s_te)})")

    blob = {"model_version": CAL_VERSION, "method": "isotonic regression",
            "fitted_on": "validation split", "dataset_version": DATASET_VERSION,
            "decision_settings_frozen_at": settings["frozen_at"],
            "created": datetime.now().isoformat(timespec="seconds"), "maps": cal}
    (PROCESSED / "calibration.json").write_text(json.dumps(blob, indent=2))
    (RESULTS / "week5_calibration.json").write_text(json.dumps(report, indent=2))
    print(f"-> {PROCESSED / 'calibration.json'}")


if __name__ == "__main__":
    main()
