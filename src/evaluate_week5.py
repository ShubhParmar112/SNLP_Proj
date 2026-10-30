"""Week 5: evaluation and error analysis of the final asset on the TEST split.

    python src/calibrate.py        # first: score calibration (fitted on validation)
    python src/evaluate_week5.py

The model and every setting were frozen in Week 4; nothing here changes them. This is
measurement only, so reading the test split's errors is allowed by the protocol (no
decision is taken on it). The opening is still logged in results/test_openings.log.

Sections of the output (results/week5_eval.json):
  overall     tuple-F1 with 95% CI, precision, recall, polarity accuracy, aspect F1
  categories  per-aspect P/R/F1 with bootstrap CIs, and what kind of mistake each makes
  confusion   polarity confusion on correctly found aspects; document-sentiment confusion
  slices      where the errors are: comment length, review type, negation, contrast words
  robustness  the same 900 comments with typos, all lowercase, and no punctuation
  probes      32 hand-written sentences, each testing one behaviour (data/probes/)
Every test error is exported to results/week5_test_errors.json for hand analysis.
"""

import csv
import json
import random
import re
import string
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from config import ASPECTS, DOC_SENTIMENTS, POLARITIES, PROCESSED, RANDOM_SEED, RAW, RESULTS
from metrics import (absa_tuple_f1, absa_tuple_f1_per_category, aspect_detection_metrics,
                     bootstrap_ci, polarity_accuracy_on_detected, sentiment_metrics)
from pipeline import FeedbackAnalyzer
from train_transformer import load_split
from transformer_absa import decide_slots, decode_slots, predict
from week4_common import SEED_MODELS

NEGATION = re.compile(r"\b(not|no|never|n't|nothing|hardly|barely|without)\b|n't\b", re.I)
CONTRAST = re.compile(r"\b(but|however|although|though|yet|whereas)\b", re.I)


def run(fa, texts, thresholds):
    """Raw model pass + the frozen v1.1 decision rule -> sets of (aspect, polarity)."""
    preds = predict(fa.model, fa.encoder(texts), fa.pad_id, fa.device)
    sets = [decode_slots(decide_slots(p["slot_probs"], thresholds)) for p in preds]
    docs = [DOC_SENTIMENTS[int(p["doc_probs"].argmax())] for p in preds]
    return sets, docs


def f1(g, p):
    return absa_tuple_f1(g, p)["f1"]


# --- robustness perturbations (seeded, so every run perturbs identically) -----

def add_typos(text, rng, rate=0.10):
    """One character edit (swap, drop or double a letter) in ~10% of words of 4+ letters."""
    def edit(w):
        if len(w) < 4 or not w.isalpha() or rng.random() >= rate:
            return w
        i = rng.randrange(1, len(w) - 1)
        kind = rng.choice(("swap", "drop", "double"))
        if kind == "swap":
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        if kind == "drop":
            return w[:i] + w[i + 1:]
        return w[:i] + w[i] + w[i:]
    return re.sub(r"[A-Za-z]+", lambda m: edit(m.group(0)), text)


def no_punct(text):
    return re.sub(r"\s+", " ", text.translate(str.maketrans(string.punctuation,
                                                            " " * len(string.punctuation))))


def per_category_ci(gold, pred, n=1000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    k = len(gold)
    draws = {a: [] for a in ASPECTS}
    for _ in range(n):
        idx = rng.integers(0, k, k)
        per = absa_tuple_f1_per_category([gold[i] for i in idx], [pred[i] for i in idx], ASPECTS)
        for a in ASPECTS:
            draws[a].append(per[a]["f1"])
    return {a: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for a, v in draws.items()}


def error_profile(gold, pred):
    """Per aspect: right, wrong polarity, missed, spurious (counted per comment)."""
    prof = {a: Counter() for a in ASPECTS}
    for g, p in zip(gold, pred):
        g, p = dict(g), dict(p)
        for a in ASPECTS:
            if a in g and a in p:
                prof[a]["correct" if g[a] == p[a] else "wrong_polarity"] += 1
            elif a in g:
                prof[a]["missed"] += 1
            elif a in p:
                prof[a]["spurious"] += 1
    return {a: dict(c) for a, c in prof.items()}


def main():
    settings = json.loads((PROCESSED / "decision_settings.json").read_text())
    th = settings["thresholds"]
    with open(RESULTS / "test_openings.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}  test split opened  "
                f"evaluate_week5.py (error analysis and robustness; no decisions)\n")

    te = load_split(str(RAW / "annotations.csv"), names=("test",))["test"]
    gold, texts = te["tuples"], te["texts"]
    fa = FeedbackAnalyzer(SEED_MODELS[42])
    pred, docs = run(fa, texts, th)
    out = {"split": "test", "n": len(gold), "model_version": fa.model_version}

    # --- overall --------------------------------------------------------------
    d = absa_tuple_f1(gold, pred)
    Yg = np.array([[int(any(x == a for x, _ in s)) for a in ASPECTS] for s in gold])
    Yp = np.array([[int(any(x == a for x, _ in s)) for a in ASPECTS] for s in pred])
    out["overall"] = {"tuple_f1": d["f1"], "precision": d["precision"], "recall": d["recall"],
                      "ci": bootstrap_ci(gold, pred, f1),
                      "polarity_acc_on_detected": polarity_accuracy_on_detected(gold, pred)["accuracy"],
                      "aspect_micro_f1": aspect_detection_metrics(Yg, Yp, ASPECTS)["micro_f1"],
                      "exact_comment_match": float(np.mean([set(g) == set(p)
                                                            for g, p in zip(gold, pred)]))}

    # --- categories, error profile, confusions -----------------------------------
    per = absa_tuple_f1_per_category(gold, pred, ASPECTS)
    ci = per_category_ci(gold, pred)
    out["categories"] = {a: {**per[a], "f1_ci": ci[a]} for a in ASPECTS}
    out["error_profile"] = error_profile(gold, pred)
    conf = Counter()
    for g, p in zip(gold, pred):
        g, p = dict(g), dict(p)
        for a in set(g) & set(p):
            conf[(g[a], p[a])] += 1
    out["polarity_confusion"] = {gp: {pp: conf[(gp, pp)] for pp in POLARITIES} for gp in POLARITIES}
    out["doc_sentiment"] = sentiment_metrics(te["doc"], docs, DOC_SENTIMENTS)
    dconf = Counter(zip(te["doc"], docs))
    out["doc_confusion"] = {g: {p: dconf[(g, p)] for p in DOC_SENTIMENTS} for g in DOC_SENTIMENTS}

    # --- slices --------------------------------------------------------------------
    notes = pd.read_csv(RAW / "annotations.csv", dtype=str).groupby("feedback_id")["notes"].first()
    source = [notes[i].split(";")[0].replace("source=", "") for i in te["ids"]]
    words = [len(t.split()) for t in texts]

    def slice_by(name, keys):
        res = {}
        for key in sorted(set(keys), key=str):
            idx = [i for i, k in enumerate(keys) if k == key]
            gs, ps = [gold[i] for i in idx], [pred[i] for i in idx]
            res[str(key)] = {"n": len(idx), "tuple_f1": f1(gs, ps),
                             "exact_comment_match": float(np.mean([set(g) == set(p)
                                                                   for g, p in zip(gs, ps)]))}
        out.setdefault("slices", {})[name] = res

    slice_by("length_words", ["1-10" if w <= 10 else "11-40" if w <= 40 else
                              "41-80" if w <= 80 else "81+" for w in words])
    slice_by("review_type", source)
    slice_by("gold_aspects", [min(len(g), 3) for g in gold])
    slice_by("has_negation", [bool(NEGATION.search(t)) for t in texts])
    slice_by("has_contrast_word", [bool(CONTRAST.search(t)) for t in texts])

    # --- robustness ------------------------------------------------------------------
    rng = random.Random(RANDOM_SEED)
    variants = {"typos (10% of longer words)": [add_typos(t, rng) for t in texts],
                "all lowercase": [t.lower() for t in texts],
                "no punctuation": [no_punct(t) for t in texts]}
    out["robustness"] = {"original": {"tuple_f1": d["f1"], "unchanged_share": 1.0}}
    for name, vt in variants.items():
        vp, _ = run(fa, vt, th)
        out["robustness"][name] = {
            "tuple_f1": f1(gold, vp), "ci": bootstrap_ci(gold, vp, f1),
            "unchanged_share": float(np.mean([set(a) == set(b) for a, b in zip(pred, vp)])),
            "example": vt[3]}

    # --- probe set ------------------------------------------------------------------
    with open(RAW.parent / "probes" / "probe_set.csv", encoding="utf-8") as fh:
        probes = list(csv.DictReader(fh))
    pres = fa.analyze_many([p["text"] for p in probes], [p["probe_id"] for p in probes])
    rows = []
    for p, r in zip(probes, pres):
        exp = {tuple(x.split(":")) for x in p["expected"].split(";") if x}
        got = {(a["aspect"], a["polarity"]) for a in r["aspects"]}
        rows.append({"probe_id": p["probe_id"], "type": p["type"], "text": p["text"],
                     "expected": sorted(f"{a}:{b}" for a, b in exp),
                     "predicted": sorted(f"{a}:{b}" for a, b in got),
                     "pass": exp == got,
                     "scores": [(a["aspect"], a["aspect_score"], a["polarity_score"])
                                for a in r["aspects"]]})
    by_type = {}
    for t in dict.fromkeys(p["type"] for p in probes):
        rs = [r for r in rows if r["type"] == t]
        g = [set(tuple(x.split(":")) for x in r["expected"]) for r in rs]
        q = [set(tuple(x.split(":")) for x in r["predicted"]) for r in rs]
        by_type[t] = {"n": len(rs), "passed": sum(r["pass"] for r in rs), "tuple_f1": f1(g, q)}
    out["probes"] = {"by_type": by_type, "total_passed": sum(r["pass"] for r in rows),
                     "total": len(rows)}
    (RESULTS / "week5_probe_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- error export -------------------------------------------------------------------
    errs = [{"feedback_id": i, "source": s, "text": t,
             "gold": sorted(f"{a}:{b}" for a, b in g),
             "pred": sorted(f"{a}:{b}" for a, b in p),
             "missed": sorted(f"{a}:{b}" for a, b in set(g) - set(p)),
             "spurious": sorted(f"{a}:{b}" for a, b in set(p) - set(g))}
            for i, s, t, g, p in zip(te["ids"], source, texts, gold, pred) if set(g) != set(p)]
    out["n_error_comments"] = len(errs)
    (RESULTS / "week5_test_errors.json").write_text(
        json.dumps(errs, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "week5_eval.json").write_text(json.dumps(out, indent=2, default=str))

    o = out["overall"]
    print(f"TEST n={len(gold)}  tuple-F1 {o['tuple_f1']:.3f} [{o['ci']['ci_low']:.3f}, "
          f"{o['ci']['ci_high']:.3f}]  exact-comment {o['exact_comment_match']:.3f}  "
          f"errors in {len(errs)} comments  ({fa.model_version})")
    for a in ASPECTS:
        c = out["categories"][a]
        print(f"  {a:<20} F1 {c['f1']:.3f} [{c['f1_ci'][0]:.3f}, {c['f1_ci'][1]:.3f}] "
              f"n={c['support']}  {out['error_profile'][a]}")
    print("  polarity confusion:", out["polarity_confusion"])
    print("  doc confusion:", out["doc_confusion"])
    for name, res in out["slices"].items():
        print(f"  slice {name}: " + "; ".join(f"{k} n={v['n']} F1 {v['tuple_f1']:.3f}"
                                             for k, v in res.items()))
    for name, res in out["robustness"].items():
        print(f"  robust {name:<28} F1 {res['tuple_f1']:.3f} unchanged {res['unchanged_share']:.2f}")
    for t, v in by_type.items():
        print(f"  probes {t:<10} {v['passed']}/{v['n']}  tuple-F1 {v['tuple_f1']:.3f}")


if __name__ == "__main__":
    main()
