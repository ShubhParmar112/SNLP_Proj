"""Week 3 evaluation: pipeline v1 vs the Week 1-2 baseline, on the VALIDATION split.

    python src/evaluate_week3.py [--model DIR]

The test split stays closed until the Week 4 comparison (Evaluation_Protocol.md 4.2:
"the test set is opened once" per comparison, and no decision is made on it). Every
figure here is a development figure on the 900 validation comments.

All transformer numbers are computed from the pipeline's own JSON output, so this script
also exercises pipeline v1 end to end on 900 real comments.
"""

import argparse
import json
import random
import time
from collections import Counter

import numpy as np

from baselines import ABSABaseline, SentimentBaseline
from config import ASPECTS, DOC_SENTIMENTS, RANDOM_SEED, RAW, RESULTS
from data import aspect_matrix, to_model_format, validate
from metrics import (absa_tuple_f1, absa_tuple_f1_per_category, aspect_detection_metrics,
                     bootstrap_ci, majority_tuple_baseline, polarity_accuracy_on_detected,
                     sentiment_metrics)
from pipeline import DEFAULT_MODEL, FeedbackAnalyzer
from train_transformer import load_split

HANDOUT_EXAMPLE = ("The lectures were very informative, but the pace was too fast and "
                   "the assignments were difficult.")
# Guidelines section 7 worked examples: qualitative demonstration only, never a metric.
GUIDELINE_EXAMPLES = [
    "The lectures were informative but the pace was too fast.",
    "Assignments were difficult but genuinely useful.",
    "We barely get enough time to finish everything.",
    "There were four assignments this semester.",
    "Another three-hour lecture, exactly what we needed.",
    "Good course.",
    "Sir is always available for doubts but his slides are unreadable.",
    "The content is relevant, though it is covered too quickly.",
    "Please upload the recordings earlier.",
    "Not very informative.",
]


def paired_bootstrap(gold, pred_a, pred_b, n=1000, seed=RANDOM_SEED):
    """95% CI for F1(b) - F1(a), resampling comments jointly for both systems."""
    rng = np.random.default_rng(seed)
    k = len(gold)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        g = [gold[i] for i in idx]
        diffs.append(absa_tuple_f1(g, [pred_b[i] for i in idx])["f1"]
                     - absa_tuple_f1(g, [pred_a[i] for i in idx])["f1"])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": absa_tuple_f1(gold, pred_b)["f1"] - absa_tuple_f1(gold, pred_a)["f1"],
            "ci_low": float(lo), "ci_high": float(hi),
            "p_b_not_better": float(np.mean(np.array(diffs) <= 0))}


def token_jaccard(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / len(a | b) if a | b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default="week3_eval.json")
    ap.add_argument("--light", action="store_true",
                    help="metrics and spans only (for ablation checkpoints)")
    a = ap.parse_args()

    data = load_split(str(RAW / "annotations.csv"))
    tr, va = data["train"], data["validation"]
    gold = va["tuples"]
    out = {"split": "validation", "n": len(gold), "model_dir": a.model}

    # --- references and the Week 1-2 baseline, refitted on train -------------
    df = to_model_format(validate(str(RAW / "annotations.csv"), strict=False)[0])
    Y = aspect_matrix(df.set_index("feedback_id").loc[tr["ids"]].reset_index())
    base = ABSABaseline().fit(tr["texts"], Y, tr["tuples"])
    p_base = base.predict_sets(va["texts"])
    p_maj = majority_tuple_baseline(tr["tuples"], len(gold))
    sent = SentimentBaseline().fit(tr["texts"], tr["doc"])
    base_doc = list(sent.predict(va["texts"]))

    # --- pipeline v1 on every validation comment -----------------------------
    fa = FeedbackAnalyzer(a.model, settings=None,       # v1 decisions and raw scores,
                          calibration=None)               # as reported in Week 3
    fa.analyze_many(va["texts"][:32])                       # warm-up (CUDA kernels)
    t0 = time.time()
    res = fa.analyze_many(va["texts"], va["ids"])
    batch_ms = 1000 * (time.time() - t0) / len(va["texts"])
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(range(len(va["texts"])), 50)
    t1 = time.time()
    for i in sample:
        fa.analyze(va["texts"][i])
    single_ms = 1000 * (time.time() - t1) / len(sample)
    p_tr = [{(x["aspect"], x["polarity"]) for x in r["aspects"]} for r in res]
    tr_doc = [r["doc_sentiment"]["label"] for r in res]
    out["latency_ms"] = {"single_call_mean": round(single_ms, 1),
                         "batched_per_comment": round(batch_ms, 1),
                         "device": str(fa.device)}

    f1 = lambda g, p: absa_tuple_f1(g, p)["f1"]
    rows = []
    for name, preds in [("Majority class", p_maj),
                        ("TF-IDF LogReg baseline (Week 1-2)", p_base),
                        ("Transformer ABSA v1 (Week 3)", p_tr)]:
        d = absa_tuple_f1(gold, preds)
        pol = polarity_accuracy_on_detected(gold, preds)
        Yg = np.array([[int(any(x == a_ for x, _ in s)) for a_ in ASPECTS] for s in gold])
        Yp = np.array([[int(any(x == a_ for x, _ in s)) for a_ in ASPECTS] for s in preds])
        rows.append({"model": name, "micro_f1": d["f1"], "precision": d["precision"],
                     "recall": d["recall"], "ci": bootstrap_ci(gold, preds, f1),
                     "polarity_acc_on_detected": pol["accuracy"],
                     "n_detected": pol["n_detected"],
                     "aspect_micro_f1": aspect_detection_metrics(Yg, Yp, ASPECTS)["micro_f1"],
                     "per_category": absa_tuple_f1_per_category(gold, preds, ASPECTS)})
    out["task_c"] = rows
    out["paired_diff_transformer_minus_baseline"] = paired_bootstrap(gold, p_base, p_tr)
    out["task_a"] = {"majority": sentiment_metrics(va["doc"], ["positive"] * len(gold),
                                                   DOC_SENTIMENTS),
                     "baseline": sentiment_metrics(va["doc"], base_doc, DOC_SENTIMENTS),
                     "transformer": sentiment_metrics(va["doc"], tr_doc, DOC_SENTIMENTS)}

    # --- evidence spans vs gold spans (preliminary; formal span F1 is Week 4) --
    jac, n_span = [], 0
    for r, sp in zip(res, va["spans"]):
        for x in r["aspects"]:
            if x["aspect"] in sp and x["span"]:
                n_span += 1
                jac.append(token_jaccard(x["span"]["text"], sp[x["aspect"]]))
    out["spans"] = {"n_compared": n_span,
                    "share_jaccard_ge_0.5": float(np.mean(np.array(jac) >= 0.5)) if jac else None,
                    "share_any_overlap": float(np.mean(np.array(jac) > 0)) if jac else None,
                    "mean_jaccard": float(np.mean(jac)) if jac else None}

    if a.light:
        (RESULTS / a.out).write_text(json.dumps(out, indent=2, default=str))
        print(json.dumps({"task_c": [(r["model"], round(r["micro_f1"], 3)) for r in rows],
                          "spans": out["spans"]}, indent=1))
        return

    # --- the 40 baseline errors coded by hand in Week 2: which are fixed? ------
    coded = json.loads((RESULTS / "errors_coded_week2.json").read_text(encoding="utf-8"))
    text_to_i = {t: i for i, t in enumerate(va["texts"])}
    recheck = []
    for c in coded:
        i = text_to_i.get(c["text"])
        if i is None:
            continue
        g, pt = set(gold[i]), p_tr[i]
        recheck.append({"sample_rank": c["sample_rank"], "category": c["category"],
                        "text": c["text"], "gold": c["gold"], "baseline": c["pred"],
                        "transformer": sorted(f"{x}:{y}" for x, y in pt),
                        "fixed": pt == g,
                        "tuple_f1_baseline": absa_tuple_f1([g], [p_base[i]])["f1"],
                        "tuple_f1_transformer": absa_tuple_f1([g], [pt])["f1"]})
    by_cat = {}
    for r in recheck:
        s = by_cat.setdefault(r["category"], {"n": 0, "fixed": 0, "improved": 0, "worse": 0})
        s["n"] += 1
        s["fixed"] += r["fixed"]
        s["improved"] += r["tuple_f1_transformer"] > r["tuple_f1_baseline"]
        s["worse"] += r["tuple_f1_transformer"] < r["tuple_f1_baseline"]
    out["week2_error_recheck"] = {"by_category": by_cat, "items": recheck}

    # --- transformer errors on validation, for the Week 3 error table ---------
    errs = [{"feedback_id": fid, "text": t, "gold": sorted(f"{x}:{y}" for x, y in g),
             "pred": sorted(f"{x}:{y}" for x, y in p)}
            for fid, t, g, p in zip(va["ids"], va["texts"], gold, p_tr) if set(g) != p]
    out["transformer_error_count"] = len(errs)
    (RESULTS / "week3_errors_for_analysis.json").write_text(
        json.dumps(errs, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- demonstrations ---------------------------------------------------------
    demo = {"handout_example": fa.analyze(HANDOUT_EXAMPLE, "HANDOUT-1"),
            "guideline_examples": fa.analyze_many(GUIDELINE_EXAMPLES,
                                                  [f"G{i + 1}" for i in range(10)]),
            "batch_report_validation_sample": fa.analyze_batch(
                [{"feedback_id": va["ids"][i], "text": va["texts"][i]}
                 for i in range(120)], course_id="VALIDATION-SAMPLE-120")}
    (RESULTS / "week3_demo_outputs.json").write_text(
        json.dumps(demo, indent=2, ensure_ascii=False), encoding="utf-8")

    (RESULTS / a.out).write_text(json.dumps(out, indent=2, default=str))

    print(f"VALIDATION split, n={len(gold)} (test split untouched)")
    for r in rows:
        print(f"  {r['model']:<36} tuple-F1 {r['micro_f1']:.3f} "
              f"[{r['ci']['ci_low']:.3f}, {r['ci']['ci_high']:.3f}]  P={r['precision']:.3f} "
              f"R={r['recall']:.3f}  pol-acc={r['polarity_acc_on_detected']:.3f}  "
              f"aspect-F1={r['aspect_micro_f1']:.3f}")
    d = out["paired_diff_transformer_minus_baseline"]
    print(f"  difference {d['diff']:+.3f}  95% CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
    for k, v in out["task_a"].items():
        print(f"  doc sentiment {k:<12} acc {v['accuracy']:.3f}  macro-F1 {v['macro_f1']:.3f}")
    print(f"  spans: {out['spans']}")
    print(f"  latency: {out['latency_ms']}")
    print(f"  week-2 error recheck: {by_cat}")


if __name__ == "__main__":
    main()
