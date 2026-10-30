"""Week 4, step 2 of 2: the baseline-vs-improved comparison on the TEST split.

    python src/tune_week4.py       # first: every choice made on validation, then frozen
    python src/evaluate_week4.py   # then: the test split is opened, once

Evaluation_Protocol.md 4.2: the test set is opened at the end of Week 2 and at the end
of Week 4, and no decision is made on it. This script enforces the second half: it
refuses to run unless results/week4_frozen_settings.json exists and still matches
data/processed/decision_settings.json byte for byte, and it appends every opening to
results/week4_test_openings.log so the number of looks is on the record.

Every system is scored on the same 900 test comments with the same metric code.
"""

import hashlib
import json
from datetime import datetime

import numpy as np

from aspect_lexicon import AspectLexicon
from baselines import ABSABaseline, SentimentBaseline
from config import (ASPECTS, DATASET_TAG, DOC_SENTIMENTS, PROCESSED, RANDOM_SEED, RAW,
                    RESULTS)
from data import aspect_matrix, to_model_format, validate
from embedding_baseline import EmbeddingABSA, embed
from evaluate_week3 import HANDOUT_EXAMPLE, paired_bootstrap
from metrics import (absa_tuple_f1, absa_tuple_f1_per_category, aspect_detection_metrics,
                     bootstrap_ci, majority_tuple_baseline, polarity_accuracy_on_detected,
                     sentiment_metrics, stratified_random_tuple_baseline)
from pipeline import FeedbackAnalyzer
from train_transformer import load_split
from week4_common import (NO_SPAN_MODEL, SEED_MODELS, sets_from_probs, span_agreement,
                          transformer_outputs)


TAG = f"week4_{DATASET_TAG}"     # result files: the Week 4 procedure on this dataset version
HANDOUT_EXAMPLE_2 = "The lectures were good but the pace was fast."   # handout Section 21


def check_frozen():
    frozen = json.loads((RESULTS / f"{TAG}_frozen_settings.json").read_text())
    blob = (PROCESSED / "decision_settings.json").read_text()
    if hashlib.sha256(blob.encode()).hexdigest() != frozen["decision_settings_sha256"]:
        raise SystemExit("decision_settings.json changed after it was frozen - re-run "
                         "tune_week4.py on validation; the test split stays closed")
    return frozen


def row(name, gold, preds, group):
    d = absa_tuple_f1(gold, preds)
    per = absa_tuple_f1_per_category(gold, preds, ASPECTS)
    Yg = np.array([[int(any(x == a for x, _ in s)) for a in ASPECTS] for s in gold])
    Yp = np.array([[int(any(x == a for x, _ in s)) for a in ASPECTS] for s in preds])
    return {"model": name, "group": group, "micro_f1": d["f1"], "precision": d["precision"],
            "recall": d["recall"],
            "ci": bootstrap_ci(gold, preds, lambda g, p: absa_tuple_f1(g, p)["f1"]),
            "macro_f1": float(np.mean([v["f1"] for v in per.values()])),
            "polarity_acc_on_detected": polarity_accuracy_on_detected(gold, preds)["accuracy"],
            "aspect_micro_f1": aspect_detection_metrics(Yg, Yp, ASPECTS)["micro_f1"],
            "per_category": per}


def main():
    frozen = check_frozen()
    settings = frozen["decision_settings"]
    opened = datetime.now().isoformat(timespec="seconds")
    with open(RESULTS / "test_openings.log", "a", encoding="utf-8") as f:
        f.write(f"{opened}  test split opened  evaluate_week4.py ({TAG}), settings sha256 "
                f"{frozen['decision_settings_sha256'][:16]}\n")

    data = load_split(str(RAW / "annotations.csv"), names=("train", "test"))
    tr, te = data["train"], data["test"]
    gold = te["tuples"]
    out = {"split": "test", "n": len(gold), "opened": opened, "frozen_settings": frozen}
    rows, preds = [], {}

    # --- references and Week 1-2 baselines (refitted on train, as in Week 2) --
    df = to_model_format(validate(str(RAW / "annotations.csv"), strict=False)[0])
    by_id = df.set_index("feedback_id")
    Y_tr = aspect_matrix(by_id.loc[tr["ids"]].reset_index())
    preds["majority"] = majority_tuple_baseline(tr["tuples"], len(gold))
    preds["random"] = stratified_random_tuple_baseline(tr["tuples"], len(gold))
    lex = AspectLexicon().build_from_training(tr["texts"], by_id.loc[tr["ids"], "aspects"].tolist())
    preds["lexicon"] = ABSABaseline(use_lexicon=True, lexicon=lex).fit(
        tr["texts"], Y_tr, tr["tuples"]).predict_sets(te["texts"])
    preds["tfidf"] = ABSABaseline().fit(tr["texts"], Y_tr, tr["tuples"]).predict_sets(te["texts"])
    tfidf_doc = list(SentimentBaseline().fit(tr["texts"], tr["doc"]).predict(te["texts"]))

    # --- sentence-embedding model (frozen choices from validation) -------------
    X_tr, X_te = embed(tr["texts"]), embed(te["texts"])
    cw = frozen["embedding"]["class_weight"]
    emb = EmbeddingABSA(class_weight=None if cw == "None" else cw).fit(
        X_tr, tr["tuples"], tr["doc"])
    preds["emb_argmax"] = emb.predict_sets(X_te)
    preds["emb_tuned"] = emb.predict_sets(X_te, frozen["embedding"]["thresholds"])
    emb_doc = emb.doc_predict(X_te)

    # --- fine-tuned DeBERTa ------------------------------------------------------
    outs = {s: transformer_outputs(p, te["texts"]) for s, p in SEED_MODELS.items()}
    for s, o in outs.items():
        preds[f"v1_s{s}"] = sets_from_probs(o["slot_probs"])
    preds["v1.1"] = sets_from_probs(outs[42]["slot_probs"], settings["thresholds"])
    if frozen.get("ensemble_thresholds"):          # only if all three seeds exist
        P_ens = np.mean([o["slot_probs"] for o in outs.values()], axis=0)
        preds["ensemble"] = sets_from_probs(P_ens, frozen["ensemble_thresholds"])
    if (NO_SPAN_MODEL / "model.pt").exists():
        nospan = transformer_outputs(NO_SPAN_MODEL, te["texts"])
        preds["no_span_sup"] = sets_from_probs(nospan["slot_probs"])
    tr_doc = [DOC_SENTIMENTS[int(p.argmax())] for p in outs[42]["doc_probs"]]

    for key, name, group in [
            ("majority", "Majority class", "reference"),
            ("random", "Stratified random", "reference"),
            ("lexicon", "Lexicon + LogReg polarity (Week 2)", "baseline"),
            ("tfidf", "TF-IDF + LogReg (Week 2 baseline)", "baseline"),
            ("emb_argmax", "MiniLM embeddings + LogReg", "comparison"),
            ("emb_tuned", "MiniLM embeddings + LogReg, tuned thresholds", "comparison"),
            ("v1_s42", "DeBERTa ABSA v1 (Week 3)", "improved"),
            ("v1.1", "DeBERTa ABSA v1.1 (Week 4, the asset)", "improved"),
            ("ensemble", "3-seed ensemble, tuned (experiment)", "experiment"),
            ("no_span_sup", "v1 without span supervision (ablation)", "experiment")]:
        if key in preds:
            rows.append(row(name, gold, preds[key], group))
    out["task_c"] = rows
    seeds = [absa_tuple_f1(gold, preds[f"v1_s{s}"])["f1"] for s in SEED_MODELS]
    if len(seeds) > 1:
        out["v1_seeds"] = {"f1": seeds, "mean": float(np.mean(seeds)),
                           "sd": float(np.std(seeds, ddof=1))}

    out["paired"] = {k: paired_bootstrap(gold, preds[a], preds[b]) for k, (a, b) in {
        "v1.1 - TF-IDF baseline": ("tfidf", "v1.1"),
        "v1 - TF-IDF baseline": ("tfidf", "v1_s42"),
        "v1.1 - v1": ("v1_s42", "v1.1"),
        "embeddings(tuned) - TF-IDF baseline": ("tfidf", "emb_tuned"),
        "v1.1 - embeddings(tuned)": ("emb_tuned", "v1.1"),
        "ensemble - v1.1": ("v1.1", "ensemble"),
        "v1 - no span supervision": ("no_span_sup", "v1_s42")}.items()
        if a in preds and b in preds}

    out["task_a"] = {"majority": sentiment_metrics(te["doc"], ["positive"] * len(gold),
                                                   DOC_SENTIMENTS),
                     "tfidf": sentiment_metrics(te["doc"], tfidf_doc, DOC_SENTIMENTS),
                     "embeddings": sentiment_metrics(te["doc"], emb_doc, DOC_SENTIMENTS),
                     "transformer": sentiment_metrics(te["doc"], tr_doc, DOC_SENTIMENTS)}

    out["spans"] = {"v1 rule (peak, rel 0.5)": span_agreement(outs[42], preds["v1.1"],
                                                             te["spans"], {"rel": 0.5}),
                    f"v1.1 rule ({settings['span_rule_name']})": span_agreement(
                        outs[42], preds["v1.1"], te["spans"], settings["span_rule"])}

    # --- pipeline v1.1 end to end: must reproduce the v1.1 row exactly --------
    fa = FeedbackAnalyzer(SEED_MODELS[42])                 # loads the frozen settings
    res = fa.analyze_many(te["texts"], te["ids"])
    p_pipe = [{(x["aspect"], x["polarity"]) for x in r["aspects"]} for r in res]
    out["pipeline_check"] = {"model_version": fa.model_version,
                             "tuple_f1": absa_tuple_f1(gold, p_pipe)["f1"],
                             "matches_v1.1_row": p_pipe == [set(s) for s in preds["v1.1"]]}
    demo = {"handout_example_v1.1": fa.analyze(HANDOUT_EXAMPLE, "HANDOUT-1"),
            "handout_example_section21": fa.analyze(HANDOUT_EXAMPLE_2, "HANDOUT-2"),
            "course_report_all_test_comments": fa.analyze_batch(
                [{"feedback_id": i, "text": t} for i, t in zip(te["ids"][:500],
                                                               te["texts"][:500])],
                course_id="TEST-SPLIT-500")}
    (RESULTS / f"{TAG}_demo_outputs.json").write_text(
        json.dumps(demo, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / f"{TAG}_test_eval.json").write_text(json.dumps(out, indent=2, default=str))

    print(f"TEST split, n={len(gold)}  (opened {opened})")
    for r in rows:
        print(f"  {r['model']:<48} {r['micro_f1']:.3f} [{r['ci']['ci_low']:.3f}, "
              f"{r['ci']['ci_high']:.3f}]  P={r['precision']:.3f} R={r['recall']:.3f} "
              f"pol={r['polarity_acc_on_detected']:.3f} asp={r['aspect_micro_f1']:.3f} "
              f"macro={r['macro_f1']:.3f}")
    if len(seeds) > 1:
        print(f"  v1 seeds {['%.3f' % x for x in seeds]} mean {np.mean(seeds):.3f} "
              f"sd {np.std(seeds, ddof=1):.3f}")
    for k, d in out["paired"].items():
        print(f"  {k:<38} {d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
    for k, v in out["task_a"].items():
        print(f"  doc {k:<12} acc {v['accuracy']:.3f} macro-F1 {v['macro_f1']:.3f}")
    for k, v in out["spans"].items():
        print(f"  spans {k:<36} J>=0.5 {v['share_jaccard_ge_0.5']:.3f} overlap "
              f"{v['share_any_overlap']:.3f} words {v['median_words_pred']:.0f}")
    print(f"  pipeline check: {out['pipeline_check']}")
    for k in ("handout_example_v1.1", "handout_example_section21"):
        print(f"  {k}: {[(x['aspect'], x['polarity']) for x in demo[k]['aspects']]}")


if __name__ == "__main__":
    main()
