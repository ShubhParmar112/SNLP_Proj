"""End-to-end Week 1-2 baseline run.

    python src/run_baseline.py                          # smoke test on the fixture
    python src/run_baseline.py data/raw/annotations.csv # the real corpus

Produces every number the Week 1-2 report needs, including the reference baselines
and bootstrap confidence intervals, and writes them to results/ as JSON plus a
Markdown table ready to paste into the report.
"""

import json
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aspect_lexicon import AspectLexicon
from baselines import ABSABaseline, AspectBaseline, SentimentBaseline
from config import (ASPECTS, CV_FOLDS, DOC_SENTIMENTS, FIXTURES, PROCESSED,
                    RANDOM_SEED, RESULTS, SPLIT_PROPORTIONS)
from data import aspect_matrix, to_model_format, validate
from metrics import (absa_tuple_f1, absa_tuple_f1_per_category,
                     aspect_detection_metrics, bootstrap_ci, format_ci,
                     macro_tuple_f1, majority_tuple_baseline,
                     polarity_accuracy_on_detected, sentiment_metrics,
                     stratified_random_tuple_baseline)
from splitting import (group_near_duplicates, iterative_stratification,
                       report_split_balance)


def hr(title=""):
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def main(path=None, is_fixture=None):
    path = path or str(FIXTURES / "smoke_test_annotations.csv")
    if is_fixture is None:
        is_fixture = "fixture" in path or "smoke" in path

    if is_fixture:
        print("\n" + "!" * 72)
        print("!!  SMOKE TEST - NOT A RESULT".ljust(70) + "!!")
        print("!!  Numbers below prove the pipeline runs. They are computed on".ljust(70) + "!!")
        print("!!  fixture data and must NOT appear in the report as findings.".ljust(70) + "!!")
        print("!" * 72)

    # --- 1. Load and validate ------------------------------------------------
    hr("1. LOAD AND VALIDATE")
    df_long, problems = validate(path, strict=True)
    if problems:
        print("Fix the validation problems above before continuing.")
        return None
    print(f"Validation passed: {path}")

    df = to_model_format(df_long)
    Y = aspect_matrix(df)
    print(f"Comments            : {len(df)}")
    print(f"Aspect annotations  : {int(Y.sum())}")
    print(f"Comments w/o aspect : {int((Y.sum(axis=1) == 0).sum())}")
    print(f"Mean aspects/comment: {Y.sum(axis=1).mean():.2f}")

    print("\nDocument sentiment distribution:")
    for k, v in df["doc_sentiment"].value_counts().items():
        print(f"  {k:<10} {v:4d}  ({v/len(df):.1%})")

    print("\nAspect frequency:")
    for i, a in enumerate(ASPECTS):
        print(f"  {a:<22} {int(Y[:, i].sum()):4d}")

    pol_counts = pd.Series([p for s in df["tuples"] for _, p in s]).value_counts()
    print("\nPolarity distribution (aspect level):")
    for k, v in pol_counts.items():
        print(f"  {k:<10} {v:4d}  ({v/pol_counts.sum():.1%})")

    # --- 2. Split ------------------------------------------------------------
    hr("2. SPLIT (iterative stratification, near-duplicate grouped)")
    groups = group_near_duplicates(df["feedback_text"].tolist())
    n_groups = len(set(groups))
    if n_groups < len(df):
        print(f"Near-duplicate groups: {len(df)} comments -> {n_groups} groups "
              f"(kept together to prevent leakage)")

    # The split is computed ONCE and then reused (Evaluation_Protocol 4.1). Relabelling
    # the corpus (dataset v1.1) must not move comments between splits, or the test set
    # would change and no before/after comparison would mean anything.
    frozen_path = PROCESSED / "splits.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    id_to_row = {fid: i for i, fid in enumerate(df["feedback_id"])}
    if (not is_fixture and frozen
            and all(f in id_to_row for ids in frozen.values() for f in ids)):
        splits = {s: sorted(id_to_row[f] for f in ids) for s, ids in frozen.items()}
        print(f"Using the frozen split in {frozen_path}")
    else:
        splits = iterative_stratification(Y, SPLIT_PROPORTIONS, groups=groups,
                                          seed=RANDOM_SEED)
    for s, idx in splits.items():
        print(f"  {s:<11} {len(idx):4d}  ({len(idx)/len(df):.1%})")

    print("\nPer-label balance across splits:")
    print(f"  {'label':<22}{'total':>7}{'train':>7}{'val':>6}{'test':>6}")
    for row in report_split_balance(Y, splits, ASPECTS):
        print(f"  {row['label']:<22}{row['total']:>7}{row['train']:>7}"
              f"{row['validation']:>6}{row['test']:>6}")

    if is_fixture:      # never let a smoke test overwrite the real frozen split
        print("\n(smoke test: split not written)")
    elif not frozen:
        frozen_path.write_text(json.dumps(
            {s: df.iloc[i]["feedback_id"].tolist() for s, i in splits.items()}, indent=2))
        print(f"\nSplit frozen -> {frozen_path} (commit this file)")
    if not is_fixture:
        with open(RESULTS / "test_openings.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')}  test split opened  "
                    f"run_baseline.py on {path}\n")

    tr, va, te = splits["train"], splits["validation"], splits["test"]
    X_tr = df.iloc[tr]["feedback_text"].tolist()
    X_va = df.iloc[va]["feedback_text"].tolist()
    X_te = df.iloc[te]["feedback_text"].tolist()
    g_tr = df.iloc[tr]["tuples"].tolist()
    g_va = df.iloc[va]["tuples"].tolist()
    g_te = df.iloc[te]["tuples"].tolist()

    results = {"generated": datetime.now().isoformat(timespec="seconds"),
               "source": path, "is_smoke_test": is_fixture,
               "seed": RANDOM_SEED, "n_comments": len(df)}

    # --- 3. Lexicon ----------------------------------------------------------
    hr("3. ASPECT LEXICON (built from TRAINING SPLIT ONLY)")
    lex = AspectLexicon().build_from_training(X_tr, df.iloc[tr]["aspects"].tolist())
    lex.save(PROCESSED / "aspect_lexicon.json",
             provenance_path=PROCESSED / "lexicon_provenance.md")
    print(f"Terms per aspect: "
          + ", ".join(f"{a}={len(v)}" for a, v in sorted(lex.lexicon.items())))
    print(f"Provenance written -> {PROCESSED / 'lexicon_provenance.md'}")

    # --- 4. Task A -----------------------------------------------------------
    hr("4. TASK A - DOCUMENT SENTIMENT")
    y_tr = df.iloc[tr]["doc_sentiment"].tolist()
    y_te = df.iloc[te]["doc_sentiment"].tolist()

    maj = max(set(y_tr), key=y_tr.count)
    m_maj = sentiment_metrics(y_te, [maj] * len(y_te), DOC_SENTIMENTS)
    print(f"Majority class ({maj}): acc={m_maj['accuracy']:.3f}  "
          f"macro-F1={m_maj['macro_f1']:.3f}   <-- the number to beat")

    sent = SentimentBaseline().fit(X_tr, y_tr)
    m_sent = sentiment_metrics(y_te, sent.predict(X_te), DOC_SENTIMENTS)
    print(f"TF-IDF + LogReg      : acc={m_sent['accuracy']:.3f}  "
          f"macro-F1={m_sent['macro_f1']:.3f}")
    print("\n  per class:")
    for lbl, v in m_sent["per_class"].items():
        print(f"    {lbl:<10} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1']:.3f} n={v['support']}")
    results["task_a"] = {"majority": m_maj, "tfidf_logreg": m_sent}

    # --- 5. Task B -----------------------------------------------------------
    hr("5. TASK B - ASPECT CATEGORY DETECTION")
    asp_model = AspectBaseline().fit(X_tr, Y[tr])
    Y_pred = asp_model.predict(X_te)
    m_asp = aspect_detection_metrics(Y[te], Y_pred, ASPECTS)
    print(f"Learned (OvR LogReg) : micro-F1={m_asp['micro_f1']:.3f}  "
          f"macro-F1(all)={m_asp['macro_f1_all']:.3f}  "
          f"macro-F1(adequate support)={m_asp['macro_f1_adequate_support']:.3f}")

    lex_sets = lex.predict_sets(X_te)
    Y_lex = np.array([[int(a in s) for a in ASPECTS] for s in lex_sets])
    m_lex = aspect_detection_metrics(Y[te], Y_lex, ASPECTS)
    print(f"Lexicon only         : micro-F1={m_lex['micro_f1']:.3f}")

    print("\n  per category (learned model):")
    for a, v in m_asp["per_category"].items():
        flag = "  [LOW SUPPORT - do not read as a result]" if v["low_support"] else ""
        print(f"    {a:<22} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1']:.3f} n={v['support']}{flag}")
    results["task_b"] = {"learned": m_asp, "lexicon": m_lex}

    # --- 6. Task C -----------------------------------------------------------
    hr("6. TASK C - ABSA TUPLE EXTRACTION  (the headline task)")
    f1_of = lambda g, p: absa_tuple_f1(g, p)["f1"]

    p_maj = majority_tuple_baseline(g_tr, len(g_te))
    p_rnd = stratified_random_tuple_baseline(g_tr, len(g_te))

    absa = ABSABaseline().fit(X_tr, Y[tr], g_tr)
    p_absa = absa.predict_sets(X_te)

    absa_lex = ABSABaseline(use_lexicon=True, lexicon=lex).fit(X_tr, Y[tr], g_tr)
    p_lex = absa_lex.predict_sets(X_te)

    rows = []
    for name, preds in [("Majority class", p_maj), ("Stratified random", p_rnd),
                        ("Lexicon + LogReg polarity", p_lex),
                        ("TF-IDF LogReg (full baseline)", p_absa)]:
        ci = bootstrap_ci(g_te, preds, f1_of)
        d = absa_tuple_f1(g_te, preds)
        per = absa_tuple_f1_per_category(g_te, preds, ASPECTS)
        pol = polarity_accuracy_on_detected(g_te, preds)
        rows.append({"model": name, "micro_f1": d["f1"], "precision": d["precision"],
                     "recall": d["recall"], "ci": ci,
                     "macro_f1": macro_tuple_f1(per),
                     "polarity_acc_on_detected": pol["accuracy"],
                     "n_detected": pol["n_detected"], "per_category": per})
        print(f"{name:<32} micro-F1 = {format_ci(ci)}   "
              f"P={d['precision']:.3f} R={d['recall']:.3f}")

    print("\nPolarity accuracy on correctly detected aspects "
          "(separates 'wrong aspect' from 'wrong sentiment'):")
    for r in rows:
        if r["n_detected"]:
            print(f"  {r['model']:<32} {r['polarity_acc_on_detected']:.3f} "
                  f"(n={r['n_detected']})")
    results["task_c"] = rows

    best = max(rows, key=lambda r: r["micro_f1"])
    ref = max(rows[0]["micro_f1"], rows[1]["micro_f1"])
    if best["micro_f1"] <= ref:
        print("\n  WARNING: no model beats the reference baselines. Report this as a "
              "negative result with a diagnosis - do not hide it.")
    elif best["ci"]["ci_low"] <= ref:
        print(f"\n  NOTE: best model ({best['micro_f1']:.3f}) beats the reference "
              f"({ref:.3f}) but its 95% CI lower bound ({best['ci']['ci_low']:.3f}) "
              f"does not. State this as suggestive, not established.")

    # --- 7. Cross-validation -------------------------------------------------
    hr(f"7. {CV_FOLDS}-FOLD CV ON TRAIN+VAL (development estimate)")
    dev = tr + va
    Xd = df.iloc[dev]["feedback_text"].tolist()
    gd = df.iloc[dev]["tuples"].tolist()
    Yd = Y[dev]
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(dev))
    folds = np.array_split(order, CV_FOLDS)

    scores = []
    for k in range(CV_FOLDS):
        vi = folds[k]
        ti = np.concatenate([folds[j] for j in range(CV_FOLDS) if j != k])
        try:
            m = ABSABaseline().fit([Xd[i] for i in ti], Yd[ti], [gd[i] for i in ti])
            scores.append(absa_tuple_f1([gd[i] for i in vi],
                                        m.predict_sets([Xd[i] for i in vi]))["f1"])
        except Exception as e:
            print(f"  fold {k+1} failed: {e}")
    if scores:
        print(f"ABSA micro-F1 = {np.mean(scores):.3f} +/- {np.std(scores):.3f} "
              f"over {len(scores)} folds")
        print(f"  folds: {', '.join(f'{s:.3f}' for s in scores)}")
        print("  Report the mean +/- sd, not a single split. A wide spread here means "
              "the corpus is too small for a confident single-split claim.")
        results["cv"] = {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
                         "folds": [float(s) for s in scores]}

    # --- 8. Ablations --------------------------------------------------------
    hr("8. ABLATIONS (evidence, not assumption)")
    for label, kwargs in [("negation marking OFF", {"negation": False}),
                          ("negation marking ON ", {"negation": True})]:
        m = ABSABaseline(**kwargs).fit(X_tr, Y[tr], g_tr)
        f = absa_tuple_f1(g_te, m.predict_sets(X_te))["f1"]
        results.setdefault("ablations", {})[label.strip()] = f
        print(f"  {label}: micro-F1 = {f:.3f}")
    ablations = {}
    for label, kw in [("word n-grams only    ", {"use_char": False}),
                      ("word + char n-grams  ", {"use_char": True})]:
        m = ABSABaseline(**kw).fit(X_tr, Y[tr], g_tr)
        f = absa_tuple_f1(g_te, m.predict_sets(X_te))["f1"]
        ablations[label.strip()] = f
        print(f"  {label}: micro-F1 = {f:.3f}")
    for label, cw in [("class_weight=None    ", None),
                      ("class_weight=balanced", "balanced")]:
        m = SentimentBaseline(class_weight=cw).fit(X_tr, y_tr)
        mm = sentiment_metrics(y_te, m.predict(X_te), DOC_SENTIMENTS)
        results.setdefault("ablations", {})[label.strip()] = {
            "accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"]}
        print(f"  {label}: acc={mm['accuracy']:.3f} macro-F1={mm['macro_f1']:.3f}")
    results["ablations"].update(ablations)

    # --- 9. Error sampling ---------------------------------------------------
    hr("9. ERROR SAMPLE FOR MANUAL ANALYSIS (validation split)")
    p_va = absa.predict_sets(X_va)
    errs = []
    for text, gold, pred in zip(X_va, g_va, p_va):
        if set(gold) != set(pred):
            errs.append({"text": text,
                         "gold": sorted(f"{a}:{p}" for a, p in gold),
                         "pred": sorted(f"{a}:{p}" for a, p in pred),
                         "missed": sorted(f"{a}:{p}" for a, p in set(gold) - set(pred)),
                         "spurious": sorted(f"{a}:{p}" for a, p in set(pred) - set(gold)),
                         "category": ""})
    (RESULTS / "errors_for_analysis.json").write_text(json.dumps(errs, indent=2))
    print(f"{len(errs)} error(s) written -> {RESULTS / 'errors_for_analysis.json'}")
    print("Fill the 'category' field by hand using Evaluation_Protocol.md §6.")
    for e in errs[:3]:
        print(f"\n  text : {e['text']}")
        print(f"  gold : {e['gold']}")
        print(f"  pred : {e['pred']}")

    # --- 10. Write outputs ---------------------------------------------------
    hr("10. OUTPUTS")
    (RESULTS / "metrics.json").write_text(
        json.dumps(results, indent=2, default=str))

    caption = ("**SMOKE TEST - NOT A RESULT.** Computed on fixture data to verify the "
               "pipeline runs. Replace with real-corpus numbers before submission."
               if is_fixture else
               f"Test split, n={len(te)}, seed={RANDOM_SEED}. "
               f"95% bootstrap CI over 1000 resamples.")
    md = ["## Task C - ABSA tuple extraction (test split)", "", caption, "",
          "| Model | micro tuple-F1 | 95% CI | Precision | Recall | Polarity acc. on detected |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['micro_f1']:.3f} | "
                  f"[{r['ci']['ci_low']:.3f}, {r['ci']['ci_high']:.3f}] | "
                  f"{r['precision']:.3f} | {r['recall']:.3f} | "
                  f"{r['polarity_acc_on_detected']:.3f} |")
    md += ["", "## Task A - document sentiment", "",
           "| Model | Accuracy | Macro-F1 |", "|---|---|---|",
           f"| Majority class | {m_maj['accuracy']:.3f} | {m_maj['macro_f1']:.3f} |",
           f"| TF-IDF + LogReg | {m_sent['accuracy']:.3f} | {m_sent['macro_f1']:.3f} |"]
    (RESULTS / "results_tables.md").write_text("\n".join(md))

    print(f"  {RESULTS / 'metrics.json'}")
    print(f"  {RESULTS / 'results_tables.md'}  <- paste into the report")
    print(f"  {RESULTS / 'errors_for_analysis.json'}")
    print(f"  {PROCESSED / 'splits.json'}")
    print(f"  {PROCESSED / 'aspect_lexicon.json'}")

    if is_fixture:
        print("\n" + "!" * 72)
        print("!!  Reminder: every number above is a SMOKE TEST on fixture data.".ljust(70) + "!!")
        print("!" * 72)
    return results


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
