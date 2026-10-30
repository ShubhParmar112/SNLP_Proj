"""Week 4, step 1 of 2: choose every decision setting on the VALIDATION split, then freeze.

    python src/tune_week4.py

Nothing here reads the test split. What is chosen:
  * the embedding model's class weighting (None vs balanced);
  * presence thresholds for the embedding model, the DeBERTa model (-> v1.1) and the
    3-seed ensemble, by the same procedure (week4_common.tune_thresholds);
  * the evidence-span growth rule (fraction of the attention peak a neighbour needs).
The minimum-mentions rule for course reports is fixed by design, not tuned.

Writes data/processed/decision_settings.json (what pipeline v1.1 loads) and
results/week4_frozen_settings.json (every choice plus its SHA-256). evaluate_week4.py
refuses to open the test split without the frozen file.
"""

import hashlib
import json
from datetime import datetime

import numpy as np

from config import ASPECTS, DATASET_TAG, DATASET_VERSION, PROCESSED, RAW, RESULTS
from embedding_baseline import MODEL_NAME, EmbeddingABSA, embed
from metrics import absa_tuple_f1, polarity_accuracy_on_detected
from train_transformer import load_split
from week4_common import (SEED_MODELS, sets_from_probs, span_agreement, transformer_outputs,
                          tune_thresholds)

# Week 3 rule (attention-peak growth) at its default and at its widest useful setting,
# against the Week 4 clause rule at several word caps.
SPAN_RULES = {"peak rel=0.5 (v1)": {"rel": 0.5},
              "peak rel=0.01": {"rel": 0.01},
              "clause, max 6 words": {"rel": 0.5, "clause_max_words": 6},
              "clause, max 8 words": {"rel": 0.5, "clause_max_words": 8},
              "clause, max 10 words": {"rel": 0.5, "clause_max_words": 10},
              "clause, max 14 words": {"rel": 0.5, "clause_max_words": 14},
              "clause, no cap": {"rel": 0.5, "clause_max_words": 0}}
MIN_MENTIONS = {"absolute": 3, "share": 0.05}   # design rule: >= 3 and >= 5% of comments
TAG = f"week4_{DATASET_TAG}"     # result files: the Week 4 procedure on this dataset version


def summary(gold, pred):
    t = absa_tuple_f1(gold, pred)
    return {"tuple_f1": t["f1"], "precision": t["precision"], "recall": t["recall"],
            "polarity_acc_on_detected": polarity_accuracy_on_detected(gold, pred)["accuracy"]}


def main():
    data = load_split(str(RAW / "annotations.csv"), names=("train", "validation"))
    tr, va = data["train"], data["validation"]
    gold = va["tuples"]
    support = {a: sum(any(x == a for x, _ in s) for s in gold) for a in ASPECTS}
    log = {"split": "validation", "n": len(gold), "support": support}

    # --- 1. embedding model ---------------------------------------------------
    X_tr, X_va = embed(tr["texts"]), embed(va["texts"])
    cands = {}
    for cw in (None, "balanced"):
        m = EmbeddingABSA(class_weight=cw).fit(X_tr, tr["tuples"], tr["doc"])
        cands[str(cw)] = (summary(gold, m.predict_sets(X_va)), m)
    cw_best = max(cands, key=lambda k: cands[k][0]["tuple_f1"])
    emb = cands[cw_best][1]
    th_emb, tl_emb = tune_thresholds(emb.slot_probs(X_va), gold, support)
    log["embedding"] = {"encoder": MODEL_NAME,
                        "class_weight_candidates": {k: v[0] for k, v in cands.items()},
                        "class_weight": cw_best, "thresholds": th_emb, "tuning": tl_emb,
                        "tuned": summary(gold, emb.predict_sets(X_va, th_emb))}
    print(f"embedding: class_weight={cw_best}  argmax {tl_emb['argmax_v1']:.3f} -> "
          f"tuned {tl_emb['per_aspect']['f1']:.3f}")

    # --- 2. DeBERTa seeds, v1.1 thresholds, ensemble --------------------------
    outs = {s: transformer_outputs(p, va["texts"]) for s, p in SEED_MODELS.items()}
    P42 = outs[42]["slot_probs"]
    th_v11, tl_v11 = tune_thresholds(P42, gold, support)
    log["transformer_v1.1"] = {"thresholds": th_v11, "tuning": tl_v11,
                               "v1": summary(gold, sets_from_probs(P42)),
                               "v1.1": summary(gold, sets_from_probs(P42, th_v11))}
    print(f"v1 {tl_v11['argmax_v1']:.3f} -> v1.1 {tl_v11['per_aspect']['f1']:.3f}")
    th_ens = None
    if len(outs) >= 3:                 # ensemble only when all three seeds were trained
        P_ens = np.mean([o["slot_probs"] for o in outs.values()], axis=0)
        th_ens, tl_ens = tune_thresholds(P_ens, gold, support)
        log["ensemble_3_seeds"] = {"thresholds": th_ens, "tuning": tl_ens,
                                   "argmax": summary(gold, sets_from_probs(P_ens)),
                                   "tuned": summary(gold, sets_from_probs(P_ens, th_ens))}
        print(f"ensemble {tl_ens['argmax_v1']:.3f} -> {tl_ens['per_aspect']['f1']:.3f}")

    # --- 3. evidence-span growth rule -----------------------------------------
    sets_v11 = sets_from_probs(P42, th_v11)
    spans = {name: span_agreement(outs[42], sets_v11, va["spans"], rule)
             for name, rule in SPAN_RULES.items()}
    best_name = max(spans, key=lambda r: spans[r]["share_jaccard_ge_0.5"])
    log["span_rule"] = {"candidates": spans, "chosen": best_name,
                        "criterion": "share of spans with token Jaccard >= 0.5"}
    for name, v in spans.items():
        print(f"  {name:<22} J>=0.5 {v['share_jaccard_ge_0.5']:.3f}  overlap "
              f"{v['share_any_overlap']:.3f}  words {v['median_words_pred']:.0f} "
              f"(gold {v['median_words_gold']:.0f})")
    print("span rule ->", best_name)

    # --- 4. freeze ---------------------------------------------------------------
    settings = {"model_version": "transformer-absa-0.2.1",
                "dataset_version": DATASET_VERSION,
                "weights": str(SEED_MODELS[42]), "thresholds": th_v11,
                "span_rule": SPAN_RULES[best_name], "span_rule_name": best_name,
                "min_mentions": MIN_MENTIONS,
                "chosen_on": f"validation split ({len(gold)} comments)",
                "frozen_at": datetime.now().isoformat(timespec="seconds")}
    blob = json.dumps(settings, indent=2, sort_keys=True)
    (PROCESSED / "decision_settings.json").write_text(blob)
    frozen = {"decision_settings": settings,
              "decision_settings_sha256": hashlib.sha256(blob.encode()).hexdigest(),
              "embedding": {"encoder": MODEL_NAME, "class_weight": cw_best,
                            "thresholds": th_emb},
              "ensemble_thresholds": th_ens,
              "asset_uses_ensemble": False,
              "asset_uses_ensemble_reason": "3x inference cost and 3x GPU memory; "
                                            "reported as an experiment only"}
    (RESULTS / f"{TAG}_frozen_settings.json").write_text(json.dumps(frozen, indent=2))
    (RESULTS / f"{TAG}_tuning.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"frozen -> {PROCESSED / 'decision_settings.json'}")


if __name__ == "__main__":
    main()
