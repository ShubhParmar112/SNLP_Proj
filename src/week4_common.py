"""Shared Week 4 helpers: model outputs, threshold tuning and span scoring.

Used by tune_week4.py (validation only) and evaluate_week4.py (the one test run), so
both apply exactly the same decision rules.
"""

import gc
from pathlib import Path

import numpy as np
import torch

from config import ASPECTS, CHECKPOINT_ROOT, DATASET_TAG
from metrics import absa_tuple_f1
from pipeline import FeedbackAnalyzer
from transformer_absa import decide_slots, decode_slots, evidence_span, predict

CHECKPOINTS = CHECKPOINT_ROOT     # see config.py
# Checkpoints trained on the current dataset version. Only seed 42 (the asset) is
# required; extra seeds and the no-span ablation are used only if they exist.
SEED_MODELS = {s: CHECKPOINTS / f"deberta-v3-base_sw0.5_s{s}_{DATASET_TAG}"
               for s in (42, 43, 44)}
SEED_MODELS = {s: p for s, p in SEED_MODELS.items()
               if s == 42 or (p / "model.pt").exists()}
NO_SPAN_MODEL = CHECKPOINTS / f"deberta-v3-base_sw0_s42_{DATASET_TAG}"

GRID = [round(x, 2) for x in np.arange(0.20, 0.81, 0.05)]
MIN_SUPPORT_FOR_OWN_THRESHOLD = 50       # rarer aspects share the global threshold


def transformer_outputs(model_dir, texts):
    """Raw slot probabilities, document probabilities, attention and token offsets."""
    fa = FeedbackAnalyzer(model_dir, settings=None, calibration=None)
    items = fa.encoder(texts)
    preds = predict(fa.model, items, fa.pad_id, fa.device)
    out = {"slot_probs": np.stack([p["slot_probs"] for p in preds]),
           "doc_probs": np.stack([p["doc_probs"] for p in preds]),
           "alpha": [p["alpha"] for p in preds],
           "items": items}
    del fa
    gc.collect()
    torch.cuda.empty_cache()
    return out


def sets_from_probs(P, thresholds=None):
    return [decode_slots(decide_slots(p, thresholds)) for p in P]


def tune_thresholds(P, gold, support):
    """Presence thresholds chosen on validation.

    1. One global threshold, searched over GRID.
    2. Aspects with at least MIN_SUPPORT_FOR_OWN_THRESHOLD validation examples then get
       their own threshold by coordinate ascent (two passes). Rarer aspects keep the
       global one: a cut-off fitted to 21 pace comments would be fitted to noise.
    Returns (thresholds, log of validation micro tuple-F1 at each step).
    """
    f1 = lambda th: absa_tuple_f1(gold, sets_from_probs(P, th))["f1"]
    log = {"argmax_v1": absa_tuple_f1(gold, sets_from_probs(P, None))["f1"]}
    best_g = max(GRID, key=lambda g: f1({a: g for a in ASPECTS}))
    th = {a: best_g for a in ASPECTS}
    log["global"] = {"threshold": best_g, "f1": f1(th)}
    own = [a for a in ASPECTS if support[a] >= MIN_SUPPORT_FOR_OWN_THRESHOLD]
    for _ in range(2):
        for a in own:
            scores = {g: f1({**th, a: g}) for g in GRID}
            th[a] = max(scores, key=scores.get)
    log["per_aspect"] = {"f1": f1(th), "tuned_aspects": own,
                         "shared_global": [a for a in ASPECTS if a not in own]}
    return th, log


def token_jaccard(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / len(a | b) if a | b else 0.0


def span_agreement(out, pred_sets, gold_spans, rule):
    """Agreement between evidence spans and gold spans on correctly detected aspects."""
    jac, lens = [], []
    for k, (ps, sp) in enumerate(zip(pred_sets, gold_spans)):
        item = out["items"][k]
        for aspect, _ in ps:
            if aspect not in sp:
                continue
            ev = evidence_span(out["alpha"][k][ASPECTS.index(aspect)], item["offsets"],
                               item["text"], **rule)
            if ev:
                jac.append(token_jaccard(ev["text"], sp[aspect]))
                lens.append((len(ev["text"].split()), len(sp[aspect].split())))
    j, L = np.array(jac), np.array(lens)
    return {"n": int(len(j)), "share_jaccard_ge_0.5": float((j >= 0.5).mean()),
            "share_any_overlap": float((j > 0).mean()), "mean_jaccard": float(j.mean()),
            "median_words_pred": float(np.median(L[:, 0])),
            "median_words_gold": float(np.median(L[:, 1]))}
