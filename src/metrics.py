"""Evaluation metrics (see docs/Evaluation_Protocol.md).

The key function is absa_tuple_f1: an ABSA prediction is correct only when BOTH the
aspect category and its polarity match. Reporting aspect-detection F1 and polarity
accuracy separately, as the original plan did, overstates performance because it never
penalises getting the right aspect with the wrong sentiment.
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from config import LOW_SUPPORT_THRESHOLD, N_BOOTSTRAP, RANDOM_SEED


# --- Task C: ABSA tuple extraction -------------------------------------------

def absa_tuple_f1(gold_sets, pred_sets):
    """Exact-match micro P/R/F1 over (aspect, polarity) tuples.

    gold_sets, pred_sets: lists of sets of (aspect, polarity) tuples, one per comment.
    """
    tp = fp = fn = 0
    for gold, pred in zip(gold_sets, pred_sets):
        gold, pred = set(gold), set(pred)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def absa_tuple_f1_per_category(gold_sets, pred_sets, aspects):
    """Per-category tuple F1 with support, flagging low-support categories."""
    counts = {a: {"tp": 0, "fp": 0, "fn": 0} for a in aspects}
    for gold, pred in zip(gold_sets, pred_sets):
        gold, pred = set(gold), set(pred)
        for a in aspects:
            g = {t for t in gold if t[0] == a}
            p = {t for t in pred if t[0] == a}
            counts[a]["tp"] += len(g & p)
            counts[a]["fp"] += len(p - g)
            counts[a]["fn"] += len(g - p)

    out = {}
    for a, c in counts.items():
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = c["tp"] + c["fn"]
        out[a] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": support,
                  "low_support": support < LOW_SUPPORT_THRESHOLD}
    return out


def macro_tuple_f1(per_category, exclude_low_support=True):
    """Macro average, excluding low-support categories from the headline figure."""
    vals = [v["f1"] for v in per_category.values()
            if not (exclude_low_support and v["low_support"])]
    return float(np.mean(vals)) if vals else 0.0


def polarity_accuracy_on_detected(gold_sets, pred_sets):
    """Polarity accuracy restricted to aspects the model detected correctly.

    Separates 'found the wrong aspect' from 'found the aspect, misjudged sentiment'.
    A model can have poor tuple F1 for either reason and the fix is different.
    """
    correct = total = 0
    for gold, pred in zip(gold_sets, pred_sets):
        gold_map = dict(gold)
        for aspect, polarity in pred:
            if aspect in gold_map:
                total += 1
                correct += (gold_map[aspect] == polarity)
    return {"accuracy": correct / total if total else 0.0, "n_detected": total}


# --- Tasks A and B -----------------------------------------------------------

def sentiment_metrics(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": f1_score(y_true, y_pred, labels=labels,
                             average="macro", zero_division=0),
        "per_class": {
            lbl: {"precision": float(p[i]), "recall": float(r[i]),
                  "f1": float(f[i]), "support": int(s[i])}
            for i, lbl in enumerate(labels)
        },
    }


def aspect_detection_metrics(Y_true, Y_pred, aspects):
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    p, r, f, s = precision_recall_fscore_support(
        Y_true, Y_pred, average=None, zero_division=0,
        labels=list(range(len(aspects)))
    )
    per = {
        a: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
            "support": int(s[i]), "low_support": int(s[i]) < LOW_SUPPORT_THRESHOLD}
        for i, a in enumerate(aspects)
    }
    macro_pool = [v["f1"] for v in per.values() if not v["low_support"]]
    return {
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "macro_f1_all": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "macro_f1_adequate_support": float(np.mean(macro_pool)) if macro_pool else 0.0,
        "per_category": per,
    }


# --- Uncertainty -------------------------------------------------------------

def bootstrap_ci(gold_sets, pred_sets, metric_fn, n=N_BOOTSTRAP,
                 seed=RANDOM_SEED, alpha=0.05):
    """Percentile bootstrap CI over comments.

    Mandatory on a corpus of a few hundred comments: a point estimate alone cannot
    support a claim that one model beats another.
    """
    rng = np.random.default_rng(seed)
    gold_sets, pred_sets = list(gold_sets), list(pred_sets)
    k = len(gold_sets)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        scores.append(metric_fn([gold_sets[i] for i in idx],
                                [pred_sets[i] for i in idx]))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": metric_fn(gold_sets, pred_sets),
            "ci_low": float(lo), "ci_high": float(hi)}


def format_ci(result, digits=3):
    return (f"{result['point']:.{digits}f} "
            f"[{result['ci_low']:.{digits}f}, {result['ci_high']:.{digits}f}]")


# --- Reference baselines -----------------------------------------------------

def majority_tuple_baseline(train_gold_sets, n_test):
    """Always predict the single most frequent (aspect, polarity) tuple."""
    counts = defaultdict(int)
    for s in train_gold_sets:
        for t in s:
            counts[t] += 1
    if not counts:
        return [set() for _ in range(n_test)]
    top = max(counts, key=counts.get)
    return [{top} for _ in range(n_test)]


def stratified_random_tuple_baseline(train_gold_sets, n_test, seed=RANDOM_SEED):
    """Sample a tuple set from the empirical training distribution of tuple sets."""
    rng = np.random.default_rng(seed)
    pool = [frozenset(s) for s in train_gold_sets]
    if not pool:
        return [set() for _ in range(n_test)]
    return [set(pool[i]) for i in rng.integers(0, len(pool), n_test)]
