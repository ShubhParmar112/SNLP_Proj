"""Inter-annotator agreement.

Computes the three figures required by docs/Annotation_Guidelines.md §6. The absence
of any IAA number is the most common and most costly gap in student ABSA projects —
without it there is no evidence that the corpus is internally consistent.

Usage:
    python src/iaa.py data/raw/annotations.csv
"""

import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from config import ASPECTS, IAA_TARGET_KAPPA


def interpret(kappa: float) -> str:
    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


def load_long(path):
    """Load long-format annotations: one row per (feedback_id, annotator_id, aspect)."""
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"feedback_id", "annotator_id", "aspect_category", "polarity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"annotation file is missing columns: {sorted(missing)}")
    return df[df["annotator_id"] != "adjudicated"]


def aspect_kappa(df):
    """Per-category Cohen's kappa over the present/absent decision, plus pairwise F1."""
    by_ann = defaultdict(lambda: defaultdict(set))
    for _, r in df.iterrows():
        if r["aspect_category"]:
            by_ann[r["annotator_id"]][r["feedback_id"]].add(r["aspect_category"])

    annotators = sorted(by_ann)
    results, f1s = {}, []

    for a1, a2 in combinations(annotators, 2):
        shared = sorted(set(by_ann[a1]) & set(by_ann[a2]))
        if len(shared) < 10:
            continue
        Y1 = np.array([[int(a in by_ann[a1][f]) for a in ASPECTS] for f in shared])
        Y2 = np.array([[int(a in by_ann[a2][f]) for a in ASPECTS] for f in shared])
        f1s.append(f1_score(Y1, Y2, average="micro", zero_division=0))
        for i, asp in enumerate(ASPECTS):
            if len(set(Y1[:, i]) | set(Y2[:, i])) < 2:
                continue  # category absent for both — kappa undefined
            results.setdefault(asp, []).append(
                cohen_kappa_score(Y1[:, i], Y2[:, i])
            )

    per_cat = {a: float(np.mean(v)) for a, v in results.items() if v}
    return {
        "per_category": per_cat,
        "mean_kappa": float(np.mean(list(per_cat.values()))) if per_cat else float("nan"),
        "pairwise_micro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "n_pairs": len(f1s),
    }


def polarity_kappa(df):
    """Cohen's kappa on polarity, restricted to aspects both annotators identified."""
    by_ann = defaultdict(dict)
    for _, r in df.iterrows():
        if r["aspect_category"] and r["polarity"]:
            by_ann[r["annotator_id"]][(r["feedback_id"], r["aspect_category"])] = r["polarity"]

    kappas, n = [], 0
    for a1, a2 in combinations(sorted(by_ann), 2):
        shared = sorted(set(by_ann[a1]) & set(by_ann[a2]))
        if len(shared) < 10:
            continue
        y1 = [by_ann[a1][k] for k in shared]
        y2 = [by_ann[a2][k] for k in shared]
        if len(set(y1) | set(y2)) < 2:
            continue
        kappas.append(cohen_kappa_score(y1, y2))
        n += len(shared)
    return {"mean_kappa": float(np.mean(kappas)) if kappas else float("nan"),
            "n_shared_aspects": n, "n_pairs": len(kappas)}


def span_agreement(df, threshold=0.5):
    """Proportion of shared aspects whose spans overlap with Jaccard >= threshold.

    Exact span match is too strict for human annotators; character-level Jaccard is
    the standard relaxation.
    """
    by_ann = defaultdict(dict)
    for _, r in df.iterrows():
        try:
            s, e = int(r.get("span_start", -1)), int(r.get("span_end", -1))
        except (ValueError, TypeError):
            continue
        if r["aspect_category"] and s >= 0 and e > s:
            by_ann[r["annotator_id"]][(r["feedback_id"], r["aspect_category"])] = (s, e)

    hits = total = 0
    for a1, a2 in combinations(sorted(by_ann), 2):
        for k in set(by_ann[a1]) & set(by_ann[a2]):
            s1, e1 = by_ann[a1][k]
            s2, e2 = by_ann[a2][k]
            inter = max(0, min(e1, e2) - max(s1, s2))
            union = max(e1, e2) - min(s1, s2)
            total += 1
            hits += (union > 0 and inter / union >= threshold)
    return {"proportion_overlapping": hits / total if total else float("nan"),
            "n_compared": total}


def report(path):
    df = load_long(path)
    dbl = df[df.get("double_annotated", "0").astype(str) == "1"] \
        if "double_annotated" in df.columns else df

    asp = aspect_kappa(dbl)
    pol = polarity_kappa(dbl)
    spn = span_agreement(dbl)

    print("=" * 66)
    print("INTER-ANNOTATOR AGREEMENT")
    print("=" * 66)
    print(f"Double-annotated rows : {len(dbl)}")
    print(f"Annotators            : {sorted(dbl['annotator_id'].unique())}")

    print(f"\nAspect detection  mean kappa = {asp['mean_kappa']:.3f} "
          f"({interpret(asp['mean_kappa'])})")
    print(f"                  pairwise micro-F1 = {asp['pairwise_micro_f1']:.3f}")
    for a, k in sorted(asp["per_category"].items(), key=lambda x: x[1]):
        flag = "  <-- below target" if k < IAA_TARGET_KAPPA else ""
        print(f"    {a:<22} {k:6.3f}  {interpret(k)}{flag}")

    print(f"\nPolarity          kappa = {pol['mean_kappa']:.3f} "
          f"({interpret(pol['mean_kappa'])}) on {pol['n_shared_aspects']} shared aspects")
    print(f"Span overlap      {spn['proportion_overlapping']:.1%} at Jaccard>=0.5 "
          f"({spn['n_compared']} compared)")

    ok = (asp["mean_kappa"] >= IAA_TARGET_KAPPA and pol["mean_kappa"] >= IAA_TARGET_KAPPA)
    print("\n" + ("PASS - proceed to production annotation." if ok else
                  f"BELOW TARGET ({IAA_TARGET_KAPPA}) - revise the guidelines, "
                  "re-run the calibration round."))
    print("Categories with the lowest kappa are where the guidelines need"
          " sharper boundary rules.")
    return {"aspect": asp, "polarity": pol, "span": spn}


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "data/raw/annotations.csv")
