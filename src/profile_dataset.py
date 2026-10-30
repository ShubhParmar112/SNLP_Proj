"""Dataset quality profile for the Week 1-2 report (Sections 5.2, 10 and 11.1).

    python src/profile_dataset.py [data/raw/annotations.csv]

Run after run_baseline.py, which freezes data/processed/splits.json. Every figure in
the report's dataset tables comes from results/dataset_profile.json, written here.
"""

import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

from config import ASPECTS, NEAR_DUPLICATE_THRESHOLD, POLARITIES, PROCESSED, RAW, RESULTS
from data import aspect_matrix, to_model_format, validate
from splitting import group_near_duplicates


def main(path):
    df_long, problems = validate(path, strict=True)
    if problems:
        sys.exit("fix validation problems first")
    df = to_model_format(df_long)
    Y = aspect_matrix(df)
    texts = df["feedback_text"].tolist()
    n_tokens = np.array([len(t.split()) for t in texts])

    groups = group_near_duplicates(texts)
    group_sizes = Counter(groups)
    in_near_dup_groups = sum(s for s in group_sizes.values() if s > 1)
    exact_dupes = int(pd.Series([t.strip().lower() for t in texts]).duplicated().sum())

    pol = Counter(p for s in df["tuples"] for _, p in s)
    asp = {a: int(Y[:, i].sum()) for i, a in enumerate(ASPECTS)}
    doc = Counter(df["doc_sentiment"])

    split_ids = json.loads((PROCESSED / "splits.json").read_text())
    id_to_row = {fid: i for i, fid in enumerate(df["feedback_id"])}
    splits = {s: [id_to_row[f] for f in ids] for s, ids in split_ids.items()}
    balance = []
    for i, a in enumerate(ASPECTS):
        row = {"label": a, "total": asp[a]}
        for s, idx in splits.items():
            row[s] = int(Y[idx, i].sum())
        balance.append(row)
    split_pol = {s: dict(Counter(p for i in idx for _, p in df.iloc[i]["tuples"]))
                 for s, idx in splits.items()}
    split_doc = {s: dict(Counter(df.iloc[idx]["doc_sentiment"])) for s, idx in splits.items()}

    notes = df_long.groupby("feedback_id")["notes"].first()
    source_type = Counter(n.split(";")[0].replace("source=", "") for n in notes)

    profile = {
        "n_comments": len(df),
        "n_aspect_annotations": int(Y.sum()),
        "mean_aspects_per_comment": float(Y.sum(axis=1).mean()),
        "comments_no_aspect": int((Y.sum(axis=1) == 0).sum()),
        "source_type": dict(source_type),
        "doc_sentiment": dict(doc),
        "aspect_counts": asp,
        "polarity_counts": {p: pol.get(p, 0) for p in POLARITIES},
        "tokens": {"median": float(np.median(n_tokens)), "mean": float(n_tokens.mean()),
                   "p95": float(np.percentile(n_tokens, 95)), "max": int(n_tokens.max())},
        "quality": {
            "missing_or_empty": int(sum(1 for t in texts if not t.strip())),
            "exact_duplicates": exact_dupes,
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "near_duplicate_groups": int(sum(1 for s in group_sizes.values() if s > 1)),
            "comments_in_near_duplicate_groups": int(in_near_dup_groups),
            "very_short_lt3_tokens": int((n_tokens < 3).sum()),
            "very_long_gt100_tokens": int((n_tokens > 100).sum()),
            "aspect_imbalance_max_over_min": round(max(asp.values()) / max(1, min(asp.values())), 1),
            "polarity_share": {p: round(pol.get(p, 0) / sum(pol.values()), 3) for p in POLARITIES},
            "non_ascii_comments": int(sum(1 for t in texts if any(ord(c) > 127 for c in t))),
            "uncertain_rows": int((df_long["uncertain"].astype(str) == "1").sum()),
        },
        "split_sizes": {s: len(i) for s, i in splits.items()},
        "split_balance": balance,
        "split_polarity": split_pol,
        "split_doc_sentiment": split_doc,
    }
    out = RESULTS / "dataset_profile.json"
    out.write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(RAW / "annotations.csv"))
