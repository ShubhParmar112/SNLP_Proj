"""Multi-label splitting.

sklearn's train_test_split(stratify=y) handles single-label targets only. Our aspect
labels are multi-label, so we use iterative stratification (Sechidis, Tsoumakas &
Vlahavas, 2011), which greedily places each example into the split with the largest
remaining need for its rarest label.

Also groups near-duplicate comments so that a near-copy of a training comment cannot
leak into the test set and inflate scores.
"""

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import NEAR_DUPLICATE_THRESHOLD, RANDOM_SEED


def group_near_duplicates(texts, threshold: float = NEAR_DUPLICATE_THRESHOLD):
    """Return a group id per text; near-duplicates share a group id.

    Uses character 3-5 gram TF-IDF cosine similarity, which catches paraphrases and
    typo variants that word-level matching misses. Union-find over the similarity
    graph, so transitively-similar comments end up in one group.
    """
    if len(texts) < 2:
        return list(range(len(texts)))

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)

    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i, j in zip(*np.where(np.triu(sim, k=1) >= threshold)):
        union(int(i), int(j))

    return [find(i) for i in range(len(texts))]


def iterative_stratification(Y, proportions, groups=None, seed=RANDOM_SEED):
    """Split a multi-label indicator matrix, preserving per-label distribution.

    Y           : (n_samples, n_labels) binary array
    proportions : dict of split name -> proportion, must sum to 1
    groups      : optional group ids; all members of a group go to the same split
    returns     : dict of split name -> list of row indices
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=int)
    n, n_labels = Y.shape
    names = list(proportions)

    if groups is None:
        groups = list(range(n))
    group_rows = defaultdict(list)
    for i, g in enumerate(groups):
        group_rows[g].append(i)
    units = list(group_rows.values())

    # Aggregate each group's label counts; the group is placed as one unit.
    unit_counts = np.array([Y[rows].sum(axis=0) for rows in units])
    unit_sizes = np.array([len(rows) for rows in units])

    desired_size = {s: proportions[s] * unit_sizes.sum() for s in names}
    desired_label = {
        s: unit_counts.sum(axis=0) * proportions[s] for s in names
    }
    assigned = {s: [] for s in names}

    # Process the rarest labels first — they have the least slack.
    remaining = set(range(len(units)))
    label_order = np.argsort(unit_counts.sum(axis=0))

    while remaining:
        target_label = None
        for lbl in label_order:
            if any(unit_counts[u, lbl] > 0 for u in remaining):
                target_label = lbl
                break

        if target_label is None:
            candidates = list(remaining)
        else:
            candidates = [u for u in remaining if unit_counts[u, target_label] > 0]

        for u in sorted(candidates, key=lambda x: -unit_sizes[x]):
            if target_label is None:
                scores = {s: desired_size[s] for s in names}
            else:
                scores = {s: desired_label[s][target_label] for s in names}
            best = max(scores.values())
            tied = [s for s in names if scores[s] == best]
            if len(tied) > 1:
                best_size = max(desired_size[s] for s in tied)
                tied = [s for s in tied if desired_size[s] == best_size]
            chosen = tied[0] if len(tied) == 1 else str(rng.choice(tied))

            assigned[chosen].append(u)
            desired_size[chosen] -= unit_sizes[u]
            desired_label[chosen] = desired_label[chosen] - unit_counts[u]
            remaining.discard(u)

    return {
        s: sorted(i for u in us for i in units[u])
        for s, us in assigned.items()
    }


def report_split_balance(Y, splits, label_names):
    """Per-label proportion in each split — include this table in the report."""
    Y = np.asarray(Y, dtype=int)
    rows = []
    for li, name in enumerate(label_names):
        row = {"label": name, "total": int(Y[:, li].sum())}
        for s, idx in splits.items():
            row[s] = int(Y[idx, li].sum()) if idx else 0
        rows.append(row)
    return rows
