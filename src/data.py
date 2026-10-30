"""Loading and validating the long-format annotation file.

Long format (one row per feedback_id x annotator_id x aspect) is used instead of
JSON-in-a-cell because six people will be annotating in a spreadsheet, and a
malformed JSON string in row 240 of an Excel export is a much worse failure mode
than an extra row.
"""

import json
import sys

import numpy as np
import pandas as pd

from config import ASPECTS, DOC_SENTIMENTS, POLARITIES

COLUMNS = ["feedback_id", "feedback_text", "annotator_id", "doc_sentiment",
           "aspect_category", "polarity", "aspect_term", "span_start", "span_end",
           "uncertain", "double_annotated", "notes"]


def validate(path, strict=True):
    """Check the annotation file. Returns (dataframe, list_of_problems)."""
    df = pd.read_csv(path, dtype=str).fillna("")
    problems = []

    for c in COLUMNS:
        if c not in df.columns:
            problems.append(f"MISSING COLUMN: {c}")
    if problems:
        return df, problems

    for i, r in df.iterrows():
        ln = i + 2  # +1 header, +1 to 1-index
        if r["doc_sentiment"] and r["doc_sentiment"] not in DOC_SENTIMENTS:
            problems.append(f"line {ln}: bad doc_sentiment {r['doc_sentiment']!r}")
        if r["aspect_category"]:
            if r["aspect_category"] not in ASPECTS:
                problems.append(f"line {ln}: unknown aspect {r['aspect_category']!r}")
            if r["polarity"] not in POLARITIES:
                problems.append(f"line {ln}: bad polarity {r['polarity']!r} "
                                f"(must be one of {POLARITIES})")
        elif r["polarity"]:
            problems.append(f"line {ln}: polarity given with no aspect_category")

        # Span integrity: feedback_text[start:end] must equal aspect_term exactly.
        try:
            s, e = int(r["span_start"] or -1), int(r["span_end"] or -1)
        except ValueError:
            problems.append(f"line {ln}: non-integer span offsets")
            continue
        if s >= 0 and e > s and r["aspect_term"]:
            if r["feedback_text"][s:e] != r["aspect_term"]:
                problems.append(
                    f"line {ln}: span mismatch - text[{s}:{e}]="
                    f"{r['feedback_text'][s:e]!r} but aspect_term="
                    f"{r['aspect_term']!r}")

    # One comment must not carry two different document sentiments.
    for fid, g in df.groupby("feedback_id"):
        for ann, ga in g.groupby("annotator_id"):
            if ga["doc_sentiment"].replace("", np.nan).dropna().nunique() > 1:
                problems.append(f"{fid}/{ann}: conflicting doc_sentiment values")
        dupes = g.groupby(["annotator_id", "aspect_category"]).size()
        for (ann, asp), n in dupes.items():
            if asp and n > 1:
                problems.append(f"{fid}/{ann}: aspect {asp!r} annotated {n} times")

    if strict and problems:
        print(f"VALIDATION FAILED - {len(problems)} problem(s):")
        for p in problems[:40]:
            print("  " + p)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
    return df, problems


def to_model_format(df, annotator_priority=("adjudicated",)):
    """Collapse the long annotation file into one row per comment.

    Where a comment is double-annotated, an adjudicated row wins; otherwise the
    first annotator in sort order is used and the choice is reported, so the
    resolution rule is never silent.
    """
    df = df[df["feedback_id"] != ""].copy()
    records, unresolved = [], 0

    for fid, g in df.groupby("feedback_id", sort=True):
        annotators = list(g["annotator_id"].unique())
        chosen = next((a for a in annotator_priority if a in annotators), None)
        if chosen is None:
            if len(annotators) > 1:
                unresolved += 1
            chosen = sorted(annotators)[0]
        gg = g[g["annotator_id"] == chosen]

        tuples, aspects = set(), set()
        for _, r in gg.iterrows():
            if r["aspect_category"]:
                tuples.add((r["aspect_category"], r["polarity"]))
                aspects.add(r["aspect_category"])

        doc = gg["doc_sentiment"].replace("", np.nan).dropna()
        records.append({
            "feedback_id": fid,
            "feedback_text": gg.iloc[0]["feedback_text"],
            "doc_sentiment": doc.iloc[0] if len(doc) else "neutral",
            "aspects": sorted(aspects),
            "tuples": tuples,
            "annotator_id": chosen,
        })

    out = pd.DataFrame(records)
    if unresolved:
        print(f"  NOTE: {unresolved} double-annotated comment(s) had no adjudicated "
              f"row; used first annotator in sort order. Adjudicate these before "
              f"reporting final results.")
    return out


def aspect_matrix(df):
    """Multi-label indicator matrix, shape (n_comments, len(ASPECTS))."""
    return np.array([[int(a in row) for a in ASPECTS] for row in df["aspects"]])


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/annotations.csv"
    _, probs = validate(path)
    if not probs:
        print(f"VALIDATION PASSED: {path}")
    sys.exit(1 if probs else 0)
