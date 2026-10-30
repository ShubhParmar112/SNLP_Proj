"""Week 7: the Group 5 -> Group 6 (Early Warning) handoff.

Group 6 asked for feedback comments as {"items": [{"id": "FB001", "text": "..."}]} and
return, per id, a support-signal category, urgency, confidence and recommended action.

    python src/integration_group6.py request          # build the file we send them
    python src/integration_group6.py mock             # stand-in reply, until theirs arrives
    python src/integration_group6.py merge integration/group6_response_MOCK.json

request  Picks comments from the TRAIN split only (validation and test stay untouched),
         plus the hand-written comments in data/integration/group6_synthetic.csv, which are
         marked "synthetic": true. Comments that may name a person are skipped, and the
         rest are redacted with the course-report rule (pipeline.redact_quote). Writes:
           integration/group6_request.json                exactly Group 6's format
           integration/group6_request_with_analysis.json  + our aspects per comment
           integration/group6_key.json                    local only: where each id came from
mock     A keyword stand-in for Group 6, so the merge step can be built and tested now.
         Its output is labelled MOCK everywhere and is never reported as Group 6's result.
merge    Reads a reply, checks every id against what we sent, and writes counts only
         (per category, per urgency, and which of our aspects co-occur) to results/.
         No comment text and no id appears in the merged summary.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from config import DATA, PROCESSED, RANDOM_SEED, RAW, RESULTS, ROOT

OUT = ROOT / "integration"
SYNTHETIC = DATA / "integration" / "group6_synthetic.csv"
N_TRAIN_NEGATIVE = 15        # train comments with at least one negative aspect
N_TRAIN_OTHER = 15           # and the same number drawn from the rest
MAX_CHARS_SENT = 400         # keeps the sample readable for a demo
CATEGORIES = ["academic", "financial", "mental_health", "harassment", "administrative"]


# --- request -------------------------------------------------------------------

def may_contain_name(text, lowercase_vocab):
    """True if a word looks like a name: capitalised mid-sentence ("went to Martin's
    office"), or capitalised anywhere and never written in lowercase in the whole corpus
    ("Matt is great"). Deliberately over-cautious: it also drops "Japanese" or
    "Biology", which costs nothing because the train split has thousands of comments.
    Names written in lowercase still get through; the Week 7 report says so."""
    for sentence in re.split(r"[.!?]+\s+", text):
        for k, tok in enumerate(re.findall(r"[A-Za-z][\w']*", sentence)):
            if not tok[0].isupper() or tok.isupper() or tok == "I" or tok.startswith("I'"):
                continue
            if k > 0 or tok.lower().split("'")[0] not in lowercase_vocab:
                return True
    return False


def build_request():
    from pipeline import FeedbackAnalyzer, redact_quote

    ann = pd.read_csv(RAW / "annotations.csv")
    train_ids = set(json.loads((PROCESSED / "splits.json").read_text())["train"])
    ann = ann[ann["feedback_id"].isin(train_ids)]
    # Redaction misses bare names ("Kimsey is great"), so comments that may name a person
    # are not sent at all, nor any comment EduRABSA files under Personal traits.
    traits = set(ann.loc[ann["notes"].str.contains("Personal traits", na=False),
                         "feedback_id"])
    all_texts = pd.read_csv(RAW / "annotations.csv")["feedback_text"].drop_duplicates()
    vocab = {w for t in all_texts for w in re.findall(r"\b[a-z][a-z']*", t)}
    vocab = {w.split("'")[0] for w in vocab}
    texts = ann.groupby("feedback_id")["feedback_text"].first()
    texts = texts[(texts.str.len() <= MAX_CHARS_SENT) & ~texts.index.isin(traits)
                  & ~texts.map(lambda t: may_contain_name(t, vocab))]
    has_neg = set(ann.loc[ann["polarity"] == "negative", "feedback_id"]) & set(texts.index)

    rng = random.Random(RANDOM_SEED)
    neg = rng.sample(sorted(has_neg), N_TRAIN_NEGATIVE)
    rest = rng.sample(sorted(set(texts.index) - has_neg), N_TRAIN_OTHER)
    picked = neg + rest
    rng.shuffle(picked)

    rows = []
    for fid in picked:
        raw = texts[fid]
        sent = redact_quote(raw)
        rows.append({"source_id": fid, "split": "train", "synthetic": False,
                     "text": sent, "redacted": sent != raw, "expected_category": None})
    with open(SYNTHETIC, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"source_id": r["synthetic_id"], "split": None, "synthetic": True,
                         "text": r["text"], "redacted": False,
                         "expected_category": r["expected_category"],
                         "source_note": r["source"]})
    for i, r in enumerate(rows, 1):
        r["id"] = f"FB{i:03d}"

    fa = FeedbackAnalyzer()
    analyses = fa.analyze_many([r["text"] for r in rows], [r["id"] for r in rows],
                               include_spans=False)

    OUT.mkdir(exist_ok=True)
    plain = {"items": [{"id": r["id"], "text": r["text"]} for r in rows]}
    rich = {"from": "Group 5, Student Feedback Intelligence",
            "model_version": fa.model_version,
            "notes": ["Comments are public course and teacher reviews (EduRABSA, CC BY-NC-SA "
                      "4.0) with names and emails redacted, plus hand-written examples "
                      "marked synthetic: true. None comes from an NMIMS student.",
                      "Advisory only: no result may be used to identify or act on an "
                      "individual."],
            "items": [{"id": r["id"], "text": r["text"], "synthetic": r["synthetic"],
                       "group5_analysis": {
                           "doc_sentiment": a["doc_sentiment"]["label"],
                           "aspects": [{"aspect": x["aspect"], "polarity": x["polarity"],
                                        "polarity_score": x["polarity_score"]}
                                       for x in a["aspects"]]}}
                      for r, a in zip(rows, analyses)]}
    key = {"created": date.today().isoformat(), "seed": RANDOM_SEED,
           "items": [{k: r.get(k) for k in ("id", "source_id", "split", "synthetic",
                                            "redacted", "expected_category", "source_note")}
                     for r in rows]}
    for name, obj in (("group6_request.json", plain),
                      ("group6_request_with_analysis.json", rich),
                      ("group6_key.json", key)):
        (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {"n_items": len(rows),
               "n_train": sum(not r["synthetic"] for r in rows),
               "n_train_with_negative_aspect": N_TRAIN_NEGATIVE,
               "n_synthetic": sum(r["synthetic"] for r in rows),
               "n_redacted": sum(r["redacted"] for r in rows),
               "splits_used": ["train"],
               "files": ["integration/group6_request.json",
                         "integration/group6_request_with_analysis.json"]}
    (RESULTS / "week7_group6_request.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# --- mock Group 6 -------------------------------------------------------------

MOCK_RULES = [  # (category, urgency, recommended action, trigger words)
    ("mental_health", "high", "Counsellor follow-up", ["anxious", "unable to sleep", "depress",
                                                       "stress", "overwhelm"]),
    ("harassment", "high", "Refer to the grievance committee", ["uncomfortable", "harass",
                                                                "bully"]),
    ("academic", "high", "Academic advisor follow-up", ["dropping out", "drop out",
                                                        "fallen behind", "not be able to "
                                                        "complete", "fail"]),
    ("financial", "medium", "Financial aid office", ["fees", "part-time job", "afford"]),
    ("administrative", "medium", "Route to administration", ["scholarship", "office",
                                                             "form", "pending"]),
]


def mock_response(request_path):
    req = json.loads(Path(request_path).read_text(encoding="utf-8"))
    out = []
    for it in req["items"]:
        low = it["text"].lower()
        hit = next((r for r in MOCK_RULES if any(w in low for w in r[3])), None)
        out.append({"id": it["id"],
                    "category": hit[0] if hit else "none",
                    "urgency": hit[1] if hit else "low",
                    "confidence": 0.9 if hit else 0.6,
                    "recommended_action": hit[2] if hit else "No action"})
    resp = {"source": "MOCK - Group 5 keyword stand-in for Group 6; not Group 6's system",
            "items": out}
    path = OUT / "group6_response_MOCK.json"
    path.write_text(json.dumps(resp, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)} ({len(out)} items, MOCK)")


# --- merge ---------------------------------------------------------------------

def _items(resp):
    if isinstance(resp, list):
        return resp
    for k in ("items", "results", "data"):
        if isinstance(resp.get(k), list):
            return resp[k]
    raise ValueError("reply has no list of items (looked for items / results / data)")


def merge(response_path, tag=None):
    resp = json.loads(Path(response_path).read_text(encoding="utf-8"))
    is_mock = isinstance(resp, dict) and "MOCK" in str(resp.get("source", ""))
    tag = tag or ("mock" if is_mock else "group6")
    sent = json.loads((OUT / "group6_request_with_analysis.json").read_text(encoding="utf-8"))
    key = {k["id"]: k for k in json.loads((OUT / "group6_key.json").read_text())["items"]}
    ours = {it["id"]: it for it in sent["items"]}

    got = {}
    problems = Counter()
    for it in _items(resp):
        i = it.get("id")
        if i not in ours:
            problems["unknown_id"] += 1
            continue
        if i in got:
            problems["duplicate_id"] += 1
        cat = (it.get("category") or "none").strip().lower().replace(" ", "_")
        got[i] = {"category": cat, "urgency": str(it.get("urgency", "unknown")).lower(),
                  "confidence": it.get("confidence")}
    problems["missing_id"] = len(set(ours) - set(got))

    flagged = {i: g for i, g in got.items() if g["category"] not in ("none", "no_signal")}
    aspect_cooc = {}
    for i, g in flagged.items():
        c = aspect_cooc.setdefault(g["category"], Counter())
        for a in ours[i]["group5_analysis"]["aspects"]:
            c[f"{a['aspect']}:{a['polarity']}"] += 1

    synth = [i for i in got if key[i]["synthetic"]]
    synth_match = sum(got[i]["category"] == key[i]["expected_category"] for i in synth)

    summary = {
        "reply_file": str(Path(response_path).name),
        "is_mock": is_mock,
        "n_sent": len(ours),
        "n_returned_valid": len(got),
        "problems": dict(problems),
        "n_flagged": len(flagged),
        "by_category": dict(Counter(g["category"] for g in got.values())),
        "by_urgency_flagged": dict(Counter(g["urgency"] for g in flagged.values())),
        "flagged_train_comments": sum(not key[i]["synthetic"] for i in flagged),
        "flagged_synthetic_comments": sum(key[i]["synthetic"] for i in flagged),
        "synthetic_category_matches": f"{synth_match}/{len(synth)}",
        "group5_aspects_in_flagged_comments": {k: dict(v.most_common(5))
                                               for k, v in aspect_cooc.items()},
        "note": ("Counts only: no comment text or id is reported. Advisory: Group 6's "
                 "categories are suggestions for human review, not decisions."
                 + (" MOCK reply: tests the plumbing only, says nothing about Group 6's "
                    "system." if is_mock else "")),
    }
    path = RESULTS / f"week7_group6_merge_{tag}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {path.relative_to(ROOT)}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("request")
    sub.add_parser("mock")
    m = sub.add_parser("merge")
    m.add_argument("reply")
    m.add_argument("--tag")
    a = ap.parse_args()
    if a.cmd == "request":
        build_request()
    elif a.cmd == "mock":
        mock_response(OUT / "group6_request.json")
    else:
        merge(a.reply, a.tag)
