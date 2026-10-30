"""Build data/raw/annotations.csv from the public EduRABSA corpus.

    python src/build_dataset.py            # downloads EduRABSA if it is not already present
    python src/build_dataset.py --src DIR  # DIR holds the two Annotation_raw_output JSON files

Why EduRABSA (docs/Dataset.md has the full reasoning):
    6,500 real, public student reviews of courses, teaching staff and universities, each
    already annotated with (aspect, opinion, category, sentiment) quadruplets by a domain
    expert. It is the "public course-review dataset" the Week 1-2 plan asked for, and it
    arrives annotated, so the corpus no longer depends on the team writing its own
    comments - the failure mode REVIEW.md warned about.

Licence: CC BY-NC-SA 4.0 (Hua, Denny, Wicker & Taskova, 2025). Non-commercial academic
use is permitted with attribution; anything derived from it that we share must carry the
same licence. The source texts themselves are CC0 (Waterloo, Exeter) and CC BY 4.0 (RMP).

The EduRABSA label scheme (24 entity-attribute categories, 3 polarities) is mapped onto
our 8-category, 4-polarity scheme by the fixed rules below. Every rule is written down
here and in docs/Dataset.md so a reader can audit it; nothing is decided per comment.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from config import DATA, DATASET_VERSION, RAW, RESULTS

ZENODO_URL = ("https://zenodo.org/api/records/18500517/files/"
              "EduRABSA_Dataset.zip/content")
ZENODO_MD5 = "f2f8b534e0ea83b6f3f7915237542b56"
SOURCE_FILES = ["Train_validation_Merged_ASQE_N4000.json",
                "DEV_TEST_Merged_ASQE_N2500.json"]
DEFAULT_SRC = DATA / "external" / "edurabsa"

ANNOTATOR_ID = "edurabsa_expert"

# --- Rule 1: which reviews are in scope ---------------------------------------
# The asset analyses course and teaching feedback. University-wide reviews are about
# accommodation, nightlife, cost and the city; none of that fits the taxonomy.
KEEP_TYPES = {"course review", "teacher review"}

# --- Rule 2: category mapping ------------------------------------------------
# None = the quadruplet carries no aspect in our scheme (its sentiment still counts
# towards the document-level label).
CATEGORY_MAP = {
    "Staff - Teaching": "teaching",
    "Staff - Knowledge & skills": "teaching",       # "subject command" (Guidelines 2.1)
    "Staff - Overall": "teaching",                  # "Amazing teacher" = teaching quality
    "Staff - Attitude": "faculty_interaction",      # "attitude" is listed under it
    "Staff - Helpfulness": "faculty_interaction",   # doubt resolution, responsiveness
    "Staff - Personal traits": None,                # see note below
    "Course - Content": "course_content",
    "Course - Difficulty": "course_content",        # "intrinsic difficulty" (2.1)
    "Course - Assessment": "assessment",
    "Course - Workload": "workload",
    "Course - Course materials": "resources",
    "Course - Technology & tools": "resources",
    "Course - Learning activity": "other",          # labs, tutorials, readings; see note
    "Course - Overall": None,                       # Guidelines R9: "Good course." has no aspect
}
# University-level categories (programme, campus, cost, services ...) occasionally
# appear inside course and teacher reviews (31 quadruplets). They are institution-level,
# outside a course-feedback taxonomy, so they carry no aspect here.
UNIVERSITY_PREFIX = "University - "
# Notes on the two judgement calls:
#  * Staff - Personal traits ("funny", "hot", "his accent") is not an academic aspect,
#    and a model that learns to score instructors on looks or accent is a bias risk.
#    Dropped from the aspect layer on responsible-AI grounds (Guidelines R10 in spirit).
#  * Course - Learning activity is a clear evaluative academic aspect outside the seven
#    named labels, which is exactly the definition of `other`. It occurs well over 5
#    times, so under the `other` promotion rule it is a candidate for a 9th category
#    (`learning_activity`) - a guidelines-v2.1 decision for the team, not made here.

# --- Rule 2b (dataset v1.1): lectures --------------------------------------------
# EduRABSA files most lecture comments ("the lectures were boring") under Course -
# Learning activity, which Rule 2 sends to `other`. Our own guidelines put lecture
# delivery under `teaching` (Guidelines 2.1), and the handout's two Group 5 examples
# expect "lectures" to be a teaching-type aspect. So a Learning-activity quadruplet whose
# aspect term names lectures or lessons becomes `teaching` - unless it is really about
# a resource (video lectures, lecture slides) or an assessment (lecture quizzes), which
# keep their Rule 2 label.
LECTURE_TERMS = re.compile(r"\b(lecture|lectures|lecturing|lesson|lessons)\b", re.I)
LECTURE_EXCLUDE = re.compile(r"\b(slide|slides|notes|recording|recordings|recorded|video|"
                             r"videos|podcast|podcasts|powerpoint|ppt|quiz|quizzes|exam|"
                             r"exams|test|tests)\b", re.I)


# --- Rule 3: pace ------------------------------------------------------------
# EduRABSA has no pace category; pace complaints sit inside Staff - Teaching, Course -
# Content, etc. A quadruplet is relabelled `pace` when its human-marked aspect or
# opinion term uses delivery-speed language AND its source category is about delivery.
# The category guard matters: "responds to emails quickly" (Staff - Helpfulness) and
# "Dr. Pace" (Staff - Overall) must not become pace.
PACE_TERMS = re.compile(
    r"\b(pace|paced|pacing|fast|faster|too quick|quickly|rushed|rushes|rushing|rush|"
    r"speed|speeds|slow|slowly|slower|hurr\w*)\b", re.I)
PACE_EXCLUDE = re.compile(r"\b(email|emails|respond\w*|repl\w*|grade\w*|grading|"
                          r"mark\w*|papers|turn ?around|posting|fix\w*|grasp\w*|"
                          r"pick\w* up|downhill|hardware|completed|early morning|"
                          r"come to you)\b", re.I)
# Instructors whose surname is Pace ("Dr. Pace", "Miss Pace") - checked case-sensitively.
PACE_SURNAME = re.compile(r"\b[A-Z][a-z]*\.? Pace\b")
PACE_ELIGIBLE = {"Staff - Teaching", "Staff - Personal traits", "Course - Content",
                 "Course - Overall", "Course - Learning activity", "Course - Difficulty"}

# --- Rule 4: polarity --------------------------------------------------------
# EduRABSA polarity is per quadruplet; ours is per (comment, category). When several
# quadruplets map to one category the SemEval-2014 convention decides:
#   positive and negative both present -> conflict
#   one of positive/negative (+/- neutral) -> that polarity
#   neutral only -> neutral
POL = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}
DOC = {"Pos": "positive", "Neg": "negative", "Neu": "neutral"}


def resolve_polarity(pols):
    s = set(pols)
    if "positive" in s and "negative" in s:
        return "conflict"
    if "positive" in s:
        return "positive"
    if "negative" in s:
        return "negative"
    return "neutral"


def map_quad(q):
    """Return our aspect label for one EduRABSA quadruplet, or None."""
    cat = q["category"]
    terms = f"{q['aspect']} {q['opinion']}"
    if (cat in PACE_ELIGIBLE and PACE_TERMS.search(terms)
            and not PACE_EXCLUDE.search(terms) and not PACE_SURNAME.search(terms)):
        return "pace"
    if cat.startswith(UNIVERSITY_PREFIX):
        return None
    if (cat == "Course - Learning activity" and LECTURE_TERMS.search(q["aspect"])
            and not LECTURE_EXCLUDE.search(q["aspect"])):
        return "teaching"
    if cat not in CATEGORY_MAP:
        raise KeyError(f"unmapped EduRABSA category {cat!r} - add it to CATEGORY_MAP")
    return CATEGORY_MAP[cat]


def find_span(text, q):
    """Character span in the ORIGINAL text supporting the quadruplet.

    Covers the aspect term and the opinion term when both are present and close
    together (Guidelines section 4: the shortest span that supports the judgement);
    otherwise whichever one is present. Implicit aspects ("null") fall back to the
    opinion term. Returns (term, start, end) or ("", -1, -1).
    """
    def locate(term):
        term = (term or "").strip()
        if not term or term.lower() == "null":
            return None
        i = text.find(term)
        return (i, i + len(term)) if i >= 0 else None

    a, o = locate(q["aspect"]), locate(q["opinion"])
    if a and o and abs(a[0] - o[0]) <= 120:
        s, e = min(a[0], o[0]), max(a[1], o[1])
    elif o:
        s, e = o
    elif a:
        s, e = a
    else:
        return "", -1, -1
    return text[s:e], s, e


def ensure_source(src: Path):
    """Download and unpack the two annotated JSON files if they are missing."""
    if all((src / f).exists() for f in SOURCE_FILES):
        return
    src.mkdir(parents=True, exist_ok=True)
    print(f"Downloading EduRABSA from Zenodo (31 MB) ...")
    blob = urllib.request.urlopen(ZENODO_URL, timeout=120).read()
    md5 = hashlib.md5(blob).hexdigest()
    if md5 != ZENODO_MD5:
        raise RuntimeError(f"checksum mismatch: got {md5}, expected {ZENODO_MD5}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if "Annotation_raw_output/" in name and base in SOURCE_FILES:
                (src / base).write_bytes(z.read(name))
    print(f"  saved {', '.join(SOURCE_FILES)} -> {src}")


def build(src: Path, out: Path):
    ensure_source(src)
    rows, stats = [], Counter()
    source_cats, mapped_cats = Counter(), Counter()
    pace_examples = []

    for fname in SOURCE_FILES:
        entries = json.loads((src / fname).read_text(encoding="utf-8"))
        for key in sorted(entries, key=int):
            e = entries[key]
            stats["source_entries"] += 1
            if e["type"] not in KEEP_TYPES:
                stats["dropped_out_of_scope_type"] += 1
                continue
            if e.get("overall_sentiment") not in DOC:
                stats["dropped_no_doc_sentiment"] += 1
                continue

            text = e["text"]
            if not text.strip():
                stats["dropped_empty_text"] += 1
                continue
            fid = f"ER{int(e['id']):05d}"
            by_aspect = defaultdict(list)
            for q in e["Quadruplets"]:
                source_cats[q["category"]] += 1
                aspect = map_quad(q)
                if aspect is None:
                    stats["quads_without_aspect"] += 1
                    continue
                if aspect == "pace":
                    pace_examples.append((q["category"], q["aspect"], q["opinion"],
                                          q["sentiment"]))
                mapped_cats[aspect] += 1
                by_aspect[aspect].append(q)

            base = {"feedback_id": fid, "feedback_text": text,
                    "annotator_id": ANNOTATOR_ID,
                    "doc_sentiment": DOC[e["overall_sentiment"]],
                    "uncertain": 0, "double_annotated": 0}
            if not by_aspect:
                rows.append({**base, "aspect_category": "", "polarity": "",
                             "aspect_term": "", "span_start": -1, "span_end": -1,
                             "notes": f"source={e['type']}; original_id={e['original_id']}"})
                stats["comments_no_aspect"] += 1
            for aspect in sorted(by_aspect):
                qs = by_aspect[aspect]
                pol = resolve_polarity(POL[q["sentiment"]] for q in qs)
                term, s, en = find_span(text, qs[0])
                rows.append({**base, "aspect_category": aspect, "polarity": pol,
                             "aspect_term": term, "span_start": s, "span_end": en,
                             "notes": (f"source={e['type']}; original_id={e['original_id']}; "
                                       f"from={'|'.join(sorted({q['category'] for q in qs}))}")})
                stats[f"polarity_{pol}"] += 1
            stats["comments_kept"] += 1
            stats[f"type_{e['type'].replace(' ', '_')}"] += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["feedback_id", "feedback_text", "annotator_id", "doc_sentiment",
            "aspect_category", "polarity", "aspect_term", "span_start", "span_end",
            "uncertain", "double_annotated", "notes"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    profile = {"dataset_version": DATASET_VERSION, "source": "EduRABSA v2.1 (Zenodo 18500517)",
               "stats": dict(stats), "source_categories": dict(source_cats.most_common()),
               "mapped_aspects": dict(mapped_cats.most_common()),
               "n_pace_relabelled": len(pace_examples)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dataset_build.json").write_text(json.dumps(profile, indent=2))
    (RESULTS / "pace_relabel_audit.tsv").write_text(
        "source_category\taspect_term\topinion_term\tsentiment\n" +
        "\n".join("\t".join(x) for x in pace_examples), encoding="utf-8")

    print(f"Wrote {len(rows)} rows for {stats['comments_kept']} comments -> {out}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<32} {v}")
    print(f"  pace relabelled                  {len(pace_examples)} "
          f"(audit list: results/pace_relabel_audit.tsv)")
    return profile


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=RAW / "annotations.csv")
    a = ap.parse_args()
    build(a.src, a.out)
