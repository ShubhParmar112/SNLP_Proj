# Data Dictionary — Student Feedback Intelligence

**SNLP Course Project 2026 — Group 5** · Schema version 2.0

Two files, deliberately separated. Mixing them is how annotation projects lose track of
who labelled what.

---

## 1. Annotation file — `data/raw/annotations.csv`

**Long format: one row per (feedback_id × annotator_id × aspect).**

A comment with two aspects produces two rows. A comment with no aspect produces one row
with an empty `aspect_category`. This is used instead of a JSON blob per comment because
six people will be editing this in a spreadsheet, and a malformed JSON string in row 240
of an Excel export is a far worse failure than an extra row.

| Field | Type | Required | Values / constraints |
|---|---|---|---|
| `feedback_id` | string | yes | Unique per comment, e.g. `F001`. **Never** a roll number. Repeats across rows of the same comment. |
| `feedback_text` | string | yes | Original text, PII already redacted. Identical on every row of a comment. |
| `annotator_id` | string | yes | Pseudonym, e.g. `A_E029`, or `adjudicated` for a resolved row |
| `doc_sentiment` | categorical | yes | `positive` / `neutral` / `negative`. Repeated on each row of a comment; must not conflict. |
| `aspect_category` | categorical | no | One of the 8 taxonomy labels, or empty when the comment names no aspect |
| `polarity` | categorical | conditional | `positive` / `negative` / `neutral` / `conflict`. Required iff `aspect_category` is non-empty. |
| `aspect_term` | string | no | Literal span from `feedback_text` that triggered the label |
| `span_start` | int | no | Zero-indexed char offset, or `-1` if diffuse |
| `span_end` | int | no | End-exclusive char offset, or `-1` |
| `uncertain` | 0/1 | yes | Annotator could not decide. **Metadata, never a training label.** |
| `double_annotated` | 0/1 | yes | Row is part of the IAA subset |
| `notes` | string | no | Free text for the adjudicator |

**Hard constraint, enforced by `src/data.py`:**
`feedback_text[span_start:span_end] == aspect_term`. The loader rejects the file otherwise.
This catches the most common annotation error — offsets drifting after someone edits the
text in a spreadsheet.

---

## 2. Model file — `data/processed/dataset.csv`

Derived from the annotation file by `src/data.py::to_model_format`. One row per comment.
Never edited by hand.

| Field | Type | Description |
|---|---|---|
| `feedback_id` | string | Key back to the annotation file |
| `feedback_text` | string | Original redacted text, **retained unmodified** |
| `text_classical` | string | Branch A: lowercased, negation-marked, stopword-filtered (TF-IDF input) |
| `text_transformer` | string | Branch B: whitespace-normalised only (transformer input) |
| `doc_sentiment` | categorical | Resolved document-level label |
| `aspects` | list | Resolved aspect set |
| `tuples` | list | Resolved `(aspect, polarity)` pairs — the ABSA target |
| `split` | categorical | `train` / `validation` / `test`, from the frozen `splits.json` |
| `duplicate_group` | int | Near-duplicate group; all members share a split |
| `resolution` | categorical | `single` / `adjudicated` / `first_annotator` |
| `dataset_version` | string | e.g. `v0.3` |

**Why two preprocessed columns.** The original single-pipeline design fed lemmatised,
stopword-stripped text to both the classical and the transformer model. That destroys
casing, punctuation and function words — exactly what a pretrained transformer relies on,
and a reliable way to make the Week 4 "improved" model underperform the baseline for
reasons that have nothing to do with the model.

**`feedback_text` is never overwritten.** Every downstream artefact, including API output
spans, refers to offsets in the original.

---

## 3. Vocabularies

**Aspects (8):** `teaching`, `faculty_interaction`, `course_content`, `pace`,
`assessment`, `workload`, `resources`, `other`

**Polarity (4):** `positive`, `negative`, `neutral`, `conflict` — the SemEval-2014 Task 4
set, adopted so results stay comparable to published ABSA work.

**Document sentiment (3):** `positive`, `neutral`, `negative`

Definitions and boundary rules: `docs/Annotation_Guidelines.md` §2.

---

## 4. Privacy

- No names, roll numbers, emails, phone numbers or instructor names in `feedback_text`.
  Redact to `[NAME]`, `[ID]`, `[EMAIL]`, `[PHONE]`, `[INSTRUCTOR]`.
- `src/preprocess.py::redact_pii` runs as a second safety pass; it is a backstop, not a
  substitute for redacting at source.
- Any mapping from `feedback_id` to a source record lives in
  `data/private/id_map.csv`, which is in `.gitignore` and never leaves the team's machines.
- `annotator_id` is a pseudonym so that agreement figures can be reported without
  identifying which team member was the outlier.

---

## 5. Versioning

Record all three in every results file, because a metric is meaningless without them:

| Version | Bump when |
|---|---|
| `dataset_version` | Comments are added, removed or their text edited |
| `annotation_version` | The label set or a guideline rule changes |
| `preprocessing_version` | Any function in `preprocess.py` changes |

The train/validation/test assignment is computed **once** into
`data/processed/splits.json` and committed. Recomputing a split makes every earlier
number incomparable. `run_baseline.py` reuses the frozen file whenever it exists.

### Dataset versions (EduRABSA corpus)

For the EduRABSA corpus the labels are produced by the mapping rules in
`src/build_dataset.py`, so `dataset_version` also changes when a mapping rule changes.

| Version | Change | Effect |
|---|---|---|
| v1.1-edurabsa (2 Oct 2026) | Rule 2b: lecture comments filed by EduRABSA under *Course – Learning activity* map to `teaching` (Guidelines 2.1: teaching covers lecture delivery), except video/slide/quiz mentions | 221 of 5,998 comments relabelled (172 train, 22 validation, 27 test); same frozen split |
| v1.0-edurabsa (25 Sep 2026) | First build: 24 EduRABSA categories mapped onto the 8-aspect taxonomy | 5,998 comments, 11,271 aspect labels |

## Changelog

| Version | Change |
|---|---|
| 2.0 | Split into annotation file and model file; moved to long format; added spans with an enforced integrity constraint; separated `uncertain` from `neutral`; added `conflict` polarity; added the two preprocessing branches; added duplicate grouping and resolution provenance |
| 1.0 | Initial draft |
