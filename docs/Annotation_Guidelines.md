# Annotation Guidelines — Student Feedback Intelligence

**SNLP Course Project 2026 — Group 5**
Members: E029, E030, E031, E033, E034, E035
Guidelines version: **v2.0** · Applies to dataset version: `v0.1+`

---

## 1. Purpose and scope

These guidelines define how the team produces the gold-standard labels used to train and
evaluate the aspect-based sentiment analysis (ABSA) component.

Annotation is done at **three levels** for every feedback comment:

| Level | What is labelled | Used for |
|---|---|---|
| L1 | Overall document sentiment | Sentiment-classification baseline |
| L2 | Aspect categories present (multi-label) | Aspect-category detection |
| L3 | Polarity per aspect category, plus the text span that triggered it | ABSA (the core task) |

L3 is the deliverable that matters. L1 exists only so we can report a document-level
sentiment baseline and show why it is insufficient.

> **Why spans are mandatory from day one.** The Week 3–4 improved approach uses a
> transformer ABSA model, which needs the aspect term's character offsets for token
> classification. Retrofitting spans later means re-annotating the entire corpus. Recording
> them now costs a few seconds per comment.

---

## 2. Label sets

### 2.1 Aspect categories (8 labels — closed set)

The taxonomy is deliberately **flat and small**. A 17-label taxonomy over a corpus of a few
hundred comments produces categories with fewer than 10 examples each, which makes
macro-F1 unstable and per-class results meaningless.

| Label | Covers | Does **not** cover |
|---|---|---|
| `teaching` | Explanation clarity, teaching method, use of examples, lecture delivery, subject command | Faculty's personal availability (→ `faculty_interaction`) |
| `faculty_interaction` | Approachability, responsiveness to doubts, feedback on work, office hours, attitude | Quality of the explanation itself (→ `teaching`) |
| `course_content` | Syllabus relevance, depth, structure, sequencing, intrinsic difficulty of the material | How fast it is delivered (→ `pace`) |
| `pace` | Speed of delivery, rushing, time spent per topic, coverage vs. time | Amount of work outside class (→ `workload`) |
| `assessment` | Assignments, exams, quizzes, projects, grading, marking fairness, question quality | The *volume* of assessment (→ `workload`) |
| `workload` | Quantity of work, deadline pressure, clashing submissions, hours required | Difficulty of a single task (→ `assessment` or `course_content`) |
| `resources` | Slides, notes, textbooks, recordings, LMS, lab equipment, classroom, software/hardware | — |
| `other` | A clear evaluative statement about an academic aspect outside the seven above | Non-evaluative chatter (annotate no aspect at all) |

**Boundary rules for the three confusable pairs.** These three pairs cause most
disagreement; the deciding question is given in bold.

- `teaching` vs `pace` — **Is the complaint about *how* it was explained, or *how fast*?**
  *"He explains well but moves on too quickly"* → `teaching`:positive, `pace`:negative.
- `assessment` vs `workload` — **Is it about one task's nature, or the total amount?**
  *"The assignment questions were unfair"* → `assessment`. *"Three assignments in one
  week"* → `workload`. *"The assignment was long and unfair"* → both.
- `course_content` vs `pace` — **Would slowing down fix it?** If yes → `pace`. If the
  material itself is the problem → `course_content`.

`other` is a **holding label, not a dustbin.** Every use requires the triggering phrase in
`aspect_term`. Review all `other` rows at the end of each annotation round: if the same
theme appears **5 or more times**, promote it to a real category, bump the guidelines
version, and re-label those rows.

### 2.2 Polarity (4 labels)

We adopt the **SemEval-2014 Task 4** label set, so our results stay comparable to
published ABSA work.

| Label | Definition | Example |
|---|---|---|
| `positive` | The student evaluates the aspect favourably | "Lectures were very clear" |
| `negative` | The student evaluates the aspect unfavourably | "The pace was far too fast" |
| `neutral` | The aspect is mentioned **without evaluation**, or the statement is purely factual/descriptive | "There were four assignments this semester" |
| `conflict` | The student expresses **both** polarities about **the same** aspect | "Assignments were difficult but genuinely useful" |

### 2.3 The distinction that must not be blurred

The previous draft of these guidelines said *"if sentiment is unclear, label it Neutral."*
That rule is withdrawn. It collapsed three different situations into one label, turning
`neutral` into a garbage class and destroying macro-F1.

| Situation | Correct handling |
|---|---|
| Aspect mentioned, no evaluation given | polarity = `neutral` |
| Both polarities about the same aspect | polarity = `conflict` |
| **The annotator cannot decide** | Give your best guess **and** set `uncertain = 1` |

`uncertain` is a **metadata flag, not a label.** It never enters training as a class. It is used
to (a) route rows to adjudication, and (b) report what fraction of the corpus is genuinely
hard — which is a legitimate finding worth putting in the report.

---

## 3. Core annotation rules

**R1 — Annotate what is stated, not what is implied by a single word.**
Presence of the token *"assignment"* does not license the `assessment` label unless the
comment actually says something about assessment.

**R2 — One comment may carry multiple aspects.** There is no maximum. Annotate every
aspect that receives an evaluation.

**R3 — Polarity attaches to the aspect, never to the sentence.**
*"The professor is excellent but the assignments are excessive."*
→ `teaching`:positive **and** `workload`:negative. There is no single correct sentence-level
answer here, which is exactly the point of the project.

**R4 — Preserve negation.** *"not informative"* is negative, not positive. Never strip
negation words during annotation, and never let a preprocessing step do it either.

**R5 — Implicit aspects are annotated if a competent reader agrees on the referent.**
*"We barely get time to finish everything"* → `workload`:negative, even though the word
"workload" is absent. Record the span that carries the meaning
(`aspect_term = "barely get time to finish everything"`). If two annotators infer different
referents, the comment is genuinely ambiguous → adjudicate.

**R6 — Sarcasm is labelled by intent, not literal wording.**
*"Another three-hour lecture, exactly what we needed."* → `pace`/`workload`:negative.
Set `uncertain = 1` if the sarcastic reading is not clearly signalled.

**R7 — Comparatives across time are labelled by the current-state evaluation.**
*"Better than last semester but still too fast"* → `pace`:negative.

**R8 — Suggestions imply a negative evaluation of the current state.**
*"More practice problems would help"* → `resources`:negative.

**R9 — Non-evaluative comments get no aspect.** *"Good course."* is L1 `positive` with an
empty L2 aspect set. These rows are kept — they are a real part of the distribution and a
model that hallucinates aspects on them should be penalised.

**R10 — Never annotate personal characteristics or anything about an identifiable
student.** If a comment contains a name, roll number, email or other identifier, redact it in
`feedback_text` before the row enters the dataset.

---

## 4. Recording spans

For each aspect–polarity pair, record the shortest contiguous span of the **original,
un-preprocessed** text that supports the judgement.

```
Text:  "The lectures were informative but the pace was too fast."
                0123456789...

teaching  positive  aspect_term="lectures were informative"  start=4   end=29
pace      negative  aspect_term="the pace was too fast"      start=34  end=55
```

- Offsets are **character** offsets into `feedback_text`, zero-indexed, end-exclusive.
- For implicit aspects, span the phrase carrying the meaning, not the whole sentence.
- Spans may overlap. If an aspect is genuinely diffuse across the comment, set
  `start = -1`, `end = -1` and leave `aspect_term` empty.
- **Verify every span**: `feedback_text[start:end] == aspect_term`. The validation script
  (`src/validate_annotations.py`) enforces this and will reject the file otherwise.

---

## 5. Annotation workflow

### Round 0 — Calibration (do this before anything else)

1. All six members independently annotate the **same 40 comments** from
   `data/fixtures/calibration_set.csv`.
2. Compute agreement (§6). **Do not proceed** until κ ≥ 0.60 on both aspect detection and
   polarity.
3. Meet, walk through every disagreement, and amend these guidelines. Bump the version.
4. Repeat with 40 fresh comments if the threshold was not met.

Round 0 is not optional overhead — the agreement numbers it produces are a graded part
of the dataset methodology, and it is the cheapest possible insurance against discovering
in Week 5 that the corpus is inconsistent.

### Round 1 — Production annotation

- Split into **three pairs**: (E029, E030), (E031, E033), (E034, E035).
- Each pair is assigned a disjoint slice of the corpus.
- **Both members of a pair independently annotate 20% of their slice** (the
  double-annotated subset) and split the remaining 80% between them.
- Record `annotator_id` on every row. Double-annotated rows carry
  `double_annotated = 1` and appear twice, distinguished by `annotator_id`.

### Round 2 — Adjudication

- Disagreements on double-annotated rows, and all `uncertain = 1` rows, go to a third
  member not in that pair.
- The adjudicated label is recorded with `annotator_id = "adjudicated"`.
- Log every adjudication decision in `docs/adjudication_log.md` with the rule that settled
  it. New rules discovered here are folded into the next guidelines version.

### Guideline change control

Any change to the label set or to a rule requires: a version bump, a line in the changelog,
and re-labelling of affected rows. Never silently change a label's meaning mid-corpus —
that is the single fastest way to produce an inconsistent dataset.

---

## 6. Inter-annotator agreement

Agreement is computed by `src/iaa.py` on the double-annotated subset. Report all three
figures in the Week 1–2 report.

| Level | Metric | Why this metric |
|---|---|---|
| Aspect category detection | Per-category **Cohen's κ**, plus micro-F1 between annotators | Multi-label, so κ is computed per category as a binary present/absent decision, then averaged |
| Polarity (on aspects both annotators found) | **Cohen's κ** over the 4 polarity labels | Single-label decision on a shared item set |
| Span boundaries | Proportion of pairs with **Jaccard overlap ≥ 0.5** | Exact span match is too strict for human annotators |

**Interpretation thresholds** (Landis & Koch): < 0.40 poor, 0.40–0.60 moderate, 0.60–0.80
substantial, > 0.80 almost perfect.

**Target: κ ≥ 0.60 before production annotation begins.**

Report κ honestly. A moderate κ that is reported and discussed is worth more marks than a
high κ that is unexplained — and far more than no κ at all, which is the most common
gap in student ABSA projects.

---

## 7. Worked examples

| # | Text | L1 | L2 / L3 | Note |
|---|---|---|---|---|
| 1 | "The lectures were informative but the pace was too fast." | negative | `teaching`:positive, `pace`:negative | R3 — two aspects, opposite polarity |
| 2 | "Assignments were difficult but genuinely useful." | neutral | `assessment`:**conflict** | Same aspect, both polarities → `conflict`, **not** neutral |
| 3 | "We barely get enough time to finish everything." | negative | `workload`:negative | R5 — implicit aspect, span the whole phrase |
| 4 | "There were four assignments this semester." | neutral | `assessment`:**neutral** | Factual, no evaluation |
| 5 | "Another three-hour lecture, exactly what we needed." | negative | `pace`:negative, `uncertain=1` | R6 — sarcasm |
| 6 | "Good course." | positive | *(no aspect)* | R9 — evaluative but no aspect named |
| 7 | "Sir is always available for doubts but his slides are unreadable." | neutral | `faculty_interaction`:positive, `resources`:negative | §2.1 — interaction ≠ teaching |
| 8 | "The content is relevant, though it is covered too quickly." | neutral | `course_content`:positive, `pace`:negative | Boundary rule: slowing down would fix it → `pace` |
| 9 | "Please upload the recordings earlier." | negative | `resources`:negative | R8 — suggestion implies current-state negative |
| 10 | "Not very informative." | negative | `teaching`:negative | R4 — negation flips polarity |

---

## 8. Privacy

- No names, roll numbers, email addresses, phone numbers or other direct identifiers enter
  `feedback_text`. Redact to `[NAME]`, `[EMAIL]`, `[ID]` before the row is saved.
- Instructor names are also identifiers in this setting — redact to `[INSTRUCTOR]`.
- Any mapping from `feedback_id` back to a source record is kept in a separate access-
  controlled file that is **never** committed to the repository and never sent to any external
  service.
- Institutional feedback is used only with documented authorisation. Absent authorisation,
  use the public/synthetic corpus described in the Week 1–2 report.

---

## Changelog

| Version | Change |
|---|---|
| v2.0 | Aspect taxonomy collapsed 17 → 8 with explicit boundary rules; adopted SemEval-2014 4-way polarity including `conflict`; separated annotator uncertainty from the `neutral` label; made character-offset spans mandatory; added calibration round, pair-based workflow, adjudication protocol and quantitative IAA targets |
| v1.0 | Initial draft |
