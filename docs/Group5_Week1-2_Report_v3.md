---
title: Student Feedback Intelligence
subtitle: Aspect-Based Sentiment Analysis of Academic Feedback
week: Week 1–2: Dataset, Annotation Protocol, Baseline and Evaluation Methodology
course: SNLP Course Project 2026   •   Group 5
members: E029  ·  E030  ·  E031  ·  E033  ·  E034  ·  E035
version: Document version 3.0  •  Guidelines v2.0  •  Schema v2.0  •  Dataset v1.0-edurabsa
note: Every figure in this report was produced by a script in the repository (Appendix A). Nothing was estimated.
footer: Group 5  •  Student Feedback Intelligence  •  Week 1–2  •  
---

# Student Feedback Intelligence — Week 1–2 Report (v3.0)

*SNLP Course Project 2026 · Group 5 · E029 · E030 · E031 · E033 · E034 · E035*

Version 3.0 completes the v2.0 template: every `[TO FILL]` field is now a number produced by
`src/run_baseline.py` or `src/profile_dataset.py` on the real corpus.

## 1.  Scope and Status

This report covers the Week 1–2 deliverables: problem definition and use cases, the dataset decision, the annotation protocol and its quality controls, the preprocessing pipeline, the train/validation/test split, the baseline implementation, and the evaluation methodology under which all subsequent results will be reported.

One methodological commitment governs the whole document: no number appears here that a script did not produce. Version 2.0 left every result field marked `[TO FILL]` until the code had run on a real corpus. Version 3.0 fills them: the corpus is settled (Section 5), the baseline has run end to end on it, and each figure below traces to a file in `results/` (Appendix A). No figure in this version comes from fixture data.

### Status at end of Week 2

| Component | Status |
|---|---|
| Annotation guidelines (v2.0) | Complete — 8-aspect taxonomy, 4-way polarity, span protocol |
| Evaluation protocol | Complete — metrics, reference baselines and splitting fixed in advance |
| Baseline implementation | Complete — run on the real corpus (Section 14) |
| IAA tooling | Complete — Cohen's κ, pairwise F1, span overlap |
| Dataset acquisition | Complete — EduRABSA, 5,998 comments in scope (Section 5) |
| Annotation (calibration round) | Changed scope — gold labels now come from EduRABSA's expert annotation; the team round checks our label mapping. Sheet prepared, round pending (Section 8) |
| Baseline results on real corpus | Complete — ABSA micro tuple-F1 0.454 [0.432, 0.479] on test vs 0.216 for the majority reference |

## 2.  Institutional Context

Universities collect textual feedback on teaching, course content, pace, assessment, workload and resources, and then reduce it to numerical averages. The average is actionable only if a coordinator already knows what drove it, which is precisely what the average removes.

The problem is not that sentiment analysis is hard. It is that document-level sentiment is the wrong output shape for this data. Consider: “The professor explains concepts clearly, but the assignments take far too much time.” There is no correct single label. Forcing one discards half the information, and the half discarded is arbitrary. Aspect-based sentiment analysis produces `teaching:positive` and `workload:negative`, which is what a coordinator can act on.

This is why the project scope stops at ABSA rather than extending to summarisation and a dashboard. The brief asks for one strong NLP capability; building seven shallow ones is the most common way projects of this type lose marks on technical depth.

## 3.  Use Cases

| Use case | Input → Output | Primary user |
|---|---|---|
| Course feedback review | All comments for a course → aspect frequency, polarity per aspect, representative examples | Course coordinator |
| Teaching-quality diagnosis | Comments → polarity on teaching and faculty_interaction with supporting spans | Faculty member |
| Workload monitoring | Comments → workload mention rate and net sentiment | Department head |
| Cross-semester comparison | Two batches → change in aspect-wise net sentiment | Academic administrator |
| Group 6 advisory signal | Course batch → aggregate academic-concern signal, course-level only | Student-support workflow |

Cross-semester comparison is the brief's stretch goal. It is listed here as a use case but is explicitly out of scope for the modelling work until the core ABSA component clears its evaluation targets.

## 4.  NLP Formulation

The work is decomposed into three tasks with distinct targets and metrics. Stating them separately matters: bundling aspect extraction and sentiment classification into one undifferentiated task makes it impossible to tell whether a failure is a missed aspect or a misjudged polarity — two problems with entirely different fixes.

| Task | Target | Type | Primary metric |
|---|---|---|---|
| A — Document sentiment | One label per comment | 3-class single-label | Macro-F1 |
| B — Aspect detection | Set of aspect categories | 8-class multi-label | Micro-F1 |
| C — ABSA (core) | Set of (aspect, polarity) pairs | Structured prediction | Exact-match micro tuple-F1 |

Task C is what the project is judged on. A predicted tuple counts as correct only when both the aspect category and its polarity match a gold tuple for the same comment. Reporting aspect F1 and polarity accuracy separately overstates performance, because neither number penalises finding the right aspect and assigning it the wrong sentiment.

### Worked example of the metric

```
Gold:      {(teaching, positive), (pace, negative)}
Predicted: {(teaching, positive), (pace, neutral), (workload, negative)}
TP = 1   FP = 2   FN = 1
Precision = 0.333   Recall = 0.500   F1 = 0.400
```

A secondary figure, polarity accuracy restricted to correctly detected aspects, is reported alongside. It separates “found the wrong aspect” from “found the aspect, misjudged the sentiment”, which the headline F1 conflates.

## 5.  Dataset

### 5.1  The dataset decision

The initial plan was a team-authored synthetic corpus. That approach was rejected in v2.0: if the same six people write the comments and assign the labels, the model learns the team's own phrasing templates, and the score measures internal consistency rather than capability. Version 2.0 called for a public course-review dataset instead, and left the choice open. That choice is now made.

**Primary corpus: EduRABSA** (Hua, Denny, Wicker and Taskova, 2025). It contains 6,500 real, public English student reviews, sampled across review length and rating, from three open sources: University of Waterloo course reviews (3,000; CC0), RateMyProfessor teacher reviews (3,000; CC BY 4.0) and University of Exeter reviews (500; CC0). Every review is annotated by a domain expert with (aspect term, opinion term, category, sentiment) quadruplets plus a review-level sentiment label. It is exactly the “public course-review dataset” v2.0 asked for, and it arrives annotated: the six-person production annotation of Round 1 is no longer the critical path.

| Corpus | Role | Rationale |
|---|---|---|
| EduRABSA (public, expert-annotated) | Primary training and evaluation | Real student language: slang, misspellings, sarcasm, long multi-aspect reviews; education domain; ABSA gold labels |
| SemEval-2014 Task 4 (ABSA) | Optional cross-domain benchmark (not run) | Deferred — the in-domain corpus answers the Week 1–2 question; revisit only if time allows |
| Institutional feedback | Used only with authorisation | None obtained; the loader accepts any file in the same CSV format, so it can be added without code changes |
| Team-authored synthetic | Fixtures only | Smoke tests (`data/fixtures/`), never a reported result |

### 5.2  Mapping EduRABSA onto our taxonomy

EduRABSA uses 24 two-level categories (e.g. *Staff – Teaching*, *Course – Workload*) and three polarities. Our taxonomy has 8 flat categories and four polarities. The conversion is done by fixed, written rules in `src/build_dataset.py`; nothing is decided comment by comment.

| EduRABSA category | Our label | Quadruplets | Reason |
|---|---|---|---|
| Staff – Teaching, Knowledge & skills, Overall | teaching | 8,230 | Explanation, subject command; “amazing teacher” is teaching quality |
| Staff – Attitude, Helpfulness | faculty_interaction | 2,332 | Attitude and responsiveness are listed under this label (Guidelines 2.1) |
| Course – Content, Difficulty | course_content | 4,497 | Guidelines 2.1 puts intrinsic difficulty under course_content |
| Course – Assessment | assessment | 2,525 | Direct match |
| Course – Workload | workload | 619 | Direct match |
| Course – Course materials, Technology & tools | resources | 904 | Slides, notes, LMS, software |
| Course – Learning activity | other | 1,364 | Labs, tutorials, readings: evaluative, academic, outside the seven labels |
| Course – Overall | no aspect | 3,922 | Guidelines R9: “Good course.” carries document sentiment only |
| Staff – Personal traits | no aspect | 922 | Looks, accent and personality are not academic aspects; learning to score them is a bias risk |
| University – any | no aspect | 31 | Institution-level, outside a course-feedback taxonomy |

Counts are quadruplets before merging and before the pace rule. Three further rules complete the conversion:

- **Scope.** The 500 university-wide reviews (accommodation, city, cost) are excluded, as are 2 reviews with no overall label. 5,998 comments remain: 2,998 course reviews and 3,000 teacher reviews.
- **Pace.** EduRABSA has no pace category. A quadruplet becomes `pace` when its human-marked aspect or opinion term uses delivery-speed language (“moves too fast”, “rushed”, “well-paced”) *and* its source category is about delivery (Staff – Teaching, Course – Content, and similar). The category guard keeps “responds to emails quickly” and “Dr. Pace” out. 151 quadruplets were moved; every one is listed for audit in `results/pace_relabel_audit.tsv`.
- **Polarity.** When several quadruplets map to one category in the same comment, the SemEval-2014 convention decides: positive and negative together give `conflict`; one of them (with or without neutral) gives that polarity; neutral alone gives `neutral`.

Evidence spans are taken from the annotated terms: the shortest window covering the aspect term and the opinion term, or the opinion term alone for implicit aspects. The loader's span-integrity check (`feedback_text[start:end] == aspect_term`) passes on all 11,737 rows.

### 5.3  Dataset documentation

| Item | Value |
|---|---|
| Source and licence | EduRABSA v2.1, Zenodo record 18500517 (doi 10.5281/zenodo.16935017). CC BY-NC-SA 4.0: academic, non-commercial use with attribution; anything we share that is derived from it carries the same licence |
| Number of comments | 5,998 (6,500 minus 500 university reviews and 2 without an overall label) |
| Number of aspect annotations | 11,271 (comment, aspect) pairs |
| Mean aspects per comment | 1.88 |
| Comments with no aspect | 466 (7.8%) — kept, per Guidelines R9 |
| Unit of analysis | One student feedback comment |
| Length | median 37 words, mean 41, 95th percentile 86, maximum 442 |
| Format | CSV, long format — one row per (feedback_id × annotator_id × aspect) |
| Aspect labels | 8: teaching, faculty_interaction, course_content, pace, assessment, workload, resources, other |
| Polarity labels | 4: positive, negative, neutral, conflict (SemEval-2014 Task 4 set) |
| Document sentiment labels | 3: positive 3,013 (50.2%), negative 1,568 (26.1%), neutral 1,417 (23.6%) |
| Train / validation / test | 4,198 / 900 / 900 (70/15/15), iterative stratification, seed 42, frozen in `splits.json` |
| Language | English only (North American and UK review sites); no code-switched text |
| Privacy | Public reviews; no student identifiers. Instructor names do occur (RateMyProfessor) and are redacted at model input by `preprocess.py`; see Section 18 |
| Dataset version | v1.0-edurabsa |

Aspect-level polarity distribution: positive 5,880 (52.2%), negative 3,441 (30.5%), conflict 1,050 (9.3%), neutral 900 (8.0%).

## 6.  Aspect Taxonomy

The taxonomy was reduced from 17 labels to 8 in v2.0. A smaller taxonomy that is reliably annotated is worth more than a fine-grained one that is not. The corpus counts below are comments carrying each aspect.

| Label | Covers | Does not cover | Comments |
|---|---|---|---|
| teaching | Explanation clarity, method, examples, lecture delivery | Personal availability → faculty_interaction | 3,184 |
| faculty_interaction | Approachability, doubt resolution, responsiveness | Quality of explanation → teaching | 1,582 |
| course_content | Syllabus relevance, depth, structure, intrinsic difficulty | Delivery speed → pace | 2,718 |
| pace | Speed of delivery, rushing, coverage vs time | Out-of-class work volume → workload | 141 |
| assessment | Assignments, exams, quizzes, projects, grading fairness | Volume of assessment → workload | 1,513 |
| workload | Quantity of work, deadline pressure, clashes | Difficulty of one task → assessment | 522 |
| resources | Slides, notes, recordings, LMS, lab, classroom, equipment | — | 663 |
| other | Clear evaluative statement outside the seven above | Non-evaluative text → no aspect at all | 948 |

`other` is a holding label, not a dustbin. In this corpus it is almost entirely EduRABSA's *Learning activity* (labs, tutorials, readings). That theme occurs far more than five times, so under the promotion rule it is a candidate for a ninth category, `learning_activity`. Promoting it is a guidelines-v2.1 decision for the team; it has not been made unilaterally here.

## 7.  Annotation Scheme

Three levels per comment: document sentiment (L1), aspect categories present (L2), and polarity plus supporting character span for each aspect (L3). L3 is the deliverable; L1 exists only to demonstrate the limitation it addresses. For dataset v1.0, all three levels are derived from EduRABSA's annotation by the mapping in Section 5.2.

### 7.1  Polarity labels

| Situation | Handling |
|---|---|
| Aspect mentioned, no evaluation given | polarity = neutral |
| Both polarities about the same aspect | polarity = conflict |
| Annotator cannot decide | best guess + uncertain = 1 (metadata, never a training class) |

The four-way set is taken from SemEval-2014 Task 4 so that results remain comparable with published ABSA work. In this corpus `conflict` arises when one category receives both a positive and a negative quadruplet (1,050 cases, 9.3%). EduRABSA has no uncertainty flag, so `uncertain` is 0 throughout v1.0.

### 7.2  Spans

Every aspect–polarity pair records a supporting span as character offsets into the original text (Section 5.2). This matters from Week 3 onward: the transformer model is trained to attend to these spans and returns its own evidence span with every prediction.

### 7.3  Workflow

The Round 0 → Round 1 → Round 2 workflow (calibration, paired production annotation, adjudication) is retained for any institutional data added later. For v1.0 its Round 0 is repurposed as the mapping check described in Section 8.

## 8.  Inter-Annotator Agreement

EduRABSA was annotated by a single expert annotator (the dataset's first author), with edge cases resolved in discussion with the co-authors and a full review pass at the end. The source reports no inter-annotator agreement, so there is no κ to cite for the gold labels themselves.

What the team can and should measure is whether **our mapped labels** agree with how we would label the same comments. That is the IAA gate for v1.0. `src/make_calibration_sheet.py` has drawn 40 comments from the training split (seed 42) into `data/fixtures/calibration_sheet_blank.csv`. At least two members label them independently; `src/iaa.py` then computes member-vs-member and member-vs-gold agreement.

| Level | Metric | Target | Result |
|---|---|---|---|
| Aspect detection | Per-category Cohen's κ, averaged | ≥ 0.60 | Pending — team round |
| Aspect detection | Pairwise micro-F1 between annotators | — | Pending — team round |
| Polarity (shared aspects) | Cohen's κ over 4 labels | ≥ 0.60 | Pending — team round |
| Span boundaries | Proportion with Jaccard ≥ 0.5 | — | Pending — team round |
| Corpus difficulty | Proportion of rows with uncertain = 1 | — | 0 (flag not present in source) |

Interpretation follows Landis and Koch: below 0.40 poor, 0.40–0.60 moderate, 0.60–0.80 substantial, above 0.80 almost perfect. If agreement with the mapped gold falls below 0.60 for a category, the mapping rule for that category is revised and the dataset version bumped — the same change control the guidelines apply to annotation rules.

## 9.  Preprocessing

Two separate branches. Feeding lemmatised, stopword-stripped, lowercased text to a pretrained transformer destroys the casing, punctuation and function words that such a model depends on.

| Step | Branch A — classical (TF-IDF) | Branch B — transformer |
|---|---|---|
| PII redaction | Yes | Yes |
| Whitespace normalisation | Yes | Yes |
| Lowercasing | Yes | No — casing is signal |
| Punctuation removal | Yes | No |
| Contraction expansion | Yes | No |
| Negation scope marking | Yes — NOT_ prefix to clause boundary | No — attention handles it |
| Stopword removal | Selective — negation and contrast words retained | No |
| Lemmatisation | Optional, ablated | No — destroys subword morphology |

### 9.1  Negation handling

A general stopword list deletes *not*, *never*, *but* and *however* — the exact tokens that carry sentiment and contrast in feedback text. Our list explicitly retains them. In addition, tokens inside a negation's scope are prefixed so that a bag-of-words model can distinguish them:

```
"the lectures were not very informative"
  →  ... not  NOT_very  NOT_informative
```

Scope terminates at a clause boundary rather than a fixed window, so “not clear but the notes are good” does not wrongly negate *good*. Its contribution was measured as an ablation: micro tuple-F1 0.454 with negation marking against 0.451 without (Δ = +0.003). That difference is far inside the confidence interval (±0.024), so on this corpus negation marking has **no measurable effect** on the headline metric. It is kept because it is cheap and correct, and the claim that it helps is withdrawn.

The original feedback text is retained unmodified alongside both processed forms, because all span offsets and all API output refer to it.

## 10.  Dataset Quality Analysis

| Issue | Count / proportion | Handling |
|---|---|---|
| Missing or empty comments | 0 | — |
| Exact duplicates | 15 | Kept, and grouped so each duplicate set sits in one split |
| Near-duplicates (char 3–5 gram cosine ≥ 0.95) | 16 groups, 35 comments | Grouped; all members share a split |
| Very short comments (< 3 tokens) | 134 (2.2%) | Retained; a distinct error stratum |
| Very long comments (> 100 tokens) | 214 (3.6%) | Retained; typically multi-aspect |
| Aspect class imbalance | 22.6 : 1 (teaching 3,184 vs pace 141) | class_weight='balanced'; ablated |
| Polarity class imbalance | neutral 8.0%, conflict 9.3% of pairs | Same; per-class figures reported |
| Comments with no aspect | 466 (7.8%) | Retained — a model hallucinating aspects here is penalised |
| Spelling and code-switching | Not measured automatically; 28 comments contain non-ASCII characters; no code-switched text | Character n-grams in the vectoriser |
| Rows flagged uncertain | 0 (flag absent in source) | — |

Near-duplicate grouping is not a formality. Without it, a paraphrase of a training comment can land in the test set and inflate every score, and the inflation is invisible in the results table.

## 11.  Split and Leakage Controls

The aspect target is multi-label, so `train_test_split(stratify=y)` does not apply. We use iterative stratification (Sechidis, Tsoumakas and Vlahavas, 2011), which allocates examples so that each label's distribution is preserved across splits, with near-duplicate groups kept intact.

| Control | Implementation |
|---|---|
| Fixed seed | RANDOM_SEED = 42 in `src/config.py`, used by every stochastic step |
| Frozen split | Computed once into `data/processed/splits.json` and committed; never recomputed |
| Vectoriser fitting | Every transformation inside an sklearn Pipeline, so it can only see the training fold |
| Duplicate leakage | Near-duplicate groups assigned wholesale to one split |
| Test-set discipline | Test opened once at the end of Week 2 (this report) and next at the end of Week 4; no decision is made on it |
| Lexicon circularity | Aspect lexicon built from the training split only; per-term provenance logged in `lexicon_provenance.md` |

### 11.1  Per-label balance across splits

| Aspect | Total | Train | Val | Test |
|---|---|---|---|---|
| teaching | 3,184 | 2,229 | 478 | 477 |
| faculty_interaction | 1,582 | 1,108 | 237 | 237 |
| course_content | 2,718 | 1,902 | 408 | 408 |
| pace | 141 | 99 | 21 | 21 |
| assessment | 1,513 | 1,059 | 227 | 227 |
| workload | 522 | 366 | 78 | 78 |
| resources | 663 | 464 | 99 | 100 |
| other | 948 | 664 | 142 | 142 |

Every label is split within one comment of the 70/15/15 target. Document sentiment is also balanced without being stratified on directly: positive is 50.4% of train, 49.0% of validation and 50.7% of test.

## 12.  Baseline Implementation

| Model | Method |
|---|---|
| A — Document sentiment | TF-IDF (word 1–2 gram + char 3–5 gram) → Logistic Regression, class_weight='balanced' |
| B — Aspect detection | Same features → one-vs-rest Logistic Regression over 8 categories |
| B′ — Lexicon aspect detection | Transparent keyword/phrase lexicon, seeded from domain knowledge and expanded from training-split statistics |
| C — ABSA | Stage 1: aspect detection. Stage 2: a single shared polarity classifier over (comment, aspect) pairs with the aspect injected as an explicit token |

The Stage 2 design pools the polarity signal across categories: one model conditioned on the aspect identity, rather than eight per-aspect classifiers with too few examples each. Character n-grams are included for misspellings (*assignmnet*) and inconsistent spacing. Logistic regression is chosen because it is interpretable and fast, not because it is expected to be strong.

## 13.  Evaluation Methodology

Fixed before any result was produced, so that the Week 4 comparison is a test rather than a search for a flattering number.

### 13.1  Mandatory reference baselines

Every model figure is reported in the same table as a majority-class and a stratified-random reference. Without them no accuracy figure can be interpreted.

### 13.2  Uncertainty

- Development figures: 5-fold cross-validation on train + validation, reported as mean ± standard deviation.
- Test figures: 95% bootstrap confidence interval over 1,000 resamples, reported as F1 = 0.454 [0.432, 0.479].
- A baseline-vs-improved difference whose intervals overlap substantially is reported as inconclusive, not as an improvement.

The protocol was written for a corpus of a few hundred comments. At 5,998 comments the test split is 900 comments and the confidence intervals are about ±0.024 wide, which is narrow enough to separate the models below with confidence.

### 13.3  Low-support categories

Any aspect category with fewer than 20 test instances is reported but marked low-support and excluded from the headline macro average. With 900 test comments no category falls below that line; pace is closest, at 21.

## 14.  Initial Results

All figures are produced by `src/run_baseline.py` (seed 42, test split n = 900) and written to `results/metrics.json` and `results/results_tables.md`.

### 14.1  Task C — ABSA tuple extraction (headline)

| Model | Micro tuple-F1 | 95% CI | P | R | Polarity acc. on detected |
|---|---|---|---|---|---|
| Majority class | 0.216 | [0.197, 0.237] | 0.311 | 0.166 | 0.587 |
| Stratified random | 0.145 | [0.128, 0.164] | 0.148 | 0.143 | 0.416 |
| Lexicon + LogReg polarity | 0.332 | [0.307, 0.356] | 0.379 | 0.295 | 0.637 |
| TF-IDF LogReg (baseline) | **0.454** | [0.432, 0.479] | 0.434 | 0.477 | 0.617 |

5-fold CV (development, train + validation): micro tuple-F1 = **0.469 ± 0.003** (folds 0.466–0.475).

The baseline more than doubles the majority reference and its interval is nowhere near the reference's, so the gain is established, not suggestive. The cross-validation spread is tiny (±0.003), so the single test split is representative. The weak half is sentiment, not detection: polarity accuracy on correctly detected aspects is only 0.617 — lower than the lexicon model's 0.637 on its smaller set of detections. The error analysis (Section 15) shows why.

Per-category tuple-F1 for the baseline: teaching 0.671, faculty_interaction 0.604, pace 0.600, course_content 0.367, assessment 0.335, workload 0.264, resources 0.238, other 0.215 (macro 0.412).

### 14.2  Task A — Document sentiment

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority class (positive) | 0.507 | 0.224 |
| TF-IDF + Logistic Regression | 0.750 | 0.722 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| positive | 0.885 | 0.794 | 0.837 | 456 |
| neutral | 0.490 | 0.656 | 0.561 | 189 |
| negative | 0.794 | 0.741 | 0.767 | 255 |

Neutral is the hard class, as expected: many neutral reviews are mixed reviews whose positive and negative parts cancel, which is exactly the information a single label throws away.

### 14.3  Task B — Aspect detection, per category

Learned model (one-vs-rest LogReg): micro-F1 **0.736**, macro-F1 0.655. Lexicon only: micro-F1 0.521, macro-F1 0.444.

| Aspect | P | R | F1 | Support | Low support? |
|---|---|---|---|---|---|
| teaching | 0.891 | 0.874 | 0.883 | 477 | No |
| faculty_interaction | 0.657 | 0.840 | 0.737 | 237 | No |
| course_content | 0.780 | 0.782 | 0.781 | 408 | No |
| pace | 1.000 | 0.429 | 0.600 | 21 | No (21, closest to the line) |
| assessment | 0.664 | 0.767 | 0.712 | 227 | No |
| workload | 0.417 | 0.513 | 0.460 | 78 | No |
| resources | 0.465 | 0.590 | 0.520 | 100 | No |
| other | 0.486 | 0.627 | 0.548 | 142 | No |

Detection is strong where the vocabulary is explicit (teaching, course_content) and weak where the aspect is usually implied (workload: “doesn't weigh too heavily on your schedule”) or heterogeneous (other, resources). Pace has perfect precision but finds fewer than half its cases.

### 14.4  Ablations

| Configuration | Micro tuple-F1 / Macro-F1 | Δ |
|---|---|---|
| Negation marking ON vs OFF (Task C) | 0.454 vs 0.451 | +0.003 |
| class_weight balanced vs None (Task A macro-F1) | 0.722 vs 0.711 | +0.011 |
| Word + char n-grams vs word only (Task C) | 0.454 vs 0.440 | +0.014 |

All three changes point the expected way, but each is smaller than the ±0.024 confidence interval of a single test split, so none is established by this evidence alone. Character n-grams show the largest effect, consistent with the misspellings in real reviews. Class weighting trades a little accuracy (0.756 → 0.750) for better minority-class F1, which is the intended trade.

## 15.  Initial Error Analysis

The baseline makes 668 errors (at least one wrong or missing tuple) on the 900 validation comments; all are exported to `results/errors_for_analysis.json`. A random sample of 40 (seed 42) was categorised by hand, one primary cause each; the coded sample is `results/errors_coded_week2.json`.

| Error type | Real example from the sample | Count (of 40) | Hypothesised cause |
|---|---|---|---|
| Conflict / mixed tone copied | One `conflict` label copied onto every aspect of a mixed review | 11 | Stage 2 sees the whole comment, not the part about each aspect |
| Implicit aspect | “doesn't weigh too heavily on your schedule” (workload) | 7 | Target aspect not lexically present |
| Boundary confusion | “I love Dubins class!!!!” — no aspect under R9, predicted teaching | 7 | Overall vs aspect-specific praise; other vs assessment |
| Polarity on correct aspect | “Lectures are boring” → teaching:positive | 6 | Weak contrastive signal in bag-of-words |
| Multi-aspect | “Fair grader, tests are reasonable” missed in a list of praise | 5 | Independent per-label thresholds miss minor aspects |
| Annotation error | “office hours … answers all your questions” gold teaching; our guidelines say faculty_interaction | 2 | Source labels vs our mapping |
| Rare category | “will call you out so fast” → spurious pace | 1 | Keyword trigger on a low-support label |
| Negation | “Not to strict on the grading” → negative | 1 | Negation of a negative word |
| Sarcasm | — | 0 | None in this sample |

The dominant failure is structural, not lexical. More than a quarter of the sampled errors (11 of 40) are mixed reviews where the Stage 2 polarity classifier, which reads the entire comment, assigns the same overall tone to every aspect it found. Richer word features cannot fix this. The model needs to read *the part of the comment about each aspect*, which is the specific design goal of the Week 3 model.

The annotation-error row is reported deliberately. Both cases trace to the boundary between EduRABSA's categories and ours, which is exactly what the calibration round in Section 8 is designed to measure. The sample was coded by one person; a second member should re-code it before the Week 5 evaluation report.

## 16.  Reusable Asset Design

Specified in full in `docs/API_Specification.md`. Endpoints: `POST /v1/analyze-feedback`, `POST /v1/analyze-feedback/batch`, `GET /v1/schema` and `GET /v1/model-card`.

### 16.1  Score semantics

A keyword match has no probability attached, and `predict_proba` output is not calibrated — 0.9 does not mean “correct 90% of the time”. Every score therefore carries an explicit type:

| score_type | Meaning | Safe to threshold? |
|---|---|---|
| lexicon_match_count | Count of distinct lexicon terms matched | Ranking only |
| uncalibrated_probability | Raw model probability; monotone, not a frequency | Yes, with a locally tuned threshold |
| calibrated_probability | After Platt/isotonic calibration on held-out data | Yes |

Weeks 1–2 emit only the first two. Calibration is a Week 5 deliverable.

### 16.2  Group 6 integration

Group 5 emits course-level advisory signals carrying explicit `advisory_only` and `not_for_individual_action` flags. The component performs no individual-level inference and must never be joined to a student identifier. This is stated in the API contract rather than deferred to a Responsible AI section, because a constraint that lives only in a report is not a constraint.

## 17.  Architecture

```
Student feedback  (comment or batch)
        ↓
Validation  →  PII redaction
        ↓
Preprocessing   Branch A (TF-IDF)   |   Branch B (transformer)
        ↓
Aspect detection  →  Polarity per detected aspect
        ↓
Post-processing and aggregation  (typed scores, spans, caveats)
        ↓
POST /v1/analyze-feedback  →  Dashboard  /  Group 6 advisory signal
```

Matches the required shape: input → preprocessing → NLP model → post-processing → API asset → consuming application.

## 18.  Responsible AI

| Concern | Control |
|---|---|
| Privacy | Public, CC-licensed corpus; no student identifiers. PII redacted at model input; `data/private/` gitignored and never sent to any external service |
| Instructor names | RateMyProfessor reviews name instructors. The regex pass redacts names after an honorific (“Prof. X”, “Dr. X”) but not bare names (“Kathy is great”). Named-entity redaction is planned before any institutional data is used |
| Individual harm | Course-level aggregates only; advisory flags enforced in the API contract; no per-student inference |
| Faculty harm | Output is advisory and must not be a sole input to any evaluation of an instructor. EduRABSA's *Personal traits* category (looks, accent) is deliberately excluded from the aspect layer |
| Overconfidence | Typed scores; uncalibrated output labelled as such; caveats attached to every aggregate |
| Small-sample misreading | Aggregates over fewer than 20 comments carry an instability caveat that consumers must not strip |
| Bias | Per-category results reported in full; low-support categories flagged rather than averaged away. The corpus is North American and UK review-site text, so performance on Indian classroom feedback is untested |
| Hallucination | Extractive only; every aspect carries a supporting span; no free-text generation in the core pipeline |
| Security | Feedback text treated as untrusted input; callers warned against passing it unescaped into downstream LLM prompts |
| Reproducibility | Fixed seed, frozen split, pinned dependencies, versioned dataset, mapping, preprocessing and model |

## 19.  Limitations and Threats to Validity

Stated before the improved model's results exist, so they cannot be selected after the fact.

- **Domain shift.** Reviews come from public rating sites in Canada, the US and the UK. They are more colloquial and more extreme than solicited course feedback at NMIMS. Results here are in-domain for *education reviews*, not yet for our institution.
- **Label mapping is ours.** The source annotation is expert work, but the 24 → 8 mapping is a team decision that has not yet been checked by independent annotators (Section 8).
- **Single-annotator source.** EduRABSA reports no inter-annotator agreement, so the ceiling on achievable agreement is unknown.
- **Pace is rule-derived.** Its 141 comments come from a keyword-plus-category rule, and the seed lexicon shares some of those keywords. Lexicon results on pace are therefore optimistic.
- **Rare labels.** Pace (21 test comments) and neutral (8% of pairs) remain the thinnest classes; their per-class figures are the least stable.
- **Baseline ceiling.** Bag-of-words features cannot represent which words belong to which aspect, negation scope, sarcasm or implicit aspects. Section 15 confirms that the first of these is the largest source of error. It is the specific gap the Week 3 transformer is intended to close.

## 20.  Next Steps

| Week | Objective | Deliverable |
|---|---|---|
| 3 | Transformer ABSA with aspect-specific attention (joint aspect + polarity) | Pipeline v1, validation results against this baseline |
| 3 (team) | Mapping-check calibration round on 40 comments | κ per category; mapping revised if below 0.60 |
| 4 | Baseline vs improved on the frozen test split | Comparison with CIs and a stated verdict, including negative results |
| 5 | Full error analysis (second coder); score calibration | Evaluation report, calibrated probabilities |
| 6 | FastAPI implementation of the v1 contract | Working reusable asset + model card |
| 7 | Group 6 integration; Responsible AI review | Integrated demo |
| 8 | Tests, documentation, presentation | Final submission |

The immediate blocker in v2.0 — the dataset decision — is resolved. The one open item that needs people rather than code is the 40-comment calibration round.

## References

- Hua, Denny, Wicker and Taskova (2025). EduRABSA: An Education Review Dataset for Aspect-based Sentiment Analysis Tasks. arXiv:2508.17008. Dataset: Zenodo, doi 10.5281/zenodo.16935017.
- Pontiki et al. (2014). SemEval-2014 Task 4: Aspect Based Sentiment Analysis. Proceedings of SemEval-2014.
- Sechidis, Tsoumakas and Vlahavas (2011). On the Stratification of Multi-Label Data. ECML PKDD.
- Landis and Koch (1977). The Measurement of Observer Agreement for Categorical Data. Biometrics 33(1).

## Appendix A.  Reproducing every number

Run from the repository root (Python 3.13; `requirements.txt`). Runtime on a laptop CPU: about 3 minutes.

```
python src/build_dataset.py                          # downloads EduRABSA (31 MB), writes data/raw/annotations.csv
python src/data.py data/raw/annotations.csv          # validation, incl. span integrity
python src/run_baseline.py data/raw/annotations.csv  # split, lexicon, Tasks A-C, CV, ablations, errors
python src/profile_dataset.py                        # dataset quality tables (Sections 5, 10, 11)
python src/make_calibration_sheet.py                 # 40-comment sheet for the Section 8 round
pytest tests/ -q                                     # 6 tests
```

| Report section | Source file |
|---|---|
| 5.2 mapping counts, pace audit | `results/dataset_build.json`, `results/pace_relabel_audit.tsv` |
| 5.3, 10, 11.1 | `results/dataset_profile.json` |
| 14.1–14.4 | `results/metrics.json`, `results/results_tables.md`, `results/run_baseline_log.txt` |
| 15 | `results/errors_for_analysis.json`, `results/errors_coded_week2.json` |
