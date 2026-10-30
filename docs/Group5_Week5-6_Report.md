---
title: Student Feedback Intelligence
subtitle: Aspect-Based Sentiment Analysis of Academic Feedback
week: Weeks 5–6: Evaluation, Error Analysis and the Reusable Asset
course: SNLP Course Project 2026   •   Group 5
members: E029  ·  E030  ·  E031  ·  E033  ·  E034  ·  E035
version: Document version 1.0  •  Model transformer-absa-0.2.2  •  Dataset v1.1-edurabsa
note: Measurement only: no model weight, threshold or rule was changed after the Week 4 freeze.
footer: Group 5  •  Student Feedback Intelligence  •  Weeks 5–6  •  
---

# Student Feedback Intelligence — Weeks 5–6 Report

*SNLP Course Project 2026 · Group 5 · E029 · E030 · E031 · E033 · E034 · E035*

Weeks 5–6 deliverables: evaluation report with error cases, and a working reusable NLP asset.

## 1.  Scope and Status

The handout sets two targets for this fortnight:

- **Week 5 (3–9 October), evaluation and error analysis:** quantitative evaluation, test set, error analysis, robustness and failure cases. Deliverable: an evaluation report with metrics and at least 5–10 error cases.
- **Week 6 (10–16 October), asset engineering:** API or module, structured input and output, documentation, and confidence scores where applicable. Deliverable: a working reusable NLP asset.

The system evaluated is the one frozen at the end of Week 4: DeBERTa-v3 aspect-attention model, decision settings v1.1, dataset v1.1. Nothing about it was changed to improve the numbers below. The only addition is score calibration, which changes the confidence values the asset reports but never which aspects it reports.

### Status at end of Week 6

| Component | Status |
|---|---|
| Confidence scores | Calibrated on validation; on test, calibration error falls from 0.030 to 0.008 for aspects and from 0.116 to 0.021 for document sentiment (Section 3) |
| Test evaluation | Tuple-F1 **0.704** [0.683, 0.723] vs 0.452 for the TF-IDF baseline; per-aspect results with confidence intervals (Section 4) |
| Error slices | Long reviews, course reviews, negation and contrast words are the hard cases (Section 5) |
| Robustness | Typos cost 0.026 tuple-F1; lowercase and missing punctuation cost little (Section 6.1) |
| Behaviour probes | 17 of 32 hand-written probes fully correct; short comments 6/6, sarcasm 0/4, Hinglish 0/5 (Section 6.2) |
| Error analysis | 40 test errors coded by hand; 10 representative cases explained (Section 7) |
| Reusable asset | FastAPI service with 5 endpoints, typed request/response, error codes, interactive docs, client example (Section 8) |
| Tests | 31 passing (8 new: 7 for the API, 1 for quote redaction) |

## 2.  What Was Evaluated, and How

All figures are on the 900-comment **test split**, which has never been used to choose anything. The protocol allows measurement on it once the system is frozen; every opening is still logged in `results/test_openings.log`, and this fortnight added two entries (calibration check and evaluation, both measurement only).

The headline metric is the same as in every earlier week: exact-match micro **tuple-F1**. A prediction counts only when both the aspect and its polarity match the gold label. The system reported is the asset itself: every number in Sections 4–7 comes from the same model, settings and code the API serves.

## 3.  Confidence Scores: Calibration

Until now every score the asset returned was labelled `uncalibrated_probability`, because a raw model probability of 0.9 need not mean “right 9 times in 10”. The API specification made calibration a Week 5 deliverable.

**Method.** Each of the three reported scores gets an isotonic mapping: a monotone lookup table from raw score to observed success rate, fitted on the 900 **validation** comments (Zadrozny and Elkan, 2002). The scores are:

- **aspect score:** P(the aspect is discussed);
- **polarity score:** P(the stated feeling is right, given the aspect is right);
- **document score:** P(the whole-comment label is right).

It is then measured on the test comments with the expected calibration error (ECE: the average gap between stated confidence and actual accuracy, over 10 score bins) and the Brier score.

| Score | Test scores | ECE before | ECE after | Brier before | Brier after |
|---|---|---|---|---|---|
| Aspect | 7,200 | 0.030 | **0.008** | 0.053 | 0.051 |
| Polarity | 1,451 | 0.050 | **0.026** | 0.116 | 0.117 |
| Document sentiment | 900 | 0.116 | **0.021** | 0.123 | **0.094** |

The document head was the most over-confident. Before calibration, 818 test comments were scored 0.9 or higher, with an average stated confidence of 0.99, but only 89% of them were right. After calibration, the 465 comments in the top band average 0.99 and are 99% right:

| Document score band | Before: comments | Stated | Actual | After: comments | Stated | Actual |
|---|---|---|---|---|---|---|
| 0.5–0.6 | 21 | 0.55 | 0.43 | 104 | 0.57 | 0.52 |
| 0.8–0.9 | 35 | 0.86 | 0.57 | 159 | 0.88 | 0.88 |
| 0.9–1.0 | 818 | 0.99 | 0.89 | 465 | 0.99 | 0.99 |

Aspect scores were over-confident in the middle range: aspects scored 0.5–0.6 were really present only 38% of the time; after calibration the 0.6–0.7 band is right 61% of the time. Polarity improves less (0.050 → 0.026), mainly because neutral and conflict opinions are hard to get right at any confidence (Section 4.3).

Two properties matter for consumers. **Decisions are unchanged:** the same aspects are reported with the same polarity, so tuple-F1 is identical before and after. **Rankings are unchanged:** isotonic maps are monotone, so sorting comments by score gives the same order. The API now returns `calibrated_probability` for every score.

## 4.  Quantitative Evaluation (Test Split)

### 4.1  Overall

| Measure | Value |
|---|---|
| Tuple-F1 (headline) | **0.704** [0.683, 0.723] |
| Precision / recall | 0.699 / 0.708 |
| TF-IDF baseline, same comments | 0.452 [0.427, 0.476] |
| Aspect detection micro-F1 (aspect right, any polarity) | 0.856 |
| Polarity accuracy on correctly found aspects | 0.822 |
| Comments where every judgement is exactly right | 45.1% (406 of 900) |
| Document sentiment: accuracy / macro-F1 | 0.859 / 0.831 |

The last-but-one row is the strictest view: fewer than half of the comments are entirely right, because most carry two or three aspects and one slip makes the comment imperfect. Of the 494 comments with an error, 311 (63%) have exactly one aspect wrong (one missing, extra or wrongly-polarised aspect), and only 59 have three or more. The typical error is a single slip, not a wholly wrong reading (Section 7).

### 4.2  Per aspect

| Aspect | Test support | Precision | Recall | Tuple-F1 | 95% CI |
|---|---|---|---|---|---|
| teaching | 492 | 0.777 | 0.829 | 0.802 | [0.770, 0.833] |
| faculty_interaction | 237 | 0.745 | 0.861 | 0.798 | [0.755, 0.837] |
| assessment | 227 | 0.687 | 0.656 | 0.671 | [0.609, 0.726] |
| course_content | 408 | 0.648 | 0.654 | 0.651 | [0.608, 0.695] |
| pace | 21 | 0.750 | 0.571 | 0.649 | [0.428, 0.818] |
| workload | 78 | 0.653 | 0.603 | 0.627 | [0.530, 0.717] |
| other | 122 | 0.624 | 0.475 | 0.540 | [0.451, 0.620] |
| resources | 100 | 0.495 | 0.480 | 0.487 | [0.398, 0.576] |

The interval for pace (0.43 to 0.82) shows why a 21-comment category cannot be ranked against the others. Resources and other are the weakest well-supported categories.

What kind of mistake each aspect makes, counted per comment:

| Aspect | Correct | Wrong polarity | Missed | Spurious |
|---|---|---|---|---|
| teaching | 408 | 66 | 18 | 51 |
| faculty_interaction | 204 | 14 | 19 | 56 |
| course_content | 267 | **97** | 44 | 48 |
| assessment | 149 | 37 | 41 | 31 |
| workload | 47 | 5 | 26 | 20 |
| resources | 48 | 19 | 33 | 30 |
| other | 58 | 18 | **46** | 17 |
| pace | 12 | 2 | 7 | 2 |

Course content is found reliably but its polarity is often wrong (97 cases): content comments are where students mix praise and complaint (“interesting but very hard”). Other is mostly *missed* (46), because it is a mixed bag of labs, tutorials and readings. Faculty_interaction is over-predicted (56 spurious): “nice”, “helpful” and “responds” fire it even when the gold labels put the remark elsewhere.

### 4.3  Which polarities get confused

On the aspects the model found correctly, gold polarity (rows) against predicted polarity (columns):

| Gold \ Predicted | positive | negative | neutral | conflict |
|---|---|---|---|---|
| positive | **721** | 6 | 7 | 19 |
| negative | 10 | **385** | 14 | 25 |
| neutral | 48 | 28 | **31** | 9 |
| conflict | 56 | 34 | 2 | **56** |

Positive and negative are almost never swapped (16 swaps in 1,187). The difficulty is entirely in the two minority labels: only 31 of 116 neutral opinions and 56 of 148 conflict opinions are recognised. Most are pulled towards whichever side of the opinion is stronger. This is the main remaining weakness of the model, and it matches the error sample in Section 7.

Document sentiment shows the same pattern: positive F1 0.915, negative 0.895, neutral 0.684. Almost all confusions involve the neutral class; positive and negative comments are confused with each other only twice in 900.

## 5.  Where the Errors Are

Tuple-F1 on slices of the test set:

| Slice | Comments | Tuple-F1 | Every judgement right |
|---|---|---|---|
| 1–10 words | 109 | 0.717 | 67.9% |
| 11–40 words | 369 | 0.728 | 50.7% |
| 41–80 words | 360 | 0.707 | 38.1% |
| 81+ words | 62 | **0.611** | 12.9% |
| Teacher reviews (RateMyProfessor) | 458 | 0.759 | 47.8% |
| Course reviews (Waterloo) | 442 | **0.635** | 42.3% |
| No negation word | 474 | 0.739 | 52.1% |
| Contains a negation word | 426 | 0.675 | 37.3% |
| No contrast word | 515 | 0.760 | 55.3% |
| Contains but / however / although | 385 | **0.652** | 31.4% |

Four findings:

1. **Very long reviews are the hardest:** 0.611 above 80 words, where many aspects compete for attention and the 256-token limit can truncate the end.
2. **Course reviews are harder than teacher reviews** (0.635 vs 0.759). They spread over more aspects: content, workload, assessment and resources, where the model is weaker. Teacher reviews concentrate on teaching and faculty interaction, where it is strongest.
3. **Contrast words cost 0.108.** “But” sentences are exactly the mixed reviews where conflict and neutral polarities live.
4. **Comments with no aspect are handled well:** 73% of the 67 comments that name no aspect (“Loved it!”) correctly get an empty answer.

## 6.  Robustness

### 6.1  Perturbed input

The same 900 test comments were re-run with three automatic, seeded changes, to see whether messy real-world typing breaks the model:

| Variant | Tuple-F1 | 95% CI | Comments with unchanged output |
|---|---|---|---|
| Original | 0.704 | [0.683, 0.723] | 100% |
| Typos in ~10% of longer words | 0.678 | [0.658, 0.698] | 77.6% |
| All lowercase | 0.700 | [0.677, 0.720] | 89.6% |
| All punctuation removed | 0.692 | [0.670, 0.713] | 77.7% |

A typo-ed comment looks like: “The lectures wree very innformative, but the pce was too fast and the assignmens wree difficult.”

The model degrades gracefully. Typos cost 0.026 tuple-F1 and lowercase costs almost nothing; DeBERTa's sub-word tokens still recognise misspelled words. Removing punctuation changes a fifth of the outputs but costs only 0.012, as clause boundaries can still be inferred from words like *but*.

### 6.2  Behaviour probes

Average scores hide whether specific behaviours work. Following the CheckList approach (Ribeiro et al., 2020), 32 short sentences were written by hand, each testing one behaviour, and labelled with the annotation guidelines (`data/probes/probe_set.csv`). A probe passes only if the output is exactly right.

| Behaviour | Probes | Passed | Tuple-F1 | Example |
|---|---|---|---|---|
| Short form-style comment | 6 | **6** | 1.000 | “Too fast.” → pace: negative |
| Contrast, several aspects | 6 | 5 | 0.963 | Both handout examples pass |
| Negation | 6 | 4 | 0.857 | “The lectures were not boring at all.” → teaching: positive |
| Implicit aspect | 5 | 2 | 0.667 | “We barely get enough time to finish everything.” → nothing found |
| Sarcasm | 4 | **0** | 0.250 | “Oh great, another surprise quiz on Monday.” → nothing found |
| Hinglish | 5 | **0** | 0.500 | “Sir ka padhane ka tareeka bahut accha hai.” → nothing found |
| Total | 32 | 17 | | |

Three results stand out.

- **Short, explicit comments work (6/6).** “Too fast.” and “Notes were useful.” come out right, which matters for institutional feedback forms. Week 3's short-sentence failures were mostly *implicit* ones (“We barely get enough time to finish everything.”), and those still fail here (implicit row). So the real weakness is implicitness, not length.
- **Sarcasm and Hinglish fail completely.** Neither appears in the English review training data. For an Indian university this is the largest gap: comments like “pace bahut fast tha” are typical, and the model finds pace there but adds a spurious aspect.
- **A systematic bias:** in 5 of the 15 failed probes, the error is the same spurious **teaching: negative** added to a negative sentence about something else (“The exams were not fair.” → assessment: negative, which is right, plus teaching: negative, which is wrong). Half the training data are reviews of teachers, so the model has learned that negativity in an academic comment often concerns the teacher. In a course report this would unfairly lower an instructor's teaching score. It is a fairness issue to carry into the Responsible AI review.

## 7.  Error Analysis

### 7.1  What the errors are

494 of the 900 test comments contain at least one error. A random sample of 40 (seed 42) was read and given one primary cause each (`results/week5_errors_coded.json`):

| Cause | Count (of 40) |
|---|---|
| Boundary confusion: a neighbouring or extra category, often defensible | 16 |
| Conflict or mixed opinion flattened to one side | 8 |
| Neutral opinion read as positive or negative | 5 |
| Annotation gap: prediction correct under our guidelines, gold label missing | 4 |
| Minor aspect missed in a long review | 4 |
| Implicit aspect not inferred | 2 |
| Sarcasm or figurative language | 1 |

The profile has changed since Week 2, when the baseline's dominant error was copying one tone onto every aspect (11 of 40). That error is gone. What remains are mostly judgement calls about category boundaries and about mixed or neutral opinions: the same places where human annotators disagree most. Four of the 40 “errors” (10%) are the model being right and the gold label being incomplete, so the true quality is somewhat above the measured 0.704.

### 7.2  Ten representative error cases

Instructor names are replaced with [INSTRUCTOR].

| # | Comment (shortened) | Gold | Predicted | Why it failed |
|---|---|---|---|---|
| 1 | “She has said some nice things about my style of writing … she knows that she is disorganized & makes it up by going easy on a few things.” | teaching: conflict | teaching: positive | Mixed opinion of one aspect; the positive wording dominates. Conflict is recognised only 56 times in 148 (Section 4.3) |
| 2 | “This is an interesting course with lots of content. It isn't too difficult …” | course_content: neutral | course_content: positive | “Interesting” and “isn't too difficult” read as praise; neutral is the weakest polarity (31 of 116). The gold label is itself debatable |
| 3 | “The professor assigns a BOAT LOAD of readings that it becomes unbearable …” | workload: negative | assessment: negative | Quantity of work (workload) vs the nature of a task (assessment): the boundary the guidelines flag as hardest |
| 4 | “… when asking for help she explains it with you step by step one on one.” | faculty_interaction: positive | teaching: positive | “Explains” is the strongest teaching cue; the deciding question (explanation quality or availability?) needs the “when asking for help” context |
| 5 | “Did not like her at all. She's extremely unorganized, rude, grades based on her own opinions …” | + assessment: negative | (assessment missed) | A long, uniformly negative review: teaching and faculty interaction are found, the grading complaint is not |
| 6 | “… He cherry picked the class on who he wanted to teach things to and the rest was left out …” | faculty_interaction: negative | (missed) | Implicit aspect: no word names it; it needs the inference “ignores some students” |
| 7 | “This course is a neglected middle child who bothers you every 2 weeks.” | resources: conflict | three spurious aspects | Figurative language; the literal words point nowhere useful |
| 8 | “[INSTRUCTOR] is awesome! … His comments on my papers are SO helpful.” | (not labelled) | faculty_interaction: positive | Annotation gap: feedback on work is faculty_interaction in our guidelines, so the prediction is right and counted as an error |
| 9 | Probe: “The exams were not fair.” | assessment: negative | + teaching: negative | The learned teaching-negative bias (Section 6.2) |
| 10 | Probe: “Sir ka padhane ka tareeka bahut accha hai.” | teaching: positive | (nothing) | Hinglish is outside the training data; no English cue word for teaching |

Cases 1–2 point to polarity granularity, 3–4 to category boundaries, 5–7 to reading depth, 8 to the labels, and 9–10 to the data's coverage. Each family has a different remedy (Section 10).

## 8.  The Reusable Asset (Week 6)

### 8.1  What was built

The pipeline (Weeks 3–5) is now a web service, `src/api.py`, built with FastAPI. Another application sends JSON over HTTP and gets JSON back, without installing our model or reading our code.

```
client (any language) ─HTTP/JSON─► FastAPI service (src/api.py)
                                     │  validates the request, maps errors to codes
                                     ▼
                                   pipeline.FeedbackAnalyzer (src/pipeline.py)
                                     │  redact PII → DeBERTa ABSA → v1.1 decisions
                                     │  → calibrated scores → evidence spans → JSON
                                     ▼
                              response: aspects, polarity, scores, evidence / course report
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/analyze-feedback` | One comment: aspects, polarity, calibrated scores, evidence span, warnings |
| `POST /v1/analyze-feedback/batch` | Up to 500 comments: sentiment distribution, aspect summary, top positives, top concerns, emerging concerns, verbatim quotes, caveats |
| `GET /v1/health` | Whether the model is loaded, version, device |
| `GET /v1/schema` | The 8 aspects and 4 polarities with their definitions, so no consumer hard-codes our labels |
| `GET /v1/model-card` | Training data, test results, calibration, known failure modes, intended and out-of-scope uses, all built from the result files |

Every request and response has a typed schema, which also generates an interactive documentation page at `/docs`, where each endpoint can be tried in the browser. Errors use one documented format (`INVALID_REQUEST`, `EMPTY_TEXT`, `TEXT_TOO_LONG`, `BATCH_TOO_LARGE`, `MODEL_UNAVAILABLE`). In a batch, one bad comment is reported in `failures` and does not fail the request.

### 8.2  Using it

The handout's reusable-asset test asks what another group would need to use the component tomorrow. The answer is `docs/API.md`: start the service with `python src/api.py`, then send a request. The handout's own example through the live service:

```
POST /v1/analyze-feedback
{"text": "The lectures were very informative, but the pace was too fast and the assignments were difficult."}

teaching    positive   aspect 0.91   polarity 0.50   "The lectures were very informative"
pace        negative   aspect 0.96   polarity 0.98   "the pace was too fast"
assessment  negative   aspect 0.99   polarity 0.98   "the assignments were difficult"
```

With calibration, the scores now say something true: the model is near-certain about pace and assignments, and only 50/50 on whether the lectures remark is positive (Week 4 found the same hesitation).

`examples/client_example.py` shows the whole cycle from another program, using only the Python standard library: health check, one comment, a 6-comment batch, an error, and timing. Measured on the laptop: **45–61 ms** mean round trip per request over HTTP (GPU; two runs of 20 calls).

### 8.3  Course reports: emerging concerns

The Week 4 report showed the ranking floor hiding a small but serious concern (pace: 9 mentions, all negative). The batch response now has an `emerging_concerns` list for exactly that case: aspects below the floor but mentioned at least 3 times with net sentiment of −0.5 or lower. Every batch also carries caveats the consumer must show. Under 20 comments, an instability warning is added automatically.

On a random sample of 150 real test comments treated as one course (`examples/demo_course_report.py`, output in `results/week6_demo_course_report.txt`):

- **Top strengths:** teaching (79 mentions), course_content (64) and faculty_interaction (44).
- **Top concern:** workload, with 14 mentions and net sentiment −0.71.
- **Emerging concern:** pace, with only 3 mentions but all negative. This is exactly the case the floor would otherwise hide.

### 8.4  Tests

31 automated tests pass. The 7 new API tests check:

- the label schema and the error format, including a missing field and a batch over 500;
- that the model loads, and that empty or over-long text is rejected with the right codes;
- that the handout example comes back exactly right over HTTP, with calibrated scores and evidence offsets that point into the original text;
- the batch report shape, and that the model card states the out-of-scope uses.

## 9.  Responsible-AI Findings This Fortnight

- **Privacy gaps found and partly fixed.** Two gaps were found:
  - One test comment contains an email address written with spaces (“name@ domain. com”), which the redaction regex missed.
  - Course-report quotes were taken from the original text, so a name such as “professor [name]” could reach a coordinator.
  Quotes are now redacted with a stricter rule: titles in any case, and spaced emails (`redact_quote`, with a test). The model's own input redaction was left unchanged so that every frozen result still reproduces. Bare names without any title are still not caught; named-entity redaction is needed before real institutional data is used.
- **Fairness to instructors.** The spurious teaching-negative pattern (Section 6.2) would make course reports harsher on teaching than students intended. The API already forbids ranking instructors; the bias is documented in the model card and should be measured on more probes in Week 7.
- **Language coverage.** Hinglish comments are mostly missed. The model card states the asset is English-only, so a consumer is not misled.
- **Honest confidence.** Calibrated scores mean a “0.9” can now be trusted to mean roughly 90%, and `low_confidence` warnings fire on the uncertain judgements.

## 10.  Limitations and Next Steps

- **Neutral and conflict polarities** remain the main weakness (Section 4.3). They are rare in training (8% and 9% of labels). A class-weighted loss or more mixed-opinion examples are the obvious remedies; this is now a known limitation, not a Week 7 build item.
- **Category boundaries** account for 16 of 40 sampled errors. Several are defensible under the guidelines, so part of the fix is label quality, not the model.
- **Sarcasm, Hinglish and implicit aspects** are outside what this training data teaches.
- **One test set.** All numbers come from the same public review sites as the training data. Performance on NMIMS feedback forms is still untested.

| Week | Handout target | Plan |
|---|---|---|
| 7 | Integration, demo, Responsible AI | Group 6 advisory signal from the batch endpoint (API spec §8); demo interface; Responsible AI statement (privacy, the teaching-negative bias, language coverage) |
| 8 | Finalisation | Final report, 10-slide presentation, README, code freeze |

## References

- Guo, Pleiss, Sun and Weinberger (2017). On Calibration of Modern Neural Networks. ICML 2017.
- He, Gao and Chen (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. arXiv:2111.09543.
- Hua, Denny, Wicker and Taskova (2025). EduRABSA: An Education Review Dataset for Aspect-based Sentiment Analysis Tasks. arXiv:2508.17008.
- Ribeiro, Wu, Guestrin and Singh (2020). Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. ACL 2020.
- Zadrozny and Elkan (2002). Transforming Classifier Scores into Accurate Multiclass Probability Estimates. KDD 2002.

## Appendix A.  Reproducing every number

From the repository root, after the Week 1–4 steps (dataset v1.1, frozen split, trained model, frozen settings):

```
python src/calibrate.py               # Section 3: fit on validation, measure on test
python src/evaluate_week5.py          # Sections 4-7: metrics, slices, robustness, probes, errors
python src/api.py                     # Section 8: start the service (second terminal)
python examples/client_example.py     # Section 8.2: real calls and timing
python examples/demo_course_report.py # Section 8.3: course report for 150 comments
pytest tests/ -q                      # 31 tests
```

| Report section | Source file |
|---|---|
| 3 | `results/week5_calibration.json`, `data/processed/calibration.json` |
| 4, 5, 6.1 | `results/week5_eval.json`, `results/week5_eval_log.txt` |
| 6.2 | `results/week5_probe_results.json`, `data/probes/probe_set.csv` |
| 7 | `results/week5_test_errors.json`, `results/week5_errors_coded.json` |
| 8 | `results/week6_client_run.txt`, `results/week6_demo_course_report.txt`, `results/week6_model_card.json`, `docs/API.md` |
| 2 | `results/test_openings.log` |
