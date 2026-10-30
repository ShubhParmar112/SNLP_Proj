---
title: Student Feedback Intelligence
subtitle: Aspect-Based Sentiment Analysis of Academic Feedback
week: Week 4: Improved Model — Baseline vs Improved Comparison
course: SNLP Course Project 2026   •   Group 5
members: E029  ·  E030  ·  E031  ·  E033  ·  E034  ·  E035
version: Document version 1.1  •  Model transformer-absa-0.2.1  •  Dataset v1.1-edurabsa
note: Every setting was frozen on the validation split before the test split was opened (Section 3).
footer: Group 5  •  Student Feedback Intelligence  •  Week 4  •  
---

# Student Feedback Intelligence — Week 4 Report

*SNLP Course Project 2026 · Group 5 · E029 · E030 · E031 · E033 · E034 · E035*

Week 4 deliverable: baseline vs improved model comparison. Companion to the Week 1–2 (v3.0) and Week 3 reports.

## 1.  Scope and Status

The handout's Week 4 target (26 September – 2 October) is to *implement and compare an improved approach*, with *baseline vs improved model comparison* as the deliverable. This report gives that comparison on the held-out **test split** (900 comments), which the project's evaluation protocol keeps closed until this point.

Four things were done this week, in this order:

1. **A comparison model was added** — frozen sentence embeddings with logistic regression, the handout's own first example of an improved approach (“TF-IDF → Sentence Transformer”).
2. **Pipeline v1.1 was finalised on the validation split** — per-aspect decision thresholds, a better evidence-span rule and a report-ranking rule, then frozen.
3. **The test split was opened**, and every system was scored on it with the same code (Sections 5–8).
4. **A labelling fix was made and the comparison repeated** — lecture comments now count as `teaching`, which makes the handout's own example sentences come out right (Section 9).

As in earlier weeks, no number in this report was estimated; each traces to a file in `results/` (Appendix A).

### Status at end of Week 4

| Component | Status |
|---|---|
| Baseline vs improved comparison on test | Complete — 10 systems, same 900 comments, 95% CIs and paired tests (Section 5) |
| Headline result (dataset v1.0) | DeBERTa ABSA v1.1: tuple-F1 **0.703** [0.684, 0.723] vs **0.454** for the TF-IDF baseline; difference +0.249 [+0.225, +0.275] |
| Final result after the lectures fix (dataset v1.1) | Tuple-F1 **0.704** [0.683, 0.723] vs **0.452** for the baseline; difference +0.252 [+0.225, +0.279] (Section 9) |
| Handout example sentences | Both Group 5 examples now fully correct (Section 9.3) |
| Sentence-embedding comparison model | Complete — 0.494: only +0.039 over TF-IDF (Section 6) |
| Pipeline v1.1 | Complete — tuned decisions, phrase-length evidence spans (Section 4) |
| Evidence spans | Close match to gold rises from 12% to 55% of spans on test (Section 8) |
| Test-split discipline | Settings frozen on validation before every test run; every opening logged (Section 3) |
| Tests | 23 passing (8 new this week) |

## 2.  What Was Compared

The comparison follows the handout's progression from a simple to an improved NLP approach. All systems are trained on the same 4,198 training comments and scored on the same 900 test comments.

| System | Role | What it is |
|---|---|---|
| Majority class / stratified random | Reference | Always the most common tuple / tuples sampled from the training distribution |
| Lexicon + LogReg polarity | Week 2 baseline | Keyword lexicon for aspects, TF-IDF classifier for polarity |
| TF-IDF + LogReg | **Week 2 baseline** | Word and character n-grams; aspect detection, then one shared polarity classifier |
| MiniLM embeddings + LogReg | Week 4 comparison | Frozen pretrained sentence vector per comment (all-MiniLM-L6-v2), one 5-way classifier per aspect |
| DeBERTa ABSA v1 | Week 3 model | Fine-tuned DeBERTa-v3-base with one attention query per aspect and span supervision |
| **DeBERTa ABSA v1.1** | **Week 4 — the asset** | v1 weights with decision settings tuned on validation (Section 4) |
| 3-seed ensemble | Experiment | Average of three v1 models trained with different seeds |
| v1 without span supervision | Ablation | Same model trained without the evidence loss |

The embedding model is the key addition. It sits between the other two: it has pretrained sentence meaning like DeBERTa, but it is frozen and gives one vector per comment, like a bag of words. It uses the same 5-way-per-aspect output as the transformer, so the two differ only in the representation. If it matched DeBERTa, the Week 3 gain would be “just pretraining”. It does not (Section 6).

## 3.  How the Test Split Was Used

The protocol (Evaluation_Protocol.md §4.2) says the test set is opened at the end of Week 2 and again at the end of Week 4, and that no decision is made on it. This week enforced that in code rather than by promise:

- **Every choice was made on the validation split** by `src/tune_week4.py`: the embedding model's class weighting, the presence thresholds for three systems, and the evidence-span rule.
- **The choices were then frozen** to `data/processed/decision_settings.json`, with a SHA-256 fingerprint stored in `results/week4_frozen_settings.json`. Frozen at 18:18:05 on 2 October.
- **The test script checks the fingerprint** and refuses to run if the settings changed after freezing. Every opening is appended to `results/test_openings.log`. The main comparison (Sections 5–8) opened the test split once, at 18:19:19 on 2 October.
- **After the lectures fix (Section 9)** the same procedure was repeated: the model was retrained, every setting re-tuned on validation and frozen again (18:47:16), and then the baseline (18:38) and the retrained model (18:47) were each scored once on the same 900 test comments. The fix was decided from the handout's example sentences and our own guidelines, not from test results.
- **Decisions were committed before the test run**, including the one that could look tempting afterwards: the 3-seed ensemble is reported but not adopted into the asset, because it triples inference cost and GPU memory.
- **The pipeline was checked end to end**: running pipeline v1.1 itself on all 900 test comments reproduces the v1.1 row of the results table exactly (`pipeline_check` in `week4_test_eval.json`).

## 4.  Pipeline v1.1 — What Changed

The model weights are unchanged from Week 3. Version 1.1 changes only the decisions made on top of the model's probabilities, so it adds no model computation.

### 4.1  Per-aspect presence thresholds

Version 1 took the most probable of five slots per aspect (none, positive, negative, neutral, conflict). Version 1.1 separates the two decisions: an aspect is present when P(present) = 1 − P(none) reaches its threshold, and its polarity is then the most probable of the four polarities.

Thresholds were searched over 0.20–0.80 on validation. First one global value was chosen; then each aspect with at least 50 validation examples got its own, by coordinate ascent. Pace (21 examples) keeps the global value, since a cut-off fitted to 21 comments would be fitted to noise.

| Aspect | teaching | faculty_interaction | course_content | pace | assessment | workload | resources | other |
|---|---|---|---|---|---|---|---|---|
| Threshold | 0.20 | 0.80 | 0.40 | 0.65 (global) | 0.45 | 0.65 | 0.80 | 0.75 |

Validation tuple-F1: v1 0.712 → global threshold alone 0.712 → per-aspect thresholds 0.720. The gain is small and the thresholds are spread widely, which is itself a warning sign: the F1 surface is flat, so several of these values are near-ties. The test result (Section 5) is the real check.

### 4.2  Evidence spans: from one word to a phrase

The Week 3 report found that evidence spans pointed at the right place but were too short. Week 4 measured why: the supervised attention is almost one-hot. Even with the growth rule at its loosest setting the median span is **1 word**, against **6 words** in the gold spans, so no threshold on attention weight can produce a phrase.

The new rule keeps the attention peak as the anchor and widens it to its clause — up to the nearest punctuation mark or contrast word (*but*, *however*, *although*, *and* …) — capped at a maximum number of words. On validation:

| Span rule | Close match (Jaccard ≥ 0.5) | Any overlap | Median words (gold 6) |
|---|---|---|---|
| Attention peak, 50% rule (v1) | 13.7% | 76.9% | 1 |
| Attention peak, 1% rule | 23.1% | 80.2% | 1 |
| Clause, max 6 words | 54.1% | 86.1% | 5 |
| **Clause, max 8 words (v1.1)** | **55.2%** | 86.6% | 5 |
| Clause, max 10 words | 55.0% | 86.9% | 5 |
| Clause, max 14 words | 54.3% | 87.2% | 5 |
| Clause, no cap | 54.1% | 87.2% | 5 |

The clause rule quadruples close matches, and the result is insensitive to the cap, so the choice of 8 is not fragile. Highlighting is still anchored on what the model attended to; the clause rule only makes it readable.

### 4.3  Report ranking floor

The Week 3 course report ranked pace as a “top concern” on two mentions. In v1.1, an aspect must be mentioned by at least 3 comments *and* at least 5% of the comments analysed before it can be ranked as a top strength or concern. This is a design rule, fixed rather than tuned. Section 10 shows its cost.

## 5.  Results on the Test Split

Sections 5–8 are the full comparison on dataset v1.0, the labels used since Week 2. Section 9 repeats the key comparison after the lectures fix (dataset v1.1); the conclusions are the same.

### 5.1  Task C — ABSA tuple extraction (headline)

| System | Tuple-F1 | 95% CI | P | R | Polarity acc. on detected | Aspect F1 | Macro F1 |
|---|---|---|---|---|---|---|---|
| Majority class | 0.216 | [0.197, 0.237] | 0.311 | 0.166 | 0.587 | 0.368 | 0.051 |
| Stratified random | 0.145 | [0.128, 0.164] | 0.148 | 0.143 | 0.416 | 0.349 | 0.091 |
| Lexicon + LogReg polarity | 0.332 | [0.307, 0.356] | 0.379 | 0.295 | 0.637 | 0.521 | 0.296 |
| TF-IDF + LogReg (baseline) | 0.454 | [0.432, 0.479] | 0.434 | 0.477 | 0.617 | 0.736 | 0.412 |
| MiniLM embeddings + LogReg | 0.471 | [0.443, 0.496] | 0.620 | 0.380 | 0.720 | 0.655 | 0.275 |
| MiniLM embeddings, tuned | 0.494 | [0.470, 0.517] | 0.517 | 0.473 | 0.695 | 0.710 | 0.335 |
| DeBERTa ABSA v1 | 0.696 | [0.676, 0.715] | 0.705 | 0.687 | 0.823 | 0.845 | 0.635 |
| **DeBERTa ABSA v1.1** | **0.703** | [0.684, 0.723] | 0.714 | 0.693 | 0.816 | 0.862 | 0.644 |
| 3-seed ensemble (experiment) | 0.713 | [0.693, 0.735] | 0.737 | 0.690 | 0.828 | 0.861 | 0.651 |
| v1 without span supervision | 0.663 | [0.642, 0.684] | 0.674 | 0.652 | 0.807 | 0.821 | 0.510 |

The baseline row reproduces the Week 2 figure exactly (0.454 [0.432, 0.479]), confirming that the frozen split and code are stable between weeks. Seed stability on test: v1 trained with seeds 42, 43 and 44 scores 0.696, 0.700 and 0.702 (0.699 ± 0.003).

### 5.2  Paired comparisons

Each difference is computed by resampling the same 900 test comments for both systems together (1,000 bootstrap resamples).

| Comparison | Difference in tuple-F1 | 95% CI | Verdict |
|---|---|---|---|
| **v1.1 − TF-IDF baseline** | **+0.249** | [+0.225, +0.275] | Established |
| v1 − TF-IDF baseline | +0.241 | [+0.216, +0.267] | Established |
| Embeddings (tuned) − TF-IDF baseline | +0.039 | [+0.014, +0.064] | Established, small |
| v1.1 − embeddings (tuned) | +0.210 | [+0.186, +0.233] | Established |
| v1 − no span supervision | +0.033 | [+0.017, +0.050] | Established |
| v1.1 − v1 (threshold tuning) | +0.008 | [+0.000, +0.015] | Marginal |
| Ensemble − v1.1 | +0.009 | [−0.003, +0.021] | Inconclusive |

### 5.3  Validation versus test

| System | Validation | Test |
|---|---|---|
| TF-IDF baseline | 0.476 | 0.454 |
| MiniLM embeddings, tuned | 0.507 | 0.494 |
| DeBERTa ABSA v1 | 0.712 | 0.696 |
| DeBERTa ABSA v1.1 | 0.720 | 0.703 |
| 3-seed ensemble | 0.735 | 0.713 |

Every system drops by 0.013–0.022 from validation to test, including the untuned TF-IDF baseline, so the drop reflects the two splits differing slightly, not overfitting of the tuned settings. The ranking of systems is identical on both.

## 6.  What the Comparison Shows

**1. The improved model wins clearly.** On held-out data, the fine-tuned aspect-attention model gets an (aspect, polarity) pair right far more often than the baseline: tuple-F1 0.703 against 0.454. The paired interval [+0.225, +0.275] is nowhere near zero. Both halves of the task improve: polarity accuracy on correctly found aspects rises from 0.617 to 0.816, and aspect detection from 0.736 to 0.862.

**2. Pretraining alone is not the reason.** Frozen sentence embeddings beat TF-IDF by only 0.039, and they do it in a lopsided way: high precision (0.620) but low recall (0.380), and they do not get a single pace judgement right (Section 7). One vector per comment has to blend every aspect of a mixed review into a single point, which is the same structural limit the Week 2 error analysis found in the bag-of-words baseline. The remaining +0.210 comes from what DeBERTa adds on top: fine-tuning on the task and a separate attention for each aspect.

**3. Evidence supervision is confirmed on test.** Removing the span loss costs 0.033 tuple-F1 on test, more than the 0.022 measured on validation in Week 3, and it hurts the rare categories most (macro F1 0.510 against 0.635).

**4. Threshold tuning helped only marginally.** v1.1 gains 0.008 over v1, with an interval just touching zero. It shifts the balance towards precision and aspect detection (aspect F1 0.845 → 0.862) at a slight cost in polarity accuracy (0.823 → 0.816). It is kept because it is never worse overall and costs nothing at run time. But it is reported as marginal, not as an improvement, and the widely spread thresholds should be re-tuned whenever the model is retrained.

**5. The ensemble is not worth its cost.** Three models give +0.009 with an interval that crosses zero, for three times the inference time and memory. The decision made before the test run — report it, do not deploy it — stands.

## 7.  Per-Category Results (Test)

| Aspect | Test support | TF-IDF baseline | MiniLM tuned | DeBERTa v1 | DeBERTa v1.1 |
|---|---|---|---|---|---|
| teaching | 477 | 0.671 | 0.724 | 0.809 | 0.808 |
| faculty_interaction | 237 | 0.604 | 0.613 | 0.781 | 0.822 |
| course_content | 408 | 0.367 | 0.407 | 0.661 | 0.656 |
| pace | 21 | 0.600 | 0.000 | 0.579 | 0.595 |
| assessment | 227 | 0.335 | 0.331 | 0.636 | 0.642 |
| workload | 78 | 0.264 | 0.252 | 0.575 | 0.590 |
| resources | 100 | 0.238 | 0.187 | 0.468 | 0.491 |
| other | 142 | 0.215 | 0.165 | 0.569 | 0.547 |
| Macro average | | 0.412 | 0.335 | 0.635 | 0.644 |

v1.1 improves on the baseline in seven of eight categories, with the largest gains on the categories usually expressed implicitly or inside mixed reviews: assessment, course_content, workload and other. **Pace is again the exception**: 0.595 against 0.600, effectively a tie on 21 test comments. Its 99 training examples come from a keyword rule (Week 1–2 report, Section 5.2), and that small, rule-derived class is the likely cause. After retraining on dataset v1.1, pace scores 0.649 against the baseline's 0.600 (Section 9). A swing that size between two training runs shows that a 21-comment category is the least reliable figure in the table. It should not be read as either solved or hopeless.

## 8.  Document Sentiment and Evidence Spans (Test)

| Model | Accuracy | Macro-F1 | F1 positive | F1 neutral | F1 negative |
|---|---|---|---|---|---|
| Majority class | 0.507 | 0.224 | 0.673 | 0.000 | 0.000 |
| TF-IDF + LogReg | 0.750 | 0.722 | 0.837 | 0.561 | 0.767 |
| MiniLM embeddings + LogReg | 0.714 | 0.691 | 0.794 | 0.533 | 0.746 |
| DeBERTa (document head) | **0.863** | **0.834** | 0.917 | 0.684 | 0.902 |

For whole-comment sentiment the frozen embeddings are slightly *worse* than TF-IDF, which supports point 2 above: generic sentence meaning is not the same as knowing how students express satisfaction.

| Evidence-span rule (test, 1,436 correctly detected aspects) | Close match (Jaccard ≥ 0.5) | Any overlap | Median words (gold 6) |
|---|---|---|---|
| v1: attention peak, 50% rule | 12.0% | 79.2% | 1 |
| v1.1: clause, max 8 words | **55.2%** | **90.0%** | 6 |

The span rule chosen on validation transfers to test unchanged (55.2% on both). Spans now have the same median length as the human-marked evidence.

## 9.  The Lectures Fix (Dataset v1.1)

### 9.1  The problem

Run on the handout's own Group 5 example, the v1.0 pipeline got two of three judgements right and one wrong:

```
Input: "The lectures were very informative, but the pace was too fast and the
        assignments were difficult."                     (model 0.1.1, dataset v1.0)
  pace        negative   "the pace was too fast"
  assessment  negative   "the assignments were difficult"
  other       negative   "The lectures were very informative"      <- should be teaching:positive
```

The cause is the label mapping, not the model. EduRABSA files most lecture comments (“the lectures were boring”) under *Course – Learning activity*, and the Week 1–2 mapping sent that category to `other`. So the model learned, correctly, that lectures are usually `other`. But our own guidelines put lecture delivery under `teaching` (Annotation Guidelines §2.1). The handout also uses lectures as the example aspect twice: Group 5's example in Section 5, and the reusable-asset test in Section 21.

### 9.2  The fix

One rule was added to `src/build_dataset.py` (Rule 2b): a *Learning activity* label whose aspect term names lectures or lessons becomes `teaching`. Mentions that are really about a resource (“video lectures”, “lecture slides”) or an assessment (“lecture quizzes”) keep their old label. The rule is covered by a unit test.

| | Dataset v1.0 | Dataset v1.1 |
|---|---|---|
| Comments relabelled | — | 221 of 5,998 (172 train, 22 validation, 27 test) |
| Comments with `teaching` | 3,184 | 3,305 |
| Comments with `other` | 948 | 771 |
| Train / validation / test split | frozen | the same frozen split, same 900 test comments |

The model was retrained on the v1.1 labels (seed 42, same settings as Week 3; validation tuple-F1 0.714, against 0.712 on v1.0). Decision settings were re-tuned on validation and frozen again, then the test split was opened as described in Section 3.

### 9.3  Results after the fix (test split, dataset v1.1)

| System | Tuple-F1 | 95% CI | P | R | Polarity acc. on detected | Aspect F1 |
|---|---|---|---|---|---|---|
| Majority class | 0.216 | [0.196, 0.238] | 0.310 | 0.166 | 0.567 | 0.381 |
| TF-IDF + LogReg (baseline) | 0.452 | [0.427, 0.476] | 0.432 | 0.472 | 0.612 | 0.738 |
| MiniLM embeddings, tuned | 0.500 | [0.476, 0.523] | 0.527 | 0.475 | 0.699 | 0.715 |
| DeBERTa ABSA (argmax) | 0.704 | [0.683, 0.724] | 0.707 | 0.702 | 0.828 | 0.850 |
| **DeBERTa ABSA v1.1 (the asset)** | **0.704** | [0.683, 0.723] | 0.699 | 0.708 | 0.822 | 0.856 |

The improved model beats the baseline by **+0.252** [+0.225, +0.279], essentially the same margin as on v1.0 (+0.249). Quality did not drop: 0.704 against 0.703 before the fix. Per category, `teaching` goes from 0.656 (baseline) to 0.802 and `other` from 0.212 to 0.540. Threshold tuning made no difference this time (−0.001 [−0.006, +0.004]), which confirms Section 6's finding that it is marginal.

Both handout example sentences are now fully correct:

```
"The lectures were very informative, but the pace was too fast and the assignments
 were difficult."                                         (model 0.2.1, dataset v1.1)
  teaching    positive   aspect 0.979  polarity 0.608   "The lectures were very informative"
  pace        negative   aspect 0.994  polarity 0.995   "the pace was too fast"
  assessment  negative   aspect 0.999  polarity 0.992   "the assignments were difficult"

"The lectures were good but the pace was fast."           (handout Section 21)
  teaching    positive   aspect 0.994  polarity 0.684   "The lectures were good"
  pace        negative   aspect 0.982  polarity 0.971   "the pace was fast"
```

One honest caveat: the model is sure *that* lectures are discussed (0.98–0.99) but less sure they are *positive* (0.61–0.68), because both sentences are negative overall. The judgements are right, but they sit closer to the decision line than the negative ones. The final demo's required failure case will therefore use errors that still occur, such as sarcasm and very short form-style comments.

## 10.  Course Report on Real Comments

`analyze_batch` (final model, dataset v1.1) was run on the first 500 test comments, treated as one “course”. With 500 comments the ranking floor is 25 mentions.

| | Result |
|---|---|
| Document sentiment | positive 256, neutral 114, negative 130 |
| Top positive aspects | teaching (305 mentions, net +0.32), course_content (209, +0.42), faculty_interaction (158, +0.42) |
| Top concerns | workload (37 mentions, net −0.73) |
| Quote, workload | “The assignments in this course are a huge time sink.” |
| Below the floor | pace: 9 mentions, all negative (net −1.00) |
| Quote, pace | “Goes off on non-accounting tangents and hurries lecture.” |

The floor did its job of keeping noise out of the top lists, and it also shows its cost. Pace is mentioned rarely but always negatively here, and the floor hides it from the “top concerns” list. It is still fully listed in the aspect summary with its count and quotes. A small, severe concern is exactly what a coordinator should see, so the Week 6 API will add an *emerging concerns* list: aspects below the floor that are strongly negative, with the caveat that the count is small.

## 11.  Limitations

- **One dataset.** Training and test comments come from the same public review sites, so this comparison measures in-distribution performance. Performance on NMIMS-style form feedback is untested; Week 3 showed that short, form-like sentences are a weakness.
- **Label mapping not independently checked.** EduRABSA was labelled by a single expert annotator, and our 24 → 8 category mapping is a set of written rules. No inter-annotator agreement round was run by the team (the handout does not require one), so there is no agreement figure for the mapped labels. Section 9 shows that a mapping rule can matter.
- **Pace** is the thinnest category (21 test comments) and was weak on dataset v1.0 (Section 7).
- **Threshold tuning** is marginal (+0.008 on v1.0, −0.001 on v1.1) and its values are unstable; they are re-derived after any retraining.
- **Spans** are evaluated against EduRABSA's aspect and opinion terms, which mark *what* was evaluated rather than the full justification. Close match is measured at Jaccard ≥ 0.5 over word sets, a lenient but standard criterion.
- **Scores remain uncalibrated**; calibration is the Week 5 deliverable.

## 12.  Next Steps

| Week | Objective | Deliverable |
|---|---|---|
| 5 | Error analysis of v1.1 on test (5–10 representative errors, second coder) | Evaluation report |
| 5 | Robustness probes: negation, sarcasm, typos, short form-style comments | Probe set and results |
| 5 | Score calibration (temperature or isotonic, fitted on validation) | `calibrated_probability` scores |
| 6 | FastAPI service around `pipeline.py`, model-card endpoint, emerging-concerns list | Working reusable asset |

## References

- He, Gao and Chen (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. arXiv:2111.09543.
- Hua, Denny, Wicker and Taskova (2025). EduRABSA: An Education Review Dataset for Aspect-based Sentiment Analysis Tasks. arXiv:2508.17008.
- Pontiki et al. (2014). SemEval-2014 Task 4: Aspect Based Sentiment Analysis. Proceedings of SemEval-2014.
- Reimers and Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP 2019.
- Wang, Wei, Dong, Bao, Yang and Zhou (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. NeurIPS 2020.

## Appendix A.  Reproducing every number

The repository is set to dataset v1.1, so running it reproduces Sections 9 and 10. A CUDA GPU with 4 GB is enough; training takes about 6 minutes and each evaluation script about 2.

```
python src/build_dataset.py                          # dataset v1.1 (includes the lectures rule)
python src/run_baseline.py data/raw/annotations.csv  # baseline; reuses the frozen split
python src/train_transformer.py                      # DeBERTa, seed 42
python src/tune_week4.py                             # validation only: choose and freeze settings
python src/evaluate_week4.py                         # the test run; refuses to run if settings changed
pytest tests/ -q                                     # 23 tests
```

Sections 4–8 (dataset v1.0) were produced by the same scripts before the lectures rule was added. Their result files are kept unchanged in `results/`.

| Report section | Source file |
|---|---|
| 3 | `results/test_openings.log`, `results/week4_frozen_settings.json`, `results/week4_d1.1_frozen_settings.json` |
| 4, 5.3 (validation, v1.0) | `results/week4_tuning.json`, `results/week4_tuning_log.txt` |
| 5–8 (test, v1.0) | `results/week4_test_eval.json`, `results/week4_test_log.txt` |
| 9 (v1.1) | `results/week4_d1.1_tuning.json`, `results/week4_d1.1_test_eval.json`, `results/week4_d1.1_test_log.txt`, `results/run_baseline_log.txt` |
| 9.1, 9.3 examples; 10 | `results/week4_demo_outputs.json` (v1.0), `results/week4_d1.1_demo_outputs.json` (v1.1) |
