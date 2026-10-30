---
title: Student Feedback Intelligence
subtitle: Aspect-Based Sentiment Analysis of Academic Feedback
week: Week 3: Core NLP Model — Transformer ABSA and Pipeline v1
course: SNLP Course Project 2026   •   Group 5
members: E029  ·  E030  ·  E031  ·  E033  ·  E034  ·  E035
version: Document version 1.0  •  Model transformer-absa-0.1.0  •  Dataset v1.0-edurabsa
note: All results are on the validation split. The test split stays closed until the Week 4 comparison.
footer: Group 5  •  Student Feedback Intelligence  •  Week 3  •  
---

# Student Feedback Intelligence — Week 3 Report

*SNLP Course Project 2026 · Group 5 · E029 · E030 · E031 · E033 · E034 · E035*

Week 3 deliverable: Version 1 of the NLP pipeline. Companion to the Week 1–2 report (v3.0).

## 1.  Scope and Status

The handout's Week 3 target (19–25 September) is to *implement the primary NLP component*, with *Version 1 of the NLP pipeline* as the deliverable. For this project the primary component is the aspect-based sentiment model, and pipeline v1 is that model wrapped with preprocessing and post-processing so that raw comments go in and API-shaped JSON comes out.

As in Weeks 1–2, no number in this report was estimated; each traces to a file in `results/` (Appendix A). Every model figure is on the **validation split** (900 comments). The evaluation protocol opens the test split once per comparison, and the baseline-vs-improved comparison is the Week 4 deliverable, so the test split has not been touched this week.

### Status at end of Week 3

| Component | Status |
|---|---|
| Transformer ABSA model | Complete — DeBERTa-v3-base with aspect-specific attention, trained (Sections 3–5) |
| Pipeline v1 | Complete — raw text → redaction → model → typed scores, evidence spans, warnings → JSON (Section 6) |
| Course-level batch report | Complete — aspect summary, top positives, top concerns, verbatim quotes, caveats |
| Validation results | Tuple-F1 **0.712** [0.692, 0.733] vs 0.476 for the Week 1–2 baseline (Section 7) |
| Training stability | 3 seeds: 0.712 ± 0.002 |
| Design ablation | Evidence-span supervision: +0.022 tuple-F1 (Section 8) |
| Tests | 15 passing (6 from Weeks 1–2, 9 new) |
| Known failure | Handout example: “lectures … informative” read as other:negative (Section 10.2) |

## 2.  Why This Model

The Week 1–2 error analysis settled what the core model had to fix. The single largest error type (11 of 40 sampled errors) was structural: in a mixed review the baseline's polarity classifier reads the whole comment, so one overall tone — often `conflict` — was copied onto every aspect it found. Better word features cannot fix that. The model has to read *the part of the comment that is about each aspect*.

That gives three design requirements:

1. **Per-aspect reading.** Each aspect decides its polarity from its own region of the text, not from the whole comment.
2. **Joint decision.** Whether an aspect is present and what its polarity is should be one decision, trained directly on the (aspect, polarity) tuple the project is scored on.
3. **Evidence.** The API contract promises a supporting span for every aspect. The model should produce it, not a post-hoc keyword search.

## 3.  Model Architecture

```
comment ─► Branch B preprocessing (PII redaction, whitespace only)
        ─► DeBERTa-v3-base encoder ─► token states H  (T tokens × 768)
               │
               ├─ [CLS] ─────────────────────────────► document sentiment (3-way)
               │
               └─ 8 learned aspect queries q_a attend over H:
                     α_a = softmax(q_a · W_k H)          where does aspect a live?
                     z_a = Σ_t α_a,t H_t                what is said there
                     [z_a ; CLS] ─► aspect a's own head ─► {none, pos, neg, neu, conflict}
```

| Component | Choice | Reason |
|---|---|---|
| Encoder | microsoft/deberta-v3-base (184M parameters) | Strong on classification benchmarks; its relative-position attention suits long, loosely punctuated reviews |
| Aspect attention | One learned query per aspect over the token states | Lets `teaching` read “lectures were informative” while `pace` reads “too fast” — the Week 2 failure |
| Output per aspect | 5-way: none, positive, negative, neutral, conflict | Detection and polarity in one decision; the tuple is the training target |
| Per-aspect heads | Separate weights per aspect | Whether an aspect is present depends on which aspect it is |
| Document head | 3-way on [CLS] | Task A comes from the same forward pass |
| Evidence span | Attention peak, grown to neighbouring tokens above 50% of the peak | The span shows *which words* drove the decision |

This formulation is aspect-category sentiment analysis (ACSA), the task SemEval-2014 defines for category-level labels. The aspect-query attention follows the line of attention-based ABSA models (Wang et al., 2016), applied here on top of a pretrained transformer.

### 3.1  Supervising the evidence

The attention weights could be left free, but EduRABSA marks the words behind each label. Where such a span exists, an auxiliary loss rewards the aspect's attention for landing on it:

```
loss = CE(aspect slots) + CE(document sentiment) + λ · mean( −log Σ_{t ∈ gold span} α_a,t )
λ = 0.5 (λ = 0 is the ablation in Section 8)
```

Gold spans were located for 7,875 of 7,891 training (aspect, polarity) pairs; the other 16 were lost to redaction or truncation and simply carry no span loss.

## 4.  Training Setup

| Setting | Value |
|---|---|
| Data | 4,198 training comments; model selection on the 900 validation comments; test split not loaded |
| Input | Branch B text, max 256 tokens (27 of 5,998 comments are longer and are truncated) |
| Optimiser | AdamW, lr 3e-5 (encoder) and 5e-4 (heads), weight decay 0.01, 6% warm-up then linear decay |
| Epochs | 6; the best epoch by validation tuple-F1 is kept |
| Batching | Length-bucketed, 16 comments per batch, capped at 1,024 tokens per batch |
| Precision | fp32 weights, bf16 autocast for the forward pass |
| Hardware | RTX 3050 Laptop, 4 GB; 5.4–5.6 minutes per run, peak 3.1 GB GPU memory |
| Seeds | 42 (reported model), 43 and 44 (stability check) |

Two engineering problems had to be solved to train on a 4 GB laptop GPU, and both are recorded because they change results if repeated carelessly.

- **Loss became NaN in the first epoch.** Transformers 5 loads DeBERTa-v3 in the checkpoint's own dtype, which is fp16, and AdamW updates on fp16 weights diverge. The weights are now forced to fp32, with mixed precision applied only during the forward pass (`transformer_absa.py`). The training loop also stops immediately on a non-finite loss instead of wasting an epoch.
- **fp32 weights plus optimiser state do not fit in 4 GB.** The 128k-token embedding table is 98M of the 184M parameters. It is frozen; all 12 transformer layers and the heads (86M parameters) are trained. A per-batch token cap keeps the few 256-token comments in smaller batches. Measured peak memory on the longest batches fell from 4.0 GB (the card's entire capacity) to 3.1 GB.

## 5.  Training Behaviour

Seed 42, validation split after each epoch:

| Epoch | Train loss (slot / doc / span) | Val tuple-F1 | Val aspect-F1 | Polarity acc. on detected | Doc macro-F1 |
|---|---|---|---|---|---|
| 1 | 0.685 / 0.576 / 1.429 | 0.531 | 0.724 | 0.733 | 0.730 |
| 2 | 0.447 / 0.280 / 0.695 | 0.639 | 0.817 | 0.783 | 0.820 |
| 3 | 0.350 / 0.140 / 0.476 | 0.681 | 0.842 | 0.809 | 0.841 |
| 4 | 0.285 / 0.059 / 0.347 | 0.703 | 0.857 | 0.821 | 0.829 |
| 5 | 0.234 / 0.023 / 0.251 | **0.712** | 0.850 | 0.838 | 0.831 |
| 6 | 0.196 / 0.009 / 0.192 | 0.710 | 0.854 | 0.831 | 0.837 |

Validation tuple-F1 plateaus from epoch 4 while training loss keeps falling, so six epochs is enough; more would mostly fit the training set. The document-sentiment loss reaching 0.009 while its validation score stays flat is the clearest sign of that.

| Run | Val tuple-F1 | P | R | Best epoch |
|---|---|---|---|---|
| Seed 42 (reported model) | 0.712 | 0.718 | 0.707 | 5 |
| Seed 43 | 0.709 | 0.711 | 0.708 | 6 |
| Seed 44 | 0.714 | 0.724 | 0.703 | 6 |
| Mean ± sd | **0.712 ± 0.002** | | | |

The spread across seeds is small, so the result is a property of the method, not of a lucky initialisation.

## 6.  Pipeline v1

```
raw comment
   ↓  validate            empty → EMPTY_TEXT, > 5,000 chars → TEXT_TOO_LONG
   ↓  redact PII          emails, phones, roll numbers, "Prof./Dr. Name" → placeholders
   ↓  tokenise            DeBERTa-v3, 256 tokens, offsets kept
   ↓  transformer ABSA    8 aspect slots + document sentiment + attention per aspect
   ↓  post-process        typed scores, evidence span mapped back to the original text,
                          warnings (pii_redacted, text_truncated, low_confidence)
   ↓
JSON  (docs/API_Specification.md §2) — or, for a batch, the course report (§3)
```

`src/pipeline.py` implements both response shapes of the API specification as a Python module. Another group can call it today in-process; the FastAPI wrapper in Week 6 will expose the same two functions over HTTP without changing the contract.

### 6.1  Scores

Both scores are `uncalibrated_probability`, as the contract requires until calibration in Week 5:

- `aspect_score` = 1 − P(none): how sure the model is that the aspect is discussed.
- `polarity_score` = P(polarity | aspect present): how sure it is of the sentiment, given the aspect.

Splitting the two tells a consumer *which* judgement is weak. A `low_confidence` warning is raised when either falls below 0.5.

### 6.2  Actual output for the handout's example

```
Input: "The lectures were very informative, but the pace was too fast and the
        assignments were difficult."
doc_sentiment: negative (0.991)
aspects:
  pace        negative   aspect 0.976  polarity 0.988   span "fast"         [57:61]
  assessment  negative   aspect 0.998  polarity 0.993   span "assignments"  [70:81]
  other       negative   aspect 0.915  polarity 0.685   span "lectures"     [4:12]
warnings: []
```

Two of the three expected judgements are right, with high confidence. The third is wrong: “the lectures were very informative” should be `teaching:positive`. Section 10.2 explains why, because it is the most instructive failure of the week.

### 6.3  Course report

`analyze_batch` aggregates a set of comments into the report shape the handout asks for. On a sample of 120 validation comments:

| | Result |
|---|---|
| Top positive aspects | teaching (net +0.43), faculty_interaction (+0.40), course_content (+0.57) |
| Top concerns | assessment (net −0.14), resources (−0.17), pace (−1.00, only 2 mentions) |
| Example quote, assessment | “His tests were very difficult.” |
| Example quote, resources | “Labs were a mess, manual was awful.” |
| Example quote, pace | “You get less than a minute per question and it is very rushed.” |
| Caveats attached | uncalibrated scores; advisory only; topic clustering not in v1 |

Quotes are the full sentence around the evidence span, verbatim from the comment. Nothing in the report is generated text, so it cannot state anything a student did not write. Ranking concerns by count alone has one visible weakness: pace appears with only 2 mentions. Section 14 plans a minimum-mentions threshold.

## 7.  Results on the Validation Split

The Week 1–2 baseline was refitted on the same training split and scored on the same 900 validation comments. The transformer figures are computed from pipeline v1's JSON output, so they also verify the end-to-end pipeline, not just the model.

### 7.1  Task C — ABSA tuple extraction (headline)

| Model | Micro tuple-F1 | 95% CI | P | R | Polarity acc. on detected |
|---|---|---|---|---|---|
| Majority class | 0.225 | [0.206, 0.245] | 0.324 | 0.173 | 0.611 |
| TF-IDF LogReg baseline (Week 1–2) | 0.476 | [0.453, 0.501] | 0.459 | 0.494 | 0.638 |
| Transformer ABSA v1 (Week 3) | **0.712** | [0.692, 0.733] | 0.718 | 0.707 | **0.838** |

Paired bootstrap over the same 900 comments (1,000 resamples): the transformer is **+0.236** tuple-F1 better, 95% CI [+0.212, +0.261]. The interval is far from zero; no resample favoured the baseline.

The gain comes mainly from sentiment, which was the baseline's weak half. Polarity accuracy on correctly detected aspects rises from 0.638 to 0.838 — exactly the per-aspect reading the design targeted. Aspect detection also improves (micro-F1 0.745 → 0.850), and the model finds more aspects correctly (1,425 correct detections against 1,308).

### 7.2  Per-category tuple-F1

| Aspect | Val support | Baseline | Transformer v1 | Δ |
|---|---|---|---|---|
| teaching | 478 | 0.682 | 0.841 | +0.159 |
| faculty_interaction | 237 | 0.687 | 0.842 | +0.155 |
| course_content | 408 | 0.362 | 0.649 | +0.287 |
| pace | 21 | 0.571 | 0.514 | −0.057 |
| assessment | 227 | 0.318 | 0.620 | +0.302 |
| workload | 78 | 0.352 | 0.756 | +0.404 |
| resources | 99 | 0.278 | 0.478 | +0.200 |
| other | 142 | 0.295 | 0.539 | +0.244 |
| Macro average | | 0.443 | 0.655 | +0.212 |

Seven of eight categories improve, and the largest gains are on the categories most often expressed implicitly or in mixed reviews (workload, assessment, course_content). **Pace is the exception and is reported as such**: 0.514 against 0.571. With 21 validation instances the difference is one or two comments and is not reliable in either direction. But the transformer's pace recall is 0.429, no better than the baseline's, so pace remains the weakest category.

### 7.3  Task A — Document sentiment

| Model | Accuracy | Macro-F1 | F1 positive | F1 neutral | F1 negative |
|---|---|---|---|---|---|
| Majority class | 0.490 | 0.219 | 0.658 | 0.000 | 0.000 |
| TF-IDF + LogReg | 0.737 | 0.710 | 0.831 | 0.592 | 0.707 |
| Transformer v1 (document head) | **0.850** | **0.831** | 0.915 | 0.719 | 0.857 |

The same forward pass that produces the aspects also gives document sentiment, 0.121 macro-F1 above the baseline. Neutral is still the hardest class.

## 8.  Ablation: Does Evidence Supervision Help?

The span loss (Section 3.1) was a design choice, so it was tested rather than assumed. Same seed, same data, λ = 0.5 against λ = 0:

| Measure | λ = 0 (no span loss) | λ = 0.5 (v1) |
|---|---|---|
| Val tuple-F1 | 0.690 | **0.712** |
| Aspect detection micro-F1 | 0.825 | **0.850** |
| Polarity acc. on detected | 0.837 | 0.838 |
| Evidence span overlaps the gold span | 66.1% | **76.6%** |
| Evidence span Jaccard ≥ 0.5 with gold | **20.7%** | 13.8% |
| Mean token Jaccard with gold | **0.248** | 0.193 |

Supervision adds 0.022 tuple-F1, about ten times the seed-to-seed spread of 0.002, so the gain is real even from a single ablation run. It helps *detection*, not polarity: telling the model where an aspect lives helps it decide whether the aspect is there at all.

For the spans themselves the picture is mixed, and both halves are reported. With supervision the evidence lands on the right words more often (77% vs 66% overlap), but it is **narrower**: the attention becomes peakier, and the 50%-of-peak rule then returns one or two words (“fast”, “assignments”) where the gold span is a phrase (“pace was too fast”). So spans point in the right place but are too short to match gold boundaries. The fix is in the decoder, not the model: tune the growth rule on validation (Week 4).

## 9.  Did It Fix the Week 2 Errors?

The 40 baseline errors coded by hand in Week 2 were re-scored with pipeline v1. “Fixed” means v1's output now matches the gold exactly; “improved” means its tuple-F1 on that comment went up.

| Week 2 error type | Sampled | Fixed exactly | Improved | Worse |
|---|---|---|---|---|
| Conflict / mixed tone copied | 11 | 1 | 10 | 1 |
| Implicit aspect | 7 | 3 | 4 | 2 |
| Boundary confusion | 7 | 3 | 2 | 1 |
| Polarity on correct aspect | 6 | 2 | 5 | 0 |
| Multi-aspect | 5 | 2 | 5 | 0 |
| Annotation error | 2 | 0 | 0 | 0 |
| Rare category | 1 | 1 | 1 | 0 |
| Negation | 1 | 0 | 0 | 0 |
| **Total** | **40** | **12** | **27** | **4** |

The targeted failure responds as intended: 10 of the 11 tone-copying cases improve. Only one becomes exactly right, because these are long reviews carrying two to six aspects each, where one remaining slip still makes the comment imperfect. The two annotation-error cases cannot improve by construction, since the model would have to learn a label the guidelines disagree with.

## 10.  Error Analysis of Pipeline v1

Pipeline v1 still errs on 472 of the 900 validation comments (at least one wrong or missing tuple). All are exported to `results/week3_errors_for_analysis.json`; a random sample of 20 (seed 42) was coded by hand (`results/errors_coded_week3.json`).

### 10.1  What kind of errors remain

| Error type | Count (of 20) | Example from the sample |
|---|---|---|
| Boundary confusion | 8 | “interesting weekly lectures” → teaching; gold files it under other (Learning activity) |
| Multi-aspect | 5 | “…However … too less time for too many questions for tests” — assessment missed (teaching now right) |
| Annotation error | 3 | “No power points, he says he will send them” → resources:negative; gold omits it |
| Conflict / mixed | 2 | teaching conflict predicted positive; “needs to be MORE ORGANIZED” missed |
| Implicit aspect | 2 | “Finished it in like 3 days” = workload:positive, never named |
| Tone copied across aspects | 0 | — (was 11 of 40 for the baseline) |

The error profile has changed character. The baseline mostly got sentiment wrong; v1 mostly gets *aspect boundaries* wrong. It adds a plausible extra aspect, or files a phrase under a neighbouring category. Several of these “errors” are defensible readings under our own guidelines (slides are resources; a suggestion implies a negative evaluation), which is why three are coded as annotation errors. Precision (0.718) and recall (0.707) are balanced, so the model is neither over- nor under-predicting overall.

### 10.2  The failure case: the handout's own example

v1 reads “The lectures were very informative” as `other:negative` instead of `teaching:positive` (Section 6.2). Two causes combine, and both are diagnosable:

1. **Category.** In EduRABSA, “lectures” is usually annotated as *Course – Learning activity*, which our mapping sends to `other` (Week 1–2 report, Section 5.2). The model learned that mapping faithfully: in its training data, lectures are mostly `other`. This is a data-definition problem, not a model problem.
2. **Polarity.** With `other` a weakly defined bucket, its polarity (0.685, the least confident judgement in the output) is pulled towards the sentence's dominant negative tone. That is a milder form of the Week 2 tone-copying error.

This case will be used as the “show one failure” part of the final demo. It is **not** being patched by adjusting the mapping until this sentence passes — that would tune the dataset to one example. The principled fix is the one already pending: the team's mapping check (Week 1–2 report, Section 8), plus a decision on promoting *Learning activity* to its own category or routing lecture-related activity to `teaching`. Either change is then applied to the whole corpus and every model is retrained.

## 11.  Demonstration on the Guideline Examples

The ten worked examples from `docs/Annotation_Guidelines.md` §7 were run through pipeline v1. They were written by the team and are not a test set; this table is a qualitative check, not a metric.

| # | Text | Guideline label | Pipeline v1 | Match |
|---|---|---|---|---|
| 1 | The lectures were informative but the pace was too fast. | teaching:pos, pace:neg | pace:neg, other:neg | Partial |
| 2 | Assignments were difficult but genuinely useful. | assessment:conflict | assessment:conflict (low confidence) | Yes |
| 3 | We barely get enough time to finish everything. | workload:neg | no aspect (doc negative) | No |
| 4 | There were four assignments this semester. | assessment:neutral | no aspect | No |
| 5 | Another three-hour lecture, exactly what we needed. | pace:neg (sarcasm) | other:pos | No |
| 6 | Good course. | no aspect | no aspect | Yes |
| 7 | Sir is always available for doubts but his slides are unreadable. | faculty_interaction:pos, resources:neg | same | Yes |
| 8 | The content is relevant, though it is covered too quickly. | course_content:pos, pace:neg | same | Yes |
| 9 | Please upload the recordings earlier. | resources:neg | no aspect | No |
| 10 | Not very informative. | teaching:neg | no aspect (doc negative) | No |

Four of ten are exactly right, including both contrastive two-aspect sentences (7 and 8), which were the core motivation for the project. The misses share a pattern: very short, context-free sentences with an implicit aspect (3, 9, 10), a purely factual statement (4) and sarcasm (5). Real reviews in EduRABSA average 41 words, and v1 tends to predict no aspect at all for short fragments. That is a known domain gap, and exactly the kind of text an institutional feedback form produces. It is the main argument for adding a small set of labelled NMIMS-style feedback later.

## 12.  Reusability

Another group can use pipeline v1 now, in-process:

```
from pipeline import FeedbackAnalyzer
fa = FeedbackAnalyzer()                                 # model directory from $SNLP_MODEL
fa.analyze("The pace was too fast.")                   # one comment  -> API §2 JSON
fa.analyze_batch([{"feedback_id": "F1", "text": "..."}], course_id="DS-XI")   # -> API §3
python src/pipeline.py "any comment"                   # command line
```

| Property | Value |
|---|---|
| Latency, single call | 23 ms on the RTX 3050 GPU; about 67 ms on the laptop CPU |
| Throughput, batched | 2.7 ms per comment on GPU |
| Model files | 738 MB (`model.pt`) + tokenizer and config; kept outside the repository |
| Errors | `FeedbackError` carrying the API error codes (`EMPTY_TEXT`, `TEXT_TOO_LONG`, `BATCH_TOO_LARGE`, `MODEL_UNAVAILABLE`) |

The weights are too large for Git. For the demo and for teammates they will be shared as a single archive (Drive or a GitHub release); the loading code only needs the folder path.

## 13.  Responsible AI — What Changed in v1

| Concern | v1 behaviour |
|---|---|
| Privacy | Redaction runs before the model sees any text; output flags `pii_redacted`. Bare instructor names (without “Prof.”/“Dr.”) are still not caught — the open item from Week 1–2 |
| Hallucination | Nothing is generated. Spans and report quotes are verbatim substrings of the input, and their offsets are checked against the original text in the tests |
| Overconfidence | Two separate uncalibrated scores per aspect plus `low_confidence`; calibration is Week 5. The handout example shows why this matters: the one wrong judgement is also the least confident |
| Small samples | Every batch report carries caveats; under 20 comments an instability caveat is added automatically |
| Instructor harm | No per-instructor ranking; *Personal traits* excluded from the taxonomy; advisory-only caveat on every report |
| Domain shift | Trained on North American and UK review sites; weaker on short, form-style comments (Section 11) |

## 14.  Limitations and Next Steps

Known limitations of v1, in order of impact:

- **Category definitions leak into predictions** — lectures as `other` (Section 10.2), readings as `other` or `resources`. The fix is a team decision on the mapping, followed by a full retrain.
- **Short, form-style comments** often get no aspect (Section 11).
- **Pace** remains the weakest category (21 validation examples, recall 0.43).
- **Evidence spans** are in the right place but too narrow (Section 8).
- **Sarcasm** is read literally.
- **Report ranking** by raw counts can surface an aspect with only two mentions as a top concern.

| Week | Objective | Deliverable |
|---|---|---|
| 4 | Open the test split once: baseline vs transformer v1, with CIs and a stated verdict | Comparison table, Week 4 report |
| 4 | Tune on validation only: per-aspect presence thresholds, span-growth rule, minimum mentions for report ranking | v1.1 settings, frozen before the test run |
| 4 (team) | Mapping check on 40 comments; decide on *Learning activity* and lectures | Guidelines v2.1 if changed; dataset v1.1 and retrain |
| 5 | Full error analysis (second coder), robustness probes (negation, sarcasm, short comments), score calibration | Evaluation report with 5–10 error cases |
| 6 | FastAPI service around `pipeline.py`; model card endpoint | Working reusable asset |

## References

- He, Gao and Chen (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. arXiv:2111.09543.
- Hua, Denny, Wicker and Taskova (2025). EduRABSA: An Education Review Dataset for Aspect-based Sentiment Analysis Tasks. arXiv:2508.17008.
- Pontiki et al. (2014). SemEval-2014 Task 4: Aspect Based Sentiment Analysis. Proceedings of SemEval-2014.
- Sun, Huang and Qiu (2019). Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence. NAACL-HLT 2019.
- Wang, Huang, Zhu and Zhao (2016). Attention-based LSTM for Aspect-level Sentiment Classification. EMNLP 2016.

## Appendix A.  Reproducing every number

Run from the repository root after the Week 1–2 steps (the dataset and frozen split must exist). A CUDA GPU with 4 GB is enough; each training run takes about 6 minutes.

```
python src/train_transformer.py                          # seed 42 -> reported model
python src/train_transformer.py --seed 43                # stability runs
python src/train_transformer.py --seed 44
python src/train_transformer.py --span-weight 0          # Section 8 ablation
python src/evaluate_week3.py                             # Sections 6, 7, 9, 10
python src/evaluate_week3.py --model <..._sw0_s42> --out week3_eval_ablation_no_span_supervision.json --light
pytest tests/ -q                                         # 15 tests
```

| Report section | Source file |
|---|---|
| 5 | `results/week3_train_deberta-v3-base_sw0.5_s42.json` (and `_s43`, `_s44`) |
| 6.2, 6.3, 11 | `results/week3_demo_outputs.json` |
| 7, 9 | `results/week3_eval.json`, `results/week3_eval_log.txt` |
| 8 | `results/week3_train_deberta-v3-base_sw0_s42.json`, `results/week3_eval_ablation_no_span_supervision.json` |
| 10 | `results/week3_errors_for_analysis.json`, `results/errors_coded_week3.json` |
