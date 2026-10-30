# Evaluation Protocol — Student Feedback Intelligence

**SNLP Course Project 2026 — Group 5** · Version 1.0

This document fixes the evaluation methodology **before** any results are produced, so that
the Week 4 baseline-vs-improved comparison is a fair test rather than a post-hoc search for
a flattering number.

Anything not specified here must be decided and written down *before* looking at test-set
results.

---

## 1. Why this document exists

The original plan reported a table of `Accuracy / Precision / Recall / F1` with no
specification of:

- what a "correct" ABSA prediction actually is,
- what the metric is being compared *against*,
- whether the number is stable given the corpus size.

Each of those omissions makes the reported number uninterpretable. A stated accuracy of
0.78 means nothing if the majority class is 0.74, if the metric silently averages over
categories with 3 test examples, or if a different random seed yields 0.66.

---

## 2. The three tasks and their metrics

### Task A — Document-level sentiment classification

Single-label, 3 classes (`positive` / `neutral` / `negative`). Supporting task only.

- **Primary:** macro-F1
- **Secondary:** accuracy, per-class P/R/F1 with support counts, confusion matrix
- Macro-F1 is primary because the corpus is imbalanced; accuracy alone rewards a model
  that ignores the minority class.

### Task B — Aspect category detection

Multi-label over 8 categories. For each comment the model predicts a **set** of categories.

- **Primary:** micro-F1 over all (comment, category) positive decisions
- **Secondary:** macro-F1, per-category P/R/F1 **with support**
- Micro-F1 is primary because it weights by actual frequency; macro-F1 is reported
  alongside to expose neglected rare categories.
- **Any category with fewer than 20 test instances is reported but explicitly marked as
  low-support and excluded from the headline macro average.** Reporting an F1 of 0.00 or
  1.00 over 3 examples as if it were a finding is misleading.

### Task C — ABSA tuple extraction (the core task)

The model predicts a set of `(aspect_category, polarity)` tuples per comment. This is the
metric the project is actually judged on.

**Exact-match tuple F1.** A predicted tuple is a true positive only if **both** the category
and the polarity match a gold tuple for the same comment.

```
Gold:      {(teaching, positive), (pace, negative)}
Predicted: {(teaching, positive), (pace, neutral), (workload, negative)}

TP = 1   (teaching, positive)
FP = 2   (pace, neutral) — wrong polarity;  (workload, negative) — spurious
FN = 1   (pace, negative)

Precision = 1/3 = 0.333    Recall = 1/2 = 0.500    F1 = 0.400
```

- **Primary:** micro tuple-F1
- **Secondary:** macro tuple-F1 over categories; **polarity accuracy on correctly detected
  aspects**, which separates "found the wrong aspect" from "found it but misjudged the
  sentiment" — two very different failure modes that a single F1 conflates.
- Spans are **not** scored in Weeks 1–2 (the lexicon baseline has no meaningful span
  output). Span-level F1 with Jaccard ≥ 0.5 matching is introduced in Week 4 when the
  transformer model produces real spans.

### Task D — Topic clustering (supporting, diagnostic only)

- Silhouette score across k ∈ {3 … 10}, plus cluster sizes and top-10 TF-IDF terms per
  cluster.
- **Silhouette is reported as a diagnostic, not as evidence of quality.** It measures
  geometric separation in TF-IDF space, which is only loosely related to whether the
  clusters are thematically meaningful. The qualitative top-terms table carries more weight
  and should be read as the real result.

---

## 3. Mandatory reference baselines

Every model result is reported **in the same table** as these three references. A model that
does not beat them is a negative result, and reporting it as such is worth more marks than
hiding it.

| Reference | Tasks | Definition |
|---|---|---|
| **Majority class** | A, B, C | Always predict the most frequent training label / the most frequent category set |
| **Stratified random** | A, B, C | Sample predictions from the training label distribution |
| **Lexicon-only** | C | The aspect lexicon with a simple polarity lexicon, no learned component |

Without the majority-class row, no accuracy figure in the report can be interpreted.

---

## 4. Data splitting

### 4.1 Iterative stratification

The corpus is multi-label, so ordinary `train_test_split(stratify=y)` does not apply.
We use **iterative stratification** (Sechidis, Tsoumakas & Vlahavas, 2011), implemented in
`src/splitting.py`, which greedily allocates examples to preserve each label's distribution
across splits.

- **Split:** 70 / 15 / 15 train / validation / test
- **Seed:** `RANDOM_SEED = 42`, fixed in `src/config.py` and used everywhere
- The split is computed **once**, written to `data/processed/splits.json`, committed, and
  never recomputed. A split that moves between runs makes results incomparable.

### 4.2 Leakage rules

1. **Fit every transformation on training data only.** `TfidfVectorizer.fit` sees
   `X_train` and nothing else. All preprocessing is wrapped in an `sklearn.Pipeline` so this
   cannot be violated by accident.
2. **Near-duplicate check before splitting.** Comments with cosine similarity > 0.95 on
   character 3-gram TF-IDF are grouped, and every group is assigned to a single split.
   Otherwise a near-duplicate of a training comment can land in test and inflate scores.
3. **The test set is opened once.** All development, tuning and model selection use the
   validation split. Test results are produced at the end of Week 2 and again at the end of
   Week 4, and are not used to make any decision.
4. **The aspect lexicon is built from the training split only** (§5).

### 4.3 Small-corpus caveat

At a corpus of ~300–600 comments, a 15% test split is 45–90 comments. A single-split
point estimate is too noisy to support a claim like "the transformer beats the baseline".
Therefore:

- **Development results:** 5-fold cross-validation on train+validation, reported as
  **mean ± standard deviation**.
- **Final results:** held-out test set with a **95% bootstrap confidence interval** (1000
  resamples), reported as `F1 = 0.71 [0.63, 0.78]`.
- A baseline-vs-improved difference whose confidence intervals overlap substantially is
  reported as inconclusive, not as an improvement.

---

## 5. Avoiding circular evaluation of the aspect lexicon

If the same people write the aspect lexicon and the gold labels from the same intuitions,
lexicon precision/recall measures the team's internal consistency rather than the method's
quality. Controls:

1. The lexicon is built **only** from the training split, by members **not** annotating the
   validation or test slices.
2. No lexicon term is added by inspecting validation or test comments.
3. Terms are derived from training-split TF-IDF and the training gold spans, and the
   derivation is logged in `docs/lexicon_provenance.md` — every term records where it
   came from.
4. The lexicon is frozen before test evaluation and the frozen version is committed.

This is a genuine methodological risk, so it is stated explicitly in the report's limitations
section rather than left implicit.

---

## 6. Error analysis

Collected from Week 2 onward (not deferred to Week 5). Sample **at least 30 errors** from
the validation set, stratified across the taxonomy below, and record: input, gold,
prediction, category, and the hypothesised cause.

| Category | Description |
|---|---|
| Negation | Polarity flipped by a negation the model missed |
| Implicit aspect | Gold aspect not lexically present in the text |
| Multi-aspect | Some aspects found, others missed |
| Polarity-on-correct-aspect | Right aspect, wrong polarity |
| Conflict/mixed | `conflict` predicted as a single polarity or vice versa |
| Sarcasm / irony | Literal polarity opposite to intended |
| Boundary confusion | Confusable pair from Annotation Guidelines §2.1 (e.g. `pace` vs `teaching`) |
| Rare category | Category with low training support |
| Annotation error | On review, the gold label is wrong — count these and report the rate |

The last row matters: an honest count of gold-label errors found during error analysis is
evidence of methodological rigour, and it feeds back into the next annotation round.

### 4.4 Record of test-split openings (status, Week 6)

Every opening of the test split is appended to `results/test_openings.log` by the script
that opens it. Decisions were made only on validation and frozen before each opening:

- **Week 2:** the baseline.
- **Week 4:** the baseline vs improved comparison, plus a repeat after the lectures labelling fix (dataset v1.1).
- **Weeks 5–6:** two openings for measurement only (calibration check, error analysis); no setting was changed afterwards.

---

## 7. Reporting template

Every results table in the report uses this shape. `n` is the test-set size.

| Model | Task | Metric | Value | 95% CI | n |
|---|---|---|---|---|---|
| Majority class | C | micro tuple-F1 | | | |
| Stratified random | C | micro tuple-F1 | | | |
| TF-IDF + LogReg (baseline) | C | micro tuple-F1 | | | |
| Transformer ABSA (Week 4) | C | micro tuple-F1 | | | |

Rules:

- No number is entered without the script and seed that produced it.
- No cell is filled by estimation. `[TO FILL]` stays until the code has run.
- Any result from fixture/smoke-test data is labelled **"SMOKE TEST — NOT A RESULT"**
  in the table caption.
- Report what did not work. A failed improvement with a diagnosed cause scores better
  than a silently dropped experiment.

---

## References

- Pontiki et al. (2014). *SemEval-2014 Task 4: Aspect Based Sentiment Analysis.* — source of the 4-way polarity label set and the exact-match tuple metric.
- Sechidis, Tsoumakas & Vlahavas (2011). *On the Stratification of Multi-Label Data.* — iterative stratification.
- Landis & Koch (1977). *The Measurement of Observer Agreement for Categorical Data.* — κ interpretation thresholds.
