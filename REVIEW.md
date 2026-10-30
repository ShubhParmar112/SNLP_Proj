# What changed and why

A review of the original Week 1–2 package, and the reasoning behind each revision. Read
this first; it explains the rest.

The original package was well above average for a course project. The reframing from
"sentiment analyzer" to "reusable ABSA service", the structured JSON output, the Group 6
integration hook and the refusal to fabricate metrics were all correct calls. The problems
below are the ones that would have cost marks or, worse, produced a number that looked
good and meant nothing.

---

## The six substantive problems

### 1. The dataset plan would have produced an uninterpretable result

**Was:** a team-authored synthetic corpus as the primary dataset.

If the same six people write the comments *and* assign the labels, the model learns the
team's phrasing templates. TF-IDF + logistic regression will score very highly, and the
score measures internal consistency, not capability. This sat in direct contradiction with
the brief's demand for genuine quantitative evaluation.

**Now:** a public course-review corpus as primary, SemEval-2014 Task 4 for cross-domain
benchmarking against published numbers, institutional data if authorised, and
team-authored text demoted to fixtures and rare-aspect supplements only.

Training on SemEval restaurant/laptop ABSA and testing on course reviews is a legitimate
transfer experiment and a far more interesting result than a high score on self-written
text.

### 2. No inter-annotator agreement

**Was:** "record inter-annotator agreement if a sufficiently large double-annotated subset
is available."

Six annotators, a hand-built corpus, and IAA relegated to a conditional clause. This is
close to free marks and its absence is the first thing an examiner checks on a project
with hand-built gold data.

**Now:** a full protocol — a calibration round before production annotation, three
annotation pairs with 20% double-annotation, third-party adjudication with a logged
decision trail, and a hard κ ≥ 0.60 gate. `src/iaa.py` computes per-category Cohen's κ for
aspects, κ for polarity on shared aspects, and span-overlap agreement. Verified working.

### 3. The label set was internally contradictory

**Was:** Rule 4 said *"if sentiment is unclear, label it Neutral"*, while the examples
showed `assignments → mixed/neutral`.

That collapsed three different situations into one label:

| Situation | Was | Now |
|---|---|---|
| Aspect mentioned, no evaluation | `neutral` | `neutral` |
| Both polarities, same aspect | `neutral` or `mixed` | `conflict` |
| Annotator can't decide | `neutral` | best guess + `uncertain = 1` flag |

`neutral` had become a garbage class, which destroys macro-F1 and makes any per-class
number meaningless.

**Now:** the SemEval-2014 Task 4 four-way set (`positive` / `negative` / `neutral` /
`conflict`), with annotator uncertainty as a metadata flag that never enters training as a
class. Using the SemEval set also keeps results comparable with published ABSA work.

### 4. The evaluation would not have supported any claim

Three separate gaps, each individually fatal to interpretability:

- **No reference baseline.** "78% accuracy" is strong if the majority class is 41% and a
  failure if it's 81%. The original results table had no such row.
- **No definition of a correct ABSA prediction.** Reporting aspect F1 and polarity
  accuracy separately never penalises finding the right aspect with the wrong sentiment.
- **A single split with no uncertainty estimate.** On a few hundred comments, a 15% test
  split is a few dozen items — far too noisy for a point estimate to support a
  baseline-vs-improved claim.

**Now:** `docs/Evaluation_Protocol.md` fixes all of it in advance — exact-match micro
tuple-F1 as the headline, mandatory majority-class and stratified-random references, 95%
bootstrap CIs on test, 5-fold CV during development, and low-support categories flagged
rather than averaged into the headline. Plus polarity accuracy restricted to correctly
detected aspects, which separates "wrong aspect" from "right aspect, wrong sentiment" —
two failures with entirely different fixes.

### 5. The taxonomy was too large for the corpus

**Was:** 17 labels across five groups.

On a few hundred comments that leaves several categories with fewer than ten examples:
too few to train on, too few to evaluate on, and enough to make macro-F1 swing on two or
three test items.

**Now:** 8 flat categories with explicit boundary rules for the three confusable pairs
(`teaching`/`pace`, `assessment`/`workload`, `course_content`/`pace`), and a documented
promotion rule — if a theme appears in `other` five or more times, it becomes a real
category.

### 6. One preprocessing pipeline fed both model families

**Was:** raw → PII → lowercase → tokenise → stopwords → lemmatise → *both* the TF-IDF
baseline and the Week 3–4 transformer.

Lemmatisation, stopword removal and lowercasing destroy casing, punctuation, function
words and subword morphology — precisely what a pretrained transformer depends on. This
is a reliable way to make the Week 4 "improved" model underperform the baseline for
reasons that have nothing to do with the model, and then spend a week debugging the wrong
thing.

**Now:** two branches. Branch A normalises aggressively for TF-IDF; Branch B applies PII
redaction and whitespace normalisation only.

---

## Smaller corrections

| Issue | Fix |
|---|---|
| `"confidence": 0.91` on lexicon output | A keyword match has no probability. Every score now carries an explicit `score_type`: `lexicon_match_count`, `uncalibrated_probability` or `calibrated_probability`. `predict_proba` is not calibrated and is now labelled as such. |
| No span annotation | Character offsets are mandatory from Week 1. The Week 3–4 transformer needs token-level supervision; retrofitting means re-annotating everything. Enforced by `feedback_text[start:end] == aspect_term` in the validator. |
| Circular lexicon evaluation | Lexicon built from the training split only, by members not annotating validation or test, with per-term provenance logged to `lexicon_provenance.md`. |
| No near-duplicate handling | Char 3–5 gram cosine grouping before splitting, so a paraphrase of a training comment can't land in test and silently inflate every score. |
| `stratify=y` on multi-label data | Iterative stratification (Sechidis et al., 2011), implemented in `src/splitting.py`. |
| General stopword removal | The standard list deletes `not`, `never`, `but`, `however` — the tokens carrying sentiment and contrast. Those are now explicitly retained, plus negation-scope marking that terminates at a clause boundary. |
| Unpinned dependencies | All versions pinned, so a Week 2 metric is reproducible in Week 8. |
| Annotation format | Long format (one row per feedback × annotator × aspect) instead of JSON-in-a-cell. Six people will edit this in a spreadsheet; a malformed JSON string in row 240 of an Excel export is a much worse failure than an extra row. |
| Design choices asserted, not tested | Ablations for negation marking, class weighting and character n-grams are built into the runner. |
| Responsible AI as a Week 7 section | The `advisory_only` and `not_for_individual_action` constraints now live in the API contract. A constraint that exists only in a report is not a constraint. |
| Scope creep (7 NLP tasks) | Core is ABSA. Document sentiment is a comparison point, clustering is diagnostic. Summarisation and dashboard are explicitly out of scope — the brief asks for one strong capability. |

---

## What's new: the code

The original package had none. Its metrics tables would have stayed `[TO FILL]`
indefinitely because there was nothing to produce them. Everything below is implemented
and verified end to end on fixture data:

| File | Purpose |
|---|---|
| `config.py` | Seed, label sets, split proportions — single source of truth |
| `preprocess.py` | Both branches, PII redaction, negation-scope marking |
| `data.py` | Loading, validation, span-integrity enforcement, annotator resolution |
| `splitting.py` | Iterative stratification, near-duplicate grouping |
| `aspect_lexicon.py` | Training-split-only lexicon with provenance logging |
| `baselines.py` | Sentiment, aspect detection, ABSA — all inside sklearn Pipelines |
| `metrics.py` | Tuple F1, per-category breakdown, bootstrap CI, reference baselines |
| `iaa.py` | Cohen's κ, pairwise F1, span overlap, pass/fail against the gate |
| `make_fixture.py` | Smoke-test corpus + double-annotated calibration set |
| `run_baseline.py` | End-to-end run producing every number the report needs |

`run_baseline.py` outputs `results/results_tables.md` in the exact shape of the report's
tables, so results transfer without retyping — and without the chance to "adjust" a number
in transit.

### What the smoke test demonstrates

On the 60-comment fixture, the full baseline scores micro tuple-F1 around 0.25–0.30 with a
95% CI whose lower bound is 0.00, and 5-fold CV of 0.09 ± 0.08. That is the intended
lesson, not a defect: **nine test comments cannot support any claim whatsoever.** The
runner prints exactly that warning. It is a preview of what will happen with a small real
corpus, and the reason the CI and CV machinery is not optional.

---

## Do this first

1. **Settle the dataset.** Everything downstream is blocked on it.
2. **Run the calibration round** on 40 comments with all six annotators, then
   `python src/iaa.py`. Do not start production annotation below κ = 0.60.
3. **Annotate**, then `python src/data.py` to validate, then
   `python src/run_baseline.py data/raw/annotations.csv`.
4. **Paste `results/results_tables.md`** into Section 14 of the report.

One thing I could not verify: I don't have the actual `SNLP_Project_2026_BTI DS XI`
handout, so I could not check any of these documents against the real rubric or the stated
Week 1–2 requirements. Everything here is a judgement about what makes the NLP defensible.
Check it against the brief before submitting, and if the handout contradicts anything here,
the handout wins.
