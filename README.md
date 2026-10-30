# Student Feedback Intelligence

**SNLP Course Project 2026 — Group 5**
E029 · E030 · E031 · E033 · E034 · E035

A reusable NLP component that extracts **aspect-based sentiment** from textual student
feedback and aggregates it into course-level insight.

## If another group wanted to use our component tomorrow (handout §21)

| What they need | Answer |
|---|---|
| Our component | Student Feedback Intelligence: aspect-based sentiment analysis of student feedback |
| Input | One anonymised comment (up to 5,000 characters), or up to 500 comments for one course |
| Output | Per comment: each aspect discussed (8 aspects), its polarity (positive / negative / neutral / conflict), a calibrated confidence score and the evidence words. Per course: top strengths, top and emerging concerns, redacted quotes, caveats |
| Interface | HTTP + JSON: `POST /v1/analyze-feedback`, `POST /v1/analyze-feedback/batch`, `POST /v1/support-signals` (for Group 6); or in Python, `pipeline.FeedbackAnalyzer` |
| How to run it | Setup below, then `python src/api.py`, then open `http://127.0.0.1:8000/docs` to try every endpoint |
| Example | "The lectures were very informative, but the pace was too fast and the assignments were difficult." → teaching: positive, pace: negative, assessment: negative |
| Calling code | `examples/client_example.py` (standard library only); guide `docs/API.md`; full contract `docs/API_Specification.md` |
| How well it works | Test-split tuple-F1 0.704 [0.683, 0.723] against 0.452 for a TF-IDF baseline (`results/week4_d1.1_test_eval.json`); `GET /v1/model-card` |
| What it must not be used for | Decisions about an individual student or staff member, ranking instructors, non-English text (`docs/Responsible_AI_Statement.md`) |

---

## What this is

Given a comment such as *"The lectures were informative but the pace was too fast"*, the
component returns:

```
teaching : positive
pace     : negative
```

A single document-level sentiment label cannot represent that comment. Producing the
aspect–polarity pairs, at usable accuracy, with an honest account of where it fails, is
the whole project.

## Scope (deliberately narrow)

**Core task — what we are graded on:** aspect-based sentiment analysis.

**Supporting:** document-level sentiment (as a comparison point), topic clustering
(diagnostic only), aggregation for the batch endpoint.

**Explicitly out of scope:** summarisation, a dashboard as a primary deliverable, and any
per-student inference. The brief asks for one strong NLP capability rather than seven
shallow ones; chasing all seven is the most common way these projects lose marks.

---

## Setting up on a new laptop

1. Install Python 3.13, then from this folder:
   `python -m venv .venv`, then `.venv\Scripts\activate`, then `pip install -r requirements.txt`.
   This gives the CPU build of PyTorch, which is enough to run the API, the demo and the tests.
   Retraining needs an NVIDIA GPU; see the comment in `requirements.txt`.
2. The trained model is **not** in this folder (738 MB). Unzip
   `Group5_trained_model_s42_d1.1.zip` so that this path exists:
   `models\deberta-v3-base_sw0.5_s42_d1.1\model.pt`. The code finds `models\`
   automatically; `src/config.py` explains the lookup order.
3. Check everything: `python -m pytest tests -q` should end with **31 passed**.
4. Run the service: `python src\api.py`, then open `http://127.0.0.1:8000/docs`.

The `data/raw/annotations.csv` and `data/external/` files are included in the team
folder. In a fresh Git clone they are not, and `python src/build_dataset.py` downloads and
rebuilds them.

## Quick start

```bash
pip install -r requirements.txt

# 1. Verify the pipeline runs (fixture data, NOT results)
python src/make_fixture.py
python src/run_baseline.py

# 2. Build the real corpus from EduRABSA (downloads 31 MB from Zenodo on first run)
python src/build_dataset.py
python src/data.py data/raw/annotations.csv

# 3. Run the real baseline, then the dataset quality profile
python src/run_baseline.py data/raw/annotations.csv
python src/profile_dataset.py

# 4. Week 3: train the transformer ABSA model (GPU, ~6 min on an RTX 3050 4 GB)
#    weights go to $SNLP_ARTIFACTS (default D:/snlp-feedback/checkpoints), not the repo
python src/train_transformer.py
python src/evaluate_week3.py          # validation only; the test split stays closed

# 5. Week 4: tune decisions on validation and freeze them, then the ONE test-split run
python src/tune_week4.py
python src/evaluate_week4.py          # refuses to run if the frozen settings changed

# 6. Week 5: calibrate the scores (fit on validation), then evaluate on the test split
python src/calibrate.py
python src/evaluate_week5.py          # per-aspect results, slices, robustness, probes, errors

# 7. Week 6: run the reusable asset as a web API (interactive docs at /docs)
python src/api.py                     # http://127.0.0.1:8000
python examples/client_example.py     # how another group calls it (standard library only)

# Or use the pipeline in-process
python src/pipeline.py "The lectures were informative but the pace was too fast."
```

Re-running steps 3–5 on a new dataset version reuses the frozen split in
`data/processed/splits.json`, and every test-split opening is appended to
`results/test_openings.log`.

Outputs land in `results/`: `metrics.json`, `results_tables.md` (paste straight into the
report) and `errors_for_analysis.json`.

---

## Repository

```text
.
├── README.md
├── requirements.txt
├── docs/
│   ├── Annotation_Guidelines.md    # label sets, boundary rules, IAA protocol
│   ├── Evaluation_Protocol.md      # metrics, baselines, splitting, leakage rules
│   ├── Data_Dictionary.md          # schemas for both data files
│   ├── API_Specification.md        # the reusable-asset contract (v2.1)
│   ├── API.md                      # Week 6: how to call the API, with real examples
│   ├── adjudication_log.md         # every disagreement and the rule that settled it
│   └── architecture.png
├── data/
│   ├── external/edurabsa/          # source JSON (downloaded; gitignored)
│   ├── raw/annotations.csv         # long-format gold annotations (built from EduRABSA)
│   ├── processed/                  # splits.json, aspect_lexicon.json, decision_settings.json,
│   │                               #   calibration.json
│   ├── probes/probe_set.csv        # Week 5: 32 hand-written behaviour probes
│   ├── fixtures/                   # smoke-test data — NOT the dataset
│   └── private/                    # gitignored; never committed
├── src/
│   ├── config.py                   # seed, label sets, split proportions
│   ├── preprocess.py               # classical + transformer branches
│   ├── data.py                     # loading and validation
│   ├── splitting.py                # iterative stratification, duplicate grouping
│   ├── aspect_lexicon.py           # lexicon built from train split only
│   ├── baselines.py                # sentiment / aspect / ABSA models
│   ├── metrics.py                  # tuple F1, bootstrap CI, reference baselines
│   ├── iaa.py                      # Cohen's kappa, span agreement
│   ├── make_fixture.py             # smoke-test corpus generator
│   ├── build_dataset.py            # EduRABSA -> annotations.csv (mapping rules)
│   ├── profile_dataset.py          # dataset quality tables
│   ├── make_calibration_sheet.py   # 40-comment IAA sheet
│   ├── transformer_absa.py         # Week 3: aspect-attention DeBERTa-v3 ABSA model
│   ├── train_transformer.py        # Week 3: training (train/val only)
│   ├── pipeline.py                 # Week 3: pipeline v1, API-shaped JSON output
│   ├── run_baseline.py             # end-to-end baseline run
│   ├── evaluate_week3.py           # Week 3: v1 vs baseline on validation
│   ├── embedding_baseline.py       # Week 4: MiniLM sentence embeddings + LogReg
│   ├── week4_common.py             # Week 4: shared decision rules and span scoring
│   ├── tune_week4.py               # Week 4: choose + freeze settings (validation only)
│   ├── evaluate_week4.py           # Week 4: baseline vs improved on the test split
│   ├── calibrate.py                # Week 5: isotonic score calibration
│   ├── evaluate_week5.py           # Week 5: evaluation and error analysis on test
│   ├── api.py                      # Week 6: FastAPI service (the reusable asset)
│   └── integration_group6.py       # Week 7: comment handoff to Group 6 and back
├── examples/client_example.py      # Week 6: calling the API from another application
├── examples/demo_support_signals.py  # Week 7: what Group 6 receives from the API
├── integration/                    # Week 7: files exchanged with Group 6
├── notebooks/                      # exploration only; nothing authoritative lives here
├── results/
├── scripts/build_report.py         # report .md -> .docx + .pdf in the house style
├── scripts/draw_architecture.py    # Week 8: docs/architecture.png
├── scripts/build_presentation.js   # Week 8: the slide deck (node + pptxgenjs)
├── scripts/measure_cpu_latency.py  # Week 8: results/week8_cpu_latency.json
└── tests/                          # test_pipeline.py, test_week3.py, test_week4.py, test_api.py,
                                    #   test_integration.py
```

## Approach

| Stage | Method |
|---|---|
| Baseline (Weeks 1–2) | TF-IDF (word 1–2 gram + char 3–5 gram) → Logistic Regression for aspect detection and polarity; transparent lexicon as a second, fully interpretable baseline |
| Improved (Weeks 3–4) | Transformer encoder for joint aspect–polarity extraction, compared against the baseline on the same frozen test split |
| Asset (Week 6) | FastAPI service implementing `docs/API_Specification.md` |

---

## Methodological commitments

These are not boilerplate. Each one addresses a specific way this project could produce a
number that looks good and means nothing.

1. **Every result is reported against a majority-class and a random baseline.** An
   accuracy figure without them cannot be interpreted.
2. **Every test figure carries a 95% bootstrap CI**, and development figures are
   5-fold CV mean ± sd. On a few hundred comments a single split is too noisy to support
   a comparison claim.
3. **The split is frozen** in `data/processed/splits.json` and committed. Near-duplicates
   are grouped so a paraphrase of a training comment cannot appear in test.
4. **Every transformation is fitted inside an sklearn Pipeline**, so the vectoriser can
   only ever see training data.
5. **The aspect lexicon is built from the training split only**, by members not annotating
   validation or test, with per-term provenance logged. Otherwise lexicon precision just
   measures our own consistency.
6. **Inter-annotator agreement is measured and reported** before production annotation,
   with κ ≥ 0.60 as the gate.
7. **No number is written into a report that a script did not produce.** Fixture-derived
   figures are labelled `SMOKE TEST — NOT A RESULT` wherever they appear.
8. **Negative results are reported.** A failed improvement with a diagnosis is worth more
   than a silently dropped experiment.

## Responsible use

- Output is **advisory and course-level**. It must not drive any decision about an
  identified student or faculty member.
- The Group 6 integration receives aggregate signals only, carrying explicit
  `advisory_only` and `not_for_individual_action` flags.
- No personally identifiable information enters the modelling pipeline or any external
  service.
- Institutional feedback is used only with documented authorisation.

## Reproducibility

`RANDOM_SEED = 42`, set in `src/config.py` and used by every stochastic step. Results
files record `dataset_version`, `annotation_version`, `preprocessing_version` and
`model_version`.

## Status

**Weeks 1–2: complete.** EduRABSA corpus (5,998 comments) mapped onto the 8-aspect
taxonomy; baseline ABSA micro tuple-F1 0.454 [0.432, 0.479] on test.
Write-up: `docs/Group5_Week1-2_Report_v3.docx` (also `.pdf`, `.md`).

**Week 3: complete — pipeline v1.** DeBERTa-v3-base with aspect-specific attention and
evidence-span supervision (`src/transformer_absa.py`), wrapped as `src/pipeline.py`
(API-shaped JSON, typed scores, spans, warnings, course report). On the validation
split: tuple-F1 **0.712** [0.692, 0.733] vs 0.476 for the baseline (paired difference
+0.236 [+0.212, +0.261]); 3 seeds 0.712 ± 0.002. The test split is still closed; it is
opened once for the Week 4 comparison. Write-up: `docs/Group5_Week3_Report.docx`
(also `.pdf`, `.md`).

**Week 4: complete — baseline vs improved comparison on the test split.** Pipeline
v1.1 = the Week 3 weights plus decision settings chosen on validation and frozen
(per-aspect thresholds, clause-length evidence spans, a ranking floor for course
reports). Test split, dataset v1.0: DeBERTa ABSA v1.1 tuple-F1 **0.703** [0.684, 0.723]
vs **0.454** for the TF-IDF baseline (paired difference +0.249 [+0.225, +0.275]);
frozen MiniLM sentence embeddings reach only 0.494. Write-up:
`docs/Group5_Week4_Report.docx` (also `.pdf`, `.md`).

**Week 4, final step — dataset v1.1, the lectures fix.** EduRABSA files most lecture
comments under *Learning activity*, which v1.0 mapped to `other`, so the handout's own
examples ("The lectures were very informative, but …") came out wrong. Rule 2b in
`src/build_dataset.py` now maps them to `teaching` (Guidelines 2.1); 221 comments were
relabelled on the same frozen split. The model was retrained (seed 42), decisions re-tuned
on validation and frozen, and the test split opened once more (see
`results/test_openings.log`). Test: tuple-F1 **0.704** [0.683, 0.723] vs **0.452** for
the TF-IDF baseline on the v1.1 labels; both handout example sentences are now fully
correct. Results: `results/week4_d1.1_*`. Model version `transformer-absa-0.2.1`.

Weights (738 MB) are not in the repo: set `SNLP_MODEL` to the checkpoint folder
(default `deberta-v3-base_sw0.5_s42_d1.1`).

**Weeks 5–6: complete — evaluation, error analysis and the reusable asset.** Scores are
calibrated (isotonic, fitted on validation; test calibration error for document sentiment
0.116 → 0.021). Test evaluation: tuple-F1 0.704 [0.683, 0.723]; per-aspect results,
error slices, robustness (typos −0.026) and 32 behaviour probes (17 passed; sarcasm and
Hinglish fail); 40 test errors coded by hand. The asset runs as a FastAPI service
(`src/api.py`, guide `docs/API.md`, demo `examples/demo_course_report.py`, 31 tests passing). Write-up:
`docs/Group5_Week5-6_Report.docx` (also `.pdf`, `.md`). Mid-term guide for the team:
`docs/Group5_MidTerm_Guide.docx`.

**Week 7: complete — integration and Responsible AI.** `POST /v1/support-signals` serves
the Group 6 course-level contract; `src/integration_group6.py` builds a 38-comment
handoff in Group 6's format (train split only, possible names excluded) and reads their
reply back as counts only; one-page Responsible AI statement; 35 tests passing.
Write-up: `docs/Group5_Week7_Report.docx` (also `.pdf`, `.md`).

**Week 8: complete — final package.** Final report (`docs/Group5_Final_Report.pdf`, 16
sections), 10-slide presentation (`docs/Group5_Presentation.pptx`, built by
`scripts/build_presentation.js`), demo script (`docs/Group5_Demo_Script.pdf`), final
architecture diagram (`docs/architecture.png`, drawn by `scripts/draw_architecture.py`),
CPU latency measured on the demo laptop (`results/week8_cpu_latency.json`). Topic
clustering and a graphical UI were descoped; the report says so.

**Weeks 7–8 (handed over 24 Oct 2026).** This folder is the working copy
for the final two weeks. It is the Week 5–6 state plus one change: trained models are
looked up via `src/config.py` (`$SNLP_ARTIFACTS`, then `./models`, then the original D:
location), so the project runs on any laptop. Remaining work, deadlines and conventions:
`HANDOVER.md` in the parent project folder.

The 40-comment calibration round from the annotation guidelines was not run: the
EduRABSA labels come from the dataset's expert annotator, and the handout does not
require inter-annotator agreement. This is recorded as a limitation.
