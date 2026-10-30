---
title: Student Feedback Intelligence
subtitle: Aspect-Based Sentiment Analysis of Academic Feedback
week: Week 7: Integration, Demo and Responsible AI
course: SNLP Course Project 2026   •   Group 5
members: E029  ·  E030  ·  E031  ·  E033  ·  E034  ·  E035
version: Document version 1.0  •  Model transformer-absa-0.2.2  •  Dataset v1.1-edurabsa
note: No model weight, threshold or rule was changed this week, and the test split was not opened.
footer: Group 5  •  Student Feedback Intelligence  •  Week 7  •  
---

# Student Feedback Intelligence — Week 7 Report

*SNLP Course Project 2026 · Group 5 · E029 · E030 · E031 · E033 · E034 · E035*

Week 7 deliverables: integration with another group's component, demo, documentation and the responsible-AI analysis.

## 1.  Scope and Status

The handout's Week 7 targets, and where each one stands:

| Handout target (§14, Week 7) | Status | Where |
|---|---|---|
| API integration with another project | **Done on our side.** A course-level signal endpoint for Group 6, and a comment handoff in Group 6's own format. Group 6's reply is due 28 October | Section 2 |
| Integration with the AI University architecture | Done | Section 3 |
| Responsible-AI analysis | **Done.** One-page statement (final deliverable I) | Section 4, `docs/Responsible_AI_Statement.pdf` |
| Documentation | **Done.** README answers the handout's “reusable asset test”; API guide and specification updated | Section 5 |
| User interface / demo | The `/docs` page and the terminal demos work; one new demo for Group 6. A one-page web interface moves to Week 8 | Sections 2.2, 6 |

All 35 automated tests pass (`results/week7_tests.txt`): the 31 from Week 6, plus four for this week's work.

## 2.  Integration with Group 6 (Early Warning)

The handout names *Group 5 → Group 6: student feedback analysis → student-support signals* as an integration point (§22). We built both directions of that link.

### 2.1  What Group 6 asked for, and what we send

On 24 October Group 6 proposed the following: we send them feedback comments as `{"items": [{"id": "FB001", "text": "..."}]}`, and they return, for each id, a support-signal category (academic, financial, mental health, harassment, administrative), an urgency, a confidence and a recommended action.

`src/integration_group6.py request` builds that file (`results/week7_group6_request.json`):

| | Count |
|---|---|
| Comments sent | 38 |
| From the **train** split (validation and test untouched) | 30 |
| of which have at least one negative aspect in the gold labels | 15 |
| Hand-written, marked `"synthetic": true` | 8 |
| Comments where a titled name was redacted | 2 |

Three privacy rules shaped the sample:

- **No comment that may name a person is sent.** Our redaction misses bare names, so the script skips any comment with a capitalised word mid-sentence, or a capitalised word that never appears in lowercase in the corpus. The first draft of the sample, before this rule, contained instructor names and a remark about an instructor's disability. That is exactly the gap described in Section 4.
- Comments EduRABSA files under *Personal traits* are skipped.
- The rest are redacted with the course-report rule (`pipeline.redact_quote`).

The public reviews rarely mention money, health or harassment. The 8 synthetic comments therefore exercise Group 6's categories: two are the handout's and Group 6's own examples. A second file, `integration/group6_request_with_analysis.json`, adds our analysis (document sentiment, aspects and polarity) to every item, so Group 6 can use our output, not just the raw text.

`src/integration_group6.py merge <reply>` reads Group 6's reply back. It checks every id against what we sent and writes **counts only** to `results/`: flags per category and urgency, and which of our aspects occur in flagged comments. No comment text or id appears in the summary.

**Until Group 6's reply arrives, a stand-in is used and labelled as one.** `integration_group6.py mock` is a keyword stand-in written by us. On it, the merge step read 38 of 38 ids back with no unknown or missing id (`results/week7_group6_merge_mock.json`). This tests our plumbing only; it says nothing about Group 6's system. The real reply will be merged the same way and reported in Week 8.

One observation from the mock run concerns our own model: it tags some synthetic non-academic comments (a part-time job, a harassment report) with *teaching: negative*. This is the teaching-negative bias from Week 5, seen again in a new setting.

### 2.2  What Group 6 receives from our API: course-level signals

The contract designed in Week 1–2 (API specification §8) is now served at `POST /v1/support-signals`. The request is the same as for the course report. The response holds one `academic_concern` signal per aspect that the course report already lists as a **top concern** or an **emerging concern**, so no new threshold was invented. On the 150-comment demo course (`results/week7_support_signals_demo.txt`):

| Aspect | Basis | Negative mentions | Share of comments |
|---|---|---|---|
| workload | top concern | 12 of 150 | 8.0% |
| pace | emerging concern | 3 of 150 | 2.0% |

Each signal carries `advisory_only: true` and `not_for_individual_action: true`. No quote, comment or id is ever included, and a test enforces this. The demo comments are the same 150 test-split comments used for the Week 6 course report demo; no metric is computed on them.

### 2.3  What we need from Group 6

- Their reply to the 38 comments by **28 October**.
- One example of their response format now: field names, the list of categories and urgency levels, and the confidence range.
- A choice for the final demo: a file handoff, or a live endpoint.

## 3.  Place in the AI University Architecture

| Layer (handout §4) | Our part |
|---|---|
| University data | Anonymised course feedback (here: public EduRABSA reviews) |
| Shared NLP services | Text classification and information extraction: aspect-based sentiment with evidence spans and calibrated scores |
| Student services → Feedback | Course reports for coordinators: `POST /v1/analyze-feedback/batch` |
| Student services → Early Warning (Group 6) | Course-level concern signals (`POST /v1/support-signals`) and the comment handoff (Section 2.1) |

## 4.  Responsible AI

The one-page statement (`docs/Responsible_AI_Statement.pdf`) covers privacy, bias, hallucination, human oversight, security and limitations. Its main points:

- **Privacy.** Public, licensed data only; local model; PII redaction. The known gap is names without a title. This week it became concrete: it is why the Group 6 sample excludes any comment that may contain a name.
- **Bias.** In 5 of the 15 failed behaviour probes, the error is the same spurious *teaching: negative* (`results/week5_probe_results.json`), which is unfair to instructors. Hinglish probes pass 0 of 5 and sarcasm 0 of 4 (`results/week5_eval.json`).
- **Hallucination.** The model is extractive and generates no text; quotes are verbatim.
- **Human oversight.** Calibrated scores, `low_confidence` warnings, small-sample caveats, and flags that forbid individual action.

## 5.  Documentation

- `README.md` now opens with the handout's reusable-asset test (§21): input, output, interface, an example, how to run it, how well it works, and what it must not be used for.
- `docs/API.md` has a section for `POST /v1/support-signals`; `docs/API_Specification.md` goes to v2.2.
- The new code is documented in its module docstring (`src/integration_group6.py`), with tests in `tests/test_integration.py` and `tests/test_api.py`.

## 6.  Moved to Week 8, and Why

Week 7 was submitted late, so these items move to Week 8. None changes the model or any reported number:

| Item | Week 8 plan |
|---|---|
| One-page web interface | A single page served by the API: paste comments, see aspects and the course report |
| Topic clustering (handout: “topic coherence or cluster quality”) | Diagnostic on the train split only, as in `docs/Evaluation_Protocol.md` Task D |
| Group 6's real reply | Merge it, report the counts, show the handoff in the demo |
| Architecture diagram | Redrawn for the final system |
| Final report, slides, demo script, Git repository | Final submission, 31 October |

## Appendix A.  Reproducing every number

```
python -m pytest tests -q                                   # results/week7_tests.txt
python src/integration_group6.py request                    # results/week7_group6_request.json
python src/integration_group6.py mock
python src/integration_group6.py merge integration/group6_response_MOCK.json
python src/api.py          # then, in a second window:
python examples/demo_support_signals.py                     # results/week7_support_signals_demo.txt
```
