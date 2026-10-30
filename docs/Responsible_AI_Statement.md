---
title: Student Feedback Intelligence
subtitle: Responsible AI Statement
week: Final deliverable I (prepared in Week 7)
course: SNLP Course Project 2026   •   Group 5
members: E029  ·  E030  ·  E031  ·  E033  ·  E034  ·  E035
version: Document version 1.0  •  Model transformer-absa-0.2.2  •  Dataset v1.1-edurabsa
note: Every figure below is read from a file in results/; the file is named next to it.
footer: Group 5  •  Student Feedback Intelligence  •  Responsible AI Statement  •  
---

# Responsible AI Statement

## What the asset does, and the one rule that governs it

The asset reads student feedback and reports which parts of a course students praise or criticise. **Its output is advisory and course-level.** It must never be the basis for a decision about an identified student or staff member, and it must never be used to rank instructors. Every API response repeats this in its `caveats`, and the signals we send Group 6 carry `advisory_only` and `not_for_individual_action` flags as part of the contract.

## Privacy

- **Data.** The model is trained and tested on EduRABSA: 5,998 public course and teacher reviews (CC BY-NC-SA 4.0). No NMIMS student data was used, and none is stored in the project.
- **Redaction.** Emails, phone numbers, roll numbers and names after a title (“Prof. X”) are replaced before text reaches the model, and again, more strictly, in course-report quotes.
- **Known gap.** Names without a title (“Kimsey is great”) are *not* redacted. Named-entity redaction is needed before any real institutional feedback is processed. For the Group 6 handoff, we therefore send no comment that may contain a name at all (`src/integration_group6.py`).
- **No external services.** The model runs locally; feedback text is never sent to a public AI service.

## Bias

- **Teaching-negative bias.** In 5 of the 15 failed behaviour probes, the model adds a wrong *teaching: negative* to a negative sentence about something else (“The exams were not fair.”) (`results/week5_probe_results.json`). Half of the training data are reviews of teachers. In a course report this makes teaching look worse than students said, which is unfair to instructors. The model card states it, and instructor ranking is out of scope.
- **Language and culture.** Hinglish probes pass 0 of 5 and sarcasm probes 0 of 4 (`results/week5_eval.json`). The training reviews come from North American and UK rating sites, so accuracy on Indian classroom feedback is untested. The asset is declared English-only.
- **Uneven aspects.** Per-aspect F1 ranges from 0.802 (teaching) to 0.487 (resources) (`results/week5_eval.json`); every aspect is reported, none is averaged away.

## Hallucination

The model is **extractive**: it chooses labels from fixed lists and points to the words in the comment that support each one. It generates no free text, so it cannot invent statements. Report quotes are verbatim sentences from the comments. It can still be *wrong*: test tuple-F1 is 0.704 [0.683, 0.723], not 1.0 (`results/week5_eval.json`).

## Human oversight

- **Confidence is honest.** Scores are calibrated on validation data: document-sentiment calibration error fell from 0.116 to 0.021 on the test split (`results/week5_calibration.json`). Uncertain judgements carry a `low_confidence` warning.
- **Small samples are flagged.** Reports over fewer than 20 comments carry an instability caveat.
- **People decide.** A coordinator reads the quotes behind every concern. Group 6's support categories go to an advisor for review and never trigger an action automatically.

## Security

Feedback text is treated as untrusted input. Requests are validated (length, batch size, empty text), and consumers are told never to paste feedback unescaped into an LLM prompt.

## Limitations

Neutral and mixed opinions are often read as positive or negative. Sarcasm, Hinglish and implicit aspects fail. Every result comes from one public review corpus. These are reported, not hidden: the demo shows one failure on purpose.
