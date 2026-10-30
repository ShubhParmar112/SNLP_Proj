# Student Feedback Intelligence — API Guide

**Group 5 · SNLP Course Project 2026** · API `v1` · model `transformer-absa-0.2.2`

How another application uses our component without reading our code. The full design
contract (every field, versioning rules) is `docs/API_Specification.md`; this page is the
practical guide. Every example below is real output, captured in
`results/week6_client_run.txt`.

## The reusable-asset test (handout §21)

| | |
|---|---|
| Our component | Student Feedback Intelligence: aspect-based sentiment for student feedback |
| Input | One anonymised comment, or a batch of up to 500 for a course |
| Output | For each comment: aspects, polarity per aspect, calibrated scores and the words that show it. For a batch: top strengths, top and emerging concerns, verbatim quotes, caveats |
| Interface | `POST /v1/analyze-feedback`, `POST /v1/analyze-feedback/batch`, `POST /v1/support-signals` (HTTP + JSON), or the Python module `pipeline.FeedbackAnalyzer` |
| Example | "The lectures were good but the pace was fast." → teaching: positive, pace: negative |

## 1. Start the service

On the machine that holds the trained model (`D:\snlp-feedback\checkpoints\deberta-v3-base_sw0.5_s42_d1.1`):

```
cd "Group5_Student_Feedback_Intelligence"
python src/api.py
```

The service listens on `http://127.0.0.1:8000`. Open `http://127.0.0.1:8000/docs` for an
interactive page where every endpoint can be tried from the browser. Another team's
model directory can be used by setting `SNLP_MODEL` before starting.

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/analyze-feedback` | Analyse one comment |
| POST | `/v1/analyze-feedback/batch` | Course report for up to 500 comments |
| GET | `/v1/health` | Is the model loaded? |
| GET | `/v1/schema` | The 8 aspects and 4 polarities, with definitions |
| GET | `/v1/model-card` | Training data, test results, known failures, out-of-scope uses |

## 2. One comment — `POST /v1/analyze-feedback`

| Field | Type | Required | Notes |
|---|---|---|---|
| `feedback_id` | string | no | Echoed back; generated if missing. Never a roll number |
| `text` | string | yes | 1–5,000 characters. The service also redacts emails, phone numbers, roll numbers and "Prof./Dr. Name" |
| `options.include_spans` | bool | no | Default `true`: return the evidence words for each aspect |
| `options.min_score` | float | no | Drop aspects whose `aspect_score` is below this. Default `0.0` |

Request:

```
{"feedback_id": "F001",
 "text": "The lectures were very informative, but the pace was too fast and the assignments were difficult."}
```

Response (`200`):

```
{"feedback_id": "F001", "model_version": "transformer-absa-0.2.2", "schema_version": "1.0",
 "doc_sentiment": {"label": "negative", "score": 0.5728, "score_type": "calibrated_probability"},
 "aspects": [
  {"aspect": "teaching", "polarity": "positive", "aspect_score": 0.9128, "polarity_score": 0.5021,
   "span": {"text": "The lectures were very informative", "start": 0, "end": 34}},
  {"aspect": "pace", "polarity": "negative", "aspect_score": 0.9559, "polarity_score": 0.9804,
   "span": {"text": "the pace was too fast", "start": 40, "end": 61}},
  {"aspect": "assessment", "polarity": "negative", "aspect_score": 0.9941, "polarity_score": 0.9804,
   "span": {"text": "the assignments were difficult", "start": 66, "end": 96}}],
 "warnings": []}
```

(The `*_score_type` fields are all `calibrated_probability` and are trimmed here.)

**What the scores mean.**

- `aspect_score` is how likely it is that the comment really discusses that aspect.
- `polarity_score` is how likely the stated feeling is right, given the aspect is discussed.
- Both are calibrated on held-out comments, so 0.9 means "right about 9 times in 10". Above, the model is sure about pace and assignments (0.98) but unsure whether the lectures remark is positive (0.50), and it says so.

`span.start` and `span.end` are character positions in your original text. `aspects` is an empty list when a comment names no aspect ("Good course."): that is a valid answer, not an error.

`warnings` can contain `pii_redacted`, `text_truncated` (over about 256 words) or `low_confidence` (some score below 0.5).

## 3. A course — `POST /v1/analyze-feedback/batch`

```
{"course_id": "DEMO-101",
 "items": [{"feedback_id": "C1", "text": "Great teacher, explains every concept with real examples."},
           {"feedback_id": "C2", "text": "The pace was far too fast, I could not keep up."}, ...],
 "options": {"include_examples": true, "max_examples_per_aspect": 3}}
```

The response has:

- `sentiment_distribution`, plus an `aspect_summary` per aspect: mentions, polarity counts, net sentiment from −1 to +1, and real quotes;
- `top_positive_aspects` and `top_concerns`, both limited to aspects mentioned by at least 3 comments and 5% of the batch;
- `emerging_concerns`, for aspects below that floor that are still clearly negative;
- `n_failed` and `failures`, listing the items that could not be analysed;
- `caveats`, which must be shown to the end user.

For a 6-comment demo batch:

```
{"n_analyzed": 6, "top_positive_aspects": [], "top_concerns": ["workload"], "emerging_concerns": [],
 "caveats": ["Aggregates over fewer than 20 comments are unstable and should not drive decisions.", ...]}
```

## 4. Errors

Every error has the same shape: `{"error": {"code": ..., "message": ..., "field": ...}}`.

| Code | HTTP | Cause |
|---|---|---|
| `INVALID_REQUEST` | 400 | Malformed JSON, missing `text`, wrong type |
| `EMPTY_TEXT` | 400 | Empty or whitespace-only text |
| `TEXT_TOO_LONG` | 400 | Over 5,000 characters |
| `BATCH_TOO_LARGE` | 400 | Over 500 items |
| `MODEL_UNAVAILABLE` | 503 | Service started without the model files |

In a batch, a bad item does not fail the request; it is listed in `failures`.

## 5. Calling it

Python, standard library only (full version: `examples/client_example.py`):

```
import json, urllib.request
req = urllib.request.Request("http://127.0.0.1:8000/v1/analyze-feedback", method="POST",
      data=json.dumps({"text": "The lectures were good but the pace was fast."}).encode(),
      headers={"Content-Type": "application/json"})
print(json.loads(urllib.request.urlopen(req).read())["aspects"])
```

Command line, from Windows PowerShell (type `curl.exe`, not `curl`, and keep the single quotes):

```
curl.exe -X POST http://127.0.0.1:8000/v1/analyze-feedback -H "Content-Type: application/json" -d '{\"text\": \"Too many assignments.\"}'
```

That returns `workload: negative` with the evidence "Too many assignments".

In-process, without HTTP:

```
from pipeline import FeedbackAnalyzer
FeedbackAnalyzer().analyze("Too many assignments.")
```

A round trip takes about 45–61 ms on the laptop GPU (mean of 20 calls, two runs), and analysis alone takes about 67 ms on the CPU.

## 5a. For Group 6 — `POST /v1/support-signals` (Week 7)

Same request as the batch endpoint. Returns course-level `academic_concern` signals: one
per aspect that the course report lists as a top concern or an emerging concern. No
comment text, quote or `feedback_id` is ever included. Run `python
examples/demo_support_signals.py`; the output on the 150 demo comments is in
`results/week7_support_signals_demo.txt`:

```json
{
  "signal_type": "academic_concern",
  "course_id": "DEMO-COURSE",
  "aspect": "workload",
  "polarity": "negative",
  "basis": "top_concern",
  "mention_count": 12,
  "total_comments": 150,
  "proportion": 0.08,
  "confidence_note": "aggregate of per-comment calibrated scores; based on 150 comments",
  "advisory_only": true,
  "not_for_individual_action": true
}
```

Group 6 also asked for comment text to run their own early-warning model on. That
handoff is a file exchange, built by `src/integration_group6.py`: see the Week 7 report.

## 6. Limits and responsible use

- **Advisory only.** Never the sole basis for a decision about a student or a staff member, and never used to rank instructors.
- **Course-level.** Only aggregates leave the batch endpoint. Its quotes are redacted (titled names, emails, phone numbers, roll numbers). Under 20 comments, a caveat is attached automatically.
- **English.** Hinglish and other code-switched text is mostly missed (Week 5–6 report, Section 5).
- **Untrusted input.** Do not paste feedback text unescaped into an LLM prompt.
- `GET /v1/model-card` lists the measured accuracy and known failure modes, so a consumer can judge fitness before wiring it in.
