# API Specification — Student Feedback Intelligence

**SNLP Course Project 2026 — Group 5** · API version `v1` · Spec version 2.1 · implemented in `src/api.py` (see §9 and `docs/API.md`)

The reusability requirement is that another team can consume this component using this
document alone, without reading our source. That means the contract has to cover error
cases, versioning and the meaning of every number — not just the happy path.

---

## 1. The score field — read this before using any number

The previous draft returned `"confidence": 0.91` for every aspect. That was misleading in
two distinct ways, and both are fixed here.

1. **A keyword match has no probability.** The lexicon baseline either matched a term or
   it did not. Dressing that up as `0.91` invents precision that does not exist.
2. **`LogisticRegression.predict_proba` is not calibrated.** Its output is a decision
   score squashed to [0,1]. A value of 0.9 does **not** mean "correct 90% of the time",
   especially on a small imbalanced corpus.

So every score is returned alongside an explicit `score_type`:

| `score_type` | Meaning | Safe to threshold? |
|---|---|---|
| `lexicon_match_count` | Number of distinct lexicon terms matched. Integer ≥ 1. | For ranking only |
| `uncalibrated_probability` | Raw `predict_proba` output. Monotone with model confidence, not a frequency. | Yes, but tune the threshold on your own data |
| `calibrated_probability` | Post-Platt/isotonic calibration on a held-out set. Approximates a true frequency. | Yes |

**Status (Week 5–6):** the service implemented in `src/api.py` returns
`calibrated_probability` for every score. The calibration is isotonic, fitted on the validation split and
checked on the test split (Week 5–6 report, Section 3). Weeks 1–2 returned only
`lexicon_match_count` and `uncalibrated_probability`; `uncalibrated_probability` is still
used if the service starts without `data/processed/calibration.json`.

---

## 2. `POST /v1/analyze-feedback`

Analyse a single anonymised feedback comment.

### Request

```json
{
  "feedback_id": "F001",
  "text": "The lectures were informative but the pace was too fast.",
  "options": {
    "include_spans": true,
    "min_score": 0.0
  }
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `feedback_id` | string | no | Echoed back. If omitted, the service generates one. Must **not** be a roll number or any student identifier. |
| `text` | string | yes | 1–5000 characters. Caller is responsible for removing PII; the service also runs its own redaction pass. |
| `options.include_spans` | bool | no | Default `true` in the implemented service (spans come from the transformer model; set `false` to omit them). |
| `options.min_score` | float | no | Drop aspects below this score. Default `0.0`. |

### Response `200 OK`

```json
{
  "feedback_id": "F001",
  "model_version": "baseline-tfidf-lr-0.1.0",
  "schema_version": "1.0",
  "doc_sentiment": {
    "label": "negative",
    "score": 0.62,
    "score_type": "uncalibrated_probability"
  },
  "aspects": [
    {
      "aspect": "teaching",
      "polarity": "positive",
      "aspect_score": 2,
      "aspect_score_type": "lexicon_match_count",
      "polarity_score": 0.81,
      "polarity_score_type": "uncalibrated_probability",
      "span": {"text": "lectures were informative", "start": 4, "end": 29}
    },
    {
      "aspect": "pace",
      "polarity": "negative",
      "aspect_score": 1,
      "aspect_score_type": "lexicon_match_count",
      "polarity_score": 0.88,
      "polarity_score_type": "uncalibrated_probability",
      "span": null
    }
  ],
  "warnings": []
}
```

- `aspects` is `[]` when no aspect is detected. **This is a valid result, not an error** —
  many comments ("Good course.") genuinely name no aspect, and a consumer that treats
  an empty list as a failure will misread the data.
- `polarity` ∈ `positive | negative | neutral | conflict`. Consumers must handle
  `conflict`; collapsing it into `neutral` destroys the distinction the model exists to make.
- `span` is `null` when the model cannot localise the aspect.
- `warnings` carries non-fatal notices, e.g. `"text_truncated"`, `"pii_redacted"`,
  `"low_confidence"`.

---

## 3. `POST /v1/analyze-feedback/batch`

### Request

```json
{
  "course_id": "DS-XI-2026",
  "items": [
    {"feedback_id": "F001", "text": "..."},
    {"feedback_id": "F002", "text": "..."}
  ],
  "options": {"include_examples": true, "max_examples_per_aspect": 3}
}
```

Maximum 500 items per call. `course_id` is optional context, never a student identifier.

### Response `200 OK`

```json
{
  "course_id": "DS-XI-2026",
  "model_version": "baseline-tfidf-lr-0.1.0",
  "n_analyzed": 128,
  "n_failed": 0,
  "sentiment_distribution": {"positive": 54, "neutral": 31, "negative": 43},
  "aspect_summary": [
    {
      "aspect": "pace",
      "mentions": 47,
      "polarity_distribution":
        {"positive": 5, "negative": 38, "neutral": 2, "conflict": 2},
      "net_sentiment": -0.70,
      "representative_examples": ["The pace was too fast to follow."]
    }
  ],
  "top_positive_aspects": ["teaching", "resources"],
  "top_concerns": ["pace", "workload"],
  "topics": [
    {"topic_id": 0, "size": 41,
     "top_terms": ["assignment", "deadline", "submission", "week"]}
  ],
  "caveats": [
    "Aggregates over fewer than 20 comments are unstable and should not drive decisions.",
    "Scores are uncalibrated; do not read them as probabilities."
  ]
}
```

- `net_sentiment` = (positive − negative) / mentions, range [−1, 1].
- **`caveats` is always populated and must be surfaced to the end user**, not stripped by
  the consuming application. An aggregate over 8 comments looks identical to one over 800
  unless the caveat travels with it.

---

## 4. Errors

Standard HTTP status codes, with a consistent body.

```json
{
  "error": {
    "code": "TEXT_TOO_LONG",
    "message": "text exceeds 5000 characters",
    "field": "text"
  }
}
```

| Code | HTTP | Cause |
|---|---|---|
| `INVALID_REQUEST` | 400 | Malformed JSON or missing `text` |
| `TEXT_TOO_LONG` | 400 | Over 5000 characters |
| `EMPTY_TEXT` | 400 | Empty or whitespace-only after redaction |
| `BATCH_TOO_LARGE` | 400 | Over 500 items |
| `UNSUPPORTED_LANGUAGE` | 422 | Reserved; not emitted by v1 (there is no language detector yet). Hinglish is accepted and usually yields few or no aspects (§6) |
| `MODEL_UNAVAILABLE` | 503 | Model artefacts not loaded |

In batch mode, per-item failures do **not** fail the whole request: failed items appear in
a `failures` array with their code, and `n_failed` is non-zero.

---

## 5. Supporting endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/health` | Liveness; returns `model_version` and whether artefacts are loaded |
| `GET /v1/schema` | The aspect taxonomy, polarity label set and their definitions, so a consumer never hardcodes our labels |
| `GET /v1/model-card` | Training corpus size, evaluation scores with CIs, known failure modes, intended and out-of-scope uses |

`GET /v1/model-card` is the endpoint that makes this genuinely reusable: it lets another
group see what the component is and is not good at before wiring it into their system.

---

## 6. Scope limits a consumer must respect

1. **Advisory only.** Output must not be the sole basis for any decision affecting a
   student or a faculty member.
2. **No individual-level inference.** The Group 6 integration consumes **course-level
   aggregates**. This API must never be used to flag or profile an identified student.
3. **English only in v1.** Code-switched Hinglish input is accepted, but the Week 5 probes
   show it is mostly missed (0 of 5 probes fully correct). No automatic warning is emitted
   for it yet.
4. **Domain-bound.** Trained on academic course feedback. Behaviour on other text
   (hostel, admin, placement) is undefined.
5. **Input is untrusted.** Callers must not pass raw text into a downstream LLM prompt
   without escaping — feedback text is user-supplied and can carry injection attempts.

---

## 7. Versioning

- The path carries the major version (`/v1/`). Breaking changes bump it; the previous
  version stays up for the remainder of the course.
- `schema_version` covers the response shape. Adding a field is a minor bump; removing or
  retyping one is major.
- `model_version` changes whenever the weights change, **even when the schema does not**.
  Two runs that return different numbers under the same `model_version` are a bug.
- Consumers must ignore unknown fields rather than failing on them.

---

## 8. Group 6 integration contract

Group 5 emits **course-level advisory signals**, never per-student ones.

```json
{
  "signal_type": "academic_concern",
  "course_id": "DS-XI-2026",
  "aspect": "workload",
  "polarity": "negative",
  "mention_count": 38,
  "total_comments": 128,
  "proportion": 0.297,
  "basis": "top_concern",
  "confidence_note": "aggregate of per-comment calibrated scores; based on 128 comments",
  "advisory_only": true,
  "not_for_individual_action": true
}
```

The last two flags are part of the contract, not decoration. Group 6 must not join this
signal to any student identifier. Served since Week 7 at `POST /v1/support-signals`
(same request as §3). A signal is emitted for each aspect in the course report's
`top_concerns` (`basis: "top_concern"`) or `emerging_concerns` (`basis:
"emerging_concern"`); `mention_count` is the number of negative mentions. No new
threshold is involved.

---

## 9. Implementation status (Week 6)

Implemented in `src/api.py` (FastAPI) on top of `src/pipeline.py`; practical guide in
`docs/API.md`; interactive docs at `/docs` when the service runs. Differences from the
draft above, all additive:

| Item | Implemented behaviour |
|---|---|
| Scores | All `calibrated_probability` (§1) |
| `options.include_spans` | Defaults to `true` |
| Batch response | Adds `min_mentions_for_ranking` (top lists need ≥ 3 mentions and ≥ 5% of the batch) and `emerging_concerns` (below that floor, net sentiment ≤ −0.5) |
| `topics` | Always `[]` in v1; topic clustering is not part of the pipeline |
| `UNSUPPORTED_LANGUAGE` | Reserved, not emitted |
| Group 6 signal (§8) | Served at `POST /v1/support-signals` (Week 7) |

## Changelog

| Spec version | Change |
|---|---|
| 2.2 (Week 7) | `POST /v1/support-signals` implements the §8 contract; signals gain a `basis` field |
| 2.1 (Week 6) | Implementation status added; calibrated scores; spans on by default; ranking floor and emerging concerns in the batch response |
| 2.0 | Typed scores, error format, versioning, Group 6 contract |
