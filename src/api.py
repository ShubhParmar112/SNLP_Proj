"""Week 6: the reusable asset - Student Feedback Intelligence as a web API.

    python src/api.py                                  # http://127.0.0.1:8000
    uvicorn api:app --app-dir src --port 8000          # same thing

Interactive documentation (try every endpoint in the browser): http://127.0.0.1:8000/docs

Implements docs/API_Specification.md v1:
    POST /v1/analyze-feedback         one comment  -> aspects, polarity, scores, evidence
    POST /v1/analyze-feedback/batch   up to 500    -> course report (positives, concerns)
    POST /v1/support-signals          up to 500    -> course-level signals for Group 6 (Week 7)
    GET  /v1/health                   is the model loaded?
    GET  /v1/schema                   the label sets and what each label means
    GET  /v1/model-card               what the model was trained on, how well it does,
                                      where it fails, and what it must not be used for

All NLP work happens in src/pipeline.py; this file only adds HTTP, request validation
and the error format of the specification, so the in-process module and the API always
return identical results.
"""

import json
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import ASPECTS, DOC_SENTIMENTS, POLARITIES, RESULTS
from pipeline import MAX_BATCH, MAX_CHARS, FeedbackAnalyzer, FeedbackError

API_VERSION = "v1"
HTTP_STATUS = {"INVALID_REQUEST": 400, "TEXT_TOO_LONG": 400, "EMPTY_TEXT": 400,
               "BATCH_TOO_LARGE": 400, "MODEL_UNAVAILABLE": 503}

# Label definitions, from docs/Annotation_Guidelines.md section 2.
ASPECT_DEFINITIONS = {
    "teaching": "Explanation clarity, teaching method, examples, lecture delivery, subject command",
    "faculty_interaction": "Approachability, responsiveness to doubts, feedback on work, attitude",
    "course_content": "Syllabus relevance, depth, structure, intrinsic difficulty of the material",
    "pace": "Speed of delivery, rushing, time spent per topic",
    "assessment": "Assignments, exams, quizzes, projects, grading and its fairness",
    "workload": "Quantity of work, deadline pressure, hours required",
    "resources": "Slides, notes, textbooks, recordings, LMS, lab equipment, software",
    "other": "An evaluative academic aspect outside the seven above (mostly labs, tutorials, "
             "readings and other learning activities)",
}
POLARITY_DEFINITIONS = {
    "positive": "The student evaluates the aspect favourably",
    "negative": "The student evaluates the aspect unfavourably",
    "neutral": "The aspect is mentioned without an evaluation",
    "conflict": "The student expresses both positive and negative views about the same aspect",
}

STATE = {"fa": None, "error": None, "lock": threading.Lock()}


@asynccontextmanager
async def lifespan(app):
    try:
        STATE["fa"] = FeedbackAnalyzer()
    except FeedbackError as e:            # the API still starts; /v1/health reports why
        STATE["error"] = e.message
    yield


app = FastAPI(
    title="Student Feedback Intelligence API",
    version="1.0.0",
    description="Group 5, SNLP Course Project 2026. Aspect-based sentiment analysis of "
                "student feedback: which aspects of a course a comment discusses, how the "
                "student feels about each, the words that show it, and a course-level "
                "report. Advisory only: never a basis for decisions about an individual "
                "student or staff member.",
    lifespan=lifespan)


# --- request and response shapes (they also drive the /docs page) -----------------

class AnalyzeOptions(BaseModel):
    include_spans: bool = Field(True, description="Return the evidence span for each aspect")
    min_score: float = Field(0.0, ge=0.0, le=1.0,
                             description="Drop aspects whose aspect_score is below this")


class AnalyzeRequest(BaseModel):
    feedback_id: Optional[str] = Field(None, max_length=64,
                                       description="Echoed back. Never a roll number.")
    text: str = Field(..., description=f"One anonymised comment, up to {MAX_CHARS} characters")
    options: AnalyzeOptions = AnalyzeOptions()

    model_config = {"json_schema_extra": {"examples": [{
        "feedback_id": "F001",
        "text": "The lectures were very informative, but the pace was too fast and the "
                "assignments were difficult.",
        "options": {"include_spans": True, "min_score": 0.0}}]}}


class Span(BaseModel):
    text: str
    start: Optional[int]
    end: Optional[int]


class AspectResult(BaseModel):
    aspect: str
    polarity: str
    aspect_score: float
    aspect_score_type: str
    polarity_score: float
    polarity_score_type: str
    span: Optional[Span]


class DocSentiment(BaseModel):
    label: str
    score: float
    score_type: str


class AnalyzeResponse(BaseModel):
    feedback_id: str
    model_version: str
    schema_version: str
    doc_sentiment: DocSentiment
    aspects: list[AspectResult]
    warnings: list[str]


class BatchItem(BaseModel):
    feedback_id: Optional[str] = Field(None, max_length=64)
    text: str


class BatchOptions(BaseModel):
    include_examples: bool = True
    max_examples_per_aspect: int = Field(3, ge=1, le=10)


class BatchRequest(BaseModel):
    course_id: Optional[str] = Field(None, max_length=64,
                                     description="Course context. Never a student identifier.")
    items: list[BatchItem]
    options: BatchOptions = BatchOptions()


class AspectSummary(BaseModel):
    aspect: str
    mentions: int
    polarity_distribution: dict[str, int]
    net_sentiment: float
    representative_examples: list[str]


class BatchResponse(BaseModel):
    course_id: Optional[str]
    model_version: str
    n_analyzed: int
    n_failed: int
    failures: list[dict]
    sentiment_distribution: dict[str, int]
    aspect_summary: list[AspectSummary]
    min_mentions_for_ranking: int
    top_positive_aspects: list[str]
    top_concerns: list[str]
    emerging_concerns: list[str]
    topics: list
    caveats: list[str]


# Named examples for the /docs page: pick one from the dropdown, then Execute.
SINGLE_EXAMPLES = {
    "handout": {"summary": "Handout example (works)", "value": {
        "feedback_id": "F001",
        "text": "The lectures were very informative, but the pace was too fast and the "
                "assignments were difficult."}},
    "short": {"summary": "Short form comment (works)", "value": {"text": "Too many assignments."}},
    "sarcasm": {"summary": "Sarcasm (known failure)", "value": {
        "text": "Oh great, another surprise quiz on Monday."}},
    "hinglish": {"summary": "Hinglish (known failure)", "value": {
        "text": "Sir ka padhane ka tareeka bahut accha hai."}},
    "empty": {"summary": "Empty text (error example)", "value": {"text": "   "}},
}
BATCH_EXAMPLES = {
    "course": {"summary": "Small course batch", "value": {
        "course_id": "DEMO-101",
        "items": [
            {"feedback_id": "C1", "text": "Great teacher, explains every concept with real examples."},
            {"feedback_id": "C2", "text": "The pace was far too fast, I could not keep up."},
            {"feedback_id": "C3", "text": "Too many assignments every single week."},
            {"feedback_id": "C4", "text": "The weekly assignments took most of my weekend."},
            {"feedback_id": "C5", "text": "Slides are clear and the notes are really useful."},
            {"feedback_id": "C6", "text": "Workload is heavy but the professor is always willing to help."}]}},
}


# --- errors in the specification's format ------------------------------------------

def error_response(code, message, field="text"):
    return JSONResponse(status_code=HTTP_STATUS.get(code, 400),
                        content={"error": {"code": code, "message": message, "field": field}})


@app.exception_handler(FeedbackError)
async def feedback_error(request: Request, exc: FeedbackError):
    return error_response(exc.code, exc.message, exc.field)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(x) for x in first.get("loc", []) if x != "body") or "body"
    return error_response("INVALID_REQUEST", first.get("msg", "malformed request"), field)


def analyzer():
    if STATE["fa"] is None:
        raise FeedbackError("MODEL_UNAVAILABLE", STATE["error"] or "model not loaded", "model")
    return STATE["fa"]


# --- endpoints -----------------------------------------------------------------------
# Plain `def` (not async): FastAPI runs them in a worker thread, so a slow analysis never
# blocks the server. The lock keeps one model call on the GPU at a time.

@app.post(f"/{API_VERSION}/analyze-feedback", response_model=AnalyzeResponse,
          summary="Analyse one comment")
def analyze_feedback(req: Annotated[AnalyzeRequest, Body(openapi_examples=SINGLE_EXAMPLES)]):
    fa = analyzer()
    with STATE["lock"]:
        return fa.analyze(req.text, req.feedback_id, req.options.include_spans,
                          req.options.min_score)


@app.post(f"/{API_VERSION}/analyze-feedback/batch", response_model=BatchResponse,
          summary="Course-level report for up to 500 comments")
def analyze_batch(req: Annotated[BatchRequest, Body(openapi_examples=BATCH_EXAMPLES)]):
    if len(req.items) > MAX_BATCH:
        raise FeedbackError("BATCH_TOO_LARGE", f"over {MAX_BATCH} items", "items")
    fa = analyzer()
    with STATE["lock"]:
        return fa.analyze_batch([i.model_dump() for i in req.items], req.course_id,
                                req.options.include_examples,
                                req.options.max_examples_per_aspect)


class SupportSignal(BaseModel):
    signal_type: str
    course_id: Optional[str]
    aspect: str
    polarity: str
    basis: str
    mention_count: int
    total_comments: int
    proportion: float
    confidence_note: str
    advisory_only: bool
    not_for_individual_action: bool


class SupportSignalsResponse(BaseModel):
    course_id: Optional[str]
    model_version: str
    n_analyzed: int
    n_failed: int
    signals: list[SupportSignal]
    caveats: list[str]


@app.post(f"/{API_VERSION}/support-signals", response_model=SupportSignalsResponse,
          summary="Course-level concern signals for Group 6 (Early Warning)")
def support_signals(req: Annotated[BatchRequest, Body(openapi_examples=BATCH_EXAMPLES)]):
    if len(req.items) > MAX_BATCH:
        raise FeedbackError("BATCH_TOO_LARGE", f"over {MAX_BATCH} items", "items")
    fa = analyzer()
    with STATE["lock"]:
        return fa.support_signals([i.model_dump() for i in req.items], req.course_id)


@app.get(f"/{API_VERSION}/health", summary="Is the service ready?")
def health():
    fa = STATE["fa"]
    return {"status": "ok" if fa else "unavailable",
            "model_loaded": fa is not None,
            "model_version": fa.model_version if fa else None,
            "device": str(fa.device) if fa else None,
            "error": STATE["error"]}


@app.get(f"/{API_VERSION}/schema", summary="Label sets and their meaning")
def schema():
    return {"schema_version": "1.0",
            "aspects": [{"label": a, "definition": ASPECT_DEFINITIONS[a]} for a in ASPECTS],
            "polarities": [{"label": p, "definition": POLARITY_DEFINITIONS[p]}
                           for p in POLARITIES],
            "doc_sentiments": DOC_SENTIMENTS,
            "score_types": {
                "calibrated_probability": "Mapped on held-out data so that a score of 0.8 "
                                          "is right about 80% of the time",
                "uncalibrated_probability": "Raw model probability; ranking only"},
            "limits": {"max_chars": MAX_CHARS, "max_batch_items": MAX_BATCH}}


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@app.get(f"/{API_VERSION}/model-card", summary="What the model is, and is not, good for")
def model_card():
    fa = STATE["fa"]
    ev, cal = _load("week5_eval.json"), _load("week5_calibration.json")
    w4 = _load("week4_d1.1_test_eval.json")
    baseline = next((r["micro_f1"] for r in (w4 or {}).get("task_c", [])
                     if r["model"].startswith("TF-IDF")), None)
    card = {
        "model_version": fa.model_version if fa else None,
        "model": "DeBERTa-v3-base fine-tuned for aspect-category sentiment, with one "
                 "attention query per aspect (Week 3), decision settings tuned on "
                 "validation (Week 4), isotonic score calibration (Week 5)",
        "training_data": "EduRABSA (Hua et al., 2025; CC BY-NC-SA 4.0): 5,998 public English "
                         "course and teacher reviews, mapped to 8 aspects (dataset "
                         "v1.1-edurabsa); 4,198 train / 900 validation / 900 test",
        "intended_use": "Course-level summaries of anonymised feedback for coordinators "
                        "and faculty; advisory only",
        "out_of_scope": ["Any decision about an individual student or staff member",
                         "Ranking or comparing instructors",
                         "Non-academic text (hostel, admin, placement)",
                         "Languages other than English (Hinglish is not supported)"],
        "known_failure_modes": [
            "Sarcasm is read literally",
            "Hinglish / code-switched comments are mostly missed",
            "Neutral and mixed (conflict) opinions are often read as positive or negative",
            "Negative comments about a teacher can add a spurious teaching:negative",
            "Long reviews (80+ words) score lower; minor aspects get missed",
            "Names without any title are not redacted; emails written with spaces are "
            "redacted only in report quotes"],
    }
    if ev:
        o = ev["overall"]
        card["test_results"] = {
            "split": f"test, {ev['n']} comments, never used for any decision",
            "tuple_f1": round(o["tuple_f1"], 3),
            "tuple_f1_95ci": [round(o["ci"]["ci_low"], 3), round(o["ci"]["ci_high"], 3)],
            "baseline_tuple_f1_tfidf": round(baseline, 3) if baseline is not None else None,
            "polarity_accuracy_on_found_aspects": round(o["polarity_acc_on_detected"], 3),
            "per_aspect_f1": {a: round(v["f1"], 3) for a, v in ev["categories"].items()},
            "probe_set_passed": f"{ev['probes']['total_passed']}/{ev['probes']['total']}"}
    if cal:
        card["calibration_ece_test"] = {k: {"before": round(v["ece_before"], 3),
                                            "after": round(v["ece_after"], 3)}
                                        for k, v in cal.items()}
    return card


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
