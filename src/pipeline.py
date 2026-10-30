"""The NLP pipeline: v1 (Week 3), v1.1 decision settings (Week 4).

    raw comment -> validation -> PII redaction (Branch B) -> transformer ABSA
                -> post-processing (typed scores, evidence spans, warnings) -> JSON

Implements the response shapes of docs/API_Specification.md sections 2 and 3 as a Python
module, so another group can already call it in-process. The FastAPI wrapper around
these two functions is the Week 6 deliverable; the contract does not change.

    from pipeline import FeedbackAnalyzer
    fa = FeedbackAnalyzer()                          # loads $SNLP_MODEL
    fa.analyze("The lectures were informative but the pace was too fast.")
    fa.analyze_batch([{"feedback_id": "F1", "text": "..."}], course_id="DS-XI")

CLI:
    python src/pipeline.py "The lectures were very informative, but the pace was too fast."

Week 4 (v1.1) keeps the same weights and changes only the decisions made on top of them,
read from data/processed/decision_settings.json (tuned on the validation split by
src/tune_week4.py): a presence threshold per aspect, the evidence-span growth rule, and a
minimum-mentions rule for the course report. settings=None reproduces v1 exactly.

Week 5 adds calibrated scores (data/processed/calibration.json, fitted on validation by
src/calibrate.py): every score is then reported as `calibrated_probability`. Calibration
changes only the reported numbers, never which aspects are reported. calibration=None
keeps the raw scores. Week 6 wraps this module as the FastAPI service in src/api.py.
"""

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from config import (ASSET_MODEL_NAME, CHECKPOINT_ROOT, ASPECTS, DOC_SENTIMENTS, POLARITIES,
                    PROCESSED)
from preprocess import normalise_whitespace, preprocess_transformer, redact_pii
from transformer_absa import (SLOT_LABELS, AspectAttentionABSA, Encoder, decide_slots,
                              evidence_span, predict)

DEFAULT_MODEL = Path(os.environ.get("SNLP_MODEL", str(CHECKPOINT_ROOT / ASSET_MODEL_NAME)))
DEFAULT_SETTINGS = PROCESSED / "decision_settings.json"
DEFAULT_CALIBRATION = PROCESSED / "calibration.json"
EMERGING_MIN_MENTIONS = 3       # below the ranking floor but at least this many mentions
EMERGING_MAX_NET = -0.5         # and clearly negative: an emerging concern
SCHEMA_VERSION = "1.0"
MAX_CHARS = 5000
MAX_BATCH = 500
LOW_CONFIDENCE = 0.5
SMALL_SAMPLE = 20


class FeedbackError(ValueError):
    """Carries an API error code from docs/API_Specification.md section 4."""

    def __init__(self, code, message, field="text"):
        super().__init__(message)
        self.code, self.message, self.field = code, message, field

    def as_dict(self):
        return {"error": {"code": self.code, "message": self.message, "field": self.field}}


# Quotes in course reports go to a coordinator, not back to the comment's author, so they
# get a stricter redaction than model input: titles in any case ("professor Smith"), and
# emails written with spaces ("name@ gmail. com"). Model input keeps the frozen Week 1-2
# redaction so every reported result still reproduces exactly.
_TITLED_NAME = re.compile(r"\b(?i:prof|professor|dr|mr|mrs|ms|miss|madam)\.?\s+[A-Z][a-z]+\b")
_SPACED_EMAIL = re.compile(r"\b[\w.+-]+\s?@\s?[\w-]+\s?\.\s?[a-z]{2,4}\b", re.I)


def redact_quote(text):
    text = _SPACED_EMAIL.sub("[EMAIL]", text)
    return _TITLED_NAME.sub("[INSTRUCTOR]", redact_pii(text))


def sentence_around(text, start, end, max_len=240):
    """The sentence of `text` containing [start, end): the quote used in course reports.

    Evidence spans are deliberately tight (often one or two words), which is right for
    highlighting but useless as a quote. Reports need a readable, still verbatim, unit.
    """
    bounds = [0] + [m.end() for m in re.finditer(r"[.!?]+\s+|\n+", text)] + [len(text)]
    s = max(b for b in bounds if b <= start)
    e = min((b for b in bounds if b >= end), default=len(text))
    quote = text[s:e].strip()
    return quote if len(quote) <= max_len else quote[:max_len].rsplit(" ", 1)[0] + " ..."


class FeedbackAnalyzer:
    def __init__(self, model_dir=None, device=None, settings=DEFAULT_SETTINGS,
                 calibration=DEFAULT_CALIBRATION):
        model_dir = Path(model_dir or DEFAULT_MODEL)
        if not (model_dir / "model.pt").exists():
            raise FeedbackError("MODEL_UNAVAILABLE", f"no model.pt in {model_dir}", "model")
        self.card = json.loads((model_dir / "card.json").read_text())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = Encoder(str(model_dir / "tokenizer"), self.card["max_length"])
        cfg = AutoConfig.from_pretrained(model_dir / "encoder_config")
        self.model = AspectAttentionABSA(encoder_config=cfg)
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        self.model.to(self.device).eval()
        self.pad_id = self.encoder.tok.pad_token_id
        self.model_version = self.card["model_version"]
        # Decision layer. None = v1 (argmax, 50%-of-peak spans, no mention floor).
        self.thresholds, self.span_rule, self.mention_rule = None, {"rel": 0.5}, None
        if settings is not None and Path(settings).exists():
            st = json.loads(Path(settings).read_text())
            self.thresholds = st["thresholds"]
            self.span_rule = st["span_rule"]
            self.mention_rule = st["min_mentions"]
            self.model_version = st["model_version"]
        # Score calibration (Week 5). None = raw, uncalibrated scores.
        self.cal = None
        if calibration is not None and Path(calibration).exists():
            cb = json.loads(Path(calibration).read_text())
            self.cal = {k: (np.array(v["x"]), np.array(v["y"])) for k, v in cb["maps"].items()}
            self.model_version = cb["model_version"]
        self.score_type = "calibrated_probability" if self.cal else "uncalibrated_probability"

    # --- single comment ------------------------------------------------------

    def analyze(self, text, feedback_id=None, include_spans=True, min_score=0.0):
        return self.analyze_many([text], [feedback_id], include_spans, min_score)[0]

    def analyze_many(self, texts, feedback_ids=None, include_spans=True, min_score=0.0,
                     batch_size=32):
        """Batched inference; one API-shaped result per text (same order)."""
        feedback_ids = feedback_ids or [None] * len(texts)
        for t in texts:
            self._check(t)
        items = self.encoder(texts)
        preds = predict(self.model, items, self.pad_id, self.device, batch_size)
        return [self._format(t, fid, it, p, include_spans, min_score, k)
                for k, (t, fid, it, p) in enumerate(zip(texts, feedback_ids, items, preds))]

    @staticmethod
    def _check(text):
        if not isinstance(text, str):
            raise FeedbackError("INVALID_REQUEST", "text must be a string")
        if len(text) > MAX_CHARS:
            raise FeedbackError("TEXT_TOO_LONG", f"text exceeds {MAX_CHARS} characters")
        if not preprocess_transformer(text).strip():
            raise FeedbackError("EMPTY_TEXT", "text is empty after redaction")

    def _score(self, kind, raw):
        """Reported score: the calibrated value when calibration is loaded."""
        if self.cal is None:
            return float(raw)
        x, y = self.cal[kind]
        return float(np.interp(raw, x, y))

    def _format(self, text, fid, item, pred, include_spans, min_score, k):
        warnings = []
        if item["text"] != normalise_whitespace(text):
            warnings.append("pii_redacted")
        if item["truncated"]:
            warnings.append("text_truncated")

        doc_p = pred["doc_probs"]
        doc_j = int(doc_p.argmax())
        aspects = []
        slots = decide_slots(pred["slot_probs"], self.thresholds)
        for a_i, aspect in enumerate(ASPECTS):
            p = pred["slot_probs"][a_i]
            j = int(slots[a_i])
            if j == 0:                               # aspect not present
                continue
            aspect_score = self._score("aspect", 1.0 - p[0])            # P(present)
            pol_score = self._score("polarity", p[j] / max(1e-9, 1.0 - p[0]))  # P(pol | present)
            if aspect_score < min_score:
                continue
            span = None
            if include_spans:
                ev = evidence_span(pred["alpha"][a_i], item["offsets"], item["text"],
                                   **self.span_rule)
                if ev:
                    # Offsets must refer to the caller's original text. Redaction or
                    # whitespace normalisation can shift them, so re-locate the span.
                    s = text.find(ev["text"])
                    span = ({"text": ev["text"], "start": s, "end": s + len(ev["text"])}
                            if s >= 0 else {"text": ev["text"], "start": None, "end": None})
            if aspect_score < LOW_CONFIDENCE or pol_score < LOW_CONFIDENCE:
                if "low_confidence" not in warnings:
                    warnings.append("low_confidence")
            aspects.append({
                "aspect": aspect,
                "polarity": SLOT_LABELS[j],
                "aspect_score": round(aspect_score, 4),
                "aspect_score_type": self.score_type,
                "polarity_score": round(pol_score, 4),
                "polarity_score_type": self.score_type,
                "span": span,
            })
        return {
            "feedback_id": fid or f"auto-{k + 1:04d}",
            "model_version": self.model_version,
            "schema_version": SCHEMA_VERSION,
            "doc_sentiment": {"label": DOC_SENTIMENTS[doc_j],
                              "score": round(self._score("doc", doc_p[doc_j]), 4),
                              "score_type": self.score_type},
            "aspects": aspects,
            "warnings": warnings,
        }

    # --- batch / course report --------------------------------------------

    def analyze_batch(self, items, course_id=None, include_examples=True,
                      max_examples_per_aspect=3):
        """Course-level aggregate (API section 3). Examples are verbatim evidence spans
        from the comments, so the report quotes students rather than paraphrasing them."""
        if len(items) > MAX_BATCH:
            raise FeedbackError("BATCH_TOO_LARGE", f"over {MAX_BATCH} items", "items")
        ok, failures = [], []
        for it in items:
            try:
                self._check(it.get("text"))
                ok.append(it)
            except FeedbackError as e:
                failures.append({"feedback_id": it.get("feedback_id"), "code": e.code})
        results = self.analyze_many([it["text"] for it in ok],
                                    [it.get("feedback_id") for it in ok]) if ok else []

        sentiment = Counter(r["doc_sentiment"]["label"] for r in results)
        per_aspect = {a: [] for a in ASPECTS}
        for it, r in zip(ok, results):
            for x in r["aspects"]:
                sp = x["span"]
                quote = (redact_quote(sentence_around(it["text"], sp["start"], sp["end"]))
                         if sp and sp["start"] is not None else None)
                per_aspect[x["aspect"]].append({**x, "quote": quote})

        summary = []
        for aspect, xs in per_aspect.items():
            if not xs:
                continue
            dist = Counter(x["polarity"] for x in xs)
            net = (dist["positive"] - dist["negative"]) / len(xs)
            dominant = "negative" if net < 0 else "positive"
            ex = sorted((x for x in xs if x["polarity"] == dominant and x["quote"]),
                        key=lambda x: -x["polarity_score"])
            seen, quotes = set(), []
            for x in ex:                      # one quote per distinct sentence
                if x["quote"] not in seen:
                    seen.add(x["quote"])
                    quotes.append(x["quote"])
            summary.append({
                "aspect": aspect,
                "mentions": len(xs),
                "polarity_distribution": {p: dist.get(p, 0) for p in POLARITIES},
                "net_sentiment": round(net, 3),
                "representative_examples": (quotes[:max_examples_per_aspect]
                                            if include_examples else []),
            })
        summary.sort(key=lambda s: -s["mentions"])
        # v1.1: an aspect mentioned by only a handful of comments cannot be a "top"
        # strength or concern - Week 3's sample ranked pace on 2 mentions.
        floor = 1
        if self.mention_rule:
            floor = max(self.mention_rule["absolute"],
                        int(np.ceil(self.mention_rule["share"] * len(results))))
        ranked = [s for s in summary if s["mentions"] >= floor]
        positives = sorted((s for s in ranked if s["net_sentiment"] > 0),
                           key=lambda s: -s["polarity_distribution"]["positive"])
        concerns = sorted((s for s in ranked if s["net_sentiment"] < 0),
                          key=lambda s: -s["polarity_distribution"]["negative"])

        # Emerging concerns: too few mentions to be ranked, but clearly negative.
        # Week 4 showed the ranking floor hiding pace (9 mentions, all negative).
        emerging = [s["aspect"] for s in summary
                    if EMERGING_MIN_MENTIONS <= s["mentions"] < floor
                    and s["net_sentiment"] <= EMERGING_MAX_NET]

        caveats = [("Scores are calibrated on held-out data but remain estimates."
                    if self.cal else
                    "Scores are uncalibrated; do not read them as probabilities."),
                   "Advisory only: not a basis for decisions about an individual student "
                   "or staff member.",
                   "Topic clustering is not part of pipeline v1."]
        if len(results) < SMALL_SAMPLE:
            caveats.insert(0, f"Aggregates over fewer than {SMALL_SAMPLE} comments are "
                              f"unstable and should not drive decisions.")
        return {
            "course_id": course_id,
            "model_version": self.model_version,
            "n_analyzed": len(results),
            "n_failed": len(failures),
            "failures": failures,
            "sentiment_distribution": {s: sentiment.get(s, 0) for s in DOC_SENTIMENTS},
            "aspect_summary": summary,
            "min_mentions_for_ranking": floor,
            "top_positive_aspects": [s["aspect"] for s in positives[:3]],
            "top_concerns": [s["aspect"] for s in concerns[:3]],
            "emerging_concerns": emerging,
            "topics": [],
            "caveats": caveats,
        }

    # --- Week 7: course-level signals for Group 6 (API spec section 8) ---------

    def support_signals(self, items, course_id=None):
        """Course-level `academic_concern` signals for Group 6 (Early Warning).

        Built only from the course report's own lists - top_concerns and
        emerging_concerns - so no new threshold is introduced. A signal names an aspect
        and counts, never a comment, a quote or a feedback_id.
        """
        report = self.analyze_batch(items, course_id, include_examples=False)
        n = report["n_analyzed"]
        by_aspect = {s["aspect"]: s for s in report["aspect_summary"]}
        signals = []
        for basis, aspects in (("top_concern", report["top_concerns"]),
                               ("emerging_concern", report["emerging_concerns"])):
            for aspect in aspects:
                neg = by_aspect[aspect]["polarity_distribution"]["negative"]
                signals.append({
                    "signal_type": "academic_concern",
                    "course_id": course_id,
                    "aspect": aspect,
                    "polarity": "negative",
                    "basis": basis,
                    "mention_count": neg,
                    "total_comments": n,
                    "proportion": round(neg / n, 3) if n else 0.0,
                    "confidence_note": (f"aggregate of per-comment calibrated scores; "
                                        f"based on {n} comments"
                                        + ("; small sample, unstable" if n < SMALL_SAMPLE
                                           else "")),
                    "advisory_only": True,
                    "not_for_individual_action": True,
                })
        return {
            "course_id": course_id,
            "model_version": self.model_version,
            "n_analyzed": n,
            "n_failed": report["n_failed"],
            "signals": signals,
            "caveats": report["caveats"] + [
                "Signals describe a course, never a student. Do not join them to any "
                "student identifier."],
        }


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or ("The lectures were very informative, but the pace "
                                      "was too fast and the assignments were difficult.")
    t0 = time.time()
    fa = FeedbackAnalyzer()
    t1 = time.time()
    out = fa.analyze(text, feedback_id="CLI-1")
    print(json.dumps(out, indent=2))
    print(f"\n(model load {t1 - t0:.1f}s, analysis {1000 * (time.time() - t1):.0f} ms)",
          file=sys.stderr)
