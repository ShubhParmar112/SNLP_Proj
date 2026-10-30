"""Week 4 tests: the v1.1 decision layer and the frozen-settings guard.

The model-dependent test is skipped when the trained weights are not present.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import ASPECTS, DATASET_TAG, PROCESSED, RESULTS
from transformer_absa import clause_window, decide_slots, evidence_span


def test_no_thresholds_is_exactly_the_v1_argmax_rule():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5), size=8)
    assert (decide_slots(p, None) == p.argmax(-1)).all()


def test_threshold_rule_separates_presence_from_polarity():
    p = np.tile([0.55, 0.30, 0.10, 0.03, 0.02], (8, 1))      # argmax says "none"
    assert (decide_slots(p, None) == 0).all()
    low = {a: 0.40 for a in ASPECTS}                            # P(present)=0.45 >= 0.40
    assert (decide_slots(p, low) == 1).all()                    # -> positive
    high = {a: 0.50 for a in ASPECTS}
    assert (decide_slots(p, high) == 0).all()


def test_clause_window_stops_at_contrast_words():
    text = "The lectures were informative but the pace was too fast."
    s = text.find("fast")
    a, b = clause_window(text, s, s + 4, max_words=8)
    assert text[a:b] == "the pace was too fast"


def test_clause_window_caps_long_clauses_around_the_evidence():
    text = "one two three four five six seven eight nine ten eleven twelve"
    s = text.find("seven")
    a, b = clause_window(text, s, s + 5, max_words=4)
    assert len(text[a:b].split()) == 4 and "seven" in text[a:b]


def test_clause_rule_widens_a_one_word_evidence_span():
    text = "The pace was too fast"
    offsets = [(0, 0), (0, 3), (3, 8), (8, 12), (12, 16), (16, 21), (0, 0)]
    alpha = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])      # one-hot on "fast"
    assert evidence_span(alpha, offsets, text)["text"] == "fast"
    assert evidence_span(alpha, offsets, text, clause_max_words=8)["text"] == text


def test_decision_settings_still_match_what_was_frozen():
    frozen = json.loads((RESULTS / f"week4_{DATASET_TAG}_frozen_settings.json").read_text())
    blob = (PROCESSED / "decision_settings.json").read_text()
    assert hashlib.sha256(blob.encode()).hexdigest() == frozen["decision_settings_sha256"]


def test_pipeline_loads_v1_1_settings():
    from pipeline import DEFAULT_MODEL, FeedbackAnalyzer
    if not (DEFAULT_MODEL / "model.pt").exists():
        pytest.skip(f"no trained model at {DEFAULT_MODEL}")
    fa = FeedbackAnalyzer()
    assert fa.model_version == "transformer-absa-0.2.2" and fa.thresholds is not None
    v1 = FeedbackAnalyzer(settings=None, calibration=None)
    assert v1.model_version == "transformer-absa-0.2.0" and v1.thresholds is None


def test_report_quotes_are_redacted():
    from pipeline import redact_quote
    q = redact_quote("Very interesting class, and professor Lektzian is also helpful. "
                     "Mail me at normnorm@ gmail. com or Dr. Rao.")
    assert "Lektzian" not in q and "Rao" not in q and "gmail" not in q
    assert redact_quote("The teacher Helps a lot.") == "The teacher Helps a lot."


def test_lectures_rule_sends_lecture_comments_to_teaching():
    from build_dataset import map_quad
    q = lambda aspect, opinion="boring": {"category": "Course - Learning activity",
                                          "aspect": aspect, "opinion": opinion}
    assert map_quad(q("lectures")) == "teaching"
    assert map_quad(q("online lectures")) == "teaching"
    assert map_quad(q("video lectures")) == "other"          # a resource, not delivery
    assert map_quad(q("lecture quizzes")) == "other"         # an assessment, not delivery
    assert map_quad(q("labs")) == "other"                    # other activities unchanged
    assert map_quad(q("lectures", "way too fast")) == "pace"  # pace rule still wins
