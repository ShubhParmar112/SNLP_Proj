"""Week 3 tests: transformer ABSA helpers and the pipeline v1 contract.

The first group needs no model weights and always runs. The contract test at the end
loads the trained checkpoint ($SNLP_MODEL) and is skipped when it is not present, e.g.
on a teammate's laptop without the 740 MB weights.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformer_absa import (SLOT_LABELS, batches, decode_slots, encode_targets,
                              evidence_span, span_token_mask)


def test_slot_encoding_round_trips():
    tuples = {("teaching", "positive"), ("pace", "negative"), ("assessment", "conflict")}
    assert decode_slots(encode_targets(tuples)) == tuples


def test_empty_comment_encodes_to_all_none():
    y = encode_targets(set())
    assert (y == SLOT_LABELS.index("none")).all() and decode_slots(y) == set()


def test_span_mask_selects_overlapping_tokens_only():
    # "[CLS] The lectures were informative [SEP]" with DeBERTa-style offsets
    offsets = [(0, 0), (0, 3), (3, 12), (12, 17), (17, 29), (0, 0)]
    m = span_token_mask(offsets, 4, 12)                   # "lectures"
    assert m.tolist() == [False, False, True, False, False, False]


def test_evidence_span_grows_around_the_attention_peak():
    text = "The pace was too fast"
    offsets = [(0, 0), (0, 3), (3, 8), (8, 12), (12, 16), (16, 21), (0, 0)]
    alpha = np.array([0.0, 0.02, 0.40, 0.30, 0.05, 0.23, 0.0])
    span = evidence_span(alpha, offsets, text)
    assert span == {"text": "pace was", "start": 4, "end": 12}


def test_token_budget_caps_long_batches():
    items = [{"input_ids": [0] * n} for n in [10] * 20 + [256] * 10]
    got = list(batches(items, 16, shuffle=False, token_budget=1024))
    long = [len(b) for b in got if len(items[b[0]]["input_ids"]) == 256]
    assert max(long) <= 4 and len(got[0]) == 16


def test_pipeline_rejects_bad_input_with_api_error_codes():
    pytest.importorskip("transformers")
    from pipeline import FeedbackAnalyzer, FeedbackError
    for text, code in [("", "EMPTY_TEXT"), ("   ", "EMPTY_TEXT"),
                       ("x" * 5001, "TEXT_TOO_LONG"), (None, "INVALID_REQUEST")]:
        with pytest.raises(FeedbackError) as e:
            FeedbackAnalyzer._check(text)
        assert e.value.code == code


def _analyzer():
    from pipeline import DEFAULT_MODEL, FeedbackAnalyzer
    if not (DEFAULT_MODEL / "model.pt").exists():
        pytest.skip(f"no trained model at {DEFAULT_MODEL}")
    return FeedbackAnalyzer()


def test_pipeline_output_matches_api_contract():
    fa = _analyzer()
    text = ("The lectures were very informative, but the pace was too fast and the "
            "assignments were difficult.")
    out = fa.analyze(text, feedback_id="T1")
    assert out["feedback_id"] == "T1" and out["schema_version"] == "1.0"
    # Week 5 calibrated the scores; without calibration.json they stay uncalibrated.
    assert out["doc_sentiment"]["score_type"] in {"calibrated_probability",
                                                  "uncalibrated_probability"}
    for a in out["aspects"]:
        assert a["polarity"] in {"positive", "negative", "neutral", "conflict"}
        assert 0.0 <= a["aspect_score"] <= 1.0 and 0.0 <= a["polarity_score"] <= 1.0
        if a["span"] and a["span"]["start"] is not None:
            assert text[a["span"]["start"]:a["span"]["end"]] == a["span"]["text"]


def test_redacted_input_is_flagged():
    fa = _analyzer()
    out = fa.analyze("Prof Sharma explains well, mail him at a@b.com")
    assert "pii_redacted" in out["warnings"]


def test_batch_report_always_carries_caveats():
    fa = _analyzer()
    rep = fa.analyze_batch([{"feedback_id": "A", "text": "Great lectures."},
                            {"feedback_id": "B", "text": ""}])
    assert rep["n_analyzed"] == 1 and rep["n_failed"] == 1
    assert any("fewer than 20" in c for c in rep["caveats"])
