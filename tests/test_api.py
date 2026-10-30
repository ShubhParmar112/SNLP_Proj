"""Week 6 tests: the HTTP API keeps the contract in docs/API_Specification.md.

Uses FastAPI's in-process test client, so no server needs to be running. Tests that need
the trained model are skipped when the 738 MB weights are not present.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import api
from pipeline import DEFAULT_MODEL

HAVE_MODEL = (DEFAULT_MODEL / "model.pt").exists()
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason=f"no trained model at {DEFAULT_MODEL}")


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:          # runs the startup that loads the model
        yield c


def test_schema_lists_the_label_sets(client):
    s = client.get("/v1/schema").json()
    assert [a["label"] for a in s["aspects"]][:2] == ["teaching", "faculty_interaction"]
    assert {p["label"] for p in s["polarities"]} == {"positive", "negative", "neutral", "conflict"}


def test_bad_requests_use_the_specified_error_format(client):
    r = client.post("/v1/analyze-feedback", json={"feedback_id": "X"})       # no text
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST"
    r = client.post("/v1/analyze-feedback/batch", json={"items": [{"text": "ok"}] * 501})
    assert r.status_code == 400 and r.json()["error"]["code"] == "BATCH_TOO_LARGE"


@needs_model
def test_health_reports_the_loaded_model(client):
    h = client.get("/v1/health").json()
    assert h["status"] == "ok" and h["model_version"] == "transformer-absa-0.2.2"


@needs_model
def test_empty_and_too_long_text_are_rejected(client):
    for text, code in [("   ", "EMPTY_TEXT"), ("x" * 5001, "TEXT_TOO_LONG")]:
        r = client.post("/v1/analyze-feedback", json={"text": text})
        assert r.status_code == 400 and r.json()["error"]["code"] == code


@needs_model
def test_handout_example_through_http(client):
    text = ("The lectures were very informative, but the pace was too fast and the "
            "assignments were difficult.")
    r = client.post("/v1/analyze-feedback", json={"feedback_id": "F001", "text": text})
    assert r.status_code == 200
    out = r.json()
    got = {(a["aspect"], a["polarity"]) for a in out["aspects"]}
    assert got == {("teaching", "positive"), ("pace", "negative"), ("assessment", "negative")}
    for a in out["aspects"]:
        assert a["aspect_score_type"] == "calibrated_probability"
        assert text[a["span"]["start"]:a["span"]["end"]] == a["span"]["text"]


@needs_model
def test_batch_report_shape_and_caveats(client):
    items = [{"feedback_id": f"F{i}", "text": t} for i, t in enumerate(
        ["Great lectures.", "Too many assignments every week.", ""])]
    rep = client.post("/v1/analyze-feedback/batch",
                      json={"course_id": "TEST-101", "items": items}).json()
    assert rep["n_analyzed"] == 2 and rep["n_failed"] == 1
    assert "emerging_concerns" in rep and rep["caveats"]


@needs_model
def test_support_signals_are_course_level_and_advisory(client):
    texts = (["Too many assignments every single week."] * 4
             + ["Great lectures, very clear examples."] * 4)
    items = [{"feedback_id": f"S{i}", "text": t} for i, t in enumerate(texts)]
    r = client.post("/v1/support-signals", json={"course_id": "TEST-101", "items": items})
    assert r.status_code == 200
    out = r.json()
    assert out["n_analyzed"] == 8 and out["signals"]
    for s in out["signals"]:
        assert s["advisory_only"] is True and s["not_for_individual_action"] is True
        assert s["signal_type"] == "academic_concern" and s["polarity"] == "negative"
        assert s["basis"] in {"top_concern", "emerging_concern"}
        assert "feedback_id" not in s and "quote" not in s       # nothing per comment
    assert any(s["aspect"] in {"workload", "assessment"} for s in out["signals"])


@needs_model
def test_model_card_states_limits(client):
    card = client.get("/v1/model-card").json()
    assert "Any decision about an individual student or staff member" in card["out_of_scope"]
    assert 0.6 < card["test_results"]["tuple_f1"] < 0.8
