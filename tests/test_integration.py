"""Week 7 tests: the Group 6 handoff keeps its privacy and split promises."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import PROCESSED, ROOT
from integration_group6 import may_contain_name, merge

KEY = ROOT / "integration" / "group6_key.json"
needs_request = pytest.mark.skipif(not KEY.exists(), reason="run integration_group6.py request")


def test_name_filter_catches_bare_names():
    vocab = {"great", "is", "the", "went", "to", "office", "course", "hours"}
    assert may_contain_name("Kimsey is great.", vocab)                 # sentence start
    assert may_contain_name("I went to Martin's office hours.", vocab)  # mid-sentence
    assert not may_contain_name("Great course. The office hours help.", vocab)
    assert not may_contain_name("I think CS is great.", vocab)          # acronym, "I"


@needs_request
def test_request_uses_train_split_only_and_marks_synthetic():
    key = json.loads(KEY.read_text())["items"]
    splits = json.loads((PROCESSED / "splits.json").read_text())
    closed = set(splits["validation"]) | set(splits["test"])
    assert not any(k["source_id"] in closed for k in key)
    assert all(k["split"] == "train" for k in key if not k["synthetic"])
    sent = json.loads((ROOT / "integration" / "group6_request.json").read_text("utf-8"))
    assert set(sent) == {"items"}                                        # Group 6's format
    assert all(set(i) == {"id", "text"} for i in sent["items"])


@needs_request
def test_merge_reports_counts_only(tmp_path):
    sent = json.loads((ROOT / "integration" / "group6_request.json").read_text("utf-8"))
    ids = [i["id"] for i in sent["items"]]
    reply = {"items": [{"id": ids[0], "category": "academic", "urgency": "high",
                        "confidence": 0.8, "recommended_action": "Advisor"},
                       {"id": "NOT-SENT", "category": "academic"}]}
    p = tmp_path / "reply.json"
    p.write_text(json.dumps(reply))
    s = merge(p, tag="pytest")
    (ROOT / "results" / "week7_group6_merge_pytest.json").unlink()
    assert s["n_flagged"] == 1 and s["problems"]["unknown_id"] == 1
    assert s["problems"]["missing_id"] == len(ids) - 1
    text = json.dumps(s)
    assert not any(i in text for i in ids)                               # no ids, no text
