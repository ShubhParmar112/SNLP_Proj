"""Smoke tests. Run with: pytest tests/ -v

These are deliberately minimal but not decorative: each one guards a specific mistake
that would silently corrupt results rather than raise an error.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import absa_tuple_f1, polarity_accuracy_on_detected
from preprocess import mark_negation, preprocess_classical, redact_pii


def test_tuple_f1_exact_match():
    gold = [{("teaching", "positive"), ("pace", "negative")}]
    pred = [{("teaching", "positive"), ("pace", "neutral"), ("workload", "negative")}]
    r = absa_tuple_f1(gold, pred)
    assert r["tp"] == 1 and r["fp"] == 2 and r["fn"] == 1
    assert abs(r["f1"] - 0.4) < 1e-9


def test_right_aspect_wrong_polarity_is_not_credited():
    """The failure mode that separate aspect-F1 + polarity-accuracy reporting hides."""
    assert absa_tuple_f1([{("pace", "negative")}], [{("pace", "positive")}])["f1"] == 0.0


def test_polarity_accuracy_isolates_the_second_failure_mode():
    gold = [{("pace", "negative")}]
    pred = [{("pace", "positive")}]
    assert polarity_accuracy_on_detected(gold, pred)["accuracy"] == 0.0


def test_negation_scope_stops_at_clause_boundary():
    out = mark_negation("not clear but the notes are good".split())
    assert "NOT_clear" in out
    assert "NOT_good" not in out and "good" in out


def test_negation_words_survive_preprocessing():
    out = preprocess_classical("The lectures were not very informative.")
    assert "not" in out and "NOT_informative" in out


def test_pii_is_redacted():
    out = redact_pii("Contact E029 at a@b.com about Prof Sharma")
    assert "E029" not in out and "a@b.com" not in out and "Sharma" not in out
