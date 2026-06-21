"""The guard lets a valid short request through and keeps off-topic text out."""
from __future__ import annotations

from ontime.pipeline.guard import classify_structured, classify_text


def test_one_sentence_request_passes():
    decision = classify_text("plan my errands, last stop by 5pm")
    assert decision.in_scope
    assert decision.problem_kind == "driving"


def test_single_clear_hint_passes():
    decision = classify_text("give me a route for these deliveries")
    assert decision.in_scope
    assert decision.problem_kind == "driving"


def test_scheduling_request_passes():
    decision = classify_text("schedule these jobs on one machine")
    assert decision.in_scope
    assert decision.problem_kind == "machine_scheduling"


def test_off_topic_request_is_refused():
    decision = classify_text("write me a poem about the sea")
    assert not decision.in_scope


def test_structured_driving_passes():
    decision = classify_structured(
        {"kind": "driving", "coordinates": [[0, 0]], "time_windows": [[0, 1]]}
    )
    assert decision.in_scope
    assert decision.problem_kind == "driving"
