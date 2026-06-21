"""Both shipped examples run end to end with no model."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ontime.route import plan_route

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_driving_structured_example_runs():
    outcome = plan_route(EXAMPLES / "driving" / "stops.csv")
    assert outcome.verified, outcome.reason
    assert outcome.route[0] == 0 and outcome.route[-1] == 0


def test_machine_scheduling_structured_example_runs():
    outcome = plan_route(EXAMPLES / "machine_scheduling" / "jobs.csv")
    assert outcome.verified, outcome.reason
    assert outcome.total_time is not None


@pytest.mark.skipif(
    not os.getenv("ONTIME_LLM_BASE_URL"),
    reason="natural-language path needs a self-hosted model endpoint",
)
def test_driving_text_example_runs():
    from ontime.pipeline.model_client import OpenAICompatibleClient

    outcome = plan_route(EXAMPLES / "driving" / "day.txt", model_client=OpenAICompatibleClient())
    assert outcome.verified, outcome.reason
