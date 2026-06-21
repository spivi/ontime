"""The schedule renderer turns a verified route into readable lines."""
from __future__ import annotations

import csv
from pathlib import Path

from ontime.pipeline import schedule
from ontime.route import plan_route


def test_render_uses_names_and_times():
    text = schedule.render(
        route=[0, 2, 1, 0],
        arrivals=[0, 12, 24, 45],
        names=["home", "pharmacy", "post office"],
        time_windows=[[0, 600], [30, 90], [0, 30]],
    )
    assert "Leave home at 0 min." in text
    assert "Arrive at post office at 12 min." in text
    assert "Arrive at pharmacy at 24 min." in text
    assert "Return to home at 45 min." in text
    assert "The whole run takes 45 min." in text


def test_render_falls_back_to_stop_numbers():
    text = schedule.render(route=[0, 1, 0], arrivals=[0, 5, 12])
    assert "Leave stop 0 at 0 min." in text
    assert "Arrive at stop 1 at 5 min." in text


def test_schedule_from_a_real_run(tmp_path):
    csv_path = tmp_path / "stops.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x", "y", "open", "close", "service", "speed"])
        writer.writerow(["home", 0, 0, 0, 600, 0, 1.0])
        writer.writerow(["market", 10, 0, 0, 100, 2, 1.0])
        writer.writerow(["cafe", 4, 0, 0, 100, 2, 1.0])

    outcome = plan_route(csv_path)
    assert outcome.verified
    text = outcome.schedule_text()
    assert "Leave home" in text
    assert "market" in text and "cafe" in text
    assert text.strip().endswith("min.")


def test_schedule_text_on_refusal():
    from ontime.route import RouteOutcome

    refused = RouteOutcome(
        None, None, None, False, "stop 3 cannot be met", "structured", refused=True
    )
    assert refused.schedule_text().startswith("No schedule.")
