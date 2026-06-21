"""Structured mode verifies on every carried instance, with no model."""
from __future__ import annotations

import csv
from pathlib import Path

from ontime.route import plan_route


def _write_stops_csv(inst: dict, path: Path) -> None:
    coords = inst["coordinates"]
    windows = inst["time_windows"]
    speed = inst["speed"]
    service = inst["service_time"]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x", "y", "open", "close", "service", "speed"])
        for i, (c, w) in enumerate(zip(coords, windows)):
            svc = 0 if i == 0 else service
            writer.writerow([f"stop{i}", c[0], c[1], w[0], w[1], svc, speed])


def test_structured_driving_verifies_all(tmp_path, instances_n8):
    verified = 0
    for inst in instances_n8:
        csv_path = tmp_path / f"{inst['instance_id']}.csv"
        _write_stops_csv(inst, csv_path)
        outcome = plan_route(csv_path)
        assert outcome.verified, (inst["instance_id"], outcome.reason)
        assert outcome.total_time == inst["optimal_total_time"], inst["instance_id"]
        verified += 1
    assert verified == len(instances_n8)
