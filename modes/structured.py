"""Structured input: CSV or JSON into a parsed request.

This is the reliable path. The model barely has to work here, because the numbers
arrive already shaped. A driving CSV lists stops with coordinates and windows. A
machine scheduling CSV lists jobs with release, due, and processing times, paired
with a changeover matrix. JSON carries the same fields under explicit keys.

The parsed dict it returns is what the modeler maps into a ProblemSpec.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def detect(path: Path) -> bool:
    """Return True when the path looks like a structured input file."""
    return path.suffix.lower() in {".csv", ".json"}


def parse(path: Path) -> dict:
    """Parse a structured request file into a parsed dict."""
    if path.suffix.lower() == ".json":
        return _parse_json(path)
    return _parse_csv(path)


def _parse_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    kind = data.get("kind", "driving")
    if kind == "machine_scheduling":
        return {
            "kind": "machine_scheduling",
            "cost_matrix": [[int(v) for v in row] for row in data["cost_matrix"]],
            "time_windows": [[int(w[0]), int(w[1])] for w in data["time_windows"]],
            "service_time": _read_service_time(data["service_time"]),
        }
    return {
        "kind": "driving",
        "coordinates": [[float(c[0]), float(c[1])] for c in data["coordinates"]],
        "time_windows": [[int(w[0]), int(w[1])] for w in data["time_windows"]],
        "service_time": _read_service_time(data["service_time"]),
        "speed": float(data["speed"]),
    }


def _read_service_time(value) -> int | list[int]:
    """Read a service time that may be a single value or one value per stop."""
    if isinstance(value, list):
        out = [int(v) for v in value]
        out[0] = 0  # the start has no service time
        return out
    return int(value)


def _parse_csv(path: Path) -> dict:
    rows = list(csv.DictReader(path.read_text().splitlines()))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    header = set(rows[0].keys())
    if {"release", "due", "processing"} <= header:
        return _parse_machine_scheduling_csv(path, rows)
    return _parse_driving_csv(rows)


def _parse_driving_csv(rows: list[dict]) -> dict:
    coordinates: list[list[float]] = []
    windows: list[list[int]] = []
    service_times: list[int] = []
    for r in rows:
        x = float(r.get("lat", r.get("x")))
        y = float(r.get("lon", r.get("y")))
        coordinates.append([x, y])
        windows.append([int(r["open"]), int(r["close"])])
        service_times.append(int(r.get("service", 0)))
    service_times[0] = 0  # the start has no service time
    speed = float(rows[0].get("speed", 50.0))
    return {
        "kind": "driving",
        "coordinates": coordinates,
        "time_windows": windows,
        "service_time": service_times,
        "speed": speed,
    }


def _parse_machine_scheduling_csv(path: Path, rows: list[dict]) -> dict:
    windows: list[list[int]] = []
    processing_times: list[int] = []
    for r in rows:
        windows.append([int(r["release"]), int(r["due"])])
        processing_times.append(int(r["processing"]))
    processing_times[0] = 0  # the start state has no processing time

    matrix_path = path.with_name("changeover.csv")
    if not matrix_path.exists():
        raise ValueError(f"machine scheduling input needs a changeover matrix at {matrix_path}")
    matrix = [
        [int(v) for v in line.split(",")]
        for line in matrix_path.read_text().splitlines()
        if line.strip()
    ]
    return {
        "kind": "machine_scheduling",
        "cost_matrix": matrix,
        "time_windows": windows,
        "service_time": processing_times,
    }
