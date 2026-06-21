"""Per-stop service times survive into the spec and the arrival walk.

A driving problem can give each stop its own service time. The parser, the spec,
the solver, and the verifier all have to carry the per-stop value, not collapse
the stops onto one shared number.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ontime.modes import structured
from ontime.pipeline import modeler
from ontime.pipeline.solve_tsp_tw import solve
from ontime.pipeline.spec import ProblemSpec
from ontime.pipeline.verifier import verify


def _write_csv(path: Path, services: list[int]) -> None:
    coords = [[0, 0], [10, 0], [20, 0], [30, 0]]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x", "y", "open", "close", "service", "speed"])
        for i, (c, s) in enumerate(zip(coords, services)):
            writer.writerow([f"stop{i}", c[0], c[1], 0, 1000, s, 1.0])


def test_distinct_service_times_survive_into_spec(tmp_path):
    services = [0, 5, 1, 9]
    csv_path = tmp_path / "stops.csv"
    _write_csv(csv_path, services)

    parsed = structured.parse(csv_path)
    assert parsed["service_time"] == [0, 5, 1, 9]

    spec = modeler.model_structured(parsed)
    assert spec.service_time == [0, 5, 1, 9]
    assert [spec.service_for(i) for i in range(4)] == [0, 5, 1, 9]


def test_distinct_service_times_survive_into_the_walk(tmp_path):
    services = [0, 5, 1, 9]
    csv_path = tmp_path / "stops.csv"
    _write_csv(csv_path, services)
    spec = modeler.model_structured(structured.parse(csv_path))

    # For the in-order tour the travel is 10 + 10 + 10 + 30 = 60, and the service
    # charged on departing stops 1, 2, 3 is 5 + 1 + 9 = 15, so the walk is 75. A
    # single collapsed service time would not produce this.
    result = verify([0, 1, 2, 3, 0], spec)
    assert result.correct
    assert result.objective == 75


def test_solver_and_verifier_agree_with_per_stop_service():
    spec = ProblemSpec(
        kind="driving",
        n=4,
        service_time=[0, 5, 1, 9],
        time_windows=[[0, 1000]] * 4,
        coordinates=[[0, 0], [10, 0], [20, 0], [30, 0]],
        speed=1.0,
    )
    result = solve(spec)
    v = verify(result["tour"], spec)
    assert v.correct
    assert v.objective == result["total_time"]


def test_machine_scheduling_processing_times_are_per_stop(tmp_path):
    jobs = tmp_path / "jobs.csv"
    with jobs.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["job", "release", "due", "processing"])
        writer.writerow(["start", 0, 1000, 0])
        writer.writerow(["a", 0, 500, 4])
        writer.writerow(["b", 0, 500, 7])
        writer.writerow(["c", 0, 500, 2])
    (tmp_path / "changeover.csv").write_text(
        "0,3,3,3\n3,0,3,3\n3,3,0,3\n3,3,3,0\n"
    )
    parsed = structured.parse(jobs)
    assert parsed["service_time"] == [0, 4, 7, 2]
    spec = modeler.model_structured(parsed)
    assert [spec.service_for(i) for i in range(4)] == [0, 4, 7, 2]
