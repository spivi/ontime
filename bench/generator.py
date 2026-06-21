"""The benchmark instance generator.

It builds TSP-with-time-windows instances at a chosen size. Windows are widened
around the arrival times of an unconstrained tour so that feasible orderings
exist, which keeps the problems solvable while still constraining the order.
Ground truth comes from Held-Karp, cross-checked against OR-Tools. An instance is
kept only when both solvers agree it is feasible.

The shipped instances under bench/instances were produced by this generator. The
seeds are fixed, so re-running it reproduces them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ontime.bench import held_karp
from ontime.bench import ortools_solver


SERVICE_TIME = 10
SPEED = 50.0
DEPOT_WINDOW = (0, 1000)
WINDOW_PAD_LO = 30
WINDOW_PAD_HI = 60
N_INSTANCES = 20
ORTOOLS_TIME_LIMIT_S = 60

SEEDS = {8: 4242, 12: 1234, 16: 5678, 20: 9101}

INSTANCES_DIR = Path(__file__).resolve().parent / "instances"


def instances_path(n: int) -> Path:
    return INSTANCES_DIR / f"instances_n{n}.jsonl"


def _build_baseline_unconstrained(coords: list[list[float]], n: int) -> dict[str, Any]:
    open_windows = [(0, 10**6) for _ in range(n)]
    r = held_karp.solve(coords, SERVICE_TIME, open_windows, speed=SPEED)
    assert r["status"] == "optimal", f"baseline unconstrained TSP failed: {r}"
    return r


def generate_instance(rng: np.random.Generator, idx: int, n: int) -> dict[str, Any]:
    coords = rng.uniform(0, 100, size=(n, 2)).tolist()
    baseline = _build_baseline_unconstrained(coords, n)
    tour, arrivals = baseline["tour"], baseline["arrivals"]

    windows: list[tuple[int, int]] = [(-1, -1)] * n
    windows[0] = DEPOT_WINDOW
    for pos in range(n):
        node = tour[pos]
        if node == 0:
            continue
        t = arrivals[pos]
        lo_pad = int(rng.integers(WINDOW_PAD_LO, WINDOW_PAD_HI + 1))
        hi_pad = int(rng.integers(WINDOW_PAD_LO, WINDOW_PAD_HI + 1))
        windows[node] = (max(0, t - lo_pad), t + hi_pad)

    for c, w in enumerate(windows):
        assert w != (-1, -1), f"instance {idx}: stop {c} missing a window"

    gt = held_karp.solve(coords, SERVICE_TIME, [list(w) for w in windows], speed=SPEED)
    assert gt["status"] == "optimal", (
        f"n{n} instance {idx} infeasible with constructed windows: {gt}"
    )

    return {
        "instance_id": f"tsp_tw_n{n}_{idx:03d}",
        "subproblem": "tsp_tw",
        "n_cities": n,
        "coordinates": coords,
        "service_time": SERVICE_TIME,
        "speed": SPEED,
        "time_windows": [list(w) for w in windows],
        "optimal_tour": gt["tour"],
        "optimal_total_time": int(gt["total_time"]),
    }


def generate_instances(n: int) -> list[dict[str, Any]]:
    """Generate N_INSTANCES that both Held-Karp and OR-Tools solve."""
    rng = np.random.default_rng(SEEDS[n])
    instances: list[dict[str, Any]] = []
    attempts = 0
    idx = 0
    while len(instances) < N_INSTANCES:
        attempts += 1
        if attempts > N_INSTANCES * 20:
            raise RuntimeError(
                f"could not generate {N_INSTANCES} solvable instances at n={n} "
                f"after {attempts} attempts"
            )
        inst = generate_instance(rng, idx, n)
        ot = ortools_solver.solve(
            inst["coordinates"], inst["service_time"], inst["time_windows"],
            inst["speed"], time_limit_s=ORTOOLS_TIME_LIMIT_S, solution_limit=200,
        )
        if ot["status"] != "optimal":
            idx += 1
            continue
        inst["instance_id"] = f"tsp_tw_n{n}_{len(instances):03d}"
        instances.append(inst)
        idx += 1
    return instances


def load_instances(n: int) -> list[dict[str, Any]]:
    """Load instances for size n, generating and saving them if absent."""
    path = instances_path(n)
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    instances = generate_instances(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")
    return instances
