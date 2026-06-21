"""Grading for the benchmark.

A graded instance carries its own ground truth: the optimal completion time found
by Held-Karp and cross-checked against OR-Tools. Grading a proposed tour means
checking it against the real windows with the verifier and comparing its
completion time to the stored optimum.

The trivial-copy control is the baseline where the numbers are handed to the model
already shaped, so the only thing under test is the tool call itself. When the
control does not reach the optimum on every instance, the harness itself is
suspect, not the model.
"""
from __future__ import annotations

from dataclasses import dataclass

from ontime.bench import held_karp, ortools_solver
from ontime.pipeline.spec import ProblemSpec, provider_for
from ontime.pipeline.verifier import VerifyResult, verify


def spec_from_instance(inst: dict) -> ProblemSpec:
    """Build a driving ProblemSpec from a carried benchmark instance."""
    return ProblemSpec(
        kind="driving",
        n=inst["n_cities"],
        service_time=inst["service_time"],
        time_windows=[list(w) for w in inst["time_windows"]],
        coordinates=inst["coordinates"],
        speed=inst["speed"],
    )


@dataclass
class GradeResult:
    feasible: bool
    optimal: bool
    objective: int | None
    optimum: int
    reason: str


def grade_instance(inst: dict, tour: list[int]) -> GradeResult:
    """Grade a proposed tour against a benchmark instance."""
    spec = spec_from_instance(inst)
    optimum = int(inst["optimal_total_time"])
    v: VerifyResult = verify(tour, spec)
    if not v.correct:
        return GradeResult(False, False, None, optimum, v.reason)
    return GradeResult(True, v.objective == optimum, v.objective, optimum, "")


def cross_check(inst: dict, time_limit_s: int = 10, solution_limit: int | None = 200) -> dict:
    """Confirm the ground truth holds on one instance.

    Held-Karp is exact, so it must equal the stored optimum. OR-Tools is a
    heuristic, so it must return a feasible tour whose completion time is at least
    the optimum. The stored optimum comes from Held-Karp run at generation time.
    """
    spec = spec_from_instance(inst)
    matrix = provider_for(spec).matrix(spec)
    stored = int(inst["optimal_total_time"])
    hk = held_karp.solve(None, spec.service_time, spec.time_windows, time_matrix=matrix)
    ort = ortools_solver.solve(
        None, spec.service_time, spec.time_windows, time_matrix=matrix,
        time_limit_s=time_limit_s, solution_limit=solution_limit,
    )
    ort_feasible = ort["status"] == "optimal" and ort["total_time"] >= stored
    return {
        "instance_id": inst["instance_id"],
        "stored": stored,
        "held_karp": hk["total_time"],
        "ortools": ort["total_time"],
        "held_karp_exact": hk["total_time"] == stored,
        "ortools_feasible": ort_feasible,
    }
