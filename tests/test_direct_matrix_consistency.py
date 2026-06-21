"""The gate stays sound when there is no geography.

A machine scheduling spec carries a direct cost matrix. The solver and the
verifier read the same matrix through the same provider, so a route the solver
returns passes the gate, and changing the matrix changes the gate's objective in
step with the solver.
"""
from __future__ import annotations

from ontime.pipeline.solve_tsp_tw import solve
from ontime.pipeline.spec import ProblemSpec
from ontime.pipeline.verifier import verify


def _spec(cost):
    return ProblemSpec(
        kind="machine_scheduling",
        n=4,
        service_time=3,
        time_windows=[[0, 1000], [0, 60], [0, 80], [0, 90]],
        cost_matrix=cost,
    )


def test_machine_scheduling_route_passes_gate():
    cost = [
        [0, 5, 9, 7],
        [6, 0, 4, 8],
        [3, 5, 0, 6],
        [7, 2, 5, 0],
    ]
    spec = _spec(cost)
    result = solve(spec)
    assert result["status"] == "optimal"
    v = verify(result["tour"], spec)
    assert v.correct
    assert v.objective == result["total_time"]


def test_changing_the_matrix_moves_the_gate_objective():
    base = [
        [0, 5, 9, 7],
        [6, 0, 4, 8],
        [3, 5, 0, 6],
        [7, 2, 5, 0],
    ]
    inflated = [[v * 2 for v in row] for row in base]
    spec_base = _spec(base)
    spec_inflated = _spec(inflated)
    tour = solve(spec_base)["tour"]
    obj_base = verify(tour, spec_base).objective
    obj_inflated = verify(tour, spec_inflated).objective
    assert obj_inflated is not None
    assert obj_inflated > obj_base
