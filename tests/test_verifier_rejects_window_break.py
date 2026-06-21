"""The feasibility gate rejects a route that breaks a window."""
from __future__ import annotations

from ontime.bench.grade import spec_from_instance
from ontime.pipeline.solve_tsp_tw import solve
from ontime.pipeline.verifier import verify


def _swap_two_middle_stops(tour: list[int]) -> list[int]:
    swapped = list(tour)
    if len(swapped) >= 5:
        swapped[1], swapped[3] = swapped[3], swapped[1]
    return swapped


def test_rejects_a_late_arrival(instances_n8):
    # Find an instance where swapping two stops forces a late arrival.
    rejected_any = False
    for inst in instances_n8:
        spec = spec_from_instance(inst)
        optimal = solve(spec)["tour"]
        broken = _swap_two_middle_stops(optimal)
        if broken == optimal:
            continue
        v = verify(broken, spec)
        if not v.correct:
            assert v.late_stop is not None
            assert str(v.late_stop) in v.reason
            rejected_any = True
            break
    assert rejected_any, "expected at least one swap to break a window"


def test_rejects_missing_stop(instances_n8):
    inst = instances_n8[0]
    spec = spec_from_instance(inst)
    optimal = solve(spec)["tour"]
    dropped = optimal[:2] + optimal[3:]  # remove one stop
    v = verify(dropped, spec)
    assert not v.correct


def test_rejects_no_return_to_depot(instances_n8):
    inst = instances_n8[0]
    spec = spec_from_instance(inst)
    optimal = solve(spec)["tour"]
    no_return = optimal[:-1]  # drop the closing depot
    v = verify(no_return, spec)
    assert not v.correct


def test_infeasibility_report_names_the_stop():
    from ontime.pipeline.spec import ProblemSpec
    from ontime.pipeline.verifier import diagnose_infeasibility

    # Stop 4 is far from the start and has a window that closes at time 1, so it
    # cannot be reached in time on any route.
    spec = ProblemSpec(
        kind="driving",
        n=5,
        service_time=3,
        time_windows=[[0, 600], [30, 90], [15, 40], [0, 30], [0, 1]],
        coordinates=[[0, 0], [2, 1], [5, 4], [1, 6], [4, 2]],
        speed=0.5,
    )
    reason = diagnose_infeasibility(spec)
    assert "stop 4" in reason
