"""The feasibility gate.

Given a tour and the problem spec, this checks the tour against the real windows
and returns its true completion time. It recomputes the travel matrix from the
same cost provider the solver used, so it never trusts a matrix handed to it. A
tour that passes here is feasible under the real windows.

This is a checker, never a solver. It never produces or repairs a tour. On a
window violation it names the stop that cannot be met, which is what an
infeasibility report shows the user.
"""
from __future__ import annotations

from dataclasses import dataclass

from ontime.pipeline.spec import CostProvider, ProblemSpec, provider_for


@dataclass
class VerifyResult:
    correct: bool
    reason: str
    objective: int | None
    late_stop: int | None = None


def verify(
    tour: list[int],
    spec: ProblemSpec,
    *,
    cost_provider: CostProvider | None = None,
) -> VerifyResult:
    """Check a tour against the spec and return its true completion time."""
    provider = cost_provider if cost_provider is not None else provider_for(spec)
    matrix = provider.matrix(spec)
    n = spec.n
    windows = spec.time_windows
    service_time = int(spec.service_time)

    if not isinstance(tour, list) or any(not isinstance(c, int) for c in tour):
        return VerifyResult(False, "tour must be a list of integers", None)
    if len(tour) != n + 1:
        return VerifyResult(
            False, f"tour has {len(tour)} stops; expected {n + 1} including the return", None
        )
    if tour[0] != 0:
        return VerifyResult(False, f"tour must start at stop 0, starts at {tour[0]}", None)
    if tour[-1] != 0:
        return VerifyResult(False, f"tour must end at stop 0, ends at {tour[-1]}", None)
    visited = set(tour[:-1])
    if visited != set(range(n)):
        missing = sorted(set(range(n)) - visited)
        extra = sorted(visited - set(range(n)))
        return VerifyResult(
            False,
            f"tour must visit each stop once; missing={missing}, extra={extra}",
            None,
        )

    t = 0.0
    for k in range(1, len(tour)):
        prev, curr = tour[k - 1], tour[k]
        srv = service_time if prev != 0 else 0
        t = t + matrix[prev][curr] + srv
        lo, hi = windows[curr]
        if t > hi:
            return VerifyResult(
                False,
                f"arrives at stop {curr} at time {t:.0f}, after the window closes at {hi}",
                None,
                late_stop=curr,
            )
        if t < lo:
            t = float(lo)

    return VerifyResult(True, "", int(round(t)))


def diagnose_infeasibility(
    spec: ProblemSpec,
    *,
    cost_provider: CostProvider | None = None,
) -> str:
    """Name a stop whose window cannot be met, for an infeasibility report.

    When the solver finds no route, this looks for a stop that cannot be reached
    in time even on the fastest possible approach. It reports the earliest such
    stop it finds, or a general message when no single stop is provably the cause.
    """
    provider = cost_provider if cost_provider is not None else provider_for(spec)
    matrix = provider.matrix(spec)
    service_time = int(spec.service_time)

    worst_stop = None
    worst_slack = None
    for stop in range(1, spec.n):
        lo, hi = spec.time_windows[stop]
        earliest_direct = matrix[0][stop]  # straight from the start, no detour
        slack = hi - earliest_direct
        if slack < 0:
            return (
                f"stop {stop} cannot be met: the fastest arrival is {earliest_direct}, "
                f"after its window closes at {hi}"
            )
        if worst_slack is None or slack < worst_slack:
            worst_slack = slack
            worst_stop = stop

    if worst_stop is not None:
        lo, hi = spec.time_windows[worst_stop]
        return (
            f"no route meets every window; stop {worst_stop} is the tightest, with a "
            f"window that closes at {hi} and a fastest arrival of {matrix[0][worst_stop]} "
            "before any other stop is served"
        )
    return "no route meets every window"
