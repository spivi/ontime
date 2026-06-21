"""Exact TSP-TW solver via Held-Karp dynamic programming.

Provably optimal for the completion-time objective the verifier checks. State is
(S, i) where S is the bitmask of non-depot stops visited so far and i is the
current stop. The DP value is the earliest feasible arrival time at i after
visiting exactly the stops in S, starting from the depot at time 0.

Time complexity is O(n^2 * 2^n), which stays fast through about n=18. It depends
on no external solver, so the bench uses it as an independent ground truth that
OR-Tools is cross-checked against.

Travel times come from a precomputed integer matrix when one is supplied, or from
round(euclidean_distance / speed) otherwise. The two paths agree by construction
when the matrix is the Euclidean matrix.
"""
from __future__ import annotations

import math
from typing import Any


def _travel_time(coords: list[list[float]], speed: float, i: int, j: int) -> int:
    return round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) / speed)


def solve(
    coords: list[list[float]] | None,
    service_time: int,
    time_windows: list[tuple[int, int] | list[int]],
    speed: float = 50.0,
    time_matrix: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Return a dict with status, tour, arrivals, total_time.

    total_time is the completion time back at the depot from t=0, including any
    waiting for windows to open and service time at non-depot stops. status is
    "optimal" on success or "infeasible" when no window-respecting tour exists.
    """
    if time_matrix is not None:
        n = len(time_matrix)
        travel = [[int(time_matrix[i][j]) for j in range(n)] for i in range(n)]
    else:
        n = len(coords)
        travel = [[_travel_time(coords, speed, i, j) for j in range(n)] for i in range(n)]

    if n == 0:
        return {"status": "infeasible", "tour": None, "arrivals": None, "total_time": None}
    windows = [(int(w[0]), int(w[1])) for w in time_windows]

    INF = math.inf
    n_non = n - 1
    full_mask = (1 << n_non) - 1

    dp: list[list[float]] = [[INF] * n for _ in range(1 << n_non)]
    parent: list[list[int]] = [[-1] * n for _ in range(1 << n_non)]

    for i in range(1, n):
        arrival = max(travel[0][i], windows[i][0])
        if arrival <= windows[i][1]:
            mask = 1 << (i - 1)
            dp[mask][i] = arrival
            parent[mask][i] = 0

    for mask in range(1, 1 << n_non):
        for i in range(1, n):
            bit_i = 1 << (i - 1)
            if not (mask & bit_i):
                continue
            if dp[mask][i] == INF:
                continue
            arrival_i = dp[mask][i]
            for j in range(1, n):
                bit_j = 1 << (j - 1)
                if mask & bit_j:
                    continue
                arrival_j_raw = arrival_i + service_time + travel[i][j]
                arrival_j = max(arrival_j_raw, windows[j][0])
                if arrival_j > windows[j][1]:
                    continue
                new_mask = mask | bit_j
                if arrival_j < dp[new_mask][j]:
                    dp[new_mask][j] = arrival_j
                    parent[new_mask][j] = i

    best_total = INF
    best_last = -1
    for i in range(1, n):
        if dp[full_mask][i] == INF:
            continue
        completion = dp[full_mask][i] + service_time + travel[i][0]
        if completion < windows[0][0]:
            completion = windows[0][0]
        if completion > windows[0][1]:
            continue
        if completion < best_total:
            best_total = completion
            best_last = i

    if best_last == -1:
        return {"status": "infeasible", "tour": None, "arrivals": None, "total_time": None}

    rev: list[int] = [0]
    cur_city = best_last
    cur_mask = full_mask
    while cur_city != 0:
        rev.append(cur_city)
        prev = parent[cur_mask][cur_city]
        cur_mask = cur_mask & ~(1 << (cur_city - 1))
        cur_city = prev
    rev.append(0)
    tour = list(reversed(rev))

    arrivals: list[int] = [0]
    t = 0.0
    for k in range(1, len(tour)):
        prev, curr = tour[k - 1], tour[k]
        srv = service_time if prev != 0 else 0
        t = t + srv + travel[prev][curr]
        lo, hi = windows[curr]
        if t < lo:
            t = float(lo)
        arrivals.append(int(round(t)))

    return {
        "status": "optimal",
        "tour": tour,
        "arrivals": arrivals,
        "total_time": int(round(t)),
    }
