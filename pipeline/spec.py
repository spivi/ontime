"""The canonical problem representation every layer reads.

A ProblemSpec holds one stops-with-windows problem in the single form the solver
and the verifier both consume. It carries exactly one cost representation: either
coordinates plus a speed (driving), or a direct integer cost matrix (machine
scheduling, where the changeover matrix is the cost matrix). The provider_for
helper turns a spec into the cost provider that both the solver and the gate read,
so they always agree on travel times.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ProblemSpec:
    """One TSP-with-time-windows problem.

    For driving, set coordinates and speed and leave cost_matrix as None. For
    machine scheduling, set cost_matrix (the changeover matrix) and leave
    coordinates and speed as None. service_time is the per-stop service or
    processing time. time_windows are [open, close] for driving and
    [release, due] for machine scheduling. Index 0 is the depot or the start.
    """

    kind: str
    n: int
    service_time: int
    time_windows: list[list[int]]
    coordinates: list[list[float]] | None = None
    speed: float | None = None
    cost_matrix: list[list[int]] | None = None

    def __post_init__(self) -> None:
        has_geo = self.coordinates is not None and self.speed is not None
        has_matrix = self.cost_matrix is not None
        if has_geo == has_matrix:
            raise ValueError(
                "spec must carry exactly one of (coordinates and speed) or cost_matrix"
            )
        if len(self.time_windows) != self.n:
            raise ValueError(
                f"time_windows has {len(self.time_windows)} entries for n={self.n}"
            )
        if has_geo and len(self.coordinates) != self.n:
            raise ValueError(
                f"coordinates has {len(self.coordinates)} entries for n={self.n}"
            )
        if has_matrix:
            if len(self.cost_matrix) != self.n or any(
                len(row) != self.n for row in self.cost_matrix
            ):
                raise ValueError("cost_matrix must be square with side n")


class CostProvider(Protocol):
    """A pure function from a spec to its integer travel-time matrix."""

    def matrix(self, spec: ProblemSpec) -> list[list[int]]: ...

    def is_symmetric(self) -> bool: ...


class EuclideanProvider:
    """round(euclidean_distance / speed). Matches the carried benchmark exactly."""

    def matrix(self, spec: ProblemSpec) -> list[list[int]]:
        coords = spec.coordinates
        speed = float(spec.speed)
        n = spec.n
        m = [[0] * n for _ in range(n)]
        for i in range(n):
            xi, yi = coords[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = coords[j]
                m[i][j] = int(round(math.hypot(xi - xj, yi - yj) / speed))
        return m

    def is_symmetric(self) -> bool:
        return True


class HaversineProvider:
    """round(great_circle_distance / speed) for real lat/lon stops.

    Coordinates are [lat, lon] in degrees and speed is in distance units per time
    unit on the same scale as the radius. The default radius is kilometres, so a
    speed given in kilometres per minute yields travel times in minutes.
    """

    def __init__(self, radius: float = 6371.0) -> None:
        self.radius = radius

    def matrix(self, spec: ProblemSpec) -> list[list[int]]:
        coords = spec.coordinates
        speed = float(spec.speed)
        n = spec.n
        m = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                m[i][j] = int(round(self._distance(coords[i], coords[j]) / speed))
        return m

    def _distance(self, a: list[float], b: list[float]) -> float:
        lat1, lon1 = math.radians(a[0]), math.radians(a[1])
        lat2, lon2 = math.radians(b[0]), math.radians(b[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * self.radius * math.asin(math.sqrt(h))

    def is_symmetric(self) -> bool:
        return True


class DirectMatrixProvider:
    """Returns the spec's cost matrix verbatim. No geography is involved."""

    def matrix(self, spec: ProblemSpec) -> list[list[int]]:
        return [[int(v) for v in row] for row in spec.cost_matrix]

    def is_symmetric(self) -> bool:
        m = self.matrix
        return False  # changeover matrices are allowed to be asymmetric


def provider_for(spec: ProblemSpec, *, geo: str = "euclidean") -> CostProvider:
    """Pick the provider implied by the spec.

    A spec with a cost matrix always uses the direct provider. A spec with
    coordinates uses Euclidean by default, or Haversine when geo is "haversine".
    """
    if spec.cost_matrix is not None:
        return DirectMatrixProvider()
    if geo == "haversine":
        return HaversineProvider()
    return EuclideanProvider()
