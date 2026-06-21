"""The default cost source: straight-line distance times a speed.

This is the free default. It needs no map server and no network. For real lat/lon
stops it uses the great-circle distance. For the carried benchmark, where the
coordinates are plain points on a plane, it uses the planar distance so the
numbers match the recorded runs exactly.
"""
from __future__ import annotations

from ontime.pipeline.spec import EuclideanProvider, HaversineProvider, ProblemSpec


def cost_matrix(spec: ProblemSpec, *, radius: float = 6371.0) -> list[list[int]]:
    """Great-circle travel-time matrix for lat/lon coordinates."""
    return HaversineProvider(radius=radius).matrix(spec)


def euclidean_cost_matrix(spec: ProblemSpec) -> list[list[int]]:
    """Planar travel-time matrix. Used by the carried benchmark instances."""
    return EuclideanProvider().matrix(spec)
