"""Held-Karp and OR-Tools agree with the stored optimum."""
from __future__ import annotations

import pytest

from ontime.bench import held_karp
from ontime.bench.grade import cross_check, spec_from_instance
from ontime.pipeline.spec import provider_for


@pytest.mark.parametrize("size", [8, 12])
def test_held_karp_matches_stored(request, size):
    instances = request.getfixturevalue(f"instances_n{size}")
    for inst in instances:
        spec = spec_from_instance(inst)
        matrix = provider_for(spec).matrix(spec)
        hk = held_karp.solve(None, spec.service_time, spec.time_windows, time_matrix=matrix)
        assert hk["status"] == "optimal", inst["instance_id"]
        assert hk["total_time"] == inst["optimal_total_time"], inst["instance_id"]


def test_cross_check_agrees(instances_n8):
    for inst in instances_n8:
        result = cross_check(inst)
        assert result["held_karp_exact"], result
        assert result["ortools_feasible"], result
