"""The solver never returns a route its own gate rejects.

Feasibility is the load-bearing claim. Whatever route the engine returns, the
verifier confirms it against the real windows, and the objective the verifier
reports matches the engine's. Optimality is checked separately against the exact
Held-Karp ground truth, since the production solver is a heuristic and the
benchmark's ground truth comes from Held-Karp.
"""
from __future__ import annotations

from ontime.bench.grade import spec_from_instance
from ontime.pipeline.solve_tsp_tw import solve
from ontime.pipeline.verifier import verify


def test_engine_routes_pass_the_gate(instances_n8):
    for inst in instances_n8:
        spec = spec_from_instance(inst)
        result = solve(spec)
        assert result["status"] == "optimal", inst["instance_id"]
        v = verify(result["tour"], spec)
        assert v.correct, (inst["instance_id"], v.reason)
        assert v.objective == result["total_time"], inst["instance_id"]


def test_engine_reaches_optimum_on_small_instances(instances_n8):
    # At this size the engine reaches the Held-Karp optimum with the fast defaults.
    for inst in instances_n8:
        spec = spec_from_instance(inst)
        result = solve(spec)
        assert result["total_time"] == inst["optimal_total_time"], inst["instance_id"]
