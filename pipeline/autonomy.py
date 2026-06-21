"""The autonomy decision: commit or escalate.

This is the last gate. It returns the route only when the feasibility gate passed.
On a failure it reports which stop cannot be made and why, rather than handing
back a route that breaks a window. The consequence setting controls how strict
the gate is about the self-review signal.

The feasibility gate is the boundary of autonomy. A route the verifier rejects is
never delivered.
"""
from __future__ import annotations

from dataclasses import dataclass

from ontime.pipeline.guard import GuardDecision
from ontime.pipeline.self_review import ReviewResult
from ontime.pipeline.verifier import VerifyResult


@dataclass
class AutonomyDecision:
    deliver: bool
    reason: str


def decide(
    verify_result: VerifyResult,
    guard: GuardDecision,
    review: ReviewResult | None = None,
    *,
    consequence: str = "normal",
) -> AutonomyDecision:
    """Decide whether to deliver the route or escalate.

    consequence is "normal" or "high". On high consequence a self-review
    disagreement blocks delivery even when the gate passed, because the input was
    read in a way that is not stable. On normal consequence the gate alone decides.
    """
    if not guard.in_scope:
        return AutonomyDecision(False, guard.reason)
    if not verify_result.correct:
        return AutonomyDecision(False, verify_result.reason)
    if consequence == "high" and review is not None and not review.agrees:
        return AutonomyDecision(False, "self-review disagreed; escalating for a second look")
    return AutonomyDecision(True, "")
