"""The in-scope guard.

ontime does one thing, so the guard is a narrow gate rather than a general
classifier. It rejects a request that is not a stops-with-windows problem before
any solving happens. On the structured path it reads the shape of the parsed
input. On the natural-language path it can ask the model, and falls back to a
keyword check when no model is available.

This is scaffolding. It makes the core safe on messy input. It is not part of the
measured spine.
"""
from __future__ import annotations

from dataclasses import dataclass


DRIVING_HINTS = (
    "stop", "window", "arrive", "deliver", "errand", "address", "depot", "visit",
    "route", "pick up", "drop off", "by 5pm", "by noon", "before",
)
SCHEDULING_HINTS = (
    "job", "machine", "changeover", "release", "due", "process", "setup", "schedule",
)


@dataclass
class GuardDecision:
    in_scope: bool
    problem_kind: str  # "driving" | "machine_scheduling" | "unknown"
    reason: str


def classify_structured(parsed: dict) -> GuardDecision:
    """Decide scope from a parsed structured request."""
    kind = parsed.get("kind")
    if kind == "driving" and parsed.get("coordinates") and parsed.get("time_windows"):
        return GuardDecision(True, "driving", "structured stops with windows")
    if kind == "machine_scheduling" and parsed.get("cost_matrix") and parsed.get("time_windows"):
        return GuardDecision(True, "machine_scheduling", "structured jobs with release and due times")
    return GuardDecision(False, "unknown", "structured input is missing stops or windows")


def classify_text(text: str) -> GuardDecision:
    """Decide scope from raw text using keyword hints.

    This is the fallback when no model client is available. One clear hint is
    enough to let a request through, so a short valid request is not turned away.
    The modeler does the real reading; the guard only keeps plainly off-topic text
    out.
    """
    lowered = text.lower()
    driving = sum(1 for w in DRIVING_HINTS if w in lowered)
    scheduling = sum(1 for w in SCHEDULING_HINTS if w in lowered)
    if scheduling > driving and scheduling >= 1:
        return GuardDecision(True, "machine_scheduling", "reads as one-machine scheduling")
    if driving >= 1:
        return GuardDecision(True, "driving", "reads as stops with arrival windows")
    return GuardDecision(False, "unknown", "the request does not read as a stops-with-windows problem")
