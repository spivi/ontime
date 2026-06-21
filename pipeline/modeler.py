"""The modeler: a request becomes a ProblemSpec.

This is the measured core. The structured path maps fields straight into a spec
with no model involved. The natural-language path hands a built prompt to the
model, which reads the stops, windows, service times, start, and speed or cost
matrix into the solve_tsp_tw call, then turns the extracted parameters into a
spec.

The prompt text lives in versioned files under prompts/, so the wording can be
audited and changed without touching code.
"""
from __future__ import annotations

import json
from pathlib import Path

from ontime.pipeline.model_client import ModelResult, OpenAICompatibleClient
from ontime.pipeline.solve_tsp_tw import spec_from_tool_input
from ontime.pipeline.spec import ProblemSpec


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def _service_time(value) -> int | list[int]:
    """Carry a service time through as a single value or one value per stop."""
    if isinstance(value, list):
        return [int(v) for v in value]
    return int(value)


def model_structured(parsed: dict) -> ProblemSpec:
    """Build a spec from a parsed structured request. No model is used."""
    if parsed["kind"] == "driving":
        coords = parsed["coordinates"]
        return ProblemSpec(
            kind="driving",
            n=len(coords),
            service_time=_service_time(parsed["service_time"]),
            time_windows=[list(w) for w in parsed["time_windows"]],
            coordinates=coords,
            speed=float(parsed["speed"]),
        )
    matrix = parsed["cost_matrix"]
    return ProblemSpec(
        kind="machine_scheduling",
        n=len(matrix),
        service_time=_service_time(parsed["service_time"]),
        time_windows=[list(w) for w in parsed["time_windows"]],
        cost_matrix=matrix,
    )


def build_driving_prompt(parsed: dict) -> str:
    coords = parsed["coordinates"]
    windows = parsed["time_windows"]
    n = len(coords)
    coord_lines = "\n".join(f"  stop {i}: ({c[0]:.2f}, {c[1]:.2f})" for i, c in enumerate(coords))
    window_lines = "\n".join(f"  stop {i}: [{w[0]}, {w[1]}]" for i, w in enumerate(windows))
    return load_prompt("modeler_driving.v1.txt").format(
        n=n,
        n_minus_1=n - 1,
        coord_lines=coord_lines,
        window_lines=window_lines,
        service_time=parsed["service_time"],
        speed=parsed["speed"],
    )


def build_machine_scheduling_prompt(parsed: dict) -> str:
    matrix = parsed["cost_matrix"]
    windows = parsed["time_windows"]
    n = len(matrix)
    matrix_lines = "\n".join("  " + json.dumps(row) for row in matrix)
    window_lines = "\n".join(f"  entry {i}: [{w[0]}, {w[1]}]" for i, w in enumerate(windows))
    return load_prompt("modeler_machine_scheduling.v1.txt").format(
        n=n,
        n_minus_1=n - 1,
        matrix_lines=matrix_lines,
        window_lines=window_lines,
        service_time=parsed["service_time"],
    )


def build_prompt(parsed: dict) -> str:
    if parsed["kind"] == "machine_scheduling":
        return build_machine_scheduling_prompt(parsed)
    return build_driving_prompt(parsed)


def model_natural_language(
    text: str,
    *,
    model_client: OpenAICompatibleClient,
    parsed_hint: dict | None = None,
) -> tuple[ProblemSpec | None, ModelResult]:
    """Read a request into a spec using the model.

    parsed_hint carries a structured rendering of the same problem when the caller
    has one, which is how the prompt presents the numbers. The returned spec is
    built from the parameters the model extracted, or None when the model refused
    or extracted nothing usable. The ModelResult carries extracted_params for the
    oracle and the self-review.
    """
    prompt = build_prompt(parsed_hint) if parsed_hint is not None else text
    result = model_client.run(prompt)
    if result.extracted_params is None:
        return None, result
    try:
        spec = spec_from_tool_input(result.extracted_params)
    except (KeyError, TypeError, ValueError):
        return None, result
    return spec, result
