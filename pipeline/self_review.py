"""The self-review loop: a second independent read of the input.

After the modeler extracts parameters on the natural-language path, this runs the
extraction again and compares the two reads. When they disagree, the input was
read in a way that is not stable, which is a signal worth acting on.

State it honestly. In the article this fired on about a third of natural-language
runs and rose with problem size, so it has full recall on the dangerous class but
it is noisy. It is off by default and switched on by consequence. It is the
input-side check, not the feasibility gate.
"""
from __future__ import annotations

from dataclasses import dataclass

from ontime.pipeline.model_client import OpenAICompatibleClient
from ontime.pipeline.modeler import model_natural_language
from ontime.pipeline.solve_tsp_tw import params_match


@dataclass
class ReviewResult:
    agrees: bool
    second_pass: dict | None
    reason: str


def review(
    text: str,
    first_pass: dict,
    *,
    model_client: OpenAICompatibleClient,
    parsed_hint: dict | None = None,
) -> ReviewResult:
    """Re-read the input and compare against the first pass."""
    _, second = model_natural_language(text, model_client=model_client, parsed_hint=parsed_hint)
    if second.extracted_params is None:
        return ReviewResult(False, None, "second read did not extract parameters")
    agrees = params_match(second.extracted_params, _as_reference(first_pass))
    reason = "" if agrees else "the two reads of the input disagree"
    return ReviewResult(agrees, second.extracted_params, reason)


def _as_reference(extracted: dict) -> dict:
    """Shape a first-pass extraction so params_match can compare against it."""
    ref = {
        "time_windows": extracted.get("time_windows", []),
        "service_time": extracted.get("service_time", -1),
    }
    if extracted.get("coordinates") is not None:
        ref["coordinates"] = extracted["coordinates"]
        ref["speed"] = extracted.get("speed", -1)
    else:
        ref["cost_matrix"] = extracted.get("cost_matrix", [])
    return ref
