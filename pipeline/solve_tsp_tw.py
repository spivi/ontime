"""The one solving engine, exposed as a tool the model can call.

It wraps OR-Tools. It accepts either coordinates plus a speed or a direct cost
matrix, builds the integer travel matrix once through a cost provider, and feeds
that matrix into the routing model. The same provider feeds the verifier, so the
route the engine returns is checked against the same travel times it was built on.

TOOL_SCHEMA is the tool definition handed to the model. execute_tool is the
function the model's tool call lands in.
"""
from __future__ import annotations

from typing import Any

from ontime.bench import ortools_solver
from ontime.pipeline.spec import CostProvider, ProblemSpec, provider_for


TOOL_SCHEMA = {
    "name": "solve_tsp_tw",
    "description": (
        "Solve a route through stops with time windows and return the order, the "
        "arrival time at each stop, and the completion time. Supply either "
        "coordinates with a speed, or a direct cost matrix, never both. All "
        "numeric values must be finite."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "coordinates": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "description": "List of [x, y] coordinates, one per stop. Stop 0 is the start.",
            },
            "cost_matrix": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "Square matrix of integer travel times between stops. Use this instead of coordinates when there is no map.",
            },
            "time_windows": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "description": "List of [open, close] windows, one per stop. Stop 0 is the start.",
            },
            "service_time": {
                "type": "integer",
                "description": "Time charged when departing each non-start stop.",
            },
            "speed": {
                "type": "number",
                "description": "Speed used to turn distance into travel time. Required with coordinates.",
            },
        },
        "required": ["time_windows", "service_time"],
    },
}


def solve(
    spec: ProblemSpec,
    *,
    cost_provider: CostProvider | None = None,
    time_limit_s: int = 10,
    solution_limit: int | None = 200,
) -> dict[str, Any]:
    """Solve a spec and return status, tour, arrivals, total_time, time_matrix.

    The defaults keep an interactive run fast. The local search stops after
    solution_limit improving solutions or time_limit_s seconds, whichever comes
    first, which returns in milliseconds on the sizes this tool targets. Raise
    both for a harder instance, or set solution_limit to None to run to the time
    budget.
    """
    provider = cost_provider if cost_provider is not None else provider_for(spec)
    matrix = provider.matrix(spec)
    return ortools_solver.solve(
        coords=None,
        service_time=spec.service_time,
        time_windows=spec.time_windows,
        time_matrix=matrix,
        time_limit_s=time_limit_s,
        solution_limit=solution_limit,
    )


def spec_from_tool_input(tool_input: dict[str, Any]) -> ProblemSpec:
    """Build a ProblemSpec from raw tool-call arguments.

    Raises ValueError when the arguments are missing, malformed, or carry both a
    coordinate set and a cost matrix.
    """
    windows = tool_input["time_windows"]
    service_time = int(tool_input["service_time"])
    has_coords = tool_input.get("coordinates") is not None
    has_matrix = tool_input.get("cost_matrix") is not None
    if has_coords == has_matrix:
        raise ValueError("supply either coordinates with speed or a cost matrix, not both")

    if has_coords:
        coords = tool_input["coordinates"]
        speed = float(tool_input["speed"])
        return ProblemSpec(
            kind="driving",
            n=len(coords),
            service_time=service_time,
            time_windows=[list(w) for w in windows],
            coordinates=coords,
            speed=speed,
        )

    matrix = tool_input["cost_matrix"]
    return ProblemSpec(
        kind="machine_scheduling",
        n=len(matrix),
        service_time=service_time,
        time_windows=[list(w) for w in windows],
        cost_matrix=matrix,
    )


def execute_tool(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Run a solve_tsp_tw tool call and return a result the model can read.

    Returns a success dict with tour, total_time, and arrivals, or an error dict
    with a message the model can use to correct its call.
    """
    try:
        spec = spec_from_tool_input(tool_input)
    except (KeyError, TypeError, ValueError) as exc:
        return {"error": f"invalid tool input: {exc}"}

    if len(tool_input["time_windows"]) != spec.n:
        return {
            "error": (
                f"the problem has {spec.n} stops but time_windows has "
                f"{len(tool_input['time_windows'])} entries; they must match"
            )
        }

    try:
        result = solve(spec)
    except Exception as exc:
        return {"error": f"solver error: {type(exc).__name__}: {exc}"}

    if result["status"] != "optimal":
        return {
            "error": (
                f"solver returned {result['status']}: {result.get('message', '')}. "
                "Check that the windows are consistent and every stop is reachable."
            )
        }

    return {
        "tour": result["tour"],
        "total_time": result["total_time"],
        "arrivals": result["arrivals"],
    }


def params_match(extracted: dict[str, Any], spec_dict: dict[str, Any]) -> bool:
    """Return True when extracted parameters match the reference exactly.

    Coordinates are compared with a small float tolerance, windows are integer
    exact. spec_dict may hold either coordinates or a cost matrix.
    """
    try:
        if spec_dict.get("coordinates") is not None:
            if not _coords_match(extracted.get("coordinates", []), spec_dict["coordinates"]):
                return False
            if abs(float(extracted.get("speed", -1)) - float(spec_dict["speed"])) >= 1e-6:
                return False
        else:
            if not _matrix_match(extracted.get("cost_matrix", []), spec_dict["cost_matrix"]):
                return False
        tw_ok = _windows_match(extracted.get("time_windows", []), spec_dict["time_windows"])
        svc_ok = int(extracted.get("service_time", -1)) == int(spec_dict["service_time"])
        return tw_ok and svc_ok
    except Exception:
        return False


def _coords_match(a: list, b: list, tol: float = 0.01) -> bool:
    if len(a) != len(b):
        return False
    return all(
        abs(float(ai[0]) - float(bi[0])) < tol and abs(float(ai[1]) - float(bi[1])) < tol
        for ai, bi in zip(a, b)
    )


def _windows_match(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    return all(int(ai[0]) == int(bi[0]) and int(ai[1]) == int(bi[1]) for ai, bi in zip(a, b))


def _matrix_match(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    return all(
        len(ar) == len(br) and all(int(x) == int(y) for x, y in zip(ar, br))
        for ar, br in zip(a, b)
    )
