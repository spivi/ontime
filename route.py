"""ontime: read a stops-with-windows problem and return a verified route.

Run it on a structured file or a text request:

    python route.py examples/driving/stops.csv
    python route.py examples/driving/day.txt
    python route.py examples/machine_scheduling/jobs.csv

The structured path needs no model. The text path reads the request with a
self-hosted model through an OpenAI-compatible endpoint, set through the
ONTIME_LLM_BASE_URL and ONTIME_LLM_MODEL environment variables.

The route that comes back has been checked against the real windows. On an
infeasible problem the report names the stop and the window that cannot be met.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running this file directly from inside the repo, so `import ontime`
# resolves whether or not the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ontime.modes import natural_language, structured
from ontime.pipeline import autonomy, guard, modeler, self_review, verifier
from ontime.pipeline.solve_tsp_tw import solve
from ontime.pipeline.spec import ProblemSpec, provider_for


@dataclass
class RouteOutcome:
    route: list[int] | None
    arrivals: list[int] | None
    total_time: int | None
    verified: bool
    reason: str
    mode: str
    refused: bool


def plan_route(
    request: str | Path,
    *,
    model_client=None,
    geo: str = "euclidean",
    consequence: str = "normal",
    run_self_review: bool = False,
    time_limit_s: int = 10,
    solution_limit: int | None = 200,
) -> RouteOutcome:
    """Run the pipeline on one request and return a verified route or a refusal.

    time_limit_s and solution_limit bound the solver. The defaults return fast on
    the sizes this tool targets. Raise them for a larger or harder instance.
    """
    path = Path(request) if not isinstance(request, Path) else request
    is_file = path.exists()

    if is_file and structured.detect(path):
        return _plan_structured(
            path, geo=geo, time_limit_s=time_limit_s, solution_limit=solution_limit
        )

    text = natural_language.read(path if is_file else request)
    return _plan_natural_language(
        text,
        model_client=model_client,
        geo=geo,
        consequence=consequence,
        run_self_review=run_self_review,
        time_limit_s=time_limit_s,
        solution_limit=solution_limit,
    )


def _plan_structured(
    path: Path, *, geo: str, time_limit_s: int, solution_limit: int | None
) -> RouteOutcome:
    parsed = structured.parse(path)
    g = guard.classify_structured(parsed)
    if not g.in_scope:
        return RouteOutcome(None, None, None, False, g.reason, "structured", refused=True)

    spec = modeler.model_structured(parsed)
    return _solve_and_gate(
        spec, g, review=None, mode="structured", geo=geo, consequence="normal",
        time_limit_s=time_limit_s, solution_limit=solution_limit,
    )


def _plan_natural_language(
    text: str,
    *,
    model_client,
    geo: str,
    consequence: str,
    run_self_review: bool,
    time_limit_s: int,
    solution_limit: int | None,
) -> RouteOutcome:
    g = guard.classify_text(text)
    if not g.in_scope:
        return RouteOutcome(None, None, None, False, g.reason, "natural_language", refused=True)
    if model_client is None:
        return RouteOutcome(
            None, None, None, False,
            "natural-language input needs a model client; set ONTIME_LLM_BASE_URL or pass a structured file",
            "natural_language", refused=True,
        )

    spec, result = modeler.model_natural_language(text, model_client=model_client)
    if spec is None:
        reason = result.reason or "the model did not return usable parameters"
        return RouteOutcome(None, None, None, False, reason, "natural_language", refused=True)

    review = None
    if run_self_review:
        review = self_review.review(text, result.extracted_params, model_client=model_client)

    return _solve_and_gate(
        spec, g, review=review, mode="natural_language", geo=geo, consequence=consequence,
        time_limit_s=time_limit_s, solution_limit=solution_limit,
    )


def _solve_and_gate(
    spec: ProblemSpec,
    g: guard.GuardDecision,
    *,
    review,
    mode: str,
    geo: str,
    consequence: str,
    time_limit_s: int,
    solution_limit: int | None,
) -> RouteOutcome:
    provider = provider_for(spec, geo=geo)
    result = solve(
        spec, cost_provider=provider, time_limit_s=time_limit_s, solution_limit=solution_limit
    )
    if result["status"] != "optimal":
        reason = verifier.diagnose_infeasibility(spec, cost_provider=provider)
        return RouteOutcome(None, None, None, False, reason, mode, refused=False)

    v = verifier.verify(result["tour"], spec, cost_provider=provider)
    decision = autonomy.decide(v, g, review, consequence=consequence)
    if not decision.deliver:
        return RouteOutcome(
            result["tour"] if v.correct else None,
            result["arrivals"] if v.correct else None,
            v.objective,
            False, decision.reason, mode, refused=not v.correct,
        )

    return RouteOutcome(
        result["tour"], result["arrivals"], result["total_time"],
        True, "", mode, refused=False,
    )


def _print_outcome(outcome: RouteOutcome) -> None:
    if outcome.verified:
        print(f"verified route ({outcome.mode})")
        print(f"  order: {' -> '.join(str(s) for s in outcome.route)}")
        print(f"  arrivals: {outcome.arrivals}")
        print(f"  completion time: {outcome.total_time}")
    else:
        print(f"no verified route ({outcome.mode})")
        print(f"  reason: {outcome.reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a verified route through stops with time windows.")
    parser.add_argument("request", help="a structured file (csv or json) or a text request (txt)")
    parser.add_argument("--geo", choices=["euclidean", "haversine"], default="euclidean")
    parser.add_argument("--consequence", choices=["normal", "high"], default="normal")
    parser.add_argument("--self-review", action="store_true", help="run the second-pass input check")
    parser.add_argument(
        "--time-limit", type=int, default=10,
        help="solver time budget in seconds (raise for large instances)",
    )
    parser.add_argument(
        "--solution-limit", type=int, default=200,
        help="stop the solver after this many solutions; 0 runs to the time budget",
    )
    args = parser.parse_args(argv)
    solution_limit = None if args.solution_limit == 0 else args.solution_limit

    model_client = None
    path = Path(args.request)
    is_text = not (path.exists() and structured.detect(path))
    if is_text:
        try:
            from ontime.pipeline.model_client import OpenAICompatibleClient

            model_client = OpenAICompatibleClient()
        except Exception as exc:
            print(f"could not build the model client: {exc}", file=sys.stderr)

    outcome = plan_route(
        args.request,
        model_client=model_client,
        geo=args.geo,
        consequence=args.consequence,
        run_self_review=args.self_review,
        time_limit_s=args.time_limit,
        solution_limit=solution_limit,
    )
    _print_outcome(outcome)
    return 0 if outcome.verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
