"""Recompute every headline number from the run logs.

The README cites numbers from results/summary.json, and this script writes that
file. Every rate is recomputed from the carried JSONL logs, so the numbers are
reproducible rather than typed in by hand. Run it with:

    python -m ontime.bench.analysis

Definitions, each tied to a field in the logged rows:

  refusal rate         fail_stage == "tool_call" over all rows
  marshaling oracle    params_match_oracle == True over rows where it is set
  self-review fire     second_pass_match == False over rows where it is set
  correctness          correct == True over all rows
  engage rate          tool_called == True over all rows
  schema floor         correct == True over the schema diagnostic rows
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

FULL_RUN_RE = re.compile(r"full_run_(?P<model>\w+?)_tool_(?P<presentation>nl|trivial)_n(?P<n>\d+)\.jsonl")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def cell_summary(rows: list[dict]) -> dict:
    """Summarize one full-run cell from its rows."""
    total = len(rows)
    refused = sum(1 for r in rows if r.get("fail_stage") == "tool_call")
    engaged = sum(1 for r in rows if r.get("tool_called"))
    correct = sum(1 for r in rows if r.get("correct"))
    feasible = sum(1 for r in rows if r.get("feasible"))

    oracle_rows = [r for r in rows if r.get("params_match_oracle") is not None]
    oracle_match = sum(1 for r in oracle_rows if r["params_match_oracle"])

    review_rows = [r for r in rows if r.get("second_pass_match") is not None]
    review_fired = sum(1 for r in review_rows if r["second_pass_match"] is False)

    return {
        "rows": total,
        "refused": refused,
        "refusal_rate": _rate(refused, total),
        "engage_rate": _rate(engaged, total),
        "correct": correct,
        "correctness": _rate(correct, total),
        "feasible": feasible,
        "feasibility_rate": _rate(feasible, total),
        "oracle_rows": len(oracle_rows),
        "marshaling_oracle_rate": _rate(oracle_match, len(oracle_rows)),
        "review_rows": len(review_rows),
        "self_review_fire_rate": _rate(review_fired, len(review_rows)),
    }


def schema_floor() -> dict:
    """Correctness of the structured-assembly path from the schema diagnostic."""
    out = {}
    for path in sorted(RESULTS_DIR.glob("schema_diag_n*.jsonl")):
        rows = load_rows(path)
        n = int(re.search(r"n(\d+)", path.name).group(1))
        correct = sum(1 for r in rows if r.get("correct"))
        truncated = sum(1 for r in rows if r.get("out_tokens", 0) >= 64000)
        out[f"n{n}"] = {
            "rows": len(rows),
            "correct": correct,
            "correctness": _rate(correct, len(rows)),
            "truncated_at_64k": truncated,
        }
    return out


def aggregate(cells: dict) -> dict:
    """Pool the per-cell counts into the headline rates.

    Counts are summed across the size cells of each model and presentation, then
    turned into a rate, so the result matches a count over every row.
    """

    def total(prefix: str, field: str) -> int:
        return sum(s[field] for key, s in cells.items() if key.startswith(prefix))

    headline: dict = {}
    for model in ("sonnet", "qwen"):
        nl = f"{model}_nl_"
        tr = f"{model}_trivial_"
        headline[f"{model}_nl_refusal_rate"] = _rate(total(nl, "refused"), total(nl, "rows"))
        headline[f"{model}_nl_engage_rate"] = _rate(
            total(nl, "rows") - total(nl, "refused"), total(nl, "rows")
        )
        headline[f"{model}_nl_correctness"] = _rate(total(nl, "correct"), total(nl, "rows"))
        headline[f"{model}_trivial_correctness"] = _rate(total(tr, "correct"), total(tr, "rows"))
        headline[f"{model}_nl_marshaling_oracle_rate"] = _rate(
            _oracle_match_count(cells, nl), total(nl, "oracle_rows")
        )

    fired = _review_fire_count(cells, "sonnet_nl_")
    review_rows = total("sonnet_nl_", "review_rows")
    headline["sonnet_nl_self_review_fire_rate"] = _rate(fired, review_rows)
    return headline


def _oracle_match_count(cells: dict, prefix: str) -> int:
    count = 0
    for key, s in cells.items():
        if key.startswith(prefix) and s["marshaling_oracle_rate"] is not None:
            count += round(s["marshaling_oracle_rate"] * s["oracle_rows"])
    return count


def _review_fire_count(cells: dict, prefix: str) -> int:
    count = 0
    for key, s in cells.items():
        if key.startswith(prefix) and s["self_review_fire_rate"] is not None:
            count += round(s["self_review_fire_rate"] * s["review_rows"])
    return count


def build_summary() -> dict:
    cells: dict[str, dict] = {}
    for path in sorted(RESULTS_DIR.glob("full_run_*.jsonl")):
        m = FULL_RUN_RE.match(path.name)
        if not m:
            continue
        key = f"{m['model']}_{m['presentation']}_n{m['n']}"
        cells[key] = cell_summary(load_rows(path))

    summary = {
        "cells": cells,
        "schema_floor": schema_floor(),
        "headline": aggregate(cells),
    }
    return summary


def _print_headline(summary: dict) -> None:
    h = summary["headline"]
    print("headline rates from the run logs")
    print(f"  sonnet NL refusal rate     {h.get('sonnet_nl_refusal_rate')}")
    print(f"  qwen NL refusal rate       {h.get('qwen_nl_refusal_rate')}")
    print(f"  qwen NL engage rate        {h.get('qwen_nl_engage_rate')}")
    print(f"  sonnet NL correctness      {h.get('sonnet_nl_correctness')}")
    print(f"  sonnet trivial correctness {h.get('sonnet_trivial_correctness')}")
    print(f"  qwen trivial correctness   {h.get('qwen_trivial_correctness')}")
    print(f"  sonnet self-review fire    {h.get('sonnet_nl_self_review_fire_rate')}")
    sf = summary["schema_floor"]
    for n, vals in sf.items():
        print(f"  schema floor {n} correctness {vals['correctness']} (truncated {vals['truncated_at_64k']}/{vals['rows']})")


def main() -> int:
    summary = build_summary()
    out_path = RESULTS_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    _print_headline(summary)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
