# ontime

ontime reads a set of stops with time windows and returns a route that has been
checked against those windows. The reading runs on a free self-hosted model and
the solving runs on a free solver, OR-Tools. The default path needs no paid API
and no outbound network.

This is not a better optimizer than OR-Tools. OR-Tools does the solving and is
reliable on its own. ontime is the architecture around it made usable: a free
natural-language front door, and a verification gate that refuses to return a
route that breaks a window.

## What it does

Give ontime a set of stops, each with a location and a time window and an
optional service time, plus a start point, a start time, and a speed or a cost
source. The input can be plain text or a structured CSV or JSON file. ontime
returns an ordered route with the arrival time at each stop, a feasibility
verdict, and on an infeasible problem the exact stop and window that cannot be
met. It never returns a route that breaks a window.

## Quickstart

Install the dependencies into a virtual environment.

```
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### Structured mode

The reliable path. The numbers arrive already shaped, so no model is involved.

```
.venv/bin/python route.py examples/driving/stops.csv
.venv/bin/python route.py examples/machine_scheduling/jobs.csv
```

Each returns a verified route with arrival times and a completion time.

### Natural-language mode

The showcase path. A self-hosted model reads a text request into the structure.
Point ontime at any OpenAI-compatible endpoint, which is what a server such as
vLLM or llama.cpp exposes.

```
export ONTIME_LLM_BASE_URL=http://localhost:8000/v1
export ONTIME_LLM_MODEL=qwen2.5-7b-instruct
.venv/bin/python route.py examples/driving/day.txt
```

The open 7B declines to call the tool on a share of raw text requests, around
7.5% in the benchmark. When that happens, fall back to structured input, or
escalate that one request to a stronger model.

## Architecture

Each module is one layer of the pipeline. See
`results/figures/fig04_verification_ladder.svg` for the verification ladder that
the autonomy decision sits on.

| Module | Layer | Measured or scaffolding |
| --- | --- | --- |
| `pipeline/guard.py` | reject anything that is not a stops-with-windows problem | scaffolding |
| `pipeline/modeler.py` | read the stops, windows, service times, start, and speed or cost matrix into the tool call | measured |
| `pipeline/self_review.py` | re-read the input and flag a mismatch between the two reads | measured |
| `pipeline/solve_tsp_tw.py` | wrap OR-Tools, accept coordinates with a speed or a direct cost matrix | measured |
| `pipeline/verifier.py` | check the route against the real windows | measured |
| `pipeline/autonomy.py` | return the route, or report which stop cannot be made | scaffolding |
| `distances/haversine.py` | the free default cost source | scaffolding |
| `distances/osrm.py` | an optional road-time adapter, never required | scaffolding |
| `modes/structured.py` | parse CSV or JSON | scaffolding |
| `modes/natural_language.py` | read a text request | scaffolding |

The same `solve_tsp_tw` engine serves both a driving problem and a one-machine
scheduling problem. For driving it takes coordinates and a speed. For machine
scheduling it takes the changeover matrix as the cost matrix, the release-to-due
times as the windows, and the processing time as the service time. The solver and
the verifier read the same cost matrix through the same provider, so a route that
the solver returns is checked against the travel times it was built on.

## The honesty boundary

The measured spine is the interface jump, the marshaling mechanism, and the
feasibility gate. Those are the parts the benchmark put numbers on.

- The interface jump lives in the marshaling layer. When the numbers are handed
  to the model already shaped, the trivial-copy path reaches the optimum on every
  instance. When the model reads the same problem from prose, the marshaling
  oracle match drops to 72.5% for Sonnet, and correctness drops with it.
- The feasibility gate is the boundary of autonomy. A route the verifier rejects
  is never delivered. On an infeasible problem the report names the stop and the
  window that cannot be met.

The classifier guard and the self-review loop are scaffolding that makes the core
safe on messy input. The self-review is honest about its cost. It has full recall
on the dangerous class, but it is noisy. In the benchmark it fired on 34.6% of the
natural-language runs and rose with problem size, so it is off by default and
switched on by consequence.

All numbers in this section come from `results/summary.json`, which is generated
from the run logs by `python -m ontime.bench.analysis`. See `results/NUMBERS.md`
for the full table and the source of each rate.

## Benchmark

`bench/` holds the instances and the grading harness. Ground truth comes from
Held-Karp, cross-checked against OR-Tools, and a tour is graded by the same
verifier the tool uses. `results/` holds the run logs that back every number.

```
.venv/bin/python -m ontime.bench.analysis
.venv/bin/python -m pytest
```

## Scope

This is a complete reference implementation of the architecture, not a product
with a roadmap. It runs end to end on a self-hosted 7B with no paid dependency on
the haversine default, and it returns a verified route or a clear infeasibility
report that names the stop that cannot be made.
