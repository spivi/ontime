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

Add `--schedule` to get the same answer as a readable plan, using the names from
the file:

```
.venv/bin/python route.py examples/driving/stops.csv --schedule
```

```
Leave home at 0 min.
Arrive at post office at 12 min. The window runs 0 to 30, so you are on time.
Arrive at hardware store at 24 min. The window runs 15 to 40, so you are on time.
Arrive at bakery at 31 min. The window runs 25 to 70, so you are on time.
Arrive at pharmacy at 38 min. The window runs 30 to 90, so you are on time.
Return to home at 45 min.
The whole run takes 45 min.
```

### Natural-language mode

The showcase path. A self-hosted model reads a text request into the structure.
Point ontime at any OpenAI-compatible endpoint, which is what a server such as
vLLM or llama.cpp exposes.

```
export ONTIME_LLM_BASE_URL=http://localhost:8000/v1
export ONTIME_LLM_MODEL=qwen2.5-7b-instruct
.venv/bin/python route.py examples/driving/day.txt
```

The worked example below shows the full request next to the six phases the system
runs and the verified result. The open 7B declines to call the tool on a share of
raw text requests, around 7.5% in the benchmark. When that happens, fall back to
structured input, or escalate that one request to a stronger model.

### Serving the model on a cloud GPU

The natural-language path needs a model behind an OpenAI-compatible endpoint. A
laptop can serve a 7B through Ollama or llama.cpp, which needs no account and is
free. A cloud GPU is faster and is one command away. The serve utility runs vLLM
on a Modal GPU, waits until it is healthy, and writes the endpoint into a `.env`
file that `route.py` reads on its own.

It needs the optional serving dependency and a one-time Modal login:

```
.venv/bin/python -m pip install -r requirements-serve.txt
modal token new
```

Then bring an endpoint up, check it, run a request, and bring it down:

```
.venv/bin/python -m ontime.serve up
.venv/bin/python -m ontime.serve test
.venv/bin/python route.py examples/driving/day.txt --schedule
.venv/bin/python -m ontime.serve down
```

`up` deploys the app, waits for the endpoint to answer a health check, and writes
`ONTIME_LLM_BASE_URL` and `ONTIME_LLM_MODEL` into `.env`. Pass a model id to serve
a different model, for example
`python -m ontime.serve up meta-llama/Llama-3.1-8B-Instruct`. The default is
`Qwen/Qwen2.5-7B-Instruct`. The endpoint scales to zero two minutes after the last
request, so an idle endpoint costs nothing, and `down` stops it and clears the
endpoint from `.env`. The Modal app is defined in `serve/modal_app.py`, with tool
calling enabled, which the modeler needs.

## A worked example, end to end

Say you have a real afternoon of errands and you want an order that gets you to
each place inside its window. Here is the whole path from the real problem to a
verified route.

### Step 1: write down the problem

You have a start point and a few stops. Each stop has a location, a window when a
visit is allowed, and a few minutes of time spent there.

- Home, your start, at map point (0, 0).
- Pharmacy at (2, 1), open between minute 30 and minute 90, 3 minutes inside.
- Hardware store at (5, 4), open between 15 and 40, 3 minutes inside.
- Post office at (1, 6), open between 0 and 30, 3 minutes inside.
- Bakery at (4, 2), open between 25 and 70, 3 minutes inside.

You travel at 0.5 distance units per minute.

### Step 2: shape it into a file

Put one row per stop. The first row is the start and carries the speed. The
coordinates are plain map points here, so the default cost source measures
straight-line distance. Use `lat` and `lon` columns instead of `x` and `y` when
your points are real latitude and longitude, and add `--geo haversine`.

```
name,x,y,open,close,service,speed
home,0,0,0,600,0,0.5
pharmacy,2,1,30,90,3,0.5
hardware store,5,4,15,40,3,0.5
post office,1,6,0,30,3,0.5
bakery,4,2,25,70,3,0.5
```

This is the file at `examples/driving/stops.csv`.

### Step 3: run it

```
.venv/bin/python route.py examples/driving/stops.csv
```

```
verified route (structured)
  order: 0 -> 3 -> 2 -> 4 -> 1 -> 0
  arrivals: [0, 12, 24, 31, 38, 45]
  completion time: 45
```

### Step 4: read the result

The order maps back to your stops by row number. Stop 0 is home, stop 3 is the
post office, stop 2 is the hardware store, stop 4 is the bakery, stop 1 is the
pharmacy. So the route is home, post office, hardware store, bakery, pharmacy,
home. The arrivals line gives the minute you reach each one, and the completion
time is when you get back home, at minute 45. The word verified means the route
was checked against every window before it was printed.

### Step 5: when there is no answer

Suppose the post office instead closed at minute 1, which you cannot reach in
time. ontime says so and names the stop rather than handing back a route that
breaks the window.

```
no verified route (structured)
  reason: stop 3 cannot be met: the fastest arrival is 12, after its window closes at 1
```

### The same problem in plain language

You do not have to shape the file by hand. Describe the afternoon in words and let
a self-hosted model read it into the same structure. The file
`examples/driving/day.txt` is the errand list above written as prose:

```
I am running errands today and I want a route that gets me to each place inside
its time window. I start from home at coordinates (0, 0) at time zero, and I need
to be back home by 600 minutes. My speed is 0.5 distance units per minute and I
spend 3 minutes at each stop.

The pharmacy is at (2, 1) and is open between 30 and 90.
The hardware store is at (5, 4) and is open between 15 and 40.
The post office is at (1, 6) and is open between 0 and 30.
The bakery is at (4, 2) and is open between 25 and 70.
```

Point ontime at any OpenAI-compatible endpoint, which is what a server such as
vLLM or llama.cpp exposes, then run the same command on the text file:

```
export ONTIME_LLM_BASE_URL=http://localhost:8000/v1
export ONTIME_LLM_MODEL=qwen2.5-7b-instruct
.venv/bin/python route.py examples/driving/day.txt
```

```
verified route (natural_language)
  order: 0 -> 3 -> 2 -> 4 -> 1 -> 0
  arrivals: [0, 12, 24, 31, 38, 45]
  completion time: 45
```

The answer matches the structured run, since the text describes the same problem.

### What the system did to get there

The text request runs through six phases. Here is what each one did on this run,
with Qwen 2.5 7B reading the file.

1. Guard. The request is checked for scope. It reads as a stops-with-windows
   problem, so it passes.

   ```
   in_scope: True | kind: driving
   ```

2. Modeler. The model reads the prose and calls the solve_tsp_tw tool with the
   parameters it found. This is the step the benchmark measures, and it is where a
   weak read shows up.

   ```
   tool_called: True
   coordinates:  [[0, 0], [2, 1], [5, 4], [1, 6], [4, 2]]
   time_windows: [[0, 600], [30, 90], [15, 40], [0, 30], [25, 70]]
   service_time: 3 | speed: 0.5
   ```

3. Self-review. A second independent read of the same text checks whether the
   first read was stable. Here the two reads agree. This step is off by default
   and is shown here for the walkthrough.

   ```
   agrees: True
   ```

4. Solve. OR-Tools solves the extracted problem.

   ```
   status: optimal | tour: [0, 3, 2, 4, 1, 0] | total_time: 45
   ```

5. Verify. The feasibility gate recomputes the route against the real windows. A
   route that fails here is never returned.

   ```
   correct: True | objective: 45
   ```

6. Autonomy. With a passing gate, the route is delivered. On a failed gate, it is
   held back and the unmeetable stop is reported.

   ```
   deliver: True
   ```

The modeler is the step that can go wrong. If the prose is vague about a value,
for example leaving the time to be back home unstated, the model can read it as a
window of [0, 0] and the problem comes back infeasible. State the value and the
read is clean. When the open 7B declines to call the tool at all, which the
benchmark saw on about 7.5% of raw text requests, fall back to the structured file
or escalate that one request to a stronger model.

These runs were done with Qwen 2.5 7B both locally through Ollama on a laptop and
on a Modal cloud GPU through vLLM. The endpoint is OpenAI-compatible either way,
so the command and the result are the same. The cloud GPU is faster, and the
laptop needs no account.

### A scheduling problem is the same shape

One machine running jobs in an order is the same problem with no map. The
changeover time between jobs is the cost, the release-to-due time is the window,
and the processing time is the service time. See
`examples/machine_scheduling/jobs.csv` and its `changeover.csv`, then run:

```
.venv/bin/python route.py examples/machine_scheduling/jobs.csv
```

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
| `pipeline/schedule.py` | render the verified route as a readable plan | scaffolding |
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

The measured spine has two parts that the benchmark put numbers on: the interface
jump that the marshaling mechanism produces, and the feasibility gate that the
verifier enforces.

- The interface jump lives in the marshaling layer. When the numbers are handed
  to the model already shaped, the trivial-copy path reaches the optimum on every
  instance. When the model reads the same problem from prose, the marshaling
  oracle match drops to 72.5% for Sonnet, and correctness drops with it.
- The feasibility gate is the boundary of autonomy. A route the verifier rejects
  is never delivered. On an infeasible problem the report names the stop and the
  window that cannot be met.

The classifier guard and the self-review loop are scaffolding that makes the core
safe on messy input. The self-review is honest about its cost. It has full recall
on the dangerous class while staying noisy. In the benchmark it fired on 34.6% of
the natural-language runs and climbed toward 45% at the larger sizes, so it is off
by default and switched on by consequence.

The refusal weakness sits on the open 7B. It declined to call the tool on about
7.5% of raw text requests, where the frontier model declined on none. When the 7B
declines, fall back to the structured file or escalate that one request to the
frontier model.

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

## Tuning the solver

The solver defaults return fast on the sizes this tool targets. They stop after
200 solutions or 10 seconds, whichever comes first. A large instance may need a
longer budget to reach the optimum. Raise both limits on the command line.

```
.venv/bin/python route.py stops.csv --time-limit 60 --solution-limit 0
```

A solution limit of 0 lets the solver run to the time budget.

## Scope

This is a complete reference implementation of the architecture, not a product
with a roadmap. It plans for one vehicle that starts at the depot, visits every
stop once, and returns to the depot. Many vehicles at once is vehicle routing,
which this does not cover.

It runs end to end on a self-hosted 7B with no paid dependency on the haversine
default, and it returns a verified route or a clear infeasibility report that
names the stop that cannot be made.
