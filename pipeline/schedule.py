"""Turn a verified route into a schedule the requester can read.

The solver speaks in stop numbers and arrival times. This renders that into plain
sentences, one line per stop, using the names from the request when they are
available. It is a presentation layer over a route that has already passed the
feasibility gate.
"""
from __future__ import annotations


def _label(index: int, names: list[str] | None) -> str:
    if names is not None and 0 <= index < len(names) and names[index]:
        return names[index]
    return f"stop {index}"


def render(
    route: list[int],
    arrivals: list[int],
    *,
    names: list[str] | None = None,
    time_windows: list[list[int]] | None = None,
    unit: str = "min",
) -> str:
    """Render a verified route as a readable schedule.

    route and arrivals come straight from a verified RouteOutcome. names and
    time_windows are optional. unit is the word used after each time value.
    """
    start = _label(route[0], names)
    lines = [f"Leave {start} at 0 {unit}."]

    for position in range(1, len(route) - 1):
        stop = route[position]
        arrive = arrivals[position]
        line = f"Arrive at {_label(stop, names)} at {arrive} {unit}."
        if time_windows is not None and 0 <= stop < len(time_windows):
            open_at, close_at = time_windows[stop]
            if arrive == open_at:
                line += f" That is the moment the window opens, which runs {open_at} to {close_at}."
            else:
                line += f" The window runs {open_at} to {close_at}, so you are on time."
        lines.append(line)

    end = _label(route[-1], names)
    completion = arrivals[-1]
    lines.append(f"Return to {end} at {completion} {unit}.")
    lines.append(f"The whole run takes {completion} {unit}.")
    return "\n".join(lines)
