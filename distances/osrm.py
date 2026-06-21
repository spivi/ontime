"""An optional road-time cost source.

This adapter fetches a travel-time matrix from a self-hostable OSRM server or an
OpenRouteService endpoint. It is never required. The default haversine path runs
with no network at all. Use this only when you want real road times and you have
an endpoint to point it at.

The adapter raises OsrmUnavailable when the endpoint cannot be reached, which the
caller maps to a fall back to the haversine default.
"""
from __future__ import annotations

import json
import urllib.request

from ontime.pipeline.spec import ProblemSpec


class OsrmUnavailable(RuntimeError):
    """Raised when the OSRM endpoint cannot be reached or returns an error."""


def cost_matrix(
    spec: ProblemSpec,
    *,
    base_url: str,
    profile: str = "driving",
    timeout: float = 10.0,
) -> list[list[int]]:
    """Fetch an integer travel-time matrix from an OSRM table service.

    Coordinates are [lat, lon]. OSRM expects lon,lat order in the path, so they
    are swapped here. Durations come back in seconds and are rounded to integers.
    """
    if spec.coordinates is None:
        raise OsrmUnavailable("OSRM needs coordinates and the spec has none")

    coord_str = ";".join(f"{lon},{lat}" for lat, lon in spec.coordinates)
    url = f"{base_url.rstrip('/')}/table/v1/{profile}/{coord_str}?annotations=duration"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise OsrmUnavailable(f"could not reach OSRM at {base_url}: {exc}") from exc

    if payload.get("code") != "Ok" or "durations" not in payload:
        raise OsrmUnavailable(f"OSRM returned an unexpected response: {payload.get('code')}")

    durations = payload["durations"]
    n = spec.n
    if len(durations) != n or any(len(row) != n for row in durations):
        raise OsrmUnavailable("OSRM duration matrix has the wrong shape")

    return [[int(round(durations[i][j] or 0)) for j in range(n)] for i in range(n)]
