"""Natural-language input: a text request into a parsed request.

The free 7B reads a text request into the structure the modeler needs. This is
the showcase path and the weaker one. In the article the model declined to call
the tool on a share of raw text requests, around 7.5% for the open 7B. When that
happens the caller falls back to structured input or escalates that one case to a
stronger model.

This module holds a text request and detects the problem kind. The model call
itself happens in the modeler, which is where the prompt is built.
"""
from __future__ import annotations

from pathlib import Path

from ontime.pipeline.guard import classify_text


def detect(path_or_text: str | Path) -> bool:
    """Return True when the input is a text request rather than a file path."""
    if isinstance(path_or_text, Path):
        return path_or_text.suffix.lower() not in {".csv", ".json"}
    return True


def read(path_or_text: str | Path) -> str:
    """Return the request text from a path or a raw string."""
    if isinstance(path_or_text, Path):
        return path_or_text.read_text()
    candidate = Path(path_or_text)
    if candidate.exists() and candidate.suffix.lower() == ".txt":
        return candidate.read_text()
    return str(path_or_text)


def problem_kind(text: str) -> str:
    """Best guess of the problem kind from the text."""
    return classify_text(text).problem_kind
