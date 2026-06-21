import json
from pathlib import Path

import pytest

INSTANCES_DIR = Path(__file__).resolve().parents[1] / "bench" / "instances"


def load_instances(n: int) -> list[dict]:
    path = INSTANCES_DIR / f"instances_n{n}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="session")
def instances_n8() -> list[dict]:
    return load_instances(8)


@pytest.fixture(scope="session")
def instances_n12() -> list[dict]:
    return load_instances(12)
