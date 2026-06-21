"""Manage an on-demand model endpoint on Modal for the natural-language path.

This wraps the Modal CLI so the reading model can run on a cloud GPU instead of a
laptop. It deploys serve/modal_app.py, waits until the endpoint is healthy, and
writes the endpoint into a .env file as ONTIME_LLM_BASE_URL and ONTIME_LLM_MODEL,
which is what route.py reads.

Usage:

    python -m ontime.serve up [<model-id>]
    python -m ontime.serve status
    python -m ontime.serve test
    python -m ontime.serve down

It needs the modal package and a one-time `modal token new`. The endpoint scales
to zero when idle, so bring it up before a run and down when you are finished.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
MODAL_APP_PATH = REPO_ROOT / "serve" / "modal_app.py"
APP_NAME = "ontime-llm"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

ENV_KEYS = ("ONTIME_LLM_BASE_URL", "ONTIME_LLM_MODEL", "ONTIME_LLM_API_KEY")

URL_RE = re.compile(r"https://[^\s]+\.modal\.run", re.IGNORECASE)
FATAL_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"^\w*Error:", re.MULTILINE),
    re.compile(r"OutOfMemoryError"),
    re.compile(r"CUDA error"),
    re.compile(r"Runner failed with exception"),
    re.compile(r"Image build for .* failed"),
]
HARD_CEILING_SECONDS = 300
HEALTH_POLL_INTERVAL = 2
HEALTH_POLL_TIMEOUT = 5


# ---------------------------------------------------------------------------
# env file helpers
# ---------------------------------------------------------------------------

def read_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    out: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        if not line.strip() or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def write_env(updates: dict[str, str]) -> None:
    """Apply updates to .env, keeping comments and unrelated lines intact."""
    existing = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    seen = set()
    new_lines = []
    for line in existing:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            new_lines.append(line)
    for key, val in updates.items():
        if key not in seen:
            new_lines.append(f"{key}={val}")
    ENV_PATH.write_text("\n".join(new_lines) + "\n")


def modal_bin() -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / "modal"
    return str(candidate) if candidate.exists() else "modal"


def _run(cmd: list[str], env: dict[str, str] | None = None, check: bool = True):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.run(cmd, env=full_env, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if check and proc.returncode != 0:
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        sys.exit(proc.returncode)
    return proc


# ---------------------------------------------------------------------------
# health detection
# ---------------------------------------------------------------------------

class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.ready: float | None = None
        self.fatal: str | None = None
        self.last_health_status: str | None = None


def _poll_health(endpoint: str, state: _State, stop: threading.Event) -> None:
    health_url = endpoint + "/health"
    while not stop.is_set():
        try:
            with urllib.request.urlopen(health_url, timeout=HEALTH_POLL_TIMEOUT) as resp:
                if resp.status == 200:
                    with state.lock:
                        if state.ready is None:
                            state.ready = time.monotonic()
                            state.last_health_status = "200 OK"
                    return
                with state.lock:
                    state.last_health_status = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            with state.lock:
                state.last_health_status = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            with state.lock:
                state.last_health_status = f"connect: {exc.__class__.__name__}"
        stop.wait(HEALTH_POLL_INTERVAL)


def _tail_logs(state: _State, stop: threading.Event) -> None:
    proc = subprocess.Popen(
        [modal_bin(), "app", "logs", APP_NAME],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    try:
        for line in iter(proc.stdout.readline, ""):
            if stop.is_set() or not line:
                break
            for pattern in FATAL_PATTERNS:
                if pattern.search(line):
                    with state.lock:
                        if state.fatal is None:
                            state.fatal = line.rstrip()
                    return
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def _wait_healthy(endpoint: str) -> dict:
    state = _State()
    stop = threading.Event()
    threads = [
        threading.Thread(target=_poll_health, args=(endpoint, state, stop), daemon=True),
        threading.Thread(target=_tail_logs, args=(state, stop), daemon=True),
    ]
    for t in threads:
        t.start()

    started = time.monotonic()
    deadline = started + HARD_CEILING_SECONDS
    while time.monotonic() < deadline:
        with state.lock:
            if state.ready is not None or state.fatal is not None:
                break
        time.sleep(1)

    stop.set()
    for t in threads:
        t.join(timeout=3)

    with state.lock:
        elapsed = time.monotonic() - started
        return {
            "elapsed_seconds": round(elapsed, 1),
            "ready_at_seconds": round(state.ready - started, 1) if state.ready else None,
            "fatal_log_line": state.fatal,
            "last_health_status": state.last_health_status,
            "hard_ceiling_hit": elapsed >= HARD_CEILING_SECONDS,
        }


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_up(model: str) -> int:
    proc = _run([modal_bin(), "deploy", str(MODAL_APP_PATH)], env={"ONTIME_SERVE_MODEL": model})
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    urls = [u for u in URL_RE.findall(output) if "vllm-server" in u] or URL_RE.findall(output)
    if not urls:
        sys.stderr.write("could not find the Modal endpoint URL in the deploy output:\n" + output)
        return 1
    endpoint = urls[0].rstrip("/")
    base_url = endpoint + "/v1"

    write_env({
        "ONTIME_LLM_BASE_URL": base_url,
        "ONTIME_LLM_MODEL": model,
        "ONTIME_LLM_API_KEY": "not-needed",
    })

    health = _wait_healthy(endpoint)
    healthy = health["ready_at_seconds"] is not None and health["fatal_log_line"] is None
    result = {
        "app": APP_NAME,
        "model": model,
        "endpoint": endpoint,
        "base_url": base_url,
        "healthy": healthy,
        "health": health,
    }
    if not healthy:
        result["hint"] = (
            "the container did not reach health within the wait. "
            f"check `{modal_bin()} app logs {APP_NAME}`, then run `python -m ontime.serve down`."
        )
        print(json.dumps(result, indent=2))
        return 2
    result["next"] = "the endpoint is in .env; run `python -m ontime.serve test` or route.py on a text request"
    print(json.dumps(result, indent=2))
    return 0


def cmd_down() -> int:
    proc = _run([modal_bin(), "app", "stop", "--yes", APP_NAME], check=False)
    stopped = proc.returncode == 0
    if not stopped:
        err = (proc.stderr or "") + (proc.stdout or "")
        if "not found" in err.lower() or "no app" in err.lower():
            stopped = True
        else:
            sys.stderr.write(err)
            return proc.returncode
    write_env({k: "" for k in ENV_KEYS})
    print(json.dumps({"app": APP_NAME, "stopped": stopped, "env_cleared": list(ENV_KEYS)}, indent=2))
    return 0


def cmd_status() -> int:
    proc = _run([modal_bin(), "app", "list", "--json"], check=False)
    apps = []
    if proc.returncode == 0:
        try:
            apps = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError:
            sys.stderr.write(proc.stdout or "")
    env = read_env()
    running = next((a for a in apps if (a.get("Name") or a.get("name")) == APP_NAME), None)
    print(json.dumps({
        "app": APP_NAME,
        "running": running,
        "env": {
            "ONTIME_LLM_BASE_URL": env.get("ONTIME_LLM_BASE_URL"),
            "ONTIME_LLM_MODEL": env.get("ONTIME_LLM_MODEL"),
        },
    }, indent=2))
    return 0


def cmd_test() -> int:
    env = read_env()
    base_url = env.get("ONTIME_LLM_BASE_URL")
    if not base_url:
        sys.stderr.write("ONTIME_LLM_BASE_URL is not set; run `python -m ontime.serve up` first\n")
        return 1
    for k, v in env.items():
        if v:
            os.environ.setdefault(k, v)

    sys.path.insert(0, str(REPO_ROOT))
    from ontime.pipeline.model_client import OpenAICompatibleClient

    client = OpenAICompatibleClient()
    t0 = time.monotonic()
    result = client.run("plan a route for two stops with time windows")
    latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
    print(json.dumps({
        "endpoint": base_url,
        "model": env.get("ONTIME_LLM_MODEL"),
        "tool_called": result.extracted_params is not None,
        "refused": result.refused,
        "latency_ms": latency_ms,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "status"
    rest = args[1:]
    if command == "up":
        return cmd_up(rest[0] if rest else DEFAULT_MODEL)
    if command == "down":
        return cmd_down()
    if command == "status":
        return cmd_status()
    if command == "test":
        return cmd_test()
    sys.stderr.write(f"unknown command: {command}. use up, down, status, or test.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
