"""A Modal app that serves an open-weights model with an OpenAI-compatible API.

This runs vLLM behind the same OpenAI-compatible interface that ontime's
natural-language path expects, so the reading model can run on a cloud GPU
instead of a laptop. Tool calling is enabled, which the modeler needs.

Deploy it through the ontime serve utility:

    python -m ontime.serve up
    python -m ontime.serve up meta-llama/Llama-3.1-8B-Instruct

The model id comes from the ONTIME_SERVE_MODEL environment variable at deploy
time, so re-deploying with a different value caches that model's weights and
points the same app at it. The container scales to zero two minutes after the
last request, so an idle endpoint costs nothing.
"""
from __future__ import annotations

import os
import subprocess

import modal

MODEL_NAME = os.environ.get("ONTIME_SERVE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
APP_NAME = "ontime-llm"
VLLM_PORT = 8000

# vLLM's official image is a known-good vLLM and transformers pairing.
VLLM_DOCKER_TAG = "v0.10.0"

vllm_image = (
    modal.Image.from_registry(
        f"vllm/vllm-openai:{VLLM_DOCKER_TAG}",
        add_python=None,
    )
    .entrypoint([])
    .run_commands(
        "ln -sf $(which python3) /usr/local/bin/python",
        "python3 -m pip install 'huggingface_hub[hf_xet]>=0.34'",
        # Cache the weights into the image so a cold start skips the download.
        f"python3 -c 'from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id=\"{MODEL_NAME}\", ignore_patterns=[\"*.pt\", \"*.bin\"])'",
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    gpu="L4",
    scaledown_window=120,
    timeout=3600,
    max_containers=1,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=300)
def vllm_server():
    """Launch vLLM's OpenAI-compatible HTTP server."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--dtype", "bfloat16",
        # An L4 has 24 GB of VRAM. A 24k context length is comfortable for a
        # stops-with-windows request and the tool call it produces.
        "--max-model-len", "24576",
        "--gpu-memory-utilization", "0.92",
        "--disable-log-requests",
        # The modeler reads a request by calling a tool, so tool calling has to
        # be on. Qwen 2.5 emits Hermes-style tool calls.
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    subprocess.Popen(cmd)
