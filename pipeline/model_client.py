"""The model client for the natural-language path.

The default client talks to any OpenAI-compatible endpoint, which is what a
self-hosted server such as vLLM or llama.cpp exposes. Point it at a local Qwen
and the whole reading path runs with no paid API and no outbound network beyond
the local host.

The client hands the model the solve_tsp_tw tool, runs the tool when the model
calls it, and returns the parameters the model extracted along with the route.
The structured path never touches this client.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from ontime.pipeline.solve_tsp_tw import TOOL_SCHEMA, execute_tool


SYSTEM_PROMPT = (
    "You are a routing assistant. You are given a set of stops with time windows. "
    "Read the problem, extract every parameter, and call the solve_tsp_tw tool once "
    "with those parameters. Do not solve the problem yourself. Call the tool and "
    "return its result."
)


@dataclass
class ModelResult:
    """What one model run produced."""

    route: list[int] | None
    extracted_params: dict[str, Any] | None
    refused: bool
    reason: str
    transcript: list[dict] = field(default_factory=list)


class OpenAICompatibleClient:
    """Tool-use client for any OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_turns: int = 5,
    ) -> None:
        from openai import OpenAI

        self.model = model or os.getenv("ONTIME_LLM_MODEL", "qwen2.5-7b-instruct")
        self.base_url = base_url or os.getenv("ONTIME_LLM_BASE_URL", "http://localhost:8000/v1")
        key = api_key or os.getenv("ONTIME_LLM_API_KEY", "not-needed")
        self.client = OpenAI(base_url=self.base_url, api_key=key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_turns = max_turns

    def run(self, prompt: str) -> ModelResult:
        tool = {
            "type": "function",
            "function": {
                "name": TOOL_SCHEMA["name"],
                "description": TOOL_SCHEMA["description"],
                "parameters": TOOL_SCHEMA["input_schema"],
            },
        }
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        transcript: list[dict] = [{"role": "user", "content": prompt}]

        extracted: dict[str, Any] | None = None
        route: list[int] | None = None
        refused = False
        reason = ""
        n_turns = 0

        while n_turns < self.max_turns:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    tools=[tool],
                    tool_choice="auto",
                    messages=messages,
                )
            except Exception as exc:
                reason = f"endpoint error: {exc}"
                refused = True
                break

            n_turns += 1
            msg = resp.choices[0].message
            finish = resp.choices[0].finish_reason

            assistant_msg: dict = {"role": "assistant"}
            if msg.content:
                assistant_msg["content"] = msg.content
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)
            transcript.append(assistant_msg)

            if not msg.tool_calls:
                if finish == "stop" and route is None:
                    refused = True
                    reason = "model did not call the tool"
                break

            for tc in msg.tool_calls:
                if tc.function.name != "solve_tsp_tw":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": f"unknown tool: {tc.function.name}"}),
                        }
                    )
                    continue
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError as exc:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": f"invalid JSON in arguments: {exc}"}),
                        }
                    )
                    continue
                extracted = tool_input
                result = execute_tool(tool_input)
                if "error" not in result:
                    route = result.get("tour")
                tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
                messages.append(tool_msg)
                transcript.append(tool_msg)

            if finish != "tool_calls":
                break

        return ModelResult(
            route=route,
            extracted_params=extracted,
            refused=refused,
            reason=reason,
            transcript=transcript,
        )
