"""Anthropic Claude client wrapper."""
from __future__ import annotations

import os
from typing import Any, Optional

from .base import LLMClient, LLMResponse, LLMToolCall


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Anthropic separates the system prompt from the message list."""
    system_chunks: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_chunks.append(m["content"])
        elif role in ("user", "assistant"):
            out.append({"role": role, "content": m["content"]})
        elif role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.get("tool_call_id", ""),
                            "content": m["content"],
                        }
                    ],
                }
            )
    return ("\n\n".join(system_chunks), out)


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(
        self, model: str = "claude-3-5-sonnet-latest", api_key: Optional[str] = None
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("install anthropic>=0.42 to use AnthropicClient") from e
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = Anthropic(api_key=self.api_key) if self.api_key else None

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_model: Any = None,
    ) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")
        system, msgs = _to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {}),
                }
                for t in tools
            ]
        resp = self._client.messages.create(**kwargs)
        text_parts: list[str] = []
        tcs: list[LLMToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tcs.append(LLMToolCall(id=block.id, name=block.name, args=dict(block.input)))
        usage = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
        }
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tcs,
            raw=resp,
            model=self.model,
            usage=usage,
        )
