"""LiteLLM-backed client to support local models (Ollama, vLLM) and other providers."""
from __future__ import annotations

import json
from typing import Any, Optional

from .base import LLMClient, LLMResponse, LLMToolCall


class LiteLLMClient(LLMClient):
    name = "litellm"

    def __init__(self, model: str = "ollama/llama3.1") -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("install litellm>=1.55 to use LiteLLMClient") from e
        self.model = model

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_model: Any = None,
    ) -> LLMResponse:
        import litellm

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = litellm.completion(**kwargs)
        choice = resp.choices[0].message
        text = choice.get("content") if isinstance(choice, dict) else (choice.content or "")
        raw_tcs = (choice.get("tool_calls") if isinstance(choice, dict) else choice.tool_calls) or []
        tcs: list[LLMToolCall] = []
        for tc in raw_tcs:
            fn = tc["function"] if isinstance(tc, dict) else tc.function
            try:
                args = json.loads(fn["arguments"] if isinstance(fn, dict) else fn.arguments)
            except json.JSONDecodeError:
                args = {}
            tc_id = tc["id"] if isinstance(tc, dict) else tc.id
            tc_name = fn["name"] if isinstance(fn, dict) else fn.name
            tcs.append(LLMToolCall(id=tc_id, name=tc_name, args=args))
        return LLMResponse(text=text or "", tool_calls=tcs, raw=resp, model=self.model)
