"""OpenAI chat-completions client wrapper.

Uses the official ``openai`` SDK 1.60+. Optionally wraps with ``instructor``
for typed structured outputs when ``response_model`` is provided.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from .base import LLMClient, LLMResponse, LLMToolCall


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("install openai>=1.60 to use OpenAIClient") from e
        self._OpenAI = OpenAI
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None
        self._instructor = None

    def _ensure_client(self):
        if self._client is None:
            raise RuntimeError("OPENAI_API_KEY not configured")
        return self._client

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_model: Any = None,
    ) -> LLMResponse:
        if response_model is not None:
            return self._structured(messages, response_model, temperature, max_tokens)
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tcs: list[LLMToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tcs.append(LLMToolCall(id=tc.id, name=tc.function.name, args=args))
        usage = {}
        if resp.usage is not None:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return LLMResponse(
            text=msg.content or "",
            tool_calls=tcs,
            raw=resp,
            model=self.model,
            usage=usage,
        )

    def _structured(
        self,
        messages: list[dict[str, Any]],
        response_model: Any,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        try:
            import instructor
        except ImportError as e:  # pragma: no cover
            raise ImportError("install instructor>=1.5 for response_model support") from e
        client = self._ensure_client()
        if self._instructor is None:
            self._instructor = instructor.from_openai(client)
        obj = self._instructor.chat.completions.create(
            model=self.model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(text=obj.model_dump_json(), raw=obj, model=self.model)
