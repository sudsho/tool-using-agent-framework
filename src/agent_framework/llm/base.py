"""LLM client base interface.

Three implementations live alongside this file: ``openai_client``,
``anthropic_client`` and ``litellm_client``. They all return the same
``LLMResponse`` and accept the same ``call`` signature so nodes don't care
which provider is wired in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMToolCall:
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    raw: Any = None
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(ABC):
    name: str = "base"

    @abstractmethod
    def call(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_model: Any = None,
    ) -> LLMResponse:
        ...
