"""Deterministic, rule-based LLM client for offline runs and tests.

The real clients (openai/anthropic/litellm) need an API key and a network.
``MockLLM`` needs neither: it decides what to do by looking at the running
message history with a handful of rules, so the graph engine, tool registry,
tracing and the ReAct loop can all be exercised end to end with no keys.

Behaviour inside a ReAct loop:

1. First call - only the user question is present. The client inspects the
   question and, if a tool is available and the question warrants it, emits a
   single tool call:
     - an arithmetic-looking question -> ``calculator`` with the extracted
       expression
     - any other lookup-style question -> ``web_search`` with the question
   The assistant "thought" text makes the decision visible in the trace.
2. Second call - a tool message is now in the history. The client reads the
   tool result and composes a short final answer, with no further tool calls,
   so the loop terminates.

It is intentionally not clever. It is deterministic and dependency-free.
"""
from __future__ import annotations

import ast
import re
import uuid
from typing import Any, Optional

from .base import LLMClient, LLMResponse, LLMToolCall

# operators / function names the calculator tool understands
_OP_RE = re.compile(r"[+\-*/%]")
_FUNC_RE = re.compile(r"\b(sqrt|sin|cos|tan|log10|log2|log|exp|abs|round|min|max)\b")
_MATH_KW = ("compute", "calculate", "evaluate", "what is", "how much")
# token grammar for pulling a clean arithmetic expression out of prose
_EXPR_TOKEN = re.compile(
    r"(?:sqrt|sin|cos|tan|log10|log2|log|exp|abs|round|min|max|pi\b|\d+\.?\d*|[+\-*/%()])"
)


def _looks_like_math(text: str) -> bool:
    has_digit = bool(re.search(r"\d", text))
    has_op = bool(_OP_RE.search(text)) or bool(_FUNC_RE.search(text))
    kw = any(w in text.lower() for w in _MATH_KW)
    return (has_digit and has_op) or (kw and has_digit and has_op)


def _extract_expression(text: str) -> Optional[str]:
    """Slice the arithmetic expression out of a natural-language question."""
    matches = list(_EXPR_TOKEN.finditer(text))
    if not matches:
        return None
    span = text[matches[0].start() : matches[-1].end()].strip()
    if not re.search(r"\d", span):
        return None
    return span


def _fmt_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class MockLLM(LLMClient):
    """A rule-based stand-in for a real LLM. No keys, no network."""

    name = "mock"

    def __init__(self, model: str = "mock-react-v1") -> None:
        self.model = model
        self.calls = 0

    # ----- helpers -----
    @staticmethod
    def _last_tool_message(messages: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        for m in reversed(messages):
            if m.get("role") == "tool":
                return m
        return None

    @staticmethod
    def _first_user_question(messages: list[dict[str, Any]]) -> str:
        for m in messages:
            if m.get("role") == "user":
                return str(m.get("content", ""))
        return ""

    @staticmethod
    def _parse_tool_output(content: str) -> Any:
        try:
            return ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return content

    def _available(self, tools: Optional[list[dict[str, Any]]]) -> set[str]:
        return {t.get("name", "") for t in (tools or [])}

    # ----- final-answer synthesis (a tool result is already in history) -----
    def _answer_from_tool(self, messages: list[dict[str, Any]]) -> str:
        tool_msg = self._last_tool_message(messages)
        question = self._first_user_question(messages)
        if tool_msg is None:
            return "I could not find enough information to answer."
        name = tool_msg.get("name", "")
        payload = self._parse_tool_output(str(tool_msg.get("content", "")))

        if name == "calculator" and isinstance(payload, dict):
            if "value" in payload:
                return f"The result of `{question.strip()}` is {_fmt_number(payload['value'])}."
            if "error" in payload:
                return f"The calculator could not evaluate that: {payload['error']}."

        if name == "web_search" and isinstance(payload, dict):
            results = payload.get("results") or []
            if results:
                top = results[0]
                snippet = top.get("snippet", "").strip()
                title = top.get("title", "").strip()
                url = top.get("url", "").strip()
                cite = f" (source: {title}, {url})" if title or url else ""
                return f"{snippet}{cite}"
            if payload.get("error"):
                return f"The web search failed: {payload['error']}."

        return f"Tool `{name}` returned: {payload}"

    # ----- LLMClient contract -----
    def call(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_model: Any = None,
    ) -> LLMResponse:
        self.calls += 1
        available = self._available(tools)

        # If a tool has already run, produce the final answer and stop.
        if self._last_tool_message(messages) is not None:
            return LLMResponse(text=self._answer_from_tool(messages), model=self.model)

        question = self._first_user_question(messages)

        # Decide on a tool call from the question.
        if "calculator" in available and _looks_like_math(question):
            expr = _extract_expression(question)
            if expr:
                return LLMResponse(
                    text=f"This is an arithmetic question. I'll use the calculator on `{expr}`.",
                    tool_calls=[
                        LLMToolCall(
                            id=uuid.uuid4().hex[:8],
                            name="calculator",
                            args={"expression": expr},
                        )
                    ],
                    model=self.model,
                )

        if "web_search" in available and question:
            return LLMResponse(
                text=f"I need to look this up. I'll search the web for: {question.strip()}",
                tool_calls=[
                    LLMToolCall(
                        id=uuid.uuid4().hex[:8],
                        name="web_search",
                        args={"query": question.strip()},
                    )
                ],
                model=self.model,
            )

        # No suitable tool: answer directly from the question.
        return LLMResponse(
            text=f"I don't have a tool for that, but here is my best answer to: {question.strip()}",
            model=self.model,
        )
