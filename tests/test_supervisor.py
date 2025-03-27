"""Smoke test for supervisor template using a stub LLM."""

from typing import Any, Optional

from agent_framework import AgentState, ToolRegistry
from agent_framework.llm.base import LLMClient, LLMResponse
from agent_framework.templates import build_supervisor_team
from agent_framework.templates.supervisor_workers import Decision
from agent_framework.tools import CalculatorTool


class StaticLLM(LLMClient):
    """Returns a Decision the first time, then a finishing answer."""

    name = "static"

    def __init__(self) -> None:
        self.n = 0

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_model: Any = None,
    ) -> LLMResponse:
        self.n += 1
        if response_model is Decision:
            # supervisor: first call routes to math worker, then finishes
            payload = Decision(next="math") if self.n == 1 else Decision(next="finish")
            return LLMResponse(text=payload.model_dump_json(), raw=payload)
        # worker llm: just answer plainly without tools
        return LLMResponse(text="ok", tool_calls=[])


def test_supervisor_routes_then_finishes():
    llm = StaticLLM()
    math_tools = ToolRegistry()
    math_tools.register(CalculatorTool())
    g = build_supervisor_team(llm, {"math": math_tools})
    out = g.compile(recursion_limit=10).invoke(AgentState(input="say ok"))
    # supervisor + math worker + supervisor (finish) - must terminate
    assert out is not None
