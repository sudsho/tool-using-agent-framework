"""Integration tests for the agent templates using a stub LLM."""

from typing import Any, Optional

from agent_framework import AgentState, ToolRegistry
from agent_framework.llm.base import LLMClient, LLMResponse
from agent_framework.templates import build_react_agent
from agent_framework.tools import CalculatorTool


class ScriptedLLM(LLMClient):
    """Cycles through a fixed list of LLMResponse objects."""

    name = "scripted"

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self.calls = 0

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_model: Any = None,
    ) -> LLMResponse:
        self.calls += 1
        if not self._script:
            return LLMResponse(text="(out of script)")
        return self._script.pop(0)


def test_react_loop_finishes_without_tools():
    llm = ScriptedLLM([LLMResponse(text="42", tool_calls=[])])
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    g = build_react_agent(llm, reg)
    out = g.compile(recursion_limit=5).invoke(AgentState(input="answer?"))
    assert out.output == "42"
    assert llm.calls == 1


