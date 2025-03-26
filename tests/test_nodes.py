"""Tests for LLMNode and ToolNode using a stub LLM client."""

from typing import Any, Optional

from agent_framework import AgentState, LLMNode, ToolNode, ToolRegistry
from agent_framework.llm.base import LLMClient, LLMResponse, LLMToolCall
from agent_framework.tools import CalculatorTool


class StubLLM(LLMClient):
    """Returns a fixed sequence of canned responses."""

    name = "stub"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_model: Any = None,
    ) -> LLMResponse:
        self.calls.append({"messages": list(messages), "tools": tools})
        return self._responses.pop(0)


def test_llm_node_emits_pending_calls():
    stub = StubLLM(
        [LLMResponse(text="", tool_calls=[LLMToolCall(id="1", name="calculator", args={"expression": "2+2"})])]
    )
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    node = LLMNode("llm", client=stub, tools=reg)

    s = AgentState(input="add two")
    patch = node.run(s)
    assert len(patch["pending_calls"]) == 1
    assert patch["pending_calls"][0].name == "calculator"
    assert patch.get("output") is None


def test_llm_node_writes_output_when_no_tool_call():
    stub = StubLLM([LLMResponse(text="The answer is 4.", tool_calls=[])])
    node = LLMNode("llm", client=stub, tools=ToolRegistry())
    patch = node.run(AgentState(input="2+2?"))
    assert patch["output"] == "The answer is 4."
    assert patch["pending_calls"] == []


def test_tool_node_runs_pending_call():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    node = ToolNode("tools", registry=reg)

    from agent_framework import ToolCall
    state = AgentState(pending_calls=[ToolCall(id="x", name="calculator", args={"expression": "5*5"})])
    patch = node.run(state)

    assert patch["pending_calls"] == []
    assert patch["tool_results"][0].output == {"value": 25}
    assert patch["messages"][0].role == "tool"


def test_tool_node_handles_unknown_tool():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    from agent_framework import ToolCall
    state = AgentState(pending_calls=[ToolCall(id="x", name="ghost", args={})])
    patch = ToolNode("t", registry=reg).run(state)
    assert "unknown tool" in patch["tool_results"][0].error
