"""End-to-end offline tests: MockLLM + real calculator + canned web_search
driving a full ReAct loop with tracing. No keys, no network."""

import json

from agent_framework import AgentState, ToolRegistry, Tracer
from agent_framework.llm import MockLLM, make_client
from agent_framework.llm.mock_client import _extract_expression, _looks_like_math
from agent_framework.templates import build_react_agent
from agent_framework.tools import CalculatorTool, WebSearchTool


def offline_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(WebSearchTool(provider="mock"))
    return reg


class TestMockHeuristics:
    def test_math_detection(self):
        assert _looks_like_math("What is (12 * 8) + sqrt(256)?")
        assert _looks_like_math("compute 3 + 4")
        assert not _looks_like_math("What is the population of Iceland?")
        assert not _looks_like_math("who is the president")

    def test_expression_extraction(self):
        assert _extract_expression("What is (12 * 8) + sqrt(256)?") == "(12 * 8) + sqrt(256)"
        assert _extract_expression("no math here") is None

    def test_make_client_mock(self):
        assert isinstance(make_client("mock", "m"), MockLLM)


class TestWebSearchMock:
    def test_canned_hit(self):
        out = WebSearchTool(provider="mock").run(query="population of Iceland")
        assert "393,600" in out["results"][0]["snippet"]

    def test_canned_miss(self):
        out = WebSearchTool(provider="mock").run(query="something obscure xyz")
        assert out["results"][0]["title"] == "No canned result"


class TestReactLoopOffline:
    def test_math_task_end_to_end(self, tmp_path):
        agent = build_react_agent(MockLLM(), offline_registry())
        tracer = Tracer(out_dir=tmp_path)
        compiled = agent.compile(tracer=tracer, recursion_limit=8)

        state = compiled.invoke(AgentState(input="What is (12 * 8) + sqrt(256)?"))

        assert state.output is not None
        assert "112" in state.output
        # the calculator actually ran
        assert any(r.output == {"value": 112.0} for r in state.tool_results)
        # a trace file was written with node + event spans
        spans = _read_trace(tmp_path, tracer.current_trace)
        names = [s["name"] for s in spans]
        assert names.count("llm") == 2  # decide, then answer
        assert "tools" in names

    def test_research_task_end_to_end(self, tmp_path):
        agent = build_react_agent(MockLLM(), offline_registry())
        tracer = Tracer(out_dir=tmp_path)
        compiled = agent.compile(tracer=tracer, recursion_limit=8)

        state = compiled.invoke(AgentState(input="What is the population of Iceland?"))

        assert state.output is not None
        assert "393,600" in state.output
        assert any("web_search" == (m.name or "") for m in state.messages)

    def test_loop_terminates_and_clears_queue(self, tmp_path):
        # regression: pending_calls must be drained by the tool node, else the
        # loop would run until the recursion limit.
        agent = build_react_agent(MockLLM(), offline_registry())
        state = agent.compile(recursion_limit=8).invoke(
            AgentState(input="compute 2 + 2 * 3")
        )
        assert state.pending_calls == []
        assert "8" in (state.output or "")


def _read_trace(trace_dir, trace_id):
    path = trace_dir / f"{trace_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
