"""Offline end-to-end smoke test.

Runs two tasks through a ReAct agent driven by the rule-based ``MockLLM`` and
the offline tool set (real ``calculator`` + canned ``web_search``). No API
keys, no network, no GPU. It exercises the graph engine, the tool registry,
the tracer, and the full ReAct loop (LLM -> tool -> LLM -> answer), then prints
the final answer and the recorded trace for each task.

Run:
    python scripts/smoke.py
    make smoke
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from agent_framework import AgentState, Tracer
from agent_framework.llm import MockLLM
from agent_framework.templates import build_react_agent
from agent_framework.tools import ToolRegistry
from agent_framework.tools.calculator import CalculatorTool
from agent_framework.tools.web_search import WebSearchTool


def offline_registry() -> ToolRegistry:
    """Registry with the two offline-capable tools."""
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(WebSearchTool(provider="mock"))
    return reg


def print_trace(trace_dir: Path, trace_id: str) -> None:
    path = trace_dir / f"{trace_id}.jsonl"
    spans = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    print(f"  trace {trace_id}  ({len(spans)} spans, file: {path.name})")
    for sp in spans:
        dur = ""
        if sp.get("end_ms") is not None and sp.get("start_ms") is not None:
            dur = f"  {sp['end_ms'] - sp['start_ms']:.1f}ms"
        if sp["kind"] == "event":
            attrs = " ".join(f"{k}={v}" for k, v in sp.get("attrs", {}).items())
            print(f"    - event  {sp['name']:<8} {attrs}")
        else:
            keys = ",".join(sp.get("outputs", {}).get("keys", []))
            print(f"    - {sp['kind']:<5}  {sp['name']:<8} patch=[{keys}]{dur}")


def run_task(title: str, question: str, trace_dir: Path) -> str:
    print(f"\n=== {title} ===")
    print(f"question: {question}")

    agent = build_react_agent(MockLLM(), offline_registry())
    tracer = Tracer(out_dir=trace_dir)
    compiled = agent.compile(tracer=tracer, recursion_limit=8)

    state = compiled.invoke(AgentState(input=question))

    print("reasoning / tool trace:")
    for m in state.messages:
        content = (m.content or "").replace("\n", " ")
        if len(content) > 90:
            content = content[:87] + "..."
        print(f"    [{m.role:<9}] {content}")

    print(f"final answer: {state.output}")
    print_trace(trace_dir, tracer.current_trace or "")
    assert state.output, "agent produced no final answer"
    return state.output or ""


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="agent_smoke_") as tmp:
        trace_dir = Path(tmp)
        run_task(
            "Math task (calculator tool)",
            "What is (12 * 8) + sqrt(256)?",
            trace_dir,
        )
        run_task(
            "Research task (web_search tool)",
            "What is the population of Iceland?",
            trace_dir,
        )
    print("\nSMOKE OK: both tasks ran offline, called tools, and returned answers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
