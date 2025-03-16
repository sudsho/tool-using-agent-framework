"""Math-solver example: plan-and-execute with a calculator tool.

Run:
    python examples/math_solver.py "find the integer roots of x^3 - 6x^2 + 11x - 6"
"""
from __future__ import annotations

import os
import sys

from agent_framework import AgentState, Tracer, ToolRegistry
from agent_framework.llm import OpenAIClient
from agent_framework.templates import build_plan_act_agent
from agent_framework.tools import CalculatorTool, CodeExecutorTool


def main() -> None:
    problem = " ".join(sys.argv[1:]) or "Compute (3 * pi**2 + sqrt(50)) / log(7)"
    client = OpenAIClient(model=os.getenv("MODEL", "gpt-4o-mini"))
    tools = ToolRegistry()
    tools.register(CalculatorTool())
    tools.register(CodeExecutorTool(timeout_s=4))
    graph = build_plan_act_agent(client, tools, max_steps=5)

    tracer = Tracer(out_dir=os.getenv("TRACE_DIR", "./traces"))
    compiled = graph.compile(tracer=tracer, recursion_limit=20)
    state = compiled.invoke(AgentState(input=problem))

    print("\n--- final ---\n")
    print(state.output)
    print(f"\nplan: {state.metadata.get('plan')}")
    print(f"trace: {tracer.current_trace}")


if __name__ == "__main__":
    main()
