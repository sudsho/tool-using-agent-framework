"""Research-assistant example: ReAct loop with web_search + file_io.

Run:
    python examples/research_assistant.py "what is the population of Iceland in 2024"
"""
from __future__ import annotations

import os
import sys

from agent_framework import AgentState, Tracer
from agent_framework.llm import OpenAIClient
from agent_framework.templates import build_react_agent
from agent_framework.tools import default_registry


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Summarise the latest GPT-4o release notes."
    client = OpenAIClient(model=os.getenv("MODEL", "gpt-4o-mini"))
    tools = default_registry(code_executor=False)  # not needed here
    graph = build_react_agent(client, tools)

    tracer = Tracer(out_dir=os.getenv("TRACE_DIR", "./traces"))
    compiled = graph.compile(tracer=tracer, recursion_limit=12)

    state = compiled.invoke(AgentState(input=question))
    print("\n--- final answer ---\n")
    print(state.output or "(no output)")
    print(f"\ntrace: {tracer.current_trace}")


if __name__ == "__main__":
    main()
