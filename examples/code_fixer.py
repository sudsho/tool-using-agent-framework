"""Code-fixer example: read a file, ask the LLM to fix a bug, write it back.

Run:
    python examples/code_fixer.py path/to/buggy.py "div by zero crash on empty list"
"""
from __future__ import annotations

import os
import sys

from agent_framework import AgentState, ToolRegistry, Tracer
from agent_framework.llm import OpenAIClient
from agent_framework.templates import build_react_agent
from agent_framework.tools import CodeExecutorTool, FileIOTool


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python examples/code_fixer.py <path> <description>")
        sys.exit(1)
    path = sys.argv[1]
    desc = " ".join(sys.argv[2:])
    sandbox = os.path.dirname(os.path.abspath(path)) or "."
    rel = os.path.basename(path)

    tools = ToolRegistry()
    tools.register(FileIOTool(sandbox_root=sandbox))
    tools.register(CodeExecutorTool(timeout_s=4))

    client = OpenAIClient(model=os.getenv("MODEL", "gpt-4o-mini"))
    graph = build_react_agent(
        client,
        tools,
        system_prompt=(
            "You are a careful Python debugger. Read the source file via file_io, "
            "identify the bug from the user description, propose a fix, and write "
            "the corrected file back. Validate by running it via code_executor."
        ),
    )
    tracer = Tracer(out_dir=os.getenv("TRACE_DIR", "./traces"))
    compiled = graph.compile(tracer=tracer, recursion_limit=14)
    state = compiled.invoke(AgentState(input=f"File: {rel}\nBug: {desc}"))

    print(state.output or "(no output)")


if __name__ == "__main__":
    main()
