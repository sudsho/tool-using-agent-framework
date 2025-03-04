# tool-using-agent-framework

Minimal LangGraph-style framework for building tool-using LLM agents with full
step-by-step tracing baked in. Aims to stay small enough to read in one sitting
while supporting the patterns I actually use in production: ReAct loops,
plan-and-execute, supervisor-with-workers.

## Why

LangGraph and LangChain are great but heavy. For a lot of agent work I just want:

- a typed state object that flows through a graph
- nodes that are either an LLM call, a tool call, or a router
- conditional edges
- structured outputs (pydantic via `instructor`)
- a trace of every single step (LLM input/output, tool args/result, latency)
  written somewhere I can reload later

This is that.

## Architecture

```
                  +---------+
   user --->      | START   |
                  +----+----+
                       |
                       v
                  +---------+         +-----------+
                  |  llm    |<--------|  tool     |
                  |  node   |-------->|  node     |
                  +----+----+         +-----------+
                       |  conditional edge (router)
                       v
                  +---------+
                  |  END    |
                  +---------+
```

Every transition emits a span to the tracer. Spans are persisted as JSONL by
default; a small FastAPI dashboard reloads them.

## Quickstart

```bash
pip install -e ".[dev,dashboard]"
cp .env.example .env  # then fill OPENAI_API_KEY etc.
python examples/research_assistant.py
```

Then open `http://localhost:8080` to view traces in the dashboard.

## Built-in tools

- `web_search` (Tavily or SerpAPI)
- `calculator`
- `code_executor` (subprocess sandbox)
- `file_io` (sandboxed read/write)

## Templates

- `react` - classic ReAct loop
- `plan_act` - plan-and-execute
- `supervisor_workers` - supervisor delegating to N worker agents

## Status

Early. Building it out as I go.
