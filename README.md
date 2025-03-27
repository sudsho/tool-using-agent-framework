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

This is that. ~1.5k lines of Python, no magic, fits in a notebook.

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
python examples/research_assistant.py "what is the population of Iceland in 2024"
make dashboard       # http://localhost:8080
```

Programmatic use:

```python
from agent_framework import AgentState, Tracer
from agent_framework.llm import OpenAIClient
from agent_framework.templates import build_react_agent
from agent_framework.tools import default_registry

graph = build_react_agent(OpenAIClient(model="gpt-4o-mini"), default_registry())
compiled = graph.compile(tracer=Tracer(out_dir="./traces"))
state = compiled.invoke(AgentState(input="What is 17 * 23, then sqrt of that?"))
print(state.output)
```

## Built-in tools

| name           | description                                          |
| -------------- | ---------------------------------------------------- |
| `web_search`   | Tavily or SerpAPI                                    |
| `calculator`   | Safe AST-based arithmetic                            |
| `code_executor`| Python in a subprocess sandbox with timeout          |
| `file_io`      | Sandboxed read/write under a configured root         |

## Templates

- `react` - classic ReAct loop, LLM <-> Tool until no tool call is emitted
- `plan_act` - plan-and-execute with a `Plan` schema validated by `instructor`
- `supervisor_workers` - supervisor LLM routes to N specialised worker agents

## Tracing

`Tracer` records OTel-style spans (kind = node | llm | tool | router | event)
to JSONL. Spans nest via `with tracer.span(...)`. The dashboard renders the
trace tree per `trace_id`.

```
traces/
├── 8f2e5d...jsonl
└── b1d4af...jsonl
```

## Configuration

Edit `configs/default.yaml` or override with env vars (`MODEL`, `TRACE_DIR`).

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
tracer:
  backend: jsonl
  out_dir: ./traces
tools:
  enabled: [web_search, calculator, code_executor, file_io]
```

## Development

```bash
make dev    # install dev + dashboard extras
make test   # run pytest
make lint   # ruff check
```

CI workflow lives at `ci/test.yml.example` .  copy to `.github/workflows/`
when ready.

## Layout

```
src/agent_framework/
├── core/            graph, state, nodes, edges, tracer
├── tools/           web_search, calculator, code_executor, file_io
├── llm/             openai, anthropic, litellm
├── templates/       react, plan_act, supervisor_workers
└── dashboard/       FastAPI trace viewer
```

## License

MIT.
