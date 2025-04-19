# tool-using-agent-framework

Minimal LangGraph-style framework for building tool-using LLM agents. Small
enough to read in one sitting: a typed state object flowing through a
node/edge graph, a handful of sandboxed tools, per-node span tracing, and a
FastAPI viewer for the traces.

## Why

LangGraph and LangChain are great but heavy. For a lot of agent work I just want:

- a typed state object that flows through a graph
- nodes that are either an LLM call, a tool call, or a router
- conditional edges
- a per-node span (name, step index, latency, patch keys) written to JSONL so
  I can reload it later

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

Each node execution emits a span to the tracer. Spans are persisted as JSONL;
a small FastAPI dashboard lists them.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .
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
# invoke with an AgentState(input="...") after configuring your OpenAI creds
```

## Built-in tools

| name           | description                                          |
| -------------- | ---------------------------------------------------- |
| `web_search`   | Tavily or SerpAPI                                    |
| `calculator`   | Safe AST-based arithmetic                            |
| `code_executor`| Python in a subprocess with a wall-clock timeout     |
| `file_io`      | Sandboxed read/write under a configured root         |

## Templates

- `react` - ReAct-style scaffold that alternates an LLM node with a tool node
  via a conditional edge
- `plan_act` - plan-and-execute scaffold; the planner path relies on
  `instructor` and is wired for `OpenAIClient` only
- `supervisor_workers` - supervisor LLM routes to N specialised worker agents;
  structured routing decisions rely on `OpenAIClient` + `instructor`

## Tracing

`Tracer` records per-node spans (kind = `node` or `event`) to JSONL. Each span
carries a `trace_id`, `span_id`, `parent_id`, the node name, step index,
latency, and the keys of the patch the node returned. The current dashboard
lists traces on the index page and renders each trace as a flat table of
spans.

```
traces/
|-- 8f2e5d...jsonl
`-- b1d4af...jsonl
```

## Configuration

Edit `configs/default.yaml` or override with env vars (`MODEL`, `TRACE_DIR`).
The keys read by `cli.py` are `llm.provider`, `llm.model`, `tracer.out_dir`,
and `graph.recursion_limit`; other keys in the file are documented defaults
but are not currently wired in.

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
tracer:
  out_dir: ./traces
graph:
  recursion_limit: 25
```

## Development

```bash
make dev    # install dev + dashboard extras
make test   # run pytest
make lint   # ruff check
```

CI workflow lives at `ci/test.yml.example`. Copy to `.github/workflows/` when
you want it enabled.

## Layout

```
src/agent_framework/
|-- core/            graph, state, nodes, edges, tracer
|-- tools/           web_search, calculator, code_executor, file_io
|-- llm/             openai, anthropic, litellm
|-- templates/       react, plan_act, supervisor_workers
`-- dashboard/       FastAPI trace viewer
```

## License

MIT.
