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

## Quick start (runs offline)

No API keys, no network, no GPU. The smoke script drives a ReAct agent with a
rule-based `MockLLM` and the offline tools (real `calculator` + a canned
`web_search`), so the graph engine, tool registry, tracer, and the full ReAct
loop all run as-is.

```bash
pip install -e . --no-deps    # only pydantic is required for the core
python scripts/smoke.py       # or: make smoke
```

Real output:

```
=== Math task (calculator tool) ===
question: What is (12 * 8) + sqrt(256)?
reasoning / tool trace:
    [assistant] This is an arithmetic question. I'll use the calculator on `(12 * 8) + sqrt(256)`.
    [tool     ] {'value': 112.0}
    [assistant] The result of `What is (12 * 8) + sqrt(256)?` is 112.
final answer: The result of `What is (12 * 8) + sqrt(256)?` is 112.
  trace 46e4275416b64f5ea98b5bd1315519ea  (5 spans, file: 46e4275416b64f5ea98b5bd1315519ea.jsonl)
    - node   llm      patch=[messages,pending_calls]  1.0ms
    - event  route    src=llm dst=tools
    - node   tools    patch=[messages,tool_results,pending_calls]  0.0ms
    - node   llm      patch=[messages,pending_calls,output]  0.0ms
    - event  route    src=llm dst=__end__

=== Research task (web_search tool) ===
question: What is the population of Iceland?
reasoning / tool trace:
    [assistant] I need to look this up. I'll search the web for: What is the population of Iceland?
    [tool     ] {'results': [{'title': 'Iceland - Population', 'url': 'https://example.org/iceland', 's...
    [assistant] Iceland had an estimated population of about 393,600 people in 2024, making it the most...
final answer: Iceland had an estimated population of about 393,600 people in 2024, making it the most sparsely populated country in Europe. (source: Iceland - Population, https://example.org/iceland)
  trace 503ee0f529c449c38dc551525a68f20a  (5 spans, file: 503ee0f529c449c38dc551525a68f20a.jsonl)
    - node   llm      patch=[messages,pending_calls]  0.0ms
    - event  route    src=llm dst=tools
    - node   tools    patch=[messages,tool_results,pending_calls]  0.0ms
    - node   llm      patch=[messages,pending_calls,output]  0.0ms
    - event  route    src=llm dst=__end__

SMOKE OK: both tasks ran offline, called tools, and returned answers.
```

Tests (also offline):

```bash
python -m pytest -q
# 39 passed
```

To run against a real provider instead, wire in `OpenAIClient` / `AnthropicClient`
and set the matching key (see below).

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
| `web_search`   | Tavily, SerpAPI, or an offline `mock` corpus         |
| `calculator`   | Safe AST-based arithmetic                            |
| `code_executor`| Python in a subprocess with a wall-clock timeout     |
| `file_io`      | Sandboxed read/write under a configured root         |

## LLM clients

`OpenAIClient`, `AnthropicClient`, and `LiteLLMClient` wrap the real providers
(pick one via `make_client(provider, model)` and set the matching key).
`MockLLM` is a rule-based, dependency-free client that decides tool calls from
the query and composes a final answer from tool output. It needs no key and no
network, so it powers the offline smoke and tests above.

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
