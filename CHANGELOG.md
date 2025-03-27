# Changelog

## 0.1.0 - 2025-03-27

Initial cut.

- core: `StateGraph` builder + `CompiledGraph` runner, conditional edges, recursion limit
- core: typed `AgentState` (pydantic v2) with list-append + dict-merge semantics
- core: `Tracer` writing OTel-style spans to JSONL grouped by `trace_id`
- tools: `web_search` (Tavily/SerpAPI), `calculator` (AST safe-eval),
  `code_executor` (subprocess sandbox), `file_io` (sandboxed read/write/list)
- llm: `OpenAIClient`, `AnthropicClient`, `LiteLLMClient`
  (with optional `instructor` for structured outputs)
- templates: `react`, `plan_act`, `supervisor_workers`
- dashboard: minimal FastAPI viewer for traces
- examples: research-assistant, math-solver, code-fixer
- tests: state, graph, tracer, nodes, tools, templates, supervisor, dashboard
