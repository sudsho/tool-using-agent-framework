"""ReAct-style template scaffold: an LLM node and a tool node connected by a
conditional edge.

```
START -> llm -> [tool_use?] - yes -> tools -> llm
                            - no  -> END
```
"""
from __future__ import annotations

from typing import Optional

from ..core.edge import END
from ..core.graph import StateGraph
from ..core.node import RouterNode
from ..core.nodes import LLMNode, ToolNode
from ..core.state import AgentState
from ..llm.base import LLMClient
from ..tools.base import ToolRegistry


REACT_SYSTEM = """You are a tool-using assistant. Think step by step.
When you need information you don't have, call a tool. When you have enough
information to answer, reply directly without calling any tool.
"""


def build_react_agent(
    client: LLMClient,
    tools: ToolRegistry,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> StateGraph:
    g = StateGraph()
    g.add_node(
        LLMNode(
            "llm",
            client=client,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt or REACT_SYSTEM,
        )
    )
    g.add_node(ToolNode("tools", registry=tools))

    def route_after_llm(state: AgentState) -> str:
        return "tool" if state.pending_calls else "done"

    g.add_node(RouterNode("after_llm", route_after_llm))

    g.set_entry_point("llm")
    g.add_conditional_edges(
        "llm",
        route_after_llm,
        mapping={"tool": "tools", "done": END},
    )
    g.add_edge("tools", "llm")
    return g
