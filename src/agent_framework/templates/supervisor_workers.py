"""Supervisor + workers template.

A supervisor LLM picks which named worker (each itself a small ReAct agent)
should handle the next step, until it decides the task is done.

```
supervisor -> [pick worker] -> worker_X -> supervisor -> ... -> END
```
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.edge import END
from ..core.graph import StateGraph
from ..core.node import Node
from ..core.state import AgentState, Message
from ..llm.base import LLMClient
from ..tools.base import ToolRegistry
from .react import build_react_agent


class Decision(BaseModel):
    next: str = Field(description="Name of the next worker, or 'finish'.")
    rationale: str = Field(default="")


SUPERVISOR_SYSTEM_TMPL = """You orchestrate a team of specialised workers.
Available workers: {workers}.
Look at the conversation so far and decide which worker should run next.
Respond 'finish' once the user goal is solved."""


class SupervisorNode(Node):
    def __init__(self, name: str, client: LLMClient, worker_names: list[str]) -> None:
        super().__init__(name)
        self.client = client
        self.worker_names = worker_names

    def run(self, state: AgentState) -> dict[str, Any]:
        sys = SUPERVISOR_SYSTEM_TMPL.format(workers=", ".join(self.worker_names))
        history = "\n".join(f"{m.role}: {m.content[:400]}" for m in state.messages[-6:])
        msgs = [
            {"role": "system", "content": sys},
            {
                "role": "user",
                "content": f"Goal: {state.input}\n\nRecent:\n{history}\n\nNext worker?",
            },
        ]
        resp = self.client.call(msgs, response_model=Decision, max_tokens=256)
        try:
            decision: Decision = resp.raw
        except Exception:
            decision = Decision(next="finish")
        return {
            "metadata": {**state.metadata, "next_worker": decision.next},
            "messages": [
                Message(role="assistant", content=f"supervisor->{decision.next}: {decision.rationale}")
            ],
        }


class WorkerWrapper(Node):
    def __init__(self, name: str, agent_graph) -> None:
        super().__init__(name)
        self.compiled = agent_graph.compile()

    def run(self, state: AgentState) -> dict[str, Any]:
        new_state = self.compiled.invoke(state)
        # Fold the worker's transcript back into the parent.
        return {
            "messages": new_state.messages[len(state.messages) :],
            "output": new_state.output,
        }


def build_supervisor_team(
    client: LLMClient, workers: dict[str, ToolRegistry]
) -> StateGraph:
    g = StateGraph()
    sup = SupervisorNode("supervisor", client=client, worker_names=list(workers.keys()))
    g.add_node(sup)
    for name, registry in workers.items():
        worker_graph = build_react_agent(client, registry)
        g.add_node(WorkerWrapper(name, worker_graph))

    def route_supervisor(state: AgentState) -> str:
        nxt = state.metadata.get("next_worker", "finish")
        if nxt == "finish":
            return "finish"
        return nxt if nxt in workers else "finish"

    g.set_entry_point("supervisor")
    mapping = {n: n for n in workers}
    mapping["finish"] = END
    g.add_conditional_edges("supervisor", route_supervisor, mapping=mapping)
    for name in workers:
        g.add_edge(name, "supervisor")
    return g
