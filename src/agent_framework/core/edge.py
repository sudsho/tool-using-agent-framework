"""Edges between nodes in the graph.

Two flavours:

- A plain edge ``A -> B`` says "after A always go to B".
- A conditional edge wraps a router function ``state -> next_name`` and a
  mapping of those names to actual node names. This is the equivalent of
  LangGraph's ``add_conditional_edges``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .state import AgentState


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str


@dataclass
class ConditionalEdge:
    src: str
    router: Callable[[AgentState], str]
    mapping: dict[str, str]
    default: Optional[str] = None

    def resolve(self, state: AgentState) -> str:
        key = self.router(state)
        if key in self.mapping:
            return self.mapping[key]
        if self.default is not None:
            return self.default
        raise KeyError(
            f"router for {self.src!r} returned {key!r}, no entry in mapping and no default"
        )


# Special sentinel node names used by the graph builder.
START = "__start__"
END = "__end__"
