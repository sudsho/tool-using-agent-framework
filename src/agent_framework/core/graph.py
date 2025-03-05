"""StateGraph - the compiled execution unit.

API roughly mirrors LangGraph but kept tiny on purpose. Build a graph by
adding nodes and edges, set entry+finish points, then call ``compile()`` to
get a runnable object whose ``invoke(state)`` walks the graph.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .edge import END, START, ConditionalEdge, Edge
from .node import Node, RouterNode
from .state import AgentState
from .tracer import Tracer


class StateGraph:
    def __init__(self, state_cls: type[AgentState] = AgentState) -> None:
        self.state_cls = state_cls
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._cond: list[ConditionalEdge] = []
        self._entry: Optional[str] = None
        self._finish: Optional[str] = None

    # ----- builder API -----
    def add_node(self, node: Node) -> "StateGraph":
        if node.name in self._nodes:
            raise ValueError(f"duplicate node {node.name!r}")
        self._nodes[node.name] = node
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self._edges.append(Edge(src=src, dst=dst))
        return self

    def add_conditional_edges(
        self,
        src: str,
        router: Callable[[AgentState], str],
        mapping: dict[str, str],
        default: Optional[str] = None,
    ) -> "StateGraph":
        self._cond.append(ConditionalEdge(src=src, router=router, mapping=mapping, default=default))
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self._entry = name
        return self

    def set_finish_point(self, name: str) -> "StateGraph":
        self._finish = name
        return self

    # ----- compile -----
    def compile(
        self, tracer: Optional[Tracer] = None, recursion_limit: int = 25
    ) -> "CompiledGraph":
        if self._entry is None:
            raise ValueError("entry point not set")
        return CompiledGraph(self, tracer=tracer, recursion_limit=recursion_limit)


class CompiledGraph:
    def __init__(
        self, g: StateGraph, tracer: Optional[Tracer] = None, recursion_limit: int = 25
    ) -> None:
        self._g = g
        self.tracer = tracer
        self.recursion_limit = recursion_limit

    def _next(self, current: str, state: AgentState) -> str:
        # conditional first
        for ce in self._g._cond:
            if ce.src == current:
                return ce.resolve(state)
        for e in self._g._edges:
            if e.src == current:
                return e.dst
        return END

    def invoke(self, state: AgentState | dict[str, Any]) -> AgentState:
        if isinstance(state, dict):
            state = self._g.state_cls(**state)
        if self.tracer is not None:
            self.tracer.new_trace()

        current = self._g._entry
        steps = 0
        while current and current != END:
            steps += 1
            if steps > self.recursion_limit:
                raise RuntimeError(f"recursion limit ({self.recursion_limit}) exceeded")
            node = self._g._nodes.get(current)  # type: ignore[arg-type]
            if node is None:
                raise KeyError(f"node {current!r} not registered")

            if isinstance(node, RouterNode):
                # routers are usually used via conditional edges; if reached
                # as a plain node we just consult its router and jump.
                current = node.route(state)
                continue

            if self.tracer is not None:
                with self.tracer.span(node.name, kind="node", step=steps) as sp:
                    patch = node.run(state)
                    sp.outputs = {"keys": list(patch.keys())}
            else:
                patch = node.run(state)
            state = state.merge(patch)
            state.step = steps

            if current == self._g._finish:
                break
            current = self._next(current, state)

        if self.tracer is not None:
            self.tracer.close()
        return state


__all__ = ["StateGraph", "CompiledGraph", "START", "END"]
