"""Base node abstractions.

A node receives the running ``AgentState`` and returns a partial dict (or a new
state instance) that gets merged in. There are three convenience subclasses:

- ``LLMNode``  - calls an LLM and writes the assistant message
- ``ToolNode`` - executes a tool from the registry on pending tool calls
- ``RouterNode`` - returns the next node name (used as a conditional edge body)

Heavy logic stays in subclasses; this file is mostly the contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from .state import AgentState


class Node(ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, state: AgentState) -> dict[str, Any]:
        """Execute the node. Return a partial state dict to merge."""

    def __call__(self, state: AgentState) -> dict[str, Any]:
        return self.run(state)


class FuncNode(Node):
    """Wrap a plain function as a node."""

    def __init__(self, name: str, fn: Callable[[AgentState], dict[str, Any]]) -> None:
        super().__init__(name)
        self._fn = fn

    def run(self, state: AgentState) -> dict[str, Any]:
        return self._fn(state)


class RouterNode(Node):
    """A router decides which node runs next based on state."""

    def __init__(self, name: str, fn: Callable[[AgentState], str]) -> None:
        super().__init__(name)
        self._fn = fn

    def run(self, state: AgentState) -> dict[str, Any]:  # pragma: no cover
        # Routers are usually consumed by conditional edges and don't mutate state.
        return {}

    def route(self, state: AgentState) -> str:
        return self._fn(state)
