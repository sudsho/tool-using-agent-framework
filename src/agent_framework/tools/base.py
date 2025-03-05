"""Tool base class + a tiny registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict


class ToolSchema(BaseModel):
    """JSON-schema-ish description used to build the prompt for the LLM."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    parameters: dict[str, Any]


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        ...

    def schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description, parameters=self.parameters)


class FuncTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., Any],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        if parameters is not None:
            self.parameters = parameters
        self._fn = fn

    def run(self, **kwargs: Any) -> Any:
        return self._fn(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def schemas(self) -> list[ToolSchema]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools
