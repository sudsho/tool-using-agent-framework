"""Typed agent state.

State flows through the graph and is updated in place by nodes. Each node
returns a partial dict that is merged into the running state.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    output: Any
    error: Optional[str] = None


class AgentState(BaseModel):
    """Default state schema. Subclass for custom fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    input: str = ""
    messages: list[Message] = Field(default_factory=list)
    pending_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    output: Optional[str] = None
    step: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def merge(self, patch: dict[str, Any]) -> "AgentState":
        data = self.model_dump()
        for k, v in patch.items():
            if k in data and isinstance(data[k], list) and isinstance(v, list):
                data[k] = data[k] + v
            else:
                data[k] = v
        return type(self)(**data)
