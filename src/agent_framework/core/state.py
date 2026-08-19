"""Typed agent state.

State flows through the graph and is updated in place by nodes. Each node
returns a partial dict that is merged into the running state.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

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

    # Fields treated as ephemeral control queues: a patch *replaces* them rather
    # than appending. ``pending_calls`` is the tool-call queue that a ToolNode
    # must be able to drain by returning ``{"pending_calls": []}``; without
    # replace semantics the empty patch would append (a no-op) and the ReAct
    # loop would never clear the queue.
    _REPLACE_FIELDS: ClassVar[set[str]] = {"pending_calls"}

    input: str = ""
    messages: list[Message] = Field(default_factory=list)
    pending_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    output: Optional[str] = None
    step: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def merge(self, patch: dict[str, Any]) -> "AgentState":
        """Return a new state with ``patch`` merged in.

        - list-typed fields are appended (so messages accumulate)
        - dict-typed fields are shallow-merged (metadata)
        - fields in ``_REPLACE_FIELDS`` (e.g. ``pending_calls``) are replaced
        - everything else is replaced
        """
        data = self.model_dump()
        for k, v in patch.items():
            cur = data.get(k)
            if k in self._REPLACE_FIELDS:
                data[k] = v
            elif isinstance(cur, list) and isinstance(v, list):
                data[k] = cur + v
            elif isinstance(cur, dict) and isinstance(v, dict):
                data[k] = {**cur, **v}
            else:
                data[k] = v
        return type(self)(**data)
