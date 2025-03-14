"""Concrete node implementations for the common cases."""
from __future__ import annotations

from typing import Any, Optional

from ..llm.base import LLMClient
from ..tools.base import ToolRegistry
from .node import Node
from .state import AgentState, Message, ToolCall, ToolResult


class LLMNode(Node):
    """Calls an LLM with the running message history. If the response includes
    tool calls, they're written to ``state.pending_calls`` for the next ToolNode
    to execute."""

    def __init__(
        self,
        name: str,
        client: LLMClient,
        tools: Optional[ToolRegistry] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.client = client
        self.tools = tools
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def _messages(self, state: AgentState) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        if state.input and not state.messages:
            msgs.append({"role": "user", "content": state.input})
        for m in state.messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.name:
                entry["name"] = m.name
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            msgs.append(entry)
        return msgs

    def run(self, state: AgentState) -> dict[str, Any]:
        tool_specs = None
        if self.tools is not None and self.tools.names():
            tool_specs = [s.model_dump() for s in self.tools.schemas()]
        resp = self.client.call(
            self._messages(state),
            tools=tool_specs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        new_msgs = [Message(role="assistant", content=resp.text or "")]
        pending = [ToolCall(id=tc.id, name=tc.name, args=tc.args) for tc in resp.tool_calls]
        patch: dict[str, Any] = {"messages": new_msgs, "pending_calls": pending}
        if not pending and resp.text:
            patch["output"] = resp.text
        return patch


class ToolNode(Node):
    """Executes any pending tool calls and writes results back as tool messages."""

    def __init__(self, name: str, registry: ToolRegistry) -> None:
        super().__init__(name)
        self.registry = registry

    def run(self, state: AgentState) -> dict[str, Any]:
        results: list[ToolResult] = []
        msgs: list[Message] = []
        for call in state.pending_calls:
            if call.name not in self.registry:
                tr = ToolResult(
                    call_id=call.id, output=None, error=f"unknown tool {call.name!r}"
                )
                results.append(tr)
                msgs.append(
                    Message(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=f"ERROR: unknown tool {call.name!r}",
                    )
                )
                continue
            tool = self.registry.get(call.name)
            try:
                out = tool.run(**call.args)
                results.append(ToolResult(call_id=call.id, output=out))
                msgs.append(
                    Message(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=str(out),
                    )
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                results.append(ToolResult(call_id=call.id, output=None, error=err))
                msgs.append(
                    Message(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=f"ERROR: {err}",
                    )
                )
        # consume pending_calls so the next LLM step sees an empty queue
        return {"messages": msgs, "tool_results": results, "pending_calls": []}
