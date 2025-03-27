"""Plan-and-execute template.

```
START -> planner -> executor -> [more_steps?] -- yes --> executor
                                              \-- no  --> summarizer -> END
```

The planner produces a structured ``Plan`` (list of steps) via instructor.
The executor turns each step into a ReAct sub-loop. The summarizer composes
the final answer from the step outputs.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.edge import END
from ..core.graph import StateGraph
from ..core.node import Node
from ..core.nodes import LLMNode, ToolNode
from ..core.state import AgentState, Message
from ..llm.base import LLMClient
from ..tools.base import ToolRegistry


class PlanStep(BaseModel):
    id: int
    description: str
    tool_hint: str | None = None


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)


PLANNER_SYSTEM = """You are a planner. Break the user goal into 3-7 concrete
steps. Each step is small enough to be solved with a single tool call or
short LLM reasoning."""


class PlannerNode(Node):
    def __init__(self, name: str, client: LLMClient, max_steps: int = 7) -> None:
        super().__init__(name)
        self.client = client
        self.max_steps = max_steps

    def run(self, state: AgentState) -> dict[str, Any]:
        msgs = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": state.input},
        ]
        resp = self.client.call(msgs, response_model=Plan)
        # response_model returns json text; we stash the parsed plan in metadata
        try:
            plan_obj: Plan = resp.raw  # instructor returns the model directly
        except Exception:
            plan_obj = Plan(goal=state.input, steps=[])
        steps = plan_obj.steps[: self.max_steps]
        return {
            "metadata": {**state.metadata, "plan": [s.model_dump() for s in steps], "step_idx": 0},
            "messages": [Message(role="assistant", content=f"plan: {len(steps)} steps")],
        }


class ExecutorNode(Node):
    """Runs one step of the plan as a single LLM+tool round."""

    def __init__(
        self, name: str, client: LLMClient, tools: ToolRegistry, max_tokens: int = 1024
    ) -> None:
        super().__init__(name)
        self.llm_node = LLMNode("plan_llm", client=client, tools=tools, max_tokens=max_tokens)
        self.tool_node = ToolNode("plan_tools", registry=tools)

    def run(self, state: AgentState) -> dict[str, Any]:
        plan = state.metadata.get("plan", [])
        idx = state.metadata.get("step_idx", 0)
        if idx >= len(plan):
            return {"metadata": {**state.metadata, "step_idx": idx}}
        step = plan[idx]
        prompt = f"Working on plan step {idx + 1}: {step['description']}"
        sub_state = state.merge(
            {"messages": [Message(role="user", content=prompt)], "pending_calls": []}
        )
        before = len(sub_state.messages)
        sub = self.llm_node.run(sub_state)
        sub_state = sub_state.merge(sub)
        if sub_state.pending_calls:
            tool_patch = self.tool_node.run(sub_state)
            sub_state = sub_state.merge(tool_patch)
        new_msgs = sub_state.messages[before:]
        return {
            "messages": new_msgs,
            "metadata": {**state.metadata, "step_idx": idx + 1},
        }


class SummarizerNode(Node):
    def __init__(self, name: str, client: LLMClient) -> None:
        super().__init__(name)
        self.client = client

    def run(self, state: AgentState) -> dict[str, Any]:
        history = "\n".join(f"{m.role}: {m.content}" for m in state.messages[-10:])
        msgs = [
            {"role": "system", "content": "Summarize the work into a clean final answer."},
            {"role": "user", "content": f"Goal: {state.input}\n\nWork:\n{history}"},
        ]
        resp = self.client.call(msgs, max_tokens=512)
        return {"output": resp.text, "messages": [Message(role="assistant", content=resp.text)]}


def build_plan_act_agent(
    client: LLMClient, tools: ToolRegistry, max_steps: int = 7
) -> StateGraph:
    g = StateGraph()
    g.add_node(PlannerNode("planner", client=client, max_steps=max_steps))
    g.add_node(ExecutorNode("executor", client=client, tools=tools))
    g.add_node(SummarizerNode("summarizer", client=client))

    def more_steps(state: AgentState) -> str:
        plan = state.metadata.get("plan", [])
        idx = state.metadata.get("step_idx", 0)
        return "more" if idx < len(plan) else "done"

    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_conditional_edges(
        "executor", more_steps, mapping={"more": "executor", "done": "summarizer"}
    )
    g.add_edge("summarizer", END)
    return g
