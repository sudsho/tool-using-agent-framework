"""End-to-end graph tests with stub nodes (no LLM)."""

from agent_framework import END, AgentState, FuncNode, RouterNode, StateGraph


def test_linear_graph_runs():
    g = StateGraph()
    g.add_node(FuncNode("a", lambda s: {"messages": []}))
    g.add_node(FuncNode("b", lambda s: {"output": "done"}))
    g.add_edge("a", "b")
    g.add_edge("b", END)
    g.set_entry_point("a")

    out = g.compile().invoke(AgentState(input="x"))
    assert out.output == "done"


def test_conditional_edge_branches():
    g = StateGraph()
    g.add_node(FuncNode("start", lambda s: {"step": s.step + 1}))
    g.add_node(FuncNode("loop", lambda s: {"step": s.step + 1}))
    g.add_node(FuncNode("end", lambda s: {"output": f"steps={s.step}"}))

    def router(state):
        return "loop" if state.step < 3 else "end"

    g.set_entry_point("start")
    g.add_conditional_edges("start", router, mapping={"loop": "loop", "end": "end"})
    g.add_conditional_edges("loop", router, mapping={"loop": "loop", "end": "end"})
    g.add_edge("end", END)

    out = g.compile(recursion_limit=10).invoke(AgentState())
    assert out.output is not None
    assert out.output.startswith("steps=")


def test_recursion_limit_raises():
    g = StateGraph()
    g.add_node(FuncNode("loop", lambda s: {"step": s.step + 1}))
    g.add_edge("loop", "loop")
    g.set_entry_point("loop")

    try:
        g.compile(recursion_limit=5).invoke(AgentState())
    except RuntimeError as e:
        assert "recursion limit" in str(e)
    else:
        raise AssertionError("expected recursion limit error")


def test_router_node_jumps_directly():
    visited = []
    g = StateGraph()
    g.add_node(FuncNode("a", lambda s: visited.append("a") or {}))
    g.add_node(FuncNode("b", lambda s: visited.append("b") or {"output": "via-b"}))
    g.add_node(FuncNode("c", lambda s: visited.append("c") or {"output": "via-c"}))
    g.add_node(RouterNode("pick", lambda s: "b"))
    g.set_entry_point("a")
    g.add_edge("a", "pick")
    g.add_edge("b", END)
    g.add_edge("c", END)

    out = g.compile().invoke(AgentState())
    assert out.output == "via-b"
    assert "c" not in visited
