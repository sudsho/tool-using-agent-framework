from agent_framework.core.state import AgentState, Message, ToolCall


def test_default_state_is_empty():
    s = AgentState()
    assert s.input == ""
    assert s.messages == []
    assert s.pending_calls == []
    assert s.step == 0


def test_merge_extends_lists():
    s = AgentState(input="hi")
    out = s.merge(
        {
            "messages": [Message(role="user", content="hi")],
            "pending_calls": [ToolCall(id="1", name="calculator", args={"expression": "1+1"})],
        }
    )
    assert len(out.messages) == 1
    assert out.messages[0].role == "user"
    assert out.pending_calls[0].name == "calculator"
    # original is unchanged
    assert s.messages == []


def test_merge_replaces_scalar():
    s = AgentState(input="a")
    out = s.merge({"input": "b", "step": 3})
    assert out.input == "b"
    assert out.step == 3


def test_extra_metadata_round_trips():
    s = AgentState(metadata={"trace": "abc", "depth": 2})
    out = s.merge({"metadata": {"trace": "xyz", "depth": 3}})
    assert out.metadata["trace"] == "xyz"
    assert out.metadata["depth"] == 3
