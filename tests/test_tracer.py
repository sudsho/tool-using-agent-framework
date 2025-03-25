"""Tests for the tracer span recording."""

import json

import pytest

from agent_framework import Tracer


def test_span_writes_jsonl(tmp_path):
    t = Tracer(out_dir=tmp_path)
    t.new_trace()
    with t.span("my_node", kind="node", x=1) as sp:
        sp.outputs = {"y": 2}

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    spans = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    assert len(spans) == 1
    assert spans[0]["name"] == "my_node"
    assert spans[0]["status"] == "ok"
    assert spans[0]["inputs"] == {"x": 1}
    assert spans[0]["outputs"] == {"y": 2}
    assert spans[0]["end_ms"] >= spans[0]["start_ms"]


def test_span_records_error(tmp_path):
    t = Tracer(out_dir=tmp_path)
    t.new_trace()
    with pytest.raises(RuntimeError):
        with t.span("bad", kind="node"):
            raise RuntimeError("boom")
    files = list(tmp_path.glob("*.jsonl"))
    spans = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    assert spans[0]["status"] == "error"
    assert "RuntimeError" in spans[0]["error"]


def test_nested_spans_get_parent_id(tmp_path):
    t = Tracer(out_dir=tmp_path)
    t.new_trace()
    with t.span("outer") as outer:
        with t.span("inner") as inner:
            assert inner.parent_id == outer.span_id
        # outer has no parent
        assert outer.parent_id is None


def test_event_writes_zero_duration_span(tmp_path):
    t = Tracer(out_dir=tmp_path)
    t.new_trace()
    t.event("note", message="hello")
    files = list(tmp_path.glob("*.jsonl"))
    spans = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    assert spans[0]["kind"] == "event"
    assert spans[0]["attrs"]["message"] == "hello"
