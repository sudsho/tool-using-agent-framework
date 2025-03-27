"""Smoke tests for the FastAPI dashboard."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def test_index_lists_traces(trace_dir, monkeypatch):
    # write a fake trace
    (trace_dir / "abc.jsonl").write_text(
        '{"trace_id":"abc","span_id":"s1","parent_id":null,"name":"test",'
        '"kind":"node","start_ms":1.0,"end_ms":2.0,"status":"ok",'
        '"error":null,"inputs":{},"outputs":{},"attrs":{}}\n',
        encoding="utf-8",
    )
    # reload module so it picks up TRACE_DIR
    from agent_framework.dashboard import app as dash
    client = TestClient(dash.app)

    r = client.get("/")
    assert r.status_code == 200
    assert "abc" in r.text

    j = client.get("/api/traces").json()
    assert any(item["trace_id"] == "abc" for item in j)

    detail = client.get("/api/traces/abc").json()
    assert detail["n"] == 1
    assert detail["spans"][0]["name"] == "test"


def test_unknown_trace_404s(trace_dir):
    from agent_framework.dashboard import app as dash
    client = TestClient(dash.app)
    r = client.get("/api/traces/missing")
    assert r.status_code == 404
