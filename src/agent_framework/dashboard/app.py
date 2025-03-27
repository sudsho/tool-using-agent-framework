"""Tiny FastAPI dashboard for browsing recorded traces.

Usage:
    uvicorn agent_framework.dashboard.app:app --port 8080
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse


app = FastAPI(title="agent-framework dashboard", version="0.1.0")


def _trace_dir() -> Path:
    """Resolve the trace directory each call so env changes are picked up."""
    return Path(os.getenv("TRACE_DIR", "./traces"))


def _list_traces() -> list[dict[str, Any]]:
    td = _trace_dir()
    if not td.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(td.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
        out.append(
            {
                "trace_id": p.stem,
                "file": str(p),
                "size_bytes": p.stat().st_size,
                "modified_at": p.stat().st_mtime,
            }
        )
    return out


def _load_spans(trace_id: str) -> list[dict[str, Any]]:
    f = _trace_dir() / f"{trace_id}.jsonl"
    if not f.exists():
        raise HTTPException(404, f"unknown trace {trace_id}")
    spans: list[dict[str, Any]] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            spans.append(json.loads(line))
    return spans


@app.get("/api/traces")
def list_traces() -> list[dict[str, Any]]:
    return _list_traces()


@app.get("/api/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    spans = _load_spans(trace_id)
    return {"trace_id": trace_id, "spans": spans, "n": len(spans)}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    rows = _list_traces()
    if not rows:
        body = "<p>No traces found yet. Run an agent first.</p>"
    else:
        items = "\n".join(
            f'<li><a href="/trace/{r["trace_id"]}">{r["trace_id"]}</a> '
            f'({r["size_bytes"]} bytes)</li>'
            for r in rows
        )
        body = f"<ul>{items}</ul>"
    return _PAGE.format(title="Traces", body=body)


@app.get("/trace/{trace_id}", response_class=HTMLResponse)
def trace_view(trace_id: str) -> str:
    spans = _load_spans(trace_id)
    rows = []
    for sp in spans:
        dur = ""
        if sp.get("end_ms") and sp.get("start_ms"):
            dur = f"{sp['end_ms'] - sp['start_ms']:.1f} ms"
        status = sp.get("status", "ok")
        css = "ok" if status == "ok" else "err"
        rows.append(
            f"<tr class='{css}'><td>{sp.get('kind','')}</td>"
            f"<td>{sp.get('name','')}</td><td>{dur}</td>"
            f"<td>{status}</td><td><pre>{json.dumps(sp.get('inputs', {}), indent=1)[:400]}</pre></td>"
            f"<td><pre>{json.dumps(sp.get('outputs', {}), indent=1)[:400]}</pre></td></tr>"
        )
    table = (
        "<table><thead><tr><th>kind</th><th>name</th><th>dur</th><th>status</th>"
        "<th>inputs</th><th>outputs</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )
    return _PAGE.format(title=f"Trace {trace_id}", body=table)


_PAGE = """<!doctype html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
body {{ font-family: -apple-system, sans-serif; padding: 24px; max-width: 1100px; }}
h1 {{ font-size: 18px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
th {{ background: #f5f5f5; text-align: left; }}
tr.err {{ background: #ffecec; }}
pre {{ margin: 0; white-space: pre-wrap; }}
a {{ color: #0d6efd; text-decoration: none; }}
</style></head>
<body>
<h1>{title}</h1>
<p><a href='/'>back</a></p>
{body}
</body></html>
"""
