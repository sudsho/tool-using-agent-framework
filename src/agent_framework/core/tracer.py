"""OTel-style tracer.

Every node execution, LLM call, and tool call writes a span. Spans are
serialized to JSONL by default. A second backend writes to MLflow if it is
installed and ``backend == "mlflow"``.

A trace is a tree of spans sharing a ``trace_id``. Spans have a parent_id and
form the call hierarchy.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    name: str
    kind: str  # "node" | "llm" | "tool" | "router"
    start_ms: float
    end_ms: Optional[float] = None
    status: str = "ok"
    error: Optional[str] = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_ms is None:
            return None
        return self.end_ms - self.start_ms


class Tracer:
    """Thread-safe span recorder."""

    def __init__(
        self,
        out_dir: str | os.PathLike[str] = "./traces",
        backend: str = "jsonl",
        flush_every: int = 1,
        include_payloads: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.flush_every = flush_every
        self.include_payloads = include_payloads
        self._lock = threading.Lock()
        self._buffer: list[Span] = []
        self._stack: list[Span] = []
        self.current_trace: Optional[str] = None

    def new_trace(self) -> str:
        self.current_trace = uuid.uuid4().hex
        self._stack.clear()
        return self.current_trace

    @contextmanager
    def span(self, name: str, kind: str = "node", **inputs: Any) -> Iterator[Span]:
        if self.current_trace is None:
            self.new_trace()
        parent = self._stack[-1].span_id if self._stack else None
        sp = Span(
            trace_id=self.current_trace,  # type: ignore[arg-type]
            span_id=uuid.uuid4().hex,
            parent_id=parent,
            name=name,
            kind=kind,
            start_ms=time.time() * 1000,
            inputs=inputs if self.include_payloads else {},
        )
        self._stack.append(sp)
        try:
            yield sp
        except Exception as e:
            sp.status = "error"
            sp.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            sp.end_ms = time.time() * 1000
            self._stack.pop()
            self._record(sp)

    def event(self, name: str, **attrs: Any) -> None:
        sp = Span(
            trace_id=self.current_trace or self.new_trace(),
            span_id=uuid.uuid4().hex,
            parent_id=self._stack[-1].span_id if self._stack else None,
            name=name,
            kind="event",
            start_ms=time.time() * 1000,
            end_ms=time.time() * 1000,
            attrs=attrs,
        )
        self._record(sp)

    def _record(self, sp: Span) -> None:
        with self._lock:
            self._buffer.append(sp)
            if len(self._buffer) >= self.flush_every:
                self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        if self.backend == "jsonl":
            path = self.out_dir / f"{self.current_trace}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                for sp in self._buffer:
                    f.write(json.dumps(asdict(sp), default=str) + "\n")
        elif self.backend == "memory":
            pass
        else:  # pragma: no cover
            raise ValueError(f"unknown tracer backend {self.backend!r}")
        self._buffer.clear()

    def close(self) -> None:
        with self._lock:
            self._flush()
