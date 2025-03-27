"""Shared test fixtures."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def trace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure tests can't reach real APIs by accident."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY"):
        if var in os.environ:
            monkeypatch.setenv(var, "")
