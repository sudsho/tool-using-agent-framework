"""Tiny config loader. Reads YAML and merges with env-var overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load(path: str | os.PathLike[str] = "configs/default.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    # env overrides for the few we care about
    raw.setdefault("llm", {})
    if os.getenv("MODEL"):
        raw["llm"]["model"] = os.environ["MODEL"]
    if os.getenv("TRACE_DIR"):
        raw.setdefault("tracer", {})["out_dir"] = os.environ["TRACE_DIR"]
    return raw


def get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
