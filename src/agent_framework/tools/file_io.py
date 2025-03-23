"""Sandboxed file IO.

All paths are resolved relative to a sandbox root and any path that escapes
the root is rejected. Useful as a building block for agents that need to
write intermediate artefacts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import Tool


def _safe_join(root: Path, rel: str) -> Path:
    """Resolve ``rel`` against ``root`` and refuse if it escapes the sandbox."""
    if Path(rel).is_absolute():
        raise ValueError(f"path {rel!r} must be relative to sandbox")
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as e:
        raise ValueError(f"path {rel!r} escapes sandbox") from e
    return candidate


class FileIOTool(Tool):
    name = "file_io"
    description = "Read or write a UTF-8 text file inside the sandbox."
    parameters = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["read", "write", "list"]},
            "path": {"type": "string"},
            "content": {"type": "string", "description": "Required for write"},
        },
        "required": ["op", "path"],
    }

    def __init__(self, sandbox_root: str | os.PathLike[str] = "./sandbox") -> None:
        self.root = Path(sandbox_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run(self, op: str, path: str, content: str = "", **_: Any) -> dict[str, Any]:
        try:
            target = _safe_join(self.root, path)
        except ValueError as e:
            return {"error": str(e)}
        if op == "read":
            if not target.exists():
                return {"error": f"not found: {path}"}
            return {"content": target.read_text(encoding="utf-8")}
        if op == "write":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"bytes": len(content)}
        if op == "list":
            if not target.exists() or not target.is_dir():
                return {"error": f"not a directory: {path}"}
            return {"entries": sorted(p.name for p in target.iterdir())}
        return {"error": f"unknown op {op!r}"}
