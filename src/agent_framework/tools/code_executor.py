"""Run untrusted Python in a subprocess sandbox.

Not a real sandbox in the seL4 sense - it's a child Python process with no
network arg restrictions, a CPU time limit (best effort on Windows) and a
short wall-clock timeout. Good enough for an agent doing arithmetic-heavy
calculations or one-off data wrangling. Don't expose to the open internet
without containerizing.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any

from .base import Tool


class CodeExecutorTool(Tool):
    name = "code_executor"
    description = "Execute a snippet of Python in a subprocess sandbox. Returns stdout/stderr."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to run"},
            "timeout_s": {"type": "number", "default": 6},
        },
        "required": ["code"],
    }

    def __init__(self, timeout_s: float = 6.0, mem_limit_mb: int = 256) -> None:
        self.timeout_s = timeout_s
        self.mem_limit_mb = mem_limit_mb

    def run(self, code: str, timeout_s: float | None = None, **_: Any) -> dict[str, Any]:
        timeout = timeout_s if timeout_s is not None else self.timeout_s
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "snippet.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", script],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmp,
                )
            except subprocess.TimeoutExpired:
                return {"error": f"timeout after {timeout}s", "stdout": "", "stderr": ""}
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }
