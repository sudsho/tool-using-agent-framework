"""Tests for the built-in tools."""

import math
import os

import pytest

from agent_framework.tools import CalculatorTool, FileIOTool, ToolRegistry, default_registry
from agent_framework.tools.calculator import safe_eval
from agent_framework.tools.file_io import _safe_join


class TestCalculator:
    def test_basic_arithmetic(self):
        c = CalculatorTool()
        assert c.run(expression="2 + 3 * 4")["value"] == 14
        assert c.run(expression="(2 + 3) * 4")["value"] == 20

    def test_math_functions(self):
        c = CalculatorTool()
        assert c.run(expression="sqrt(16)")["value"] == 4.0
        assert math.isclose(c.run(expression="sin(0)")["value"], 0.0, abs_tol=1e-9)

    def test_constants(self):
        c = CalculatorTool()
        assert math.isclose(c.run(expression="pi")["value"], math.pi)

    def test_blocks_unsafe(self):
        c = CalculatorTool()
        out = c.run(expression="__import__('os').system('echo hi')")
        assert "error" in out

    def test_safe_eval_rejects_string(self):
        with pytest.raises(ValueError):
            safe_eval("'abc'")


class TestFileIO:
    def test_write_and_read(self, tmp_path):
        t = FileIOTool(sandbox_root=tmp_path)
        wr = t.run(op="write", path="hello.txt", content="hi there")
        assert wr["bytes"] == 8
        rd = t.run(op="read", path="hello.txt")
        assert rd["content"] == "hi there"

    def test_list_dir(self, tmp_path):
        t = FileIOTool(sandbox_root=tmp_path)
        t.run(op="write", path="a.txt", content="a")
        t.run(op="write", path="b.txt", content="b")
        out = t.run(op="list", path=".")
        assert set(out["entries"]) == {"a.txt", "b.txt"}

    def test_escape_blocked(self, tmp_path):
        t = FileIOTool(sandbox_root=tmp_path)
        out = t.run(op="read", path="../../../etc/passwd")
        assert "error" in out

    def test_safe_join_rejects_absolute(self, tmp_path):
        with pytest.raises(ValueError):
            _safe_join(tmp_path, os.path.abspath(__file__))


class TestRegistry:
    def test_default_registry_has_known_tools(self, tmp_path):
        reg = default_registry(sandbox_root=str(tmp_path))
        assert "calculator" in reg
        assert "web_search" in reg
        assert "code_executor" in reg
        assert "file_io" in reg

    def test_register_dup_raises(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        with pytest.raises(ValueError):
            reg.register(CalculatorTool())
