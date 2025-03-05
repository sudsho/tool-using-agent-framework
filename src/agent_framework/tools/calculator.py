"""A safe calculator tool.

Uses Python's ``ast`` module to evaluate arithmetic expressions without
giving the LLM eval/exec access. Supports +, -, *, /, **, %, parentheses,
and a small whitelist of math functions (sqrt, sin, cos, log, etc).
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from .base import Tool


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}

_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"unsupported constant {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"unknown identifier {node.id!r}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported op {type(node.op).__name__}")
        return op(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary op {type(node.op).__name__}")
        return op(_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("only whitelisted math functions allowed")
        args = [_eval(a) for a in node.args]
        return _FUNCS[node.func.id](*args)
    raise ValueError(f"unsupported expr node {type(node).__name__}")


def safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate an arithmetic expression. Supports basic ops and math functions."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"}
        },
        "required": ["expression"],
    }

    def run(self, expression: str, **_: Any) -> dict[str, Any]:
        try:
            value = safe_eval(expression)
            return {"value": value}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
