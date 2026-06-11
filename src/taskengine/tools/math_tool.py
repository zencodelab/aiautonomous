"""
Mathematical computation tool — safe expression evaluation.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool

# Allowed operators for safe math evaluation
_SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed math constants and functions
_SAFE_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
}


def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluate an AST node with only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    elif isinstance(node, ast.BinOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_NAMES:
            func = _SAFE_NAMES[node.func.id]
            args = [_safe_eval(arg) for arg in node.args]
            return func(*args)
        raise ValueError(f"Unsupported function call: {ast.dump(node.func)}")
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    else:
        raise ValueError(f"Unsupported expression: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports basic arithmetic (+, -, *, /, //, %, **), math functions
    (sqrt, log, sin, cos, etc.), and constants (pi, e).

    Use this tool for any mathematical computation — it's faster and
    more reliable than writing code for simple math.

    Args:
        expression: Mathematical expression to evaluate.
                    Examples: "2 ** 10", "sqrt(144)", "log(100, 10)",
                    "sin(pi / 4)", "round(3.14159, 2)"

    Returns:
        The result of the computation as a string.
    """
    if not expression.strip():
        return "Error: No expression provided."

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return f"{expression.strip()} = {result}"
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError) as exc:
        return f"Error evaluating '{expression}': {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"
