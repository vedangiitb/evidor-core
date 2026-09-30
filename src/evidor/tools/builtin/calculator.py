"""Safe arithmetic calculator built-in tool."""

import ast
import math
import operator

from ..core.tools import tool


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate_arithmetic(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_arithmetic(node.left)
        right = _evaluate_arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent magnitude must be 100 or less")
        result = _BINARY_OPERATORS[type(node.op)](left, right)
        if not math.isfinite(result):
            raise ValueError("Result must be finite")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        result = _UNARY_OPERATORS[type(node.op)](_evaluate_arithmetic(node.operand))
        if not math.isfinite(result):
            raise ValueError("Result must be finite")
        return result
    raise ValueError("Only basic arithmetic expressions are supported")


@tool
def calculator(expression: str) -> int | float:
    """Evaluate a basic arithmetic expression safely.

    Supports numbers, parentheses, +, -, *, /, //, %, and **.

    Args:
        expression: Arithmetic expression, such as '(12 + 3) * 4'.
    """
    if len(expression) > 256:
        raise ValueError("Expression must be 256 characters or fewer")
    try:
        parsed = ast.parse(expression, mode="eval")
        return _evaluate_arithmetic(parsed)
    except (SyntaxError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid arithmetic expression: {exc}") from exc
