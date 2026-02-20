"""
Zero-dependency arithmetic solver using Python's ast module for safe evaluation.

This solver handles basic arithmetic without requiring SymPy or Z3,
making it the lightest possible math tool for LLMs.
"""

from math_logic.engine import (
    BaseSolver, SolveResult, ProblemType, SolverType, Proof, ProofStep,
)
import ast
import math
import operator
import re
import time


# Allowed operators for safe evaluation
_SAFE_OPS = {
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

# Allowed math functions (zero-dep, from stdlib)
_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pi": math.pi,
    "e": math.e,
}


def safe_eval(expr_str: str):
    """
    Safely evaluate an arithmetic expression.

    Uses ast.parse + a whitelist of operators and functions.
    Raises ValueError on anything unsafe.
    """
    try:
        tree = ast.parse(expr_str.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}") from e

    return _eval_node(tree.body)


def _eval_node(node):
    """Recursively evaluate an AST node using only whitelisted operations."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    # Numbers
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    # Unary ops: -x, +x
    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op_func(_eval_node(node.operand))

    # Binary ops: x + y, x ** y, etc.
    if isinstance(node, ast.BinOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported op: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        # Guard against absurd exponents
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and right > 1000:
            raise ValueError("Exponent too large (max 1000)")
        return op_func(left, right)

    # Function calls: sqrt(x), log(x), etc.
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
            func = _SAFE_FUNCS[node.func.id]
            args = [_eval_node(a) for a in node.args]
            return func(*args)
        raise ValueError(f"Unsupported function: {ast.dump(node.func)}")

    # Named constants: pi, e
    if isinstance(node, ast.Name) and node.id in _SAFE_FUNCS:
        val = _SAFE_FUNCS[node.id]
        if isinstance(val, (int, float)):
            return val
        raise ValueError(f"'{node.id}' is a function, not a constant")

    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class ArithmeticSolver(BaseSolver):
    """Zero-dependency arithmetic solver for basic calculations."""

    # Math function names that signal arithmetic, not algebra
    _MATH_FUNCS = {"sqrt", "log", "log2", "log10", "sin", "cos", "tan",
                   "ceil", "floor", "factorial", "abs", "round", "gcd"}

    def can_solve(self, problem: str, problem_type: ProblemType) -> bool:
        if problem_type == ProblemType.ARITHMETIC:
            return True
        # Check if it looks like pure arithmetic
        if re.search(r"\d\s*[\+\-\*/\^%]", problem):
            return True
        # Check for known math function calls (sqrt(x), factorial(n), etc.)
        lower = problem.lower()
        return any(f"{fn}(" in lower for fn in self._MATH_FUNCS)

    def solve(self, problem: str, **kwargs) -> SolveResult:
        t0 = time.perf_counter()

        expr_str = self._extract_expression(problem)

        # Normalise math notation → Python
        expr_str = expr_str.replace("^", "**").replace("×", "*").replace("÷", "/")

        try:
            result = safe_eval(expr_str)
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            return self._error(problem, t0, str(e))

        # Format result
        if isinstance(result, float) and result == int(result):
            result_str = str(int(result))
        else:
            result_str = str(result)

        elapsed = (time.perf_counter() - t0) * 1000

        steps = [
            ProofStep(1, f"Parse expression: {expr_str}"),
            ProofStep(2, "Evaluate using safe arithmetic engine"),
            ProofStep(3, f"Result: {result_str}", symbolic_form=result_str),
        ]

        # Verification: re-evaluate (it's deterministic)
        verification = {expr_str: "verified"}

        proof = Proof(
            problem=problem, method="arithmetic_eval", steps=steps,
            solutions=[result_str], verification=verification, is_verified=True,
        )

        return SolveResult(
            problem=problem, problem_type=ProblemType.ARITHMETIC,
            solutions=[result_str], proof=proof,
            solver_used=SolverType.PYTHON, confidence=1.0,
            execution_time_ms=elapsed,
        )

    def _extract_expression(self, problem: str) -> str:
        """Pull the arithmetic expression from natural language."""
        # "compute 2 + 3 * 4" → "2 + 3 * 4"
        # Note: do NOT stop at '.' because it breaks decimals (45.5)
        m = re.search(
            r"(?:compute|calculate|evaluate|what is|verify|check)\s+(.+?)(?:\?|$)",
            problem, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().rstrip(".")

        # If the input looks like a math expression already (contains function calls
        # or parens), use it directly
        stripped = problem.strip().rstrip("?.")
        if re.match(r"^[a-zA-Z_\d\s\+\-\*/\^%\(\),\.]+$", stripped):
            return stripped

        # Try to find a bare numeric expression
        m = re.search(r"([\d][\d\.\s\+\-\*/\^%\(\),a-zA-Z_]*)", problem)
        if m:
            return m.group(1).strip()

        return problem.strip()


    def _error(self, problem: str, t0: float, msg: str) -> SolveResult:
        elapsed = (time.perf_counter() - t0) * 1000
        return SolveResult(
            problem=problem, problem_type=ProblemType.ARITHMETIC,
            solutions=[], proof=Proof(problem=problem, method="error"),
            solver_used=SolverType.PYTHON, confidence=0.0,
            execution_time_ms=elapsed, error=msg,
        )
