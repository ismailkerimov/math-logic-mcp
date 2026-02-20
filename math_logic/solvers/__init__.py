"""Solver implementations."""

from math_logic.solvers.arithmetic_solver import ArithmeticSolver

# Optional solvers — imported lazily by the engine, but expose here for direct use
try:
    from math_logic.solvers.sympy_solver import SymPySolver
except Exception:
    SymPySolver = None  # type: ignore

try:
    from math_logic.solvers.z3_solver import Z3Solver
except Exception:
    Z3Solver = None  # type: ignore

__all__ = ["ArithmeticSolver", "SymPySolver", "Z3Solver"]
