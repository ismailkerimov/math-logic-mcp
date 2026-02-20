"""
Tests for math-logic-mcp solvers and engine.

Run:  pytest tests/ -v
"""

import json
import pytest

from math_logic.engine import MathLogicEngine, ProblemType, SolverType
from math_logic.router import Router
from math_logic.solvers.arithmetic_solver import ArithmeticSolver, safe_eval


# ═══════════════════════════════════════════════════════════════
# ArithmeticSolver & safe_eval (zero-dep, always available)
# ═══════════════════════════════════════════════════════════════

class TestSafeEval:
    def test_basic_add(self):
        assert safe_eval("2 + 3") == 5

    def test_order_of_operations(self):
        assert safe_eval("2 + 3 * 4") == 14

    def test_parentheses(self):
        assert safe_eval("(2 + 3) * 4") == 20

    def test_power(self):
        assert safe_eval("2 ** 10") == 1024

    def test_negative(self):
        assert safe_eval("-5 + 3") == -2

    def test_float_division(self):
        assert safe_eval("7 / 2") == 3.5

    def test_floor_division(self):
        assert safe_eval("7 // 2") == 3

    def test_modulo(self):
        assert safe_eval("10 % 3") == 1

    def test_sqrt(self):
        assert safe_eval("sqrt(144)") == 12.0

    def test_factorial(self):
        assert safe_eval("factorial(5)") == 120

    def test_rejects_names(self):
        with pytest.raises(ValueError):
            safe_eval("__import__('os')")

    def test_rejects_huge_exponent(self):
        with pytest.raises(ValueError, match="Exponent too large"):
            safe_eval("2 ** 100000")


class TestArithmeticSolver:
    def setup_method(self):
        self.solver = ArithmeticSolver()

    def test_can_solve_arithmetic(self):
        assert self.solver.can_solve("2 + 3", ProblemType.ARITHMETIC)

    def test_simple_addition(self):
        r = self.solver.solve("compute 2 + 3")
        assert r.solutions == ["5"]
        assert r.confidence == 1.0
        assert r.solver_used == SolverType.PYTHON

    def test_complex_expression(self):
        r = self.solver.solve("calculate (10 + 5) * 2 - 3")
        assert r.solutions == ["27"]

    def test_error_handling(self):
        r = self.solver.solve("compute 1/0")
        assert r.error is not None
        assert r.confidence == 0.0


# ═══════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════

class TestRouter:
    def setup_method(self):
        self.router = Router()

    def test_classify_algebra(self):
        ptype, solvers, conf = self.router.classify("Solve x^2 + 2x - 8 = 0")
        assert ptype == ProblemType.ALGEBRA

    def test_classify_arithmetic(self):
        ptype, solvers, conf = self.router.classify("compute 2 + 3 * 4")
        assert ptype == ProblemType.ARITHMETIC

    def test_classify_logic(self):
        ptype, solvers, conf = self.router.classify("check satisfiability of p and q")
        assert ptype == ProblemType.LOGIC

    def test_classify_calculus(self):
        ptype, solvers, conf = self.router.classify("derivative of x^3 + 2x")
        assert ptype == ProblemType.CALCULUS

    def test_classify_simplification(self):
        ptype, solvers, conf = self.router.classify("simplify (x^2 - 1)/(x - 1)")
        assert ptype == ProblemType.SIMPLIFICATION

    def test_confidence_positive(self):
        ptype, solvers, conf = self.router.classify("solve x + 1 = 0")
        assert conf > 0


# ═══════════════════════════════════════════════════════════════
# Engine end-to-end (arithmetic only — no optional deps needed)
# ═══════════════════════════════════════════════════════════════

class TestEngine:
    def setup_method(self):
        self.engine = MathLogicEngine()

    def test_solve_arithmetic(self):
        r = self.engine.solve("compute 2 + 3")
        assert r.solutions == ["5"]
        assert r.solver_used == SolverType.PYTHON
        assert r.execution_time_ms > 0

    def test_solve_returns_json(self):
        r = self.engine.solve("calculate 10 * 5")
        data = json.loads(r.to_json())
        assert data["solutions"] == ["50"]
        assert "proof" in data

    def test_solve_dict(self):
        r = self.engine.solve("compute 7 - 3")
        d = r.to_dict()
        assert d["solutions"] == ["4"]
        assert d["problem_type"] == "arithmetic"

    def test_register_solver(self):
        """Engine should accept custom solver registration."""
        engine = MathLogicEngine()
        solver = ArithmeticSolver()
        engine.register_solver(solver)
        # Should not raise
        r = engine.solve("compute 1 + 1")
        assert r.solutions == ["2"]


# ═══════════════════════════════════════════════════════════════
# SymPy solver (skipped if not installed)
# ═══════════════════════════════════════════════════════════════

try:
    from math_logic.solvers.sympy_solver import SymPySolver
    HAS_SYMPY = True
except Exception:
    HAS_SYMPY = False


@pytest.mark.skipif(not HAS_SYMPY, reason="sympy not installed")
class TestSymPySolver:
    def setup_method(self):
        self.solver = SymPySolver()

    def test_solve_quadratic(self):
        r = self.solver.solve("Solve x^2 - 4 = 0")
        sols = set(r.solutions)
        assert "x = -2" in sols
        assert "x = 2" in sols
        assert r.proof.is_verified

    def test_simplify(self):
        r = self.solver.solve("simplify (x**2 - 1)/(x - 1)")
        assert "x + 1" in r.solutions[0]

    def test_derivative(self):
        r = self.solver.solve("derivative of x**3")
        assert "3*x**2" in r.solutions[0]

    def test_engine_routes_to_sympy(self):
        engine = MathLogicEngine()
        r = engine.solve("Solve x + 5 = 10")
        assert r.solver_used == SolverType.SYMPY
        assert "x = 5" in r.solutions


# ═══════════════════════════════════════════════════════════════
# Z3 solver (skipped if not installed)
# ═══════════════════════════════════════════════════════════════

try:
    from math_logic.solvers.z3_solver import Z3Solver
    # Test if Z3 actually works (native lib may be missing)
    _test_solver = Z3Solver()
    HAS_Z3 = True
    del _test_solver
except Exception:
    HAS_Z3 = False


@pytest.mark.skipif(not HAS_Z3, reason="z3-solver not installed")
class TestZ3Solver:
    def setup_method(self):
        self.solver = Z3Solver()

    def test_sat(self):
        r = self.solver.solve('Check satisfiability of "p and q"')
        assert "Satisfiable" in r.solutions[0]

    def test_unsat(self):
        r = self.solver.solve('Check satisfiability of "p and (not p)"')
        assert "Unsatisfiable" in r.solutions[0]

    def test_tautology(self):
        r = self.solver.solve('Check if tautology: "p or (not p)"')
        assert "Tautology" in r.solutions[0]

    def test_not_tautology(self):
        r = self.solver.solve('Check if tautology: "p and q"')
        assert "Not a tautology" in r.solutions[0]

    def test_engine_routes_to_z3(self):
        engine = MathLogicEngine()
        r = engine.solve('Check satisfiability of "p and q"')
        assert r.solver_used == SolverType.Z3
