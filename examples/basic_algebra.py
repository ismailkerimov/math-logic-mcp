"""
Example: Using math-logic-mcp as a Python library.

Run: python examples/basic_algebra.py
"""

from math_logic import MathLogicEngine
import json

engine = MathLogicEngine()

# ── Arithmetic (always available, zero deps) ──────────────────────
print("=== Arithmetic ===")
r = engine.solve("compute 2 + 3 * 4")
print(f"2 + 3 * 4 = {r.solutions[0]}")
print(f"  solver: {r.solver_used.value}, confidence: {r.confidence}")

r = engine.solve("calculate sqrt(144) + factorial(5)")
print(f"sqrt(144) + factorial(5) = {r.solutions[0]}")

# ── Algebra (requires: pip install sympy) ─────────────────────────
print("\n=== Algebra ===")
try:
    r = engine.solve("Solve x^2 - 5x + 6 = 0")
    print(f"x^2 - 5x + 6 = 0 → {r.solutions}")
    print(f"  verified: {r.proof.is_verified}")
except Exception as e:
    print(f"  (skipped — {e})")

# ── Simplification ───────────────────────────────────────────────
print("\n=== Simplification ===")
try:
    r = engine.solve("simplify (x**2 - 1)/(x - 1)")
    print(f"(x²-1)/(x-1) → {r.solutions[0]}")
except Exception as e:
    print(f"  (skipped — {e})")

# ── Calculus ─────────────────────────────────────────────────────
print("\n=== Calculus ===")
try:
    r = engine.solve("derivative of x**3 + 2*x**2 - 5*x + 1")
    print(f"d/dx(x³+2x²-5x+1) = {r.solutions[0]}")

    r = engine.solve("integrate 3*x**2 + 4*x")
    print(f"∫(3x²+4x)dx = {r.solutions[0]}")
except Exception as e:
    print(f"  (skipped — {e})")

# ── Logic (requires: pip install z3-solver) ──────────────────────
print("\n=== Logic ===")
try:
    r = engine.solve('Check satisfiability of "p and (not p)"')
    print(f"p ∧ ¬p → {r.solutions[0]}")
except Exception as e:
    print(f"  (skipped — {e})")

# ── JSON output ──────────────────────────────────────────────────
print("\n=== JSON output ===")
r = engine.solve("compute 7 * 8")
print(json.dumps(r.to_dict(), indent=2))
