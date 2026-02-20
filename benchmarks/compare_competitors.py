"""
Competitor comparison — math-logic-mcp vs baselines

Runs a shared problem set through:
  1. math-logic-mcp (our engine — deterministic CAS)
  2. Baseline: regex/heuristic (what an LLM does without tools)

Usage:
    python benchmarks/compare_competitors.py
    python benchmarks/compare_competitors.py -v --output results/comparison.json
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from math_logic.engine import MathLogicEngine


# ── Shared problem set ─────────────────────────────────────────

PROBLEMS = [
    # (problem, expected_answer, category)
    ("2 + 2", "4", "arithmetic"),
    ("sqrt(144)", "12", "arithmetic"),
    ("factorial(5)", "120", "arithmetic"),
    ("3 * (4 + 5) - 2", "25", "arithmetic"),
    ("2^10", "1024", "arithmetic"),

    ("Solve x^2 - 5x + 6 = 0", ["2", "3"], "algebra"),
    ("Solve 2x + 7 = 15", "4", "algebra"),
    ("Solve x^2 = 49", ["7", "-7"], "algebra"),
    ("Solve 3x - 9 = 0", "3", "algebra"),

    ("simplify (x^2 - 1)/(x - 1)", "x + 1", "simplification"),
    ("factor x^2 - 9", "(x - 3)*(x + 3)", "simplification"),
    ("expand (x + 1)*(x - 1)", "x**2 - 1", "simplification"),

    ("derivative of x^3 + 2*x^2 - 5*x + 1", "3*x**2 + 4*x - 5", "calculus"),
    ("integrate 2*x", "x**2", "calculus"),

    ("Is (p -> q) & (q -> r) -> (p -> r) a tautology?", "tautology", "logic"),
]


# ── Normalize helper ───────────────────────────────────────────

def _norm(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"^x\s*=\s*", "", s)
    s = s.replace(" ", "")
    return s.lower()


# ── Baseline "solver" (regex heuristic, no CAS) ───────────────

def baseline_solve(problem: str) -> str | None:
    """
    Simple regex heuristic — represents what a small LLM might do
    without any tool. Can only handle pure digit arithmetic.
    """
    p = problem.strip()
    if re.fullmatch(r"[\d\s\+\-\*/\(\)\.\^]+", p):
        try:
            expr = p.replace("^", "**")
            return str(eval(expr))  # noqa: S307 — intentional for baseline
        except Exception:
            return None
    return None


# ── Result tracking ────────────────────────────────────────────

@dataclass
class RunnerResult:
    name: str
    correct: int = 0
    total: int = 0
    errors: int = 0
    total_ms: float = 0.0
    per_category: dict = field(default_factory=dict)

    @property
    def accuracy(self):
        return self.correct / self.total if self.total else 0.0

    def to_dict(self):
        return {
            "runner": self.name,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "avg_ms": round(self.total_ms / max(self.total, 1), 2),
            "per_category": self.per_category,
        }


def _check(got, expected) -> bool:
    """Check if got matches expected. got can be a list of solution strings."""
    if got is None:
        return False

    # Normalize got into list of strings
    if isinstance(got, list):
        got_strs = [str(g) for g in got]
    else:
        got_strs = [str(got)]

    if isinstance(expected, list):
        # Multi-answer: all expected values must be in got
        exp_set = {_norm(e) for e in expected}
        got_set = {_norm(g) for g in got_strs}
        return exp_set.issubset(got_set)

    # Single expected answer: any got must match
    exp_n = _norm(expected)
    for g in got_strs:
        g_n = _norm(g)
        if g_n == exp_n:
            return True
        try:
            if abs(float(g_n) - float(exp_n)) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass
    return False


def _run(name: str, solve_fn, verbose: bool) -> RunnerResult:
    res = RunnerResult(name=name)
    for problem, expected, category in PROBLEMS:
        t0 = time.perf_counter()
        try:
            got = solve_fn(problem)
            elapsed = (time.perf_counter() - t0) * 1000
        except Exception as e:
            got = None
            elapsed = (time.perf_counter() - t0) * 1000
            if verbose:
                print(f"  ✗  {problem[:50]:50s}  → ERROR: {e}")

        res.total += 1
        res.total_ms += elapsed

        ok = _check(got, expected)
        if ok:
            res.correct += 1
        else:
            res.errors += 1

        if category not in res.per_category:
            res.per_category[category] = {"correct": 0, "total": 0}
        res.per_category[category]["total"] += 1
        if ok:
            res.per_category[category]["correct"] += 1

        if verbose:
            status = "✓" if ok else "✗"
            got_display = got if got else "(none)"
            print(f"  {status}  {problem[:50]:50s}  → {got_display}")

    return res


def main():
    parser = argparse.ArgumentParser(description="Competitor comparison")
    parser.add_argument("--output", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    engine = MathLogicEngine()

    def engine_solve(problem):
        r = engine.solve(problem)
        return r.solutions if r.solutions else None

    print("═══ Competitor Comparison ═══")
    print(f"Problem set: {len(PROBLEMS)} problems\n")

    print("─── math-logic-mcp (CAS-backed) ───")
    r_engine = _run("math-logic-mcp", engine_solve, verbose=args.verbose)
    print(f"  → Accuracy: {r_engine.accuracy:.0%}  ({r_engine.correct}/{r_engine.total})\n")

    print("─── baseline (regex-only, no CAS) ───")
    r_baseline = _run("baseline-regex", baseline_solve, verbose=args.verbose)
    print(f"  → Accuracy: {r_baseline.accuracy:.0%}  ({r_baseline.correct}/{r_baseline.total})\n")

    # Summary table
    print("═══ Head-to-Head Summary ═══")
    categories = sorted({c for _, _, c in PROBLEMS})
    print(f"  {'Category':<16s} │ {'math-logic-mcp':^14s} │ {'baseline':^14s}")
    print(f"  {'─'*16}─┼─{'─'*14}─┼─{'─'*14}")
    for cat in categories:
        e = r_engine.per_category.get(cat, {"correct": 0, "total": 0})
        b = r_baseline.per_category.get(cat, {"correct": 0, "total": 0})
        ea = f"{e['correct']}/{e['total']}"
        ba = f"{b['correct']}/{b['total']}"
        print(f"  {cat:<16s} │ {ea:^14s} │ {ba:^14s}")
    print(f"  {'─'*16}─┼─{'─'*14}─┼─{'─'*14}")
    print(f"  {'TOTAL':<16s} │ {r_engine.correct}/{r_engine.total:^12d} │ {r_baseline.correct}/{r_baseline.total:^12d}")
    print(f"  {'ACCURACY':<16s} │ {r_engine.accuracy:^14.0%} │ {r_baseline.accuracy:^14.0%}")

    delta = r_engine.accuracy - r_baseline.accuracy
    print(f"\n  Δ accuracy: {delta:+.0%} (math-logic-mcp vs baseline)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([r_engine.to_dict(), r_baseline.to_dict()], f, indent=2)


if __name__ == "__main__":
    main()
