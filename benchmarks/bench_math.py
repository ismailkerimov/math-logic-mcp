"""
Benchmark: MATH — Competition-level math

Evaluates math-logic-mcp on algebra/number-theory problems from the
MATH dataset (Hendrycks et al.). These are symbolic problems where our
SymPy solver should shine compared to raw LLM output.

Usage:
    python benchmarks/bench_math.py                  # quick (builtin samples)
    python benchmarks/bench_math.py --samples 200    # HuggingFace dataset
    python benchmarks/bench_math.py --output results/math.json
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from math_logic.engine import MathLogicEngine


# ── LaTeX cleanup ──────────────────────────────────────────────

def strip_latex(text: str) -> str:
    """Strip LaTeX markup so the engine can parse the expression."""
    s = text
    # Remove $ delimiters
    s = s.replace("$", "")
    # \frac{a}{b} → (a)/(b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    # \sqrt{x} → sqrt(x)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    # \left, \right, \cdot
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    # Common LaTeX commands
    s = s.replace("\\pm", "+-")
    s = re.sub(r"\\(sin|cos|tan|log|ln|exp)\b", r"\1", s)
    # Remove remaining backslashes before letters
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    # Clean up whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── Dataset loading ────────────────────────────────────────────

def load_math_dataset(max_samples: Optional[int] = None, subjects=None):
    """Load MATH dataset. Filters to algebra/number-theory by default."""
    if subjects is None:
        subjects = ["algebra", "number_theory", "prealgebra", "intermediate_algebra"]

    try:
        from datasets import load_dataset
        ds = load_dataset("hendrycks/competition_math", split="test")
        rows = [r for r in ds if r.get("type", "").lower() in subjects]
        if max_samples:
            rows = rows[:max_samples]
        return [{"problem": r["problem"], "solution": r["solution"],
                 "level": r.get("level", ""), "type": r.get("type", "")} for r in rows]
    except Exception:
        pass

    return _BUILTIN_MATH_SAMPLES[:max_samples]


def extract_boxed(text: str) -> str:
    """Extract \\boxed{...} answer from MATH dataset solutions."""
    # Handle nested braces
    idx = text.find("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i-1].strip()


def normalize_answer(s: str) -> str:
    """Normalize a MATH answer for comparison."""
    s = s.strip()
    # Remove LaTeX formatting
    s = strip_latex(s)
    # Remove x = prefix
    s = re.sub(r"^x\s*=\s*", "", s)
    # Remove whitespace
    s = s.replace(" ", "")
    return s.lower()


# ── Result tracking ────────────────────────────────────────────

@dataclass
class MathBenchResult:
    total: int = 0
    correct: int = 0
    errors: int = 0
    skipped: int = 0
    total_time_ms: float = 0.0
    by_level: dict = field(default_factory=lambda: {})
    details: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_dict(self):
        return {
            "benchmark": "MATH",
            "total": self.total,
            "correct": self.correct,
            "errors": self.errors,
            "skipped": self.skipped,
            "accuracy": round(self.accuracy, 4),
            "avg_time_ms": round(self.total_time_ms / max(self.total, 1), 2),
            "by_level": self.by_level,
        }


# ── Benchmark runner ───────────────────────────────────────────

def _check_correct(got_solutions: list[str], expected: str) -> bool:
    """Check if any returned solution matches expected."""
    exp_norm = normalize_answer(expected)

    for sol in got_solutions:
        sol_norm = normalize_answer(sol)
        # Direct string match
        if sol_norm == exp_norm:
            return True
        # Numeric match
        try:
            if abs(float(sol_norm) - float(exp_norm)) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass

    # For multi-solution problems (e.g. "2, 3" expected)
    # Check if all expected values are in solutions
    if "," in expected:
        expected_parts = {normalize_answer(p) for p in expected.split(",")}
        got_parts = {normalize_answer(s) for s in got_solutions}
        if expected_parts and expected_parts.issubset(got_parts):
            return True

    return False


def run_benchmark(samples: list, verbose: bool = False) -> MathBenchResult:
    engine = MathLogicEngine()
    result = MathBenchResult()

    for i, sample in enumerate(samples):
        raw_problem = sample["problem"]
        expected = extract_boxed(sample["solution"])
        level = sample.get("level", "?")

        if not expected:
            result.skipped += 1
            result.total += 1
            if verbose:
                print(f"  [{i+1}] SKIP — no \\boxed answer found")
            continue

        # Strip LaTeX before sending to engine
        clean_problem = strip_latex(raw_problem)
        t0 = time.perf_counter()

        try:
            r = engine.solve(clean_problem)
            elapsed = (time.perf_counter() - t0) * 1000
            result.total += 1
            result.total_time_ms += elapsed

            is_correct = _check_correct(r.solutions, expected)

            if is_correct:
                result.correct += 1
            else:
                result.errors += 1

            # Track by difficulty level
            if level not in result.by_level:
                result.by_level[level] = {"total": 0, "correct": 0}
            result.by_level[level]["total"] += 1
            if is_correct:
                result.by_level[level]["correct"] += 1

            if verbose:
                status = "✓" if is_correct else "✗"
                got = r.solutions[:2] if r.solutions else ["(none)"]
                print(f"  [{i+1}] {status} [{level}] expected={expected} got={got}")

            result.details.append({
                "index": i,
                "problem": raw_problem[:120],
                "clean_problem": clean_problem[:120],
                "expected": expected,
                "got": r.solutions,
                "correct": is_correct,
                "level": level,
                "time_ms": round(elapsed, 2),
            })
        except Exception as e:
            result.total += 1
            result.errors += 1
            if verbose:
                print(f"  [{i+1}] ERROR: {e}")

    return result


# ── Builtin samples ────────────────────────────────────────────

_BUILTIN_MATH_SAMPLES = [
    {
        "problem": "Solve for $x$: $x^2 - 5x + 6 = 0$",
        "solution": "Factoring, $(x-2)(x-3)=0$, giving $x=2$ or $x=3$. Answer: $\\boxed{2, 3}$",
        "level": "Level 1", "type": "algebra",
    },
    {
        "problem": "Simplify $\\frac{x^2 - 1}{x - 1}$.",
        "solution": "Factor: $\\frac{(x-1)(x+1)}{x-1} = x+1$. Answer: $\\boxed{x + 1}$",
        "level": "Level 1", "type": "algebra",
    },
    {
        "problem": "Find the derivative of $f(x) = x^3 + 2x^2 - 5x + 1$.",
        "solution": "$f'(x) = 3x^2 + 4x - 5$. Answer: $\\boxed{3x^2 + 4x - 5}$",
        "level": "Level 2", "type": "algebra",
    },
    {
        "problem": "Solve for $x$: $2x + 7 = 15$.",
        "solution": "$2x=8$, so $x=4$. Answer: $\\boxed{4}$",
        "level": "Level 1", "type": "prealgebra",
    },
    {
        "problem": "What is $\\sqrt{144} + \\sqrt{81}$?",
        "solution": "$12 + 9 = 21$. Answer: $\\boxed{21}$",
        "level": "Level 1", "type": "prealgebra",
    },
    {
        "problem": "Factor $x^2 - 9$.",
        "solution": "$(x-3)(x+3)$. Answer: $\\boxed{(x - 3)(x + 3)}$",
        "level": "Level 1", "type": "algebra",
    },
    {
        "problem": "Solve $x^2 = 49$.",
        "solution": "$x = \\pm 7$. Answer: $\\boxed{7}$",
        "level": "Level 1", "type": "algebra",
    },
    {
        "problem": "Compute $3! + 4!$.",
        "solution": "$6 + 24 = 30$. Answer: $\\boxed{30}$",
        "level": "Level 1", "type": "prealgebra",
    },
    {
        "problem": "Solve for $x$: $3x - 9 = 0$.",
        "solution": "$3x = 9$, so $x = 3$. Answer: $\\boxed{3}$",
        "level": "Level 1", "type": "prealgebra",
    },
    {
        "problem": "Compute the integral $\\int 2x \\, dx$.",
        "solution": "$x^2 + C$. Answer: $\\boxed{x^2}$",
        "level": "Level 2", "type": "algebra",
    },
]


def main():
    parser = argparse.ArgumentParser(description="MATH Benchmark for math-logic-mcp")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("═══ MATH Competition Benchmark ═══")
    samples = load_math_dataset(args.samples)
    print(f"Loaded {len(samples)} problems\n")

    result = run_benchmark(samples, verbose=args.verbose)

    print(f"\n{'═' * 40}")
    print(f"  Total:    {result.total}")
    print(f"  Correct:  {result.correct}")
    print(f"  Errors:   {result.errors}")
    print(f"  Skipped:  {result.skipped}")
    print(f"  Accuracy: {result.accuracy:.1%}")
    for lvl, stats in sorted(result.by_level.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"    {lvl}: {stats['correct']}/{stats['total']} ({acc:.0%})")
    avg = result.total_time_ms / max(result.total, 1)
    print(f"  Avg time: {avg:.1f} ms/problem")
    print(f"{'═' * 40}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
