"""
Real benchmark: Small LLM (alone) vs Small LLM + math-logic-mcp

Tests whether adding our CAS tool actually improves a real LLM's math accuracy.

Methodology:
  1. "LLM alone": Ask the LLM to solve the problem directly, extract its answer
  2. "LLM + tool": Ask the LLM to formulate the problem, feed it to our engine,
     then use the verified CAS result

Uses Ollama for local LLM inference (no API keys needed).

Usage:
    python benchmarks/bench_llm_vs_tool.py                     # default (phi3:mini)
    python benchmarks/bench_llm_vs_tool.py --model llama3.1:8b
    python benchmarks/bench_llm_vs_tool.py -v --output results/llm_vs_tool.json
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


# ── Problem set ────────────────────────────────────────────────

PROBLEMS = [
    # ── Arithmetic: basic (LLMs usually get these) ──────────────
    ("What is 247 * 83?", "20501", "arithmetic"),
    ("What is 1847 + 2956?", "4803", "arithmetic"),
    ("What is 144 / 12?", "12", "arithmetic"),
    ("What is 17 squared?", "289", "arithmetic"),
    ("What is the square root of 2025?", "45", "arithmetic"),
    ("What is 15 factorial divided by 13 factorial?", "210", "arithmetic"),

    # ── Arithmetic: harder (LLMs usually fail these) ────────────
    ("What is 9876 * 5432?", "53646432", "arithmetic"),
    ("What is 999 * 999?", "998001", "arithmetic"),
    ("What is 2 to the power of 16?", "65536", "arithmetic"),
    ("What is 7 to the power of 5?", "16807", "arithmetic"),
    ("What is the square root of 169?", "13", "arithmetic"),
    ("What is 10 factorial?", "3628800", "arithmetic"),
    ("What is 123456 + 789012?", "912468", "arithmetic"),
    ("What is the absolute value of -42?", "42", "arithmetic"),

    # ── Algebra: linear ─────────────────────────────────────────
    ("Solve for x: 2x + 7 = 15", "4", "algebra"),
    ("Solve for x: 5x + 3 = 2x - 9", "-4", "algebra"),
    ("Solve for x: 7x - 14 = 0", "2", "algebra"),

    # ── Algebra: quadratic ──────────────────────────────────────
    ("Solve for x: x^2 - 5x + 6 = 0", ["2", "3"], "algebra"),
    ("Solve for x: 3x^2 - 12 = 0", ["2", "-2"], "algebra"),
    ("Solve for x: x^2 + 4x + 4 = 0", "-2", "algebra"),
    ("Solve for x: x^2 - 2x - 15 = 0", ["-3", "5"], "algebra"),
    ("Solve for x: 4x^2 + 4x + 1 = 0", "-1/2", "algebra"),

    # ── Simplification ──────────────────────────────────────────
    ("Simplify the expression (x^2 - 1)/(x - 1)", "x + 1", "simplification"),
    ("Simplify the expression (x^3 - 8)/(x - 2)", "x^2 + 2*x + 4", "simplification"),
    ("Simplify sin(x)^2 + cos(x)^2", "1", "simplification"),
    ("Factor x^2 - 9", "(x - 3)*(x + 3)", "simplification"),
    ("Factor x^3 - 27", "(x - 3)*(x^2 + 3*x + 9)", "simplification"),
    ("Expand (x + 3)(x - 2)", "x^2 + x - 6", "simplification"),
    ("Expand (2x + 1)^3", "8*x^3 + 12*x^2 + 6*x + 1", "simplification"),

    # ── Calculus: derivatives ───────────────────────────────────
    ("What is the derivative of x^3 + 2x^2 - 5x + 1?", "3*x^2 + 4*x - 5", "calculus"),
    ("What is the derivative of sin(x) * x^2?", "x^2*cos(x) + 2*x*sin(x)", "calculus"),
    ("What is the derivative of e^x * cos(x)?", "-exp(x)*sin(x) + exp(x)*cos(x)", "calculus"),
    ("What is the derivative of ln(x)?", "1/x", "calculus"),
    ("What is the derivative of x^4 - 3x^3 + 2x?", "4*x^3 - 9*x^2 + 2", "calculus"),

    # ── Calculus: integrals ─────────────────────────────────────
    ("What is the integral of x^2?", "x^3/3", "calculus"),
    ("What is the integral of sin(x)?", "-cos(x)", "calculus"),

    # ── Word problems (multi-step arithmetic) ───────────────────
    ("A store has 7 boxes, each with 13 items. If 29 items are sold, how many remain?", "62", "word_problem"),
    ("If you earn $45.50 per hour and work 37 hours, what is your gross pay?", "1683.5", "word_problem"),
    ("A recipe needs 3/4 cup of sugar. If you want to make 5 batches, how many cups of sugar do you need?", "3.75", "word_problem"),
    ("A contractor buys 12 boards at $8.50 each and 3 boxes of nails at $15.75 each. What is the total cost?", "149.25", "word_problem"),
    ("A tank has 250 gallons. If 73 gallons are drained and the remainder is split equally into 4 containers, how many gallons per container?", "44.25", "word_problem"),
    ("If you invest $1000 at 5%% annual interest compounded once, what is the value after 10 years? Round to 2 decimal places.", "1628.89", "word_problem"),
    ("A factory produces 156 widgets per hour. How many widgets are produced in a 3-day period running 8 hours per day?", "3744", "word_problem"),
    ("You buy 3 items at $12.99 each with a 10%% discount on the total. What do you pay?", "35.07", "word_problem"),
]


# ── Ollama client ─────────────────────────────────────────────

def ollama_generate(prompt: str, model: str, timeout: int = 60) -> str:
    """Call Ollama's HTTP API directly (no pip dependency needed)."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
        },
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "").strip()
    except (urllib.error.URLError, TimeoutError) as e:
        return f"ERROR: {e}"


def extract_number_from_llm(text: str) -> Optional[str]:
    """Extract the final numeric answer from LLM output."""
    patterns = [
        r"(?:answer|result|equals?|=)\s*[:is]*\s*\$?\*?\*?([+-]?\d+\.?\d*)\*?\*?\$?",
        r"\*\*([+-]?\d+\.?\d*)\*\*",
        r"=\s*([+-]?\d+\.?\d*)\s*$",
        r"(?:^|\n)\s*([+-]?\d+\.?\d*)\s*$",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1]

    # Fallback: last number in the text
    nums = re.findall(r"[+-]?\d+\.?\d*", text)
    return nums[-1] if nums else None


def extract_symbolic_from_llm(text: str) -> Optional[str]:
    """Extract symbolic math answer (like 'x + 1') from LLM output."""
    patterns = [
        r"(?:answer|result|simplifies? to|equals?|=)\s*[:is]*\s*\$?\*?\*?([^\n\*\$]+)",
        r"\*\*([^\*]+)\*\*",
    ]
    for pat in patterns:
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            return m[-1].strip().rstrip(".")
    return None


# ── Normalization & checking ───────────────────────────────────

def _norm(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"^x\s*=\s*", "", s)
    s = s.replace(" ", "").replace("**", "^").replace("·", "*")
    s = s.rstrip(".")
    return s.lower()


def _check(got: Optional[str], expected) -> bool:
    if got is None:
        return False

    if isinstance(expected, list):
        got_n = _norm(got)
        for e in expected:
            if _norm(e) in got_n:
                return True
            try:
                if abs(float(_norm(e)) - float(got_n)) < 0.1:
                    return True
            except (ValueError, TypeError):
                pass
        return False

    got_n = _norm(got)
    exp_n = _norm(expected)
    if got_n == exp_n:
        return True
    try:
        return abs(float(got_n) - float(exp_n)) < 0.1
    except (ValueError, TypeError):
        pass
    return exp_n in got_n or got_n in exp_n


def _check_solutions(solutions: list[str], expected) -> bool:
    """Check if any solution in the list matches expected."""
    if not solutions:
        return False
    for sol in solutions:
        if _check(sol, expected):
            return True
    # For multi-answer: check if all expected values appear somewhere in solutions
    if isinstance(expected, list):
        all_norm = set()
        for s in solutions:
            all_norm.add(_norm(s))
            # Also try numeric comparison for each
            try:
                all_norm.add(str(round(float(_norm(s)), 6)))
            except (ValueError, TypeError):
                pass
        exp_norm = set()
        for e in expected:
            exp_norm.add(_norm(e))
        if exp_norm.issubset(all_norm):
            return True
        # Check if each expected value is close to some solution
        matched = 0
        for e in expected:
            for s in solutions:
                if _check(s, e):
                    matched += 1
                    break
        if matched == len(expected):
            return True
    return False


# ── Result tracking ────────────────────────────────────────────

@dataclass
class RunResult:
    name: str
    correct: int = 0
    total: int = 0
    total_ms: float = 0.0
    per_category: dict = field(default_factory=dict)
    details: list = field(default_factory=list)

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


# ── Mode 1: LLM alone ─────────────────────────────────────────

LLM_ALONE_PROMPT = """Solve this math problem. Give ONLY the final numerical or symbolic answer on the last line, no explanation.

Problem: {problem}

Answer:"""


def run_llm_alone(model: str, verbose: bool) -> RunResult:
    res = RunResult(name=f"{model} (alone)")

    for problem, expected, category in PROBLEMS:
        prompt = LLM_ALONE_PROMPT.format(problem=problem)
        t0 = time.perf_counter()
        raw = ollama_generate(prompt, model)
        elapsed = (time.perf_counter() - t0) * 1000

        if raw.startswith("ERROR:"):
            got = None
        elif category in ("arithmetic", "word_problem"):
            got = extract_number_from_llm(raw)
        else:
            got = extract_symbolic_from_llm(raw) or extract_number_from_llm(raw)

        ok = _check(got, expected)
        res.total += 1
        res.total_ms += elapsed
        if ok:
            res.correct += 1

        if category not in res.per_category:
            res.per_category[category] = {"correct": 0, "total": 0}
        res.per_category[category]["total"] += 1
        if ok:
            res.per_category[category]["correct"] += 1

        if verbose:
            status = "✓" if ok else "✗"
            raw_short = raw.replace("\n", " ")[:60]
            print(f"  {status}  {problem[:50]:50s}  → got={got}  (raw: {raw_short})")

        res.details.append({
            "problem": problem,
            "expected": expected,
            "got": got,
            "raw_llm": raw[:200],
            "correct": ok,
            "category": category,
            "time_ms": round(elapsed, 2),
        })

    return res


# ── Mode 2: LLM + tool ────────────────────────────────────────

LLM_FORMULATE_PROMPT = """You are a math assistant. Convert this problem into a mathematical expression or equation that a computer algebra system can solve.

Rules:
- For equations to solve, write: Solve <equation> (e.g., "Solve 2*x + 7 = 15")
- For arithmetic, write: compute <expression> (e.g., "compute 247 * 83")
- For simplification, write: simplify <expression>
- For factoring, write: factor <expression>
- For expanding, write: expand <expression>
- For derivatives, write: derivative of <expression>
- For integrals, write: integral of <expression> (do NOT include dx)
- Use ^ or ** for exponents, * for multiplication
- "n squared" means n^2 (e.g., "17 squared" = "compute 17^2")
- "n to the power of m" means n^m (e.g., "7 to the power of 5" = "compute 7^5")
- Use sqrt() for square roots, factorial() for factorials
- Use exp(x) for e^x (Euler's number raised to x)
- ALWAYS use explicit * for multiplication: write 2*x not 2x, 3*x^2 not 3x^2

Give ONLY the formatted expression on a single line, nothing else.

Problem: {problem}

Expression:"""


def run_llm_plus_tool(model: str, verbose: bool) -> RunResult:
    engine = MathLogicEngine()
    res = RunResult(name=f"{model} + math-logic-mcp")

    for problem, expected, category in PROBLEMS:
        prompt = LLM_FORMULATE_PROMPT.format(problem=problem)
        t0 = time.perf_counter()

        # Step 1: LLM formulates the expression
        formulated = ollama_generate(prompt, model)
        elapsed_llm = (time.perf_counter() - t0) * 1000

        got = None
        got_solutions = []

        if formulated.startswith("ERROR:"):
            elapsed = elapsed_llm
        else:
            # Clean the LLM output
            expr = formulated.strip().strip("`").strip()
            # Take only first line if multi-line
            expr = expr.split("\n")[0].strip()

            # Step 2: Feed to our engine
            try:
                r = engine.solve(expr)
                elapsed = (time.perf_counter() - t0) * 1000
                got_solutions = r.solutions if r.solutions else []
                got = got_solutions[0] if got_solutions else None
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                if verbose:
                    print(f"    [engine error: {exc}]")

        ok = _check_solutions(got_solutions, expected) if got_solutions else _check(got, expected)

        res.total += 1
        res.total_ms += elapsed
        if ok:
            res.correct += 1

        if category not in res.per_category:
            res.per_category[category] = {"correct": 0, "total": 0}
        res.per_category[category]["total"] += 1
        if ok:
            res.per_category[category]["correct"] += 1

        if verbose:
            status = "✓" if ok else "✗"
            expr_short = formulated.replace("\n", " ")[:45] if not formulated.startswith("ERROR") else "ERROR"
            got_display = got_solutions[:3] if got_solutions else "(none)"
            print(f"  {status}  {problem[:50]:50s}  expr={expr_short}")
            print(f"       → engine={got_display}  expected={expected}")

        res.details.append({
            "problem": problem,
            "expected": expected,
            "formulated": formulated[:200],
            "got_solutions": got_solutions,
            "got": got,
            "correct": ok,
            "category": category,
            "time_ms": round(elapsed, 2),
        })

    return res


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: LLM alone vs LLM + math-logic-mcp tool"
    )
    parser.add_argument("--model", default="phi3:mini", help="Ollama model name")
    parser.add_argument("--output", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Verify Ollama is running
    try:
        test = ollama_generate("Say OK", args.model, timeout=10)
        if test.startswith("ERROR:"):
            print(f"ERROR: Cannot reach Ollama. Is it running? (ollama serve)")
            print(f"  Response: {test}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}\nMake sure Ollama is running: ollama serve")
        sys.exit(1)

    print(f"═══ LLM vs LLM+Tool Benchmark ═══")
    print(f"Model: {args.model}")
    print(f"Problems: {len(PROBLEMS)}")
    print()

    # Run LLM alone
    print(f"─── {args.model} (alone, no tools) ───")
    r_alone = run_llm_alone(args.model, verbose=args.verbose)
    print(f"  → Accuracy: {r_alone.accuracy:.0%}  ({r_alone.correct}/{r_alone.total})")
    print(f"  → Avg time: {r_alone.total_ms / max(r_alone.total, 1):.0f}ms\n")

    # Run LLM + tool
    print(f"─── {args.model} + math-logic-mcp ───")
    r_tool = run_llm_plus_tool(args.model, verbose=args.verbose)
    print(f"  → Accuracy: {r_tool.accuracy:.0%}  ({r_tool.correct}/{r_tool.total})")
    print(f"  → Avg time: {r_tool.total_ms / max(r_tool.total, 1):.0f}ms\n")

    # Summary
    print("═══ Head-to-Head ═══")
    categories = sorted({c for _, _, c in PROBLEMS})
    print(f"  {'Category':<16s} │ {'LLM alone':^14s} │ {'LLM + tool':^14s}")
    print(f"  {'─'*16}─┼─{'─'*14}─┼─{'─'*14}")
    for cat in categories:
        a = r_alone.per_category.get(cat, {"correct": 0, "total": 0})
        t = r_tool.per_category.get(cat, {"correct": 0, "total": 0})
        print(f"  {cat:<16s} │ {a['correct']:>5d}/{a['total']:<8d} │ {t['correct']:>5d}/{t['total']:<8d}")
    print(f"  {'─'*16}─┼─{'─'*14}─┼─{'─'*14}")
    print(f"  {'TOTAL':<16s} │ {r_alone.correct:>5d}/{r_alone.total:<8d} │ {r_tool.correct:>5d}/{r_tool.total:<8d}")
    print(f"  {'ACCURACY':<16s} │ {r_alone.accuracy:^14.0%} │ {r_tool.accuracy:^14.0%}")

    delta = r_tool.accuracy - r_alone.accuracy
    print(f"\n  Δ accuracy: {delta:+.0%} ({args.model} + tool vs alone)")

    if delta > 0:
        print(f"  ✅ Tool gives +{delta:.0%} improvement")
    elif delta == 0:
        print(f"  ➡️  No difference")
    else:
        print(f"  ⚠️  Tool is {delta:.0%} worse (formulation errors?)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model,
                "problems": len(PROBLEMS),
                "llm_alone": r_alone.to_dict(),
                "llm_plus_tool": r_tool.to_dict(),
                "delta_accuracy": round(delta, 4),
                "details_alone": r_alone.details,
                "details_tool": r_tool.details,
            }, f, indent=2)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
