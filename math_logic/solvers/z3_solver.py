"""
Z3-based solver for propositional logic, SAT, and SMT problems.

Security: NO eval() — all formula parsing uses a safe recursive-descent parser.
"""

from math_logic.engine import (
    BaseSolver, SolveResult, ProblemType, SolverType, Proof, ProofStep,
)
import re
import time

try:
    from z3 import (
        Bool, And, Or, Not, Implies, Xor,
        Solver as Z3SolverEngine, sat, unsat,
        Int, Real, ForAll, Exists, simplify as z3_simplify,
    )
    HAS_Z3 = True
except Exception:
    HAS_Z3 = False


class Z3Solver(BaseSolver):
    """Solver for propositional logic and SAT/SMT problems using Z3."""

    def __init__(self):
        if not HAS_Z3:
            raise ImportError("Z3 not installed. pip install z3-solver")

    def can_solve(self, problem: str, problem_type: ProblemType) -> bool:
        if problem_type in (ProblemType.LOGIC, ProblemType.DISCRETE):
            return True
        keywords = [
            "satisfiab", "tautolog", "valid", "propositional",
            "boolean", "truth table", "implies", "contradict",
            "and", "or", "not", "prove",
        ]
        return any(kw in problem.lower() for kw in keywords)

    def solve(self, problem: str, **kwargs) -> SolveResult:
        t0 = time.perf_counter()
        lower = problem.lower()

        if "tautolog" in lower or "valid" in lower:
            return self._check_tautology(problem, t0)
        if "satisfiab" in lower:
            return self._check_sat(problem, t0)
        if "truth table" in lower:
            return self._truth_table(problem, t0)
        # Default: check satisfiability
        return self._check_sat(problem, t0)

    # -----------------------------------------------------------------
    # SAT / Tautology / Truth-table
    # -----------------------------------------------------------------

    def _check_sat(self, problem: str, t0: float) -> SolveResult:
        formula_str = self._extract_formula_text(problem)
        try:
            var_map, expr = self._parse_formula(formula_str)
        except ValueError as e:
            return self._error(problem, t0, f"Parse error: {e}")

        s = Z3SolverEngine()
        s.add(expr)
        result = s.check()

        if result == sat:
            model = s.model()
            assignments = {str(v): str(model[v]) for v in var_map.values() if model[v] is not None}
            sol_str = "Satisfiable: " + ", ".join(f"{k}={v}" for k, v in assignments.items())
            solutions = [sol_str]
            verified = True
        elif result == unsat:
            solutions = ["Unsatisfiable (no assignment makes the formula true)"]
            assignments = {}
            verified = True
        else:
            solutions = ["Unknown"]
            assignments = {}
            verified = False

        steps = [
            ProofStep(1, f"Parse formula: {formula_str}"),
            ProofStep(2, "Encode into Z3 SMT solver"),
            ProofStep(3, f"Result: {solutions[0]}"),
        ]
        proof = Proof(
            problem=problem, method="sat_check", steps=steps,
            solutions=solutions, is_verified=verified,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return SolveResult(
            problem=problem, problem_type=ProblemType.LOGIC,
            solutions=solutions, proof=proof,
            solver_used=SolverType.Z3, confidence=0.95,
            execution_time_ms=elapsed,
        )

    def _check_tautology(self, problem: str, t0: float) -> SolveResult:
        formula_str = self._extract_formula_text(problem)
        try:
            var_map, expr = self._parse_formula(formula_str)
        except ValueError as e:
            return self._error(problem, t0, f"Parse error: {e}")

        # Tautology iff negation is unsat
        s = Z3SolverEngine()
        s.add(Not(expr))
        result = s.check()

        if result == unsat:
            solutions = ["Tautology (always true)"]
            is_taut = True
        elif result == sat:
            model = s.model()
            counter = {str(v): str(model[v]) for v in var_map.values() if model[v] is not None}
            ce_str = ", ".join(f"{k}={v}" for k, v in counter.items())
            solutions = [f"Not a tautology. Counter-example: {ce_str}"]
            is_taut = False
        else:
            solutions = ["Unknown"]
            is_taut = False

        steps = [
            ProofStep(1, f"Parse formula: {formula_str}"),
            ProofStep(2, "Negate formula and check satisfiability"),
            ProofStep(3, f"Result: {solutions[0]}"),
        ]
        proof = Proof(
            problem=problem, method="tautology_check", steps=steps,
            solutions=solutions, is_verified=True,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return SolveResult(
            problem=problem, problem_type=ProblemType.LOGIC,
            solutions=solutions, proof=proof,
            solver_used=SolverType.Z3, confidence=0.95,
            execution_time_ms=elapsed,
        )

    def _truth_table(self, problem: str, t0: float) -> SolveResult:
        formula_str = self._extract_formula_text(problem)
        try:
            var_map, expr = self._parse_formula(formula_str)
        except ValueError as e:
            return self._error(problem, t0, f"Parse error: {e}")

        var_names = sorted(var_map.keys())
        vars_z3 = [var_map[n] for n in var_names]
        n = len(var_names)
        rows = []
        for i in range(2**n):
            assignment = {}
            for j, name in enumerate(var_names):
                val = bool((i >> (n - 1 - j)) & 1)
                assignment[vars_z3[j]] = val
            result_val = self._eval_z3_expr(expr, assignment)
            row_vals = [str((i >> (n - 1 - j)) & 1) for j in range(n)]
            row_vals.append("1" if result_val else "0")
            rows.append(" | ".join(row_vals))

        header = " | ".join(var_names + ["result"])
        table = [header, "-" * len(header)] + rows
        solutions = ["\n".join(table)]

        proof = Proof(
            problem=problem, method="truth_table", steps=[
                ProofStep(1, f"Parse: {formula_str}"),
                ProofStep(2, f"Enumerate all {2**n} assignments"),
            ],
            solutions=solutions, is_verified=True,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return SolveResult(
            problem=problem, problem_type=ProblemType.LOGIC,
            solutions=solutions, proof=proof,
            solver_used=SolverType.Z3, confidence=0.95,
            execution_time_ms=elapsed,
        )

    def _eval_z3_expr(self, expr, assignment):
        """Evaluate a Z3 boolean expression under a concrete assignment."""
        s = Z3SolverEngine()
        for var, val in assignment.items():
            s.add(var == val)
        s.add(expr)
        return s.check() == sat

    # -----------------------------------------------------------------
    # Safe recursive-descent formula parser (NO eval)
    # -----------------------------------------------------------------

    def _extract_formula_text(self, problem: str) -> str:
        """Pull the Boolean formula out of natural language."""
        # Try to find quoted or colon-delimited formula
        m = re.search(r"['\"](.+?)['\"]", problem)
        if m:
            return m.group(1)
        m = re.search(r":\s*(.+)", problem)
        if m:
            return m.group(1).strip()
        # Remove preamble words
        cleaned = re.sub(
            r"^.*?(?:formula|expression|proposition|check|prove|verify|whether|if|that)\s*:?\s*",
            "", problem, flags=re.IGNORECASE,
        ).strip()
        return cleaned if cleaned else problem

    def _parse_formula(self, text: str):
        """
        Safe recursive-descent parser for propositional logic.
        Returns (var_map: dict[str, z3.Bool], expr: z3 BoolRef)

        Grammar:
            expr    → impl
            impl    → or_expr (('->' | '=>' | 'implies') or_expr)*
            or_expr → and_expr (('|' | 'or' | '∨') and_expr)*
            and_expr→ not_expr (('&' | 'and' | '∧') not_expr)*
            not_expr→ ('!' | 'not' | '~' | '¬') not_expr | atom
            atom    → '(' expr ')' | IDENTIFIER | 'true' | 'false'
        """
        tokens = self._tokenize(text)
        var_map = {}
        pos = [0]  # mutable index

        def peek():
            return tokens[pos[0]] if pos[0] < len(tokens) else None

        def consume(expected=None):
            tok = peek()
            if expected and tok != expected:
                raise ValueError(f"Expected '{expected}', got '{tok}'")
            pos[0] += 1
            return tok

        def parse_expr():
            return parse_impl()

        def parse_impl():
            left = parse_or()
            while peek() in ("->", "=>", "implies"):
                consume()
                right = parse_or()
                left = Implies(left, right)
            return left

        def parse_or():
            left = parse_and()
            while peek() in ("|", "or", "∨", "||"):
                consume()
                right = parse_and()
                left = Or(left, right)
            return left

        def parse_and():
            left = parse_not()
            while peek() in ("&", "and", "∧", "&&"):
                consume()
                right = parse_not()
                left = And(left, right)
            return left

        def parse_not():
            if peek() in ("!", "not", "~", "¬"):
                consume()
                operand = parse_not()
                return Not(operand)
            return parse_atom()

        def parse_atom():
            tok = peek()
            if tok is None:
                raise ValueError("Unexpected end of formula")
            if tok == "(":
                consume("(")
                inner = parse_expr()
                consume(")")
                return inner
            if tok.lower() == "true":
                consume()
                p = Bool("__true__")
                return Or(p, Not(p))  # tautology
            if tok.lower() == "false":
                consume()
                p = Bool("__false__")
                return And(p, Not(p))  # contradiction
            # Variable
            name = consume()
            if name not in var_map:
                var_map[name] = Bool(name)
            return var_map[name]

        expr = parse_expr()
        if pos[0] < len(tokens):
            raise ValueError(f"Unexpected token: {tokens[pos[0]]}")
        return var_map, expr

    def _tokenize(self, text: str):
        """Tokenize a Boolean formula string."""
        # Normalize unicode
        text = text.replace("→", "->").replace("∧", "&").replace("∨", "|").replace("¬", "!")
        text = text.replace("=>", "->").replace("&&", "&").replace("||", "|")
        tokens = []
        i = 0
        while i < len(text):
            c = text[i]
            if c in " \t\n":
                i += 1
                continue
            if c in "()&|!~":
                tokens.append(c)
                i += 1
                continue
            if text[i:i+2] == "->":
                tokens.append("->")
                i += 2
                continue
            # Word token (variable name or keyword)
            m = re.match(r"[a-zA-Z_]\w*", text[i:])
            if m:
                tokens.append(m.group(0))
                i += m.end()
                continue
            # Skip unknown chars
            i += 1
        return tokens

    # -----------------------------------------------------------------
    # Error helper
    # -----------------------------------------------------------------

    def _error(self, problem: str, t0: float, msg: str) -> SolveResult:
        elapsed = (time.perf_counter() - t0) * 1000
        return SolveResult(
            problem=problem, problem_type=ProblemType.LOGIC,
            solutions=[], proof=Proof(problem=problem, method="error"),
            solver_used=SolverType.Z3, confidence=0.0,
            execution_time_ms=elapsed, error=msg,
        )
