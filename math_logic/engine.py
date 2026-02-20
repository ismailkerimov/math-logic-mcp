"""
Math/Logic Engine — Core orchestrator
Routes problems to solvers, generates proofs, returns verified results.

This is the heart of math-logic-mcp. It wires:
  Router.classify() -> Solver.solve() -> SolveResult (with Proof)
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProblemType(Enum):
    """Classification of mathematical/logical problem types."""
    ALGEBRA = "algebra"
    LOGIC = "logic"
    SAT = "sat"
    SMT = "smt"
    CONSTRAINT = "constraint"
    CALCULUS = "calculus"
    DISCRETE = "discrete"
    ARITHMETIC = "arithmetic"
    SIMPLIFICATION = "simplification"
    UNKNOWN = "unknown"


class SolverType(Enum):
    """Available solver backends."""
    SYMPY = "sympy"
    Z3 = "z3"
    OR_TOOLS = "or_tools"
    PYTHON = "python"  # built-in arithmetic
    LLM = "llm"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProofStep:
    """Single step in a proof."""
    step_number: int
    action: str
    explanation: str = ""
    symbolic_form: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step": self.step_number,
            "action": self.action,
        }
        if self.explanation:
            d["explanation"] = self.explanation
        if self.symbolic_form:
            d["symbolic_form"] = self.symbolic_form
        return d


@dataclass
class Proof:
    """Complete proof with optional verification."""
    problem: str
    method: str
    steps: List[ProofStep] = field(default_factory=list)
    solutions: List[str] = field(default_factory=list)
    verification: Dict[str, str] = field(default_factory=dict)
    is_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "method": self.method,
            "steps": [s.to_dict() for s in self.steps],
            "solutions": self.solutions,
            "verification": self.verification,
            "is_verified": self.is_verified,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        md = f"# Proof: {self.problem}\n\n"
        md += f"**Method:** {self.method}\n\n"
        md += "## Steps\n\n"
        for step in self.steps:
            md += f"{step.step_number}. {step.action}\n"
            if step.explanation:
                md += f"   - {step.explanation}\n"
            if step.symbolic_form:
                md += f"   `{step.symbolic_form}`\n"
        md += "\n## Solutions\n\n"
        for sol in self.solutions:
            md += f"- {sol}\n"
        if self.verification:
            md += "\n## Verification\n\n"
            for expr, result in self.verification.items():
                md += f"- {expr}: {result}\n"
        return md


@dataclass
class SolveResult:
    """Result returned by every solve operation."""
    problem: str
    problem_type: ProblemType
    solutions: List[str]
    proof: Proof
    solver_used: SolverType
    confidence: float  # 0.0 - 1.0
    execution_time_ms: float
    cache_hit: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "problem_type": self.problem_type.value,
            "solutions": self.solutions,
            "proof": self.proof.to_dict(),
            "solver": self.solver_used.value,
            "confidence": self.confidence,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "cache_hit": self.cache_hit,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Solver plugin interface
# ---------------------------------------------------------------------------

class BaseSolver(ABC):
    """Abstract base class every solver must implement."""

    @abstractmethod
    def can_solve(self, problem: str, problem_type: ProblemType) -> bool:
        """Return True if this solver can handle the problem."""

    @abstractmethod
    def solve(self, problem: str, **kwargs) -> SolveResult:
        """Solve the problem and return a SolveResult with proof."""


# ---------------------------------------------------------------------------
# Engine (orchestrator)
# ---------------------------------------------------------------------------

class MathLogicEngine:
    """
    Main orchestrator for math/logic solving.

    Flow:
      1. Router.classify(problem) -> (ProblemType, [SolverType], confidence)
      2. Pick best registered solver that can_solve()
      3. solver.solve(problem) -> SolveResult
      4. Return SolveResult (includes proof + confidence)
    """

    def __init__(self):
        self._solvers: List[BaseSolver] = []
        self._router = None  # lazy import to avoid circular deps

    # -- solver management ---------------------------------------------------

    def register_solver(self, solver: BaseSolver) -> None:
        """Register a solver plugin."""
        self._solvers.append(solver)

    def _ensure_defaults(self):
        """Lazy-load default solvers on first call."""
        if self._solvers:
            return
        from math_logic.solvers.sympy_solver import SymPySolver
        from math_logic.solvers.z3_solver import Z3Solver
        from math_logic.solvers.arithmetic_solver import ArithmeticSolver

        try:
            self._solvers.append(ArithmeticSolver())
        except Exception:
            pass
        try:
            self._solvers.append(SymPySolver())
        except Exception:
            pass
        try:
            self._solvers.append(Z3Solver())
        except Exception:
            pass

    def _get_router(self):
        if self._router is None:
            from math_logic.router import Router
            self._router = Router()
        return self._router

    # -- main entry point ----------------------------------------------------

    def solve(
        self,
        problem: str,
        problem_type: Optional[ProblemType] = None,
        timeout_seconds: float = 30.0,
    ) -> SolveResult:
        """
        Solve a math/logic problem end-to-end.

        Args:
            problem: Natural-language problem statement.
            problem_type: Optional hint (skip classification).
            timeout_seconds: Max solver time.

        Returns:
            SolveResult with solutions, proof, and confidence.
        """
        self._ensure_defaults()
        t0 = time.perf_counter()

        # 1. Classify
        router = self._get_router()
        if problem_type is None:
            ptype, suggested_solvers, route_conf = router.classify(problem)
        else:
            ptype = problem_type
            suggested_solvers = []
            route_conf = 1.0

        # 2. Find a solver that can handle this
        solver = self._pick_solver(problem, ptype)
        if solver is None:
            elapsed = (time.perf_counter() - t0) * 1000
            return SolveResult(
                problem=problem,
                problem_type=ptype,
                solutions=[],
                proof=Proof(problem=problem, method="none"),
                solver_used=SolverType.LLM,
                confidence=0.0,
                execution_time_ms=elapsed,
                error="No solver available for this problem type",
            )

        # 3. Solve
        try:
            result = solver.solve(problem, timeout_seconds=timeout_seconds)
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return SolveResult(
                problem=problem,
                problem_type=ptype,
                solutions=[],
                proof=Proof(problem=problem, method="error"),
                solver_used=SolverType.LLM,
                confidence=0.0,
                execution_time_ms=elapsed,
                error=str(exc),
            )

        # 4. Patch timing
        result.execution_time_ms = (time.perf_counter() - t0) * 1000
        return result

    def _pick_solver(self, problem: str, ptype: ProblemType) -> Optional[BaseSolver]:
        """Return the best registered solver for this problem type.
        
        Prefers solvers whose type matches the router's recommendation.
        Falls back to any solver that reports can_solve().
        """
        from math_logic.solvers.arithmetic_solver import ArithmeticSolver
        from math_logic.solvers.sympy_solver import SymPySolver

        # Build a type->solver map for priority ordering
        type_priority = {
            ProblemType.ARITHMETIC: ArithmeticSolver,
            ProblemType.ALGEBRA: SymPySolver,
            ProblemType.SIMPLIFICATION: SymPySolver,
            ProblemType.CALCULUS: SymPySolver,
        }

        preferred_cls = type_priority.get(ptype)
        
        # First pass: try the preferred solver class
        if preferred_cls:
            for s in self._solvers:
                if isinstance(s, preferred_cls) and s.can_solve(problem, ptype):
                    return s

        # Second pass: any solver that can handle it
        for s in self._solvers:
            if s.can_solve(problem, ptype):
                return s
        return None
