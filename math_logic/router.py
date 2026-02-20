"""
Query Router — Classifies problem types and selects appropriate solvers.
Uses regex heuristics with extensible pattern matching.
"""

from typing import List, Tuple, Optional
from math_logic.engine import ProblemType, SolverType
import re


class Router:
    """
    Rule-based router for problem classification.

    Returns (ProblemType, [recommended SolverTypes], confidence).
    """

    PATTERNS = {
        ProblemType.ARITHMETIC: [
            r"(?:^|\s)(?:sqrt|factorial|log|log2|log10|sin|cos|tan|ceil|floor|abs|gcd)\s*\(",
            r"(?:what is|compute|calculate|evaluate)\s+[\d\.\+\-\*\/\(\)\^\s]+",
            r"^\s*[\d\.\+\-\*\/\(\)\^\s]{3,}\s*$",
            r"(?:verify|check)\s+(?:that\s+)?[\d\.\+\-\*\/\(\)\^\s]+=",
            # Only match digit-op-digit when no letters surround it
            r"(?<![a-z])\d+\s*[\+\-\*\/\^]\s*\d+(?![a-z])",  # Only if no variable letters in problem
        ],
        ProblemType.ALGEBRA: [
            r"solve\s+\(",
            r"solve\s+(?:for\s+)?[a-z]",
            r"solve\s+.*=",
            r"find\s+(?:the\s+)?(?:value|root|zero|solution)",
            r"(?:what|find)\s+.*\bif\b.*[a-z]\s*[\+\-\*\/]",
            r"[a-z]\s*[\+\-]\s*\d+\s*=\s*\d+",
            r"\d+\s*\*?\s*[a-z]\^?\d*\s*[\+\-].*=",
            r"roots?\s+of",
            r"factor(?:i[sz]e)?\s",
            r"expand\s",
        ],
        ProblemType.SIMPLIFICATION: [
            r"simplify\s",
            r"reduce\s",
            r"rewrite\s",
            # Boost: if 'simplify' + variable present, likely simplification
            r"simplify\s+.*[a-z]",
            r"reduce\s+.*[a-z]",
        ],
        ProblemType.LOGIC: [
            r"satisf(?:y|iable)",
            r"\bsat\b",
            r"[∨∧¬→↔]",
            r"\bprop(?:ositional)?\b",
            r"\bformula\b",
            r"\b(?:AND|OR|NOT)\b",
            r"truth\s+(?:table|value)",
            r"tautolog",
            r"valid(?:ity)?\s+of",
        ],
        ProblemType.SAT: [
            r"\b3-?sat\b",
            r"\bclause\b",
            r"\bcnf\b",
            r"\bdnf\b",
        ],
        ProblemType.SMT: [
            r"\bsmt\b",
            r"\bpredicate\b",
            r"first.order",
        ],
        ProblemType.CONSTRAINT: [
            r"minimi[sz]e",
            r"maximi[sz]e",
            r"subject\s+to",
            r"schedul",
            r"allocat",
            r"knapsack",
            r"optim",
        ],
        ProblemType.CALCULUS: [
            r"derivat(?:e|ive)",
            r"integr(?:al|ate)",
            r"\blimit\b",
            r"differentiat",
            r"\bd/dx\b",
            r"taylor\s+(?:series|expansion)",
            r"partial\s+derivat",
        ],
        ProblemType.DISCRETE: [
            r"combinat",
            r"\bgraph\b",
            r"permutation",
            r"combination",
            r"fibonacci",
            r"recurrence",
            r"gcd|lcm",
            r"modular\s+arithmetic",
        ],
    }

    SOLVER_MAP = {
        ProblemType.ARITHMETIC: [SolverType.PYTHON, SolverType.SYMPY],
        ProblemType.ALGEBRA: [SolverType.SYMPY, SolverType.LLM],
        ProblemType.SIMPLIFICATION: [SolverType.SYMPY, SolverType.LLM],
        ProblemType.LOGIC: [SolverType.Z3, SolverType.LLM],
        ProblemType.SAT: [SolverType.Z3],
        ProblemType.SMT: [SolverType.Z3],
        ProblemType.CONSTRAINT: [SolverType.OR_TOOLS, SolverType.Z3],
        ProblemType.CALCULUS: [SolverType.SYMPY, SolverType.LLM],
        ProblemType.DISCRETE: [SolverType.SYMPY, SolverType.LLM],
        ProblemType.UNKNOWN: [SolverType.LLM],
    }

    def classify(
        self,
        problem: str,
        llm: Optional[object] = None,
    ) -> Tuple[ProblemType, List[SolverType], float]:
        """
        Classify a problem and recommend solvers.

        Returns:
            (problem_type, recommended_solvers, confidence 0-1)
        """
        problem_lower = problem.lower()
        scores: dict[ProblemType, float] = {pt: 0.0 for pt in ProblemType}

        for problem_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, problem_lower, re.IGNORECASE):
                    scores[problem_type] += 1.0

        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]
        confidence = min(best_score / 2.0, 1.0)

        if best_score == 0:
            best_type = ProblemType.UNKNOWN
            confidence = 0.0

        solvers = self.SOLVER_MAP.get(best_type, [SolverType.LLM])
        return best_type, solvers, confidence


__all__ = ["Router"]
