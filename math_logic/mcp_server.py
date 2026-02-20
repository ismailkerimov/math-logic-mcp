"""
MCP Server — exposes math & logic tools via the Model Context Protocol.

Usage:
    math-logic-mcp              # stdio transport (default for LLM clients)
    math-logic-mcp --http       # HTTP/SSE transport on port 8080

Requires: pip install fastmcp
"""

from __future__ import annotations

import json
import os
import sys
from typing import Annotated, Optional

from fastmcp import FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from math_logic.engine import MathLogicEngine, ProblemType

# ── server instance ──────────────────────────────────────────────────
mcp = FastMCP(
    "Math & Logic MCP Server",
    instructions=(
        "Verified symbolic math and formal logic tools for LLMs. "
        "Solves equations, simplifies expressions, checks Boolean satisfiability, "
        "evaluates arithmetic — all with proof steps."
    ),
)

# Lazy-initialised engine (shared across tool calls)
_engine: Optional[MathLogicEngine] = None


def _get_engine() -> MathLogicEngine:
    global _engine
    if _engine is None:
        _engine = MathLogicEngine()
    return _engine


# ── Tools ────────────────────────────────────────────────────────────


@mcp.tool(
    annotations={
        "title": "Solve Equation",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def solve_equation(
    equation: Annotated[
        str,
        Field(description="A math equation to solve. Examples: 'x^2 + 2x - 8 = 0', 'Solve 3x + 5 = 20', 'Find x if 2x - 7 = 3'"),
    ],
) -> str:
    """Solve an algebraic equation symbolically and return solutions with proof steps."""
    result = _get_engine().solve(equation)
    return result.to_json()


@mcp.tool(
    annotations={
        "title": "Simplify Expression",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def simplify_expression(
    expression: Annotated[
        str,
        Field(description="A math expression with an operation verb. Examples: 'simplify (x^2 - 1)/(x - 1)', 'factor x^2 - 5x + 6', 'expand (x + 1)^3'"),
    ],
) -> str:
    """Simplify, factor, or expand a mathematical expression with proof steps."""
    result = _get_engine().solve(expression)
    return result.to_json()


@mcp.tool(
    annotations={
        "title": "Compute Derivative",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def compute_derivative(
    expression: Annotated[
        str,
        Field(description="The mathematical expression to differentiate. Example: 'x^3 + 2x^2 - 5x + 1'"),
    ],
    variable: Annotated[
        str,
        Field(default="x", description="The variable to differentiate with respect to."),
    ] = "x",
) -> str:
    """Compute the derivative of a mathematical expression with proof steps."""
    problem = f"derivative of {expression} with respect to {variable}"
    result = _get_engine().solve(problem)
    return result.to_json()


@mcp.tool(
    annotations={
        "title": "Compute Integral",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def compute_integral(
    expression: Annotated[
        str,
        Field(description="The mathematical expression to integrate. Example: '3x^2 + 4x'"),
    ],
    variable: Annotated[
        str,
        Field(default="x", description="The variable to integrate with respect to."),
    ] = "x",
) -> str:
    """Compute the indefinite integral of a mathematical expression with proof steps."""
    problem = f"integrate {expression} with respect to {variable}"
    result = _get_engine().solve(problem)
    return result.to_json()


@mcp.tool(
    annotations={
        "title": "Check Logic",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def check_logic(
    formula: Annotated[
        str,
        Field(description="A Boolean formula using variables and operators (and/&, or/|, not/!, ->/implies). Examples: 'p and (q or r)', '(p -> q) and (q -> r) -> (p -> r)'"),
    ],
    check_type: Annotated[
        str,
        Field(default="satisfiability", description="Type of check: 'satisfiability', 'tautology', or 'truth_table'."),
    ] = "satisfiability",
) -> str:
    """Check a propositional logic formula for satisfiability, tautology, or generate a truth table."""
    prefix_map = {
        "satisfiability": "Check satisfiability of",
        "tautology": "Check if tautology:",
        "truth_table": "Truth table for",
    }
    prefix = prefix_map.get(check_type, "Check satisfiability of")
    problem = f'{prefix} "{formula}"'
    result = _get_engine().solve(problem)
    return result.to_json()


@mcp.tool(
    annotations={
        "title": "Verify Arithmetic",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def verify_arithmetic(
    expression: Annotated[
        str,
        Field(description="A numeric expression to evaluate safely (no eval). Examples: '2 + 3 * 4', 'sqrt(144)', '2^10', 'factorial(10)', 'log(100, 10)'"),
    ],
) -> str:
    """Safely evaluate an arithmetic expression using only whitelisted operators and functions."""
    problem = f"compute {expression}"
    result = _get_engine().solve(problem)
    return result.to_json()


# ── Resources (optional context for LLMs) ───────────────────────────


@mcp.resource("math://capabilities")
def get_capabilities() -> str:
    """List all capabilities of this math/logic server."""
    return json.dumps({
        "solvers": {
            "arithmetic": {
                "description": "Basic arithmetic (zero dependencies)",
                "examples": ["2+3*4", "sqrt(144)", "factorial(10)"],
                "requires": None,
            },
            "algebra": {
                "description": "Symbolic equation solving",
                "examples": ["x^2-4=0", "Solve 3x+5=20"],
                "requires": "sympy",
            },
            "simplification": {
                "description": "Simplify / factor / expand expressions",
                "examples": ["simplify (x^2-1)/(x-1)", "factor x^2-5x+6"],
                "requires": "sympy",
            },
            "calculus": {
                "description": "Derivatives and integrals",
                "examples": ["derivative of x^3", "integrate 2x"],
                "requires": "sympy",
            },
            "logic": {
                "description": "Propositional logic / SAT / tautology",
                "examples": ["p and (q or r)", "p -> q"],
                "requires": "z3-solver",
            },
        },
        "tools": [
            "solve_equation", "simplify_expression",
            "compute_derivative", "compute_integral",
            "check_logic", "verify_arithmetic",
        ],
    }, indent=2)


# ── HTTP routes (health check + Smithery server card) ────────────────

_TOOL_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

_SERVER_CARD = {
    "serverInfo": {
        "name": "Math & Logic MCP Server",
        "version": "0.2.0",
    },
    "authentication": {
        "required": False,
    },
    "tools": [
        {
            "name": "solve_equation",
            "description": "Solve an algebraic equation symbolically and return solutions with proof steps.",
            "annotations": _TOOL_ANNOTATIONS,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "equation": {"type": "string", "description": "A math equation to solve. Examples: 'x^2 + 2x - 8 = 0', 'Solve 3x + 5 = 20', 'Find x if 2x - 7 = 3'"}
                },
                "required": ["equation"],
            },
        },
        {
            "name": "simplify_expression",
            "description": "Simplify, factor, or expand a mathematical expression with proof steps.",
            "annotations": _TOOL_ANNOTATIONS,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A math expression with an operation verb. Examples: 'simplify (x^2 - 1)/(x - 1)', 'factor x^2 - 5x + 6', 'expand (x + 1)^3'"}
                },
                "required": ["expression"],
            },
        },
        {
            "name": "compute_derivative",
            "description": "Compute the derivative of a mathematical expression with proof steps.",
            "annotations": _TOOL_ANNOTATIONS,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The mathematical expression to differentiate. Example: 'x^3 + 2x^2 - 5x + 1'"},
                    "variable": {"type": "string", "default": "x", "description": "The variable to differentiate with respect to."},
                },
                "required": ["expression"],
            },
        },
        {
            "name": "compute_integral",
            "description": "Compute the indefinite integral of a mathematical expression with proof steps.",
            "annotations": _TOOL_ANNOTATIONS,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The mathematical expression to integrate. Example: '3x^2 + 4x'"},
                    "variable": {"type": "string", "default": "x", "description": "The variable to integrate with respect to."},
                },
                "required": ["expression"],
            },
        },
        {
            "name": "check_logic",
            "description": "Check a propositional logic formula for satisfiability, tautology, or generate a truth table.",
            "annotations": _TOOL_ANNOTATIONS,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "formula": {"type": "string", "description": "A Boolean formula using variables and operators (and/&, or/|, not/!, ->/implies). Examples: 'p and (q or r)', '(p -> q) and (q -> r) -> (p -> r)'"},
                    "check_type": {"type": "string", "default": "satisfiability", "description": "Type of check: 'satisfiability', 'tautology', or 'truth_table'."},
                },
                "required": ["formula"],
            },
        },
        {
            "name": "verify_arithmetic",
            "description": "Safely evaluate an arithmetic expression using only whitelisted operators and functions.",
            "annotations": _TOOL_ANNOTATIONS,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A numeric expression to evaluate safely (no eval). Examples: '2 + 3 * 4', 'sqrt(144)', '2^10', 'factorial(10)', 'log(100, 10)'"}
                },
                "required": ["expression"],
            },
        },
    ],
    "resources": [],
    "prompts": [],
}


@mcp.custom_route("/", methods=["GET"])
async def health_check(request: Request) -> Response:
    """Health check — lets Railway / load balancers know the server is alive."""
    return JSONResponse({
        "status": "ok",
        "server": "Math & Logic MCP Server",
        "version": "0.2.0",
        "mcp_endpoint": "/mcp",
    })


@mcp.custom_route("/.well-known/mcp/server-card.json", methods=["GET"])
async def server_card(request: Request) -> Response:
    """Smithery server card for automatic scanning."""
    return JSONResponse(_SERVER_CARD)


# ── Entry point ──────────────────────────────────────────────────────

def main():
    """CLI entry point for the MCP server."""
    if "--http" in sys.argv:
        port = int(os.environ.get("PORT", 8080))
        # Use streamable-http (MCP spec default); fall back to SSE with --sse
        transport = "sse" if "--sse" in sys.argv else "streamable-http"
        mcp.run(transport=transport, host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
