# math-logic-mcp

**MCP server that gives small LLMs verified symbolic-math & logic tools.**

[![PyPI](https://img.shields.io/pypi/v/math-logic-mcp)](https://pypi.org/project/math-logic-mcp/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Small language models (Mistral, Llama, Phi, Gemma) struggle with multi-step math and formal logic. Instead of fine-tuning, **give them tools**. This project exposes a set of verified math and logic solvers via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io), so any MCP-compatible LLM can call them as functions.

## Features

| Tool | What it does | Backend |
|------|-------------|---------|
| `verify_arithmetic` | Safe arithmetic evaluation | Python stdlib (zero deps) |
| `solve_equation` | Symbolic equation solving | SymPy (optional) |
| `simplify_expression` | Simplify / factor / expand | SymPy (optional) |
| `compute_derivative` | Differentiation | SymPy (optional) |
| `compute_integral` | Integration | SymPy (optional) |
| `check_logic` | SAT / tautology / truth tables | Z3 (optional) |

Every result includes **proof steps** and **verification** — the LLM gets a machine-checked answer, not a guess.

## Quick Start

### Install from PyPI

```bash
pip install math-logic-mcp
```

### Install (full — all solvers)

```bash
pip install "math-logic-mcp[full]"
```

### Run the MCP server

```bash
# stdio transport (for Claude Desktop, Cursor, etc.)
math-logic-mcp

# HTTP/SSE transport (for remote clients)
math-logic-mcp --http
```

### Configure in Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "math-logic": {
      "command": "math-logic-mcp"
    }
  }
}
```

### Configure in Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "math-logic": {
      "command": "math-logic-mcp"
    }
  }
}
```

## Use as a Python Library

```python
from math_logic import MathLogicEngine

engine = MathLogicEngine()

# Arithmetic (always available)
result = engine.solve("compute 2 + 3 * 4")
print(result.solutions)  # ['14']

# Algebra (requires sympy)
result = engine.solve("Solve x^2 - 4 = 0")
print(result.solutions)  # ['x = -2', 'x = 2']

# Logic (requires z3-solver)
result = engine.solve('Check satisfiability of "p and q"')
print(result.solutions)  # ['Satisfiable: p=True, q=True']
```

## Architecture

```
LLM ─── MCP Protocol ──▶ mcp_server.py
                              │
                          engine.py  (router → solver → result)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ArithmeticSolver   SymPySolver      Z3Solver
      (zero deps)       (pip: sympy)    (pip: z3-solver)
```

The **router** classifies each problem by regex patterns and routes to the best available solver. Solvers are loaded lazily — if SymPy isn't installed, algebra problems gracefully report the missing dependency.

## Installation Extras

| Extra | What it adds | Install size |
|-------|-------------|-------------|
| (none) | Arithmetic only | ~1 MB |
| `[sympy]` | + algebra, calculus, simplification | ~50 MB |
| `[z3]` | + propositional logic, SAT | ~30 MB |
| `[full]` | Everything | ~80 MB |
| `[dev]` | + pytest, ruff | ~85 MB |

```bash
pip install "math-logic-mcp[sympy]"      # algebra + calculus
pip install "math-logic-mcp[z3]"         # logic
pip install "math-logic-mcp[full]"       # everything
pip install "math-logic-mcp[full,dev]"   # everything + dev tools
```

## Development

```bash
git clone https://github.com/ismailkerimov/math-logic-mcp.git
cd math-logic-mcp
pip install -e ".[full,dev]"
pytest tests/ -v
```

## Docker

```bash
docker build -t math-logic-mcp .
docker run -p 8080:8080 math-logic-mcp
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
