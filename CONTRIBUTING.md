# Contributing to Math/Logic Engine

Thank you for your interest in contributing! This document provides guidelines for getting involved.

## Getting Started

### 1. Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/math-logic-engine.git
cd math-logic-engine
git remote add upstream https://github.com/math-logic-engine/math-logic-engine.git
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### 3. Make Changes & Test
```bash
# Make your changes
git add .
git commit -m "descriptive message"

# Run tests
pytest tests/ -v

# Check code quality
black math_logic/ tests/
flake8 math_logic/ tests/
mypy math_logic/
```

### 4. Push & Create PR
```bash
git push origin your-branch-name
# Go to GitHub and create a Pull Request
```

## Areas to Contribute

### 🔧 Core Solvers
- **SymPy Algebra Solver** (Phase 1.2)
  - Improve equation parsing
  - Add support for systems of equations
  - Generate better proofs
  - Link: `math_logic/solvers/sympy_solver.py`

- **Z3 Logic Solver** (Phase 1.3)
  - Extend formula parsing
  - Add SMT constraint support
  - Improve model extraction
  - Link: `math_logic/solvers/z3_solver.py`

### 🧭 Query Router
- Improve problem classification accuracy
- Add ML-based classification (Phase 2)
- Link: `math_logic/router.py`

### 📚 Documentation
- Write tutorials (Jupyter notebooks)
- Improve API docs
- Create blog posts on applications
- Link: `docs/`, `examples/`

### 🧪 Testing
- Write unit tests
- Add integration tests
- Create benchmarks
- Link: `tests/`

### 🎨 UI/Visualization
- Web interface for proofs (Phase 2)
- Proof visualization tool
- Link: (new, TBD)

## Code Style

We use:
- **Black** for formatting (line length: 100)
- **Flake8** for linting
- **MyPy** for type checking
- **Google-style docstrings** for documentation

Example:
```python
def solve(
    self,
    problem: str,
    timeout_seconds: float = 10.0,
) -> SolveResult:
    """
    Solve a problem and return result with proof.
    
    Args:
        problem: Problem statement (e.g., "Solve x^2 + 2x - 8 = 0")
        timeout_seconds: Max solving time in seconds
    
    Returns:
        SolveResult with solutions and proof
    
    Raises:
        ValueError: If problem cannot be parsed
    
    Example:
        >>> result = solver.solve("Solve x^2 - 1 = 0")
        >>> print(result.solutions)
        ['x = -1', 'x = 1']
    """
```

## Testing Guidelines

- Aim for >80% code coverage
- Test edge cases (no solution, infinite solutions, etc.)
- Use descriptive test names
- Run full test suite before submitting PR

```bash
pytest tests/ --cov=math_logic --cov-report=html
```

## PR Process

1. **Title:** Clear, concise description
2. **Description:** Explain what problem you're solving and how
3. **Testing:** Link to passing tests or include new tests
4. **Documentation:** Update README/docs if needed
5. **Review:** Wait for 1-2 community reviews

We aim to review PRs within **1 week**.

## Commit Messages

Use conventional commits:
```
fix: correct quadratic formula parsing bug
feat: add support for systems of equations
docs: add calculus solver tutorial
test: add edge case tests for Z3 solver
chore: update dependencies
```

## Questions?

- 💬 Start a Discussion on GitHub
- 🐛 Check existing Issues for similar work

**Thank you for contributing! 🙏**
