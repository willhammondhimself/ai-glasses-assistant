"""
Local Math Solver using SymPy.
Zero API costs for algebra, calculus, and arithmetic.
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Set

logger = logging.getLogger(__name__)

# Lazy import SymPy
sympy = None
parse_expr = None


def _load_sympy():
    """Lazy load sympy."""
    global sympy, parse_expr
    if sympy is None:
        try:
            import sympy as _sympy
            from sympy.parsing.sympy_parser import parse_expr as _parse_expr
            sympy = _sympy
            parse_expr = _parse_expr
            logger.info("SymPy loaded successfully")
        except ImportError:
            logger.warning("sympy not installed. Run: pip install sympy")
    return sympy is not None


@dataclass
class MathSolution:
    """Result of local math solving."""
    answer: str
    steps: List[str]
    method: str  # "sympy_algebra", "sympy_calculus", etc.
    numeric_answer: Optional[float] = None
    success: bool = True


class LocalMathSolver:
    """
    SymPy-powered local math solver. Zero API costs.

    Capabilities:
    - Arithmetic: 2 + 3 * 4
    - Algebra: 2x + 5 = 15 → x = 5
    - Polynomials: x^2 - 5x + 6 = 0 → x = 2, 3
    - Derivatives: d/dx(x^2 + 3x) → 2x + 3
    - Integrals: ∫(x^2)dx → x^3/3
    - Trigonometry: sin, cos, tan simplification
    """

    # Problem types we can solve locally
    SOLVABLE_LOCALLY: Set[str] = {
        "arithmetic",
        "algebra",
        "polynomial",
        "derivative",
        "integral",
        "trigonometry",
        "simplify",
    }

    # Problem types that need cloud
    REQUIRES_CLOUD: Set[str] = {
        "word_problem",
        "proof",
        "optimization",
        "statistics",
        "geometry",
    }

    def __init__(self):
        self._available = _load_sympy()

    @property
    def is_available(self) -> bool:
        """Check if SymPy is available."""
        return self._available

    def classify(self, text: str) -> str:
        """
        Classify problem type.

        Args:
            text: Math problem text

        Returns:
            Problem type string
        """
        text_lower = text.lower()

        # Check for word problem indicators
        word_problem_indicators = [
            "train", "farmer", "total", "prove", "show that",
            "if", "then", "given", "find the", "how many",
            "what is the", "calculate the", "determine"
        ]
        if any(w in text_lower for w in word_problem_indicators):
            if len(text.split()) > 10:  # Likely a word problem
                return "word_problem"

        # Check for calculus
        if any(w in text_lower for w in ["d/dx", "derivative", "differentiate"]):
            return "derivative"
        if any(w in text_lower for w in ["integrate", "integral", "∫"]):
            return "integral"

        # Check for trigonometry
        if any(w in text_lower for w in ["sin", "cos", "tan", "cot", "sec", "csc"]):
            return "trigonometry"

        # Check for simplification
        if "simplify" in text_lower:
            return "simplify"

        # Check for polynomial (has = and x^n or x**n)
        if "=" in text and (re.search(r'x\s*[\^*]+\s*\d', text) or "x**" in text):
            return "polynomial"

        # Check for algebra (has = and variable)
        if "=" in text and any(c.isalpha() for c in text.replace("=", "")):
            return "algebra"

        # Default to arithmetic if just numbers and operators
        if re.match(r'^[\d\s\+\-\*/\(\)\.\^]+$', text.strip()):
            return "arithmetic"

        return "algebra"  # Default

    def can_solve_locally(self, text: str) -> bool:
        """Check if problem can be solved locally."""
        problem_type = self.classify(text)
        return problem_type in self.SOLVABLE_LOCALLY

    async def solve(self, problem: str) -> Optional[MathSolution]:
        """
        Solve math problem locally if possible.

        Args:
            problem: Math problem text or equation

        Returns:
            MathSolution if solved, None if needs cloud
        """
        if not self._available:
            return None

        problem_type = self.classify(problem)

        if problem_type in self.REQUIRES_CLOUD:
            return None  # Escalate to DeepSeek

        try:
            if problem_type == "arithmetic":
                return self._solve_arithmetic(problem)
            elif problem_type == "algebra":
                return self._solve_algebra(problem)
            elif problem_type == "polynomial":
                return self._solve_polynomial(problem)
            elif problem_type == "derivative":
                return self._solve_derivative(problem)
            elif problem_type == "integral":
                return self._solve_integral(problem)
            elif problem_type == "trigonometry":
                return self._solve_trig(problem)
            elif problem_type == "simplify":
                return self._simplify(problem)
            else:
                return None  # Fallback to cloud

        except Exception as e:
            logger.warning(f"Local solve failed: {e}")
            return None  # Fallback to cloud

    def _solve_arithmetic(self, expression: str) -> MathSolution:
        """Solve arithmetic expression."""
        # Clean up expression
        expr = expression.replace("^", "**").strip()

        # Evaluate
        result = sympy.sympify(expr)
        numeric = float(result.evalf())

        return MathSolution(
            answer=str(result),
            steps=[f"Evaluate: {expression}", f"= {result}"],
            method="sympy_arithmetic",
            numeric_answer=numeric
        )

    def _solve_algebra(self, equation: str) -> MathSolution:
        """Solve algebraic equation for x."""
        x = sympy.Symbol('x')

        # Parse equation - handle both "2x + 5 = 15" and "2*x + 5 = 15"
        equation = equation.replace("^", "**")

        # Add implicit multiplication: "2x" → "2*x"
        equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)

        if "=" in equation:
            left, right = equation.split("=")
            # Create equation: left - right = 0
            expr = parse_expr(left.strip()) - parse_expr(right.strip())
        else:
            expr = parse_expr(equation)

        solution = sympy.solve(expr, x)

        if not solution:
            return MathSolution(
                answer="No solution",
                steps=[f"Solve: {equation}", "No real solution found"],
                method="sympy_algebra",
                success=False
            )

        # Format solution
        if len(solution) == 1:
            answer = f"x = {solution[0]}"
            numeric = float(solution[0].evalf()) if solution[0].is_number else None
        else:
            answer = "x = " + ", ".join(str(s) for s in solution)
            numeric = None

        return MathSolution(
            answer=answer,
            steps=[f"Solve: {equation}", f"Rearrange and solve", answer],
            method="sympy_algebra",
            numeric_answer=numeric
        )

    def _solve_polynomial(self, equation: str) -> MathSolution:
        """Solve polynomial equation."""
        x = sympy.Symbol('x')

        # Clean and parse
        equation = equation.replace("^", "**")
        equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)

        if "=" in equation:
            left, right = equation.split("=")
            expr = parse_expr(left.strip()) - parse_expr(right.strip())
        else:
            expr = parse_expr(equation)

        solutions = sympy.solve(expr, x)

        if not solutions:
            return MathSolution(
                answer="No solution",
                steps=[f"Solve: {equation}", "No solutions found"],
                method="sympy_polynomial",
                success=False
            )

        answer = "x = " + ", ".join(str(s) for s in solutions)

        return MathSolution(
            answer=answer,
            steps=[
                f"Polynomial: {equation}",
                f"Factor or apply quadratic formula",
                answer
            ],
            method="sympy_polynomial"
        )

    def _solve_derivative(self, expression: str) -> MathSolution:
        """Calculate derivative."""
        x = sympy.Symbol('x')

        # Extract expression from "d/dx ..." or "derivative of ..."
        expr_str = expression.lower()
        expr_str = re.sub(r'd/dx\s*[\(\[]?', '', expr_str)
        expr_str = re.sub(r'derivative\s*(of)?', '', expr_str)
        expr_str = re.sub(r'[\)\]]', '', expr_str)
        expr_str = expr_str.replace("^", "**").strip()
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)

        expr = parse_expr(expr_str)
        derivative = sympy.diff(expr, x)

        return MathSolution(
            answer=str(derivative),
            steps=[
                f"d/dx({expr})",
                f"Apply differentiation rules",
                f"= {derivative}"
            ],
            method="sympy_derivative"
        )

    def _solve_integral(self, expression: str) -> MathSolution:
        """Calculate integral."""
        x = sympy.Symbol('x')

        # Extract expression from "integral of ..." or "∫ ..."
        expr_str = expression.lower()
        expr_str = re.sub(r'(integrate|integral\s*(of)?|∫)', '', expr_str)
        expr_str = re.sub(r'dx', '', expr_str)
        expr_str = expr_str.replace("^", "**").strip()
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)

        expr = parse_expr(expr_str)
        integral = sympy.integrate(expr, x)

        return MathSolution(
            answer=f"{integral} + C",
            steps=[
                f"∫({expr})dx",
                f"Apply integration rules",
                f"= {integral} + C"
            ],
            method="sympy_integral"
        )

    def _solve_trig(self, expression: str) -> MathSolution:
        """Simplify trigonometric expression."""
        # Clean expression
        expr_str = expression.replace("^", "**")
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)

        expr = parse_expr(expr_str)
        simplified = sympy.trigsimp(expr)

        return MathSolution(
            answer=str(simplified),
            steps=[
                f"Simplify: {expression}",
                f"Apply trig identities",
                f"= {simplified}"
            ],
            method="sympy_trigonometry"
        )

    def _simplify(self, expression: str) -> MathSolution:
        """Simplify expression."""
        # Remove "simplify" keyword
        expr_str = re.sub(r'simplify\s*', '', expression.lower())
        expr_str = expr_str.replace("^", "**")
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)

        expr = parse_expr(expr_str)
        simplified = sympy.simplify(expr)

        return MathSolution(
            answer=str(simplified),
            steps=[
                f"Simplify: {expr}",
                f"= {simplified}"
            ],
            method="sympy_simplify"
        )


# Test
if __name__ == "__main__":
    import asyncio

    async def test():
        print("=== Local Math Solver Test ===\n")

        solver = LocalMathSolver()

        if not solver.is_available:
            print("SymPy not available. Install with: pip install sympy")
            return

        tests = [
            ("Arithmetic", "2 + 3 * 4"),
            ("Algebra", "2x + 5 = 15"),
            ("Polynomial", "x^2 - 5x + 6 = 0"),
            ("Derivative", "d/dx(x^2 + 3x)"),
            ("Integral", "integrate x^2 dx"),
        ]

        for name, problem in tests:
            print(f"{name}: {problem}")
            result = await solver.solve(problem)
            if result:
                print(f"  Answer: {result.answer}")
                print(f"  Method: {result.method}")
            else:
                print("  Needs cloud solver")
            print()

    asyncio.run(test())
