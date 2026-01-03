"""
MathEngine: Solves mathematical problems using SymPy (free) with Claude fallback.

Strategy:
1. Try to parse and solve with SymPy (local, free)
2. If SymPy fails or problem is a word problem, use Claude Sonnet 3.5
"""

import os
import re
from typing import Optional
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import anthropic
from dotenv import load_dotenv

load_dotenv()


class MathEngine:
    """Math problem solver with SymPy-first, Claude-fallback approach."""

    def __init__(self):
        self.anthropic_client: Optional[anthropic.Anthropic] = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)

    def solve(self, problem: str) -> dict:
        """
        Main entry point for solving math problems.

        Args:
            problem: Mathematical equation or word problem

        Returns:
            dict with keys: solution, method, steps (optional), error (if failed)
        """
        # First, try to detect if this is a pure equation vs word problem
        if self._is_pure_equation(problem):
            sympy_result = self._try_sympy(problem)
            if sympy_result is not None:
                return {
                    "solution": sympy_result["solution"],
                    "method": "sympy",
                    "steps": sympy_result.get("steps"),
                    "error": None
                }

        # Fallback to Claude for word problems or if SymPy failed
        return self._solve_with_claude(problem)

    def _is_pure_equation(self, problem: str) -> bool:
        """
        Detect if the problem is a pure mathematical equation vs a word problem.

        Pure equations contain mostly math symbols and numbers.
        Word problems contain substantial English text.
        """
        # Count words vs math characters
        words = re.findall(r'[a-zA-Z]{3,}', problem)
        math_chars = len(re.findall(r'[0-9+\-*/^()=<>√∫∑πexy]', problem))

        # If there are many words, it's likely a word problem
        word_count = len(words)

        # Keywords that indicate word problems
        word_problem_keywords = [
            'find', 'solve', 'calculate', 'what', 'how', 'many', 'much',
            'total', 'sum', 'difference', 'product', 'quotient', 'area',
            'volume', 'perimeter', 'distance', 'speed', 'time', 'rate',
            'percent', 'interest', 'profit', 'loss', 'cost', 'price'
        ]

        has_word_problem_keywords = any(
            keyword in problem.lower() for keyword in word_problem_keywords
        )

        # Heuristic: if more than 5 words or has word problem keywords, not a pure equation
        return word_count <= 5 and not has_word_problem_keywords

    def _try_sympy(self, equation: str) -> Optional[dict]:
        """
        Attempt to solve using SymPy.

        Args:
            equation: Mathematical equation string

        Returns:
            dict with solution and steps, or None if failed
        """
        try:
            # Clean up the equation
            equation = equation.strip()

            # Handle different equation formats
            if '=' in equation:
                # Equation solving (e.g., "x^2 - 4 = 0" or "2x + 3 = 7")
                return self._solve_equation(equation)
            else:
                # Expression evaluation or simplification
                return self._evaluate_expression(equation)

        except Exception as e:
            # SymPy failed, return None to trigger Claude fallback
            return None

    def _solve_equation(self, equation: str) -> Optional[dict]:
        """Solve an equation for a variable."""
        try:
            # Split by '='
            parts = equation.split('=')
            if len(parts) != 2:
                return None

            left, right = parts[0].strip(), parts[1].strip()

            # Define common symbols
            x, y, z, t, n = sympy.symbols('x y z t n')

            # Parse both sides
            transformations = standard_transformations + (implicit_multiplication_application,)

            left_expr = parse_expr(left, transformations=transformations)
            right_expr = parse_expr(right, transformations=transformations)

            # Create equation: left - right = 0
            eq = sympy.Eq(left_expr, right_expr)

            # Find free symbols to solve for
            free_syms = eq.free_symbols
            if not free_syms:
                # No variables, just check if equation is true
                is_true = sympy.simplify(left_expr - right_expr) == 0
                return {
                    "solution": str(is_true),
                    "steps": f"Simplified: {left_expr} = {right_expr} → {is_true}"
                }

            # Solve for the first free symbol
            solve_for = list(free_syms)[0]
            solutions = sympy.solve(eq, solve_for)

            if not solutions:
                return None

            # Format solution
            if len(solutions) == 1:
                solution_str = f"{solve_for} = {solutions[0]}"
            else:
                solution_str = f"{solve_for} = {solutions}"

            return {
                "solution": solution_str,
                "steps": f"Equation: {eq}\nSolved for {solve_for}: {solutions}"
            }

        except Exception:
            return None

    def _evaluate_expression(self, expression: str) -> Optional[dict]:
        """Evaluate or simplify a mathematical expression."""
        try:
            # Define common symbols
            x, y, z, t, n = sympy.symbols('x y z t n')

            # Parse expression
            transformations = standard_transformations + (implicit_multiplication_application,)
            expr = parse_expr(expression, transformations=transformations)

            # Try to evaluate numerically if no free symbols
            if not expr.free_symbols:
                result = sympy.N(expr)
                return {
                    "solution": str(result),
                    "steps": f"Evaluated: {expression} = {result}"
                }

            # Otherwise, simplify
            simplified = sympy.simplify(expr)
            return {
                "solution": str(simplified),
                "steps": f"Simplified: {expression} → {simplified}"
            }

        except Exception:
            return None

    def _solve_with_claude(self, problem: str) -> dict:
        """
        Solve using Claude Sonnet 3.5 (fallback for word problems or complex math).

        Args:
            problem: Math problem in any format

        Returns:
            dict with solution, method, steps, and optional error
        """
        if not self.anthropic_client:
            return {
                "solution": None,
                "method": "claude",
                "steps": None,
                "error": "ANTHROPIC_API_KEY not configured"
            }

        try:
            message = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Solve this math problem step by step. Be concise but clear.

Problem: {problem}

Provide:
1. The final answer clearly stated
2. Brief step-by-step solution

Format your response as:
ANSWER: [final answer]
STEPS:
[numbered steps]"""
                    }
                ]
            )

            response_text = message.content[0].text

            # Parse the response
            answer = None
            steps = None

            if "ANSWER:" in response_text:
                answer_start = response_text.index("ANSWER:") + 7
                answer_end = response_text.index("STEPS:") if "STEPS:" in response_text else len(response_text)
                answer = response_text[answer_start:answer_end].strip()

            if "STEPS:" in response_text:
                steps_start = response_text.index("STEPS:") + 6
                steps = response_text[steps_start:].strip()

            # Fallback if parsing failed
            if not answer:
                answer = response_text

            return {
                "solution": answer,
                "method": "claude",
                "steps": steps,
                "error": None
            }

        except anthropic.APIError as e:
            return {
                "solution": None,
                "method": "claude",
                "steps": None,
                "error": f"Claude API error: {str(e)}"
            }
        except Exception as e:
            return {
                "solution": None,
                "method": "claude",
                "steps": None,
                "error": f"Unexpected error: {str(e)}"
            }

    def solve_latex(self, latex: str) -> dict:
        """
        Solve a problem given in LaTeX format.

        Args:
            latex: LaTeX formatted equation

        Returns:
            dict with solution details
        """
        # Convert common LaTeX to SymPy-parseable format
        converted = self._latex_to_sympy(latex)
        return self.solve(converted)

    def _latex_to_sympy(self, latex: str) -> str:
        """Convert LaTeX notation to SymPy-parseable string."""
        result = latex

        # Remove LaTeX delimiters
        result = re.sub(r'^\$+|\$+$', '', result)
        result = re.sub(r'^\\[|\]$', '', result)

        # Common LaTeX to SymPy conversions
        conversions = [
            (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))'),
            (r'\\sqrt\{([^}]+)\}', r'sqrt(\1)'),
            (r'\\sqrt\[([^\]]+)\]\{([^}]+)\}', r'(\2)**(1/(\1))'),
            (r'\^(\d)', r'**\1'),
            (r'\^\{([^}]+)\}', r'**(\1)'),
            (r'\\cdot', '*'),
            (r'\\times', '*'),
            (r'\\div', '/'),
            (r'\\pi', 'pi'),
            (r'\\infty', 'oo'),
            (r'\\sin', 'sin'),
            (r'\\cos', 'cos'),
            (r'\\tan', 'tan'),
            (r'\\log', 'log'),
            (r'\\ln', 'ln'),
            (r'\\exp', 'exp'),
            (r'\\left', ''),
            (r'\\right', ''),
            (r'\{', '('),
            (r'\}', ')'),
        ]

        for pattern, replacement in conversions:
            result = re.sub(pattern, replacement, result)

        return result.strip()
