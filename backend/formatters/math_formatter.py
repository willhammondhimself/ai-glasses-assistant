"""
Math Slide Builder

Formats math engine results into paginated slides for AR display.
Handles equations, step-by-step solutions, and final answers.
"""

import re
from typing import List, Optional

from backend.models.ar_response import Slide, ColorScheme
from .base import BaseSlideBuilder


class MathSlideBuilder(BaseSlideBuilder):
    """
    Builds slides for math/physics problems.

    Slide structure:
    1. Problem Setup (blue)
    2-N. Solution Steps (blue)
    N+1. Final Answer (green)
    """

    def __init__(self):
        super().__init__()
        self.response_type = "math"

    def build_slides(self, engine_result: dict) -> List[Slide]:
        """
        Build slides from math engine result.

        Args:
            engine_result: Dict with 'problem', 'steps', 'solution', etc.

        Returns:
            List of formatted slides
        """
        slides = []

        # Slide 1: Problem Setup
        problem = engine_result.get("problem", engine_result.get("input", ""))
        if problem:
            slides.append(Slide(
                title="Problem",
                content=self._format_equation(problem),
                color_scheme=ColorScheme.MATH,
                voice_narration=f"Solving: {problem}",
                centered=True,
                large_font=self._is_simple_equation(problem)
            ))

        # Slides 2-N: Steps
        steps = self._parse_steps(engine_result.get("steps", ""))
        for i, step in enumerate(steps, 1):
            slides.extend(self._create_content_slides(
                content=step,
                title=f"Step {i}",
                color_scheme=ColorScheme.MATH,
                voice_prefix=f"Step {i}: "
            ))

        # Final Slide: Solution
        solution = engine_result.get("solution", "")
        if solution:
            slides.append(Slide(
                title="Solution",
                content=self._format_solution(solution),
                color_scheme=ColorScheme.RESULT_GREEN,
                voice_narration=f"The answer is {solution}",
                centered=True,
                large_font=True
            ))

        # Ensure at least one slide
        if not slides:
            slides.append(Slide(
                title="Result",
                content=str(engine_result.get("solution", "No solution")),
                color_scheme=ColorScheme.INFO
            ))

        return slides

    def _parse_steps(self, steps_text: str) -> List[str]:
        """
        Parse step-by-step solution text into individual steps.

        Handles formats:
        - Numbered: "1. First step\n2. Second step"
        - Arrows: "x = 5 → x + 3 = 8"
        - Line-separated
        """
        if not steps_text:
            return []

        # Try numbered format first
        numbered_pattern = r'(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=\n\s*\d+[\.\)]|\Z)'
        matches = re.findall(numbered_pattern, steps_text, re.DOTALL)
        if matches:
            return [m.strip() for m in matches if m.strip()]

        # Try arrow-separated
        if '→' in steps_text or '->' in steps_text:
            arrow_steps = re.split(r'\s*(?:→|->)\s*', steps_text)
            return [s.strip() for s in arrow_steps if s.strip()]

        # Fall back to line-separated
        lines = [l.strip() for l in steps_text.split('\n') if l.strip()]
        return lines

    def _format_equation(self, equation: str) -> str:
        """
        Format equation for AR display.

        Converts common notation to display-friendly format.
        """
        # Replace common LaTeX-like notation
        formatted = equation
        formatted = formatted.replace('\\cdot', '×')
        formatted = formatted.replace('\\times', '×')
        formatted = formatted.replace('\\div', '÷')
        formatted = formatted.replace('\\pm', '±')
        formatted = formatted.replace('\\sqrt', '√')
        formatted = formatted.replace('\\pi', 'π')
        formatted = formatted.replace('\\theta', 'θ')
        formatted = formatted.replace('\\alpha', 'α')
        formatted = formatted.replace('\\beta', 'β')
        formatted = formatted.replace('**', '^')

        # Clean up extra whitespace
        formatted = ' '.join(formatted.split())

        return formatted

    def _format_solution(self, solution: str) -> str:
        """Format the final solution for display."""
        formatted = self._format_equation(solution)

        # If it's a simple answer, make it prominent
        if self._is_simple_answer(formatted):
            return formatted

        # Truncate long solutions
        if len(formatted) > self.MAX_LINE_CHARS * self.MAX_CONTENT_LINES:
            return formatted[:self.MAX_LINE_CHARS * self.MAX_CONTENT_LINES - 3] + "..."

        return formatted

    def _is_simple_equation(self, text: str) -> bool:
        """Check if equation is simple enough for large font."""
        return len(text) < 25 and '\n' not in text

    def _is_simple_answer(self, answer: str) -> bool:
        """Check if answer is a simple value (number, short expression)."""
        # Remove spaces and check length
        clean = answer.replace(' ', '')
        return len(clean) < 20 and '\n' not in answer

    def _generate_summary(self, engine_result: dict) -> Optional[str]:
        """Generate one-line summary: 'problem = solution'."""
        problem = engine_result.get("problem", engine_result.get("input", ""))
        solution = engine_result.get("solution", "")

        if problem and solution:
            summary = f"{problem} = {solution}"
            if len(summary) <= self.MAX_LINE_CHARS:
                return summary
            return f"Answer: {solution}"[:self.MAX_LINE_CHARS]

        return super()._generate_summary(engine_result)
