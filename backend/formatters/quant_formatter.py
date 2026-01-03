"""
Quant/Mental Math Slide Builder

Formats mental math problems, probability questions, and quant interview
content for AR display. Optimized for speed-run mode with timers.
"""

import random
from typing import List, Optional, Union, Any

from backend.models.ar_response import (
    Slide,
    ColorScheme,
    MentalMathProblem,
    MentalMathResult,
)
from .base import BaseSlideBuilder


class QuantSlideBuilder(BaseSlideBuilder):
    """
    Builds slides for quant/mental math content.

    Optimized for:
    - Large, centered problem display
    - Timer integration
    - Streak tracking
    - Quick result feedback
    """

    # Time targets by difficulty (milliseconds)
    TIME_TARGETS = {
        1: 10000,   # Easy: 10s
        2: 15000,   # Medium: 15s
        3: 20000,   # Hard: 20s
        4: 30000,   # Very Hard: 30s
        5: 45000,   # Expert: 45s
    }

    # Difficulty names for display
    DIFFICULTY_NAMES = {
        1: "D1",
        2: "D2",
        3: "D3",
        4: "D4",
        5: "D5",
    }

    # Encouraging messages for streaks
    STREAK_MESSAGES = [
        "",           # 0-2
        "Nice!",      # 3-4
        "On fire!",   # 5-7
        "Unstoppable!",  # 8-10
        "LEGENDARY!", # 11+
    ]

    def __init__(self):
        super().__init__()
        self.response_type = "quant"

    def build_slides(self, engine_result: dict) -> List[Slide]:
        """
        Build slides from quant engine result.

        Args:
            engine_result: Dict with problem, solution, category, etc.

        Returns:
            List of formatted slides
        """
        slides = []

        # Determine the type of quant content
        category = engine_result.get("category", "arithmetic")

        if category in ["arithmetic", "mental_math"]:
            slides = self._build_mental_math_slides(engine_result)
        elif category in ["probability", "expected_value"]:
            slides = self._build_probability_slides(engine_result)
        elif category == "market_making":
            slides = self._build_market_making_slides(engine_result)
        else:
            slides = self._build_generic_quant_slides(engine_result)

        return slides if slides else [Slide(
            title="Quant",
            content=str(engine_result.get("solution", "No content")),
            color_scheme=ColorScheme.QUANT_TIMER
        )]

    def build_problem_slide(self, problem: MentalMathProblem) -> Slide:
        """
        Build a single problem slide for mental math mode.

        Optimized for large, centered display.
        """
        difficulty_label = self.DIFFICULTY_NAMES.get(problem.difficulty, "D?")

        return Slide(
            title=f"[{problem.category.upper()}]  {difficulty_label}",
            content=problem.problem,
            color_scheme=ColorScheme.MATH,
            centered=True,
            large_font=True,
            voice_narration=problem.problem
        )

    def build_result_slide(self, result: MentalMathResult) -> Slide:
        """
        Build a result slide showing correct/incorrect answer.
        """
        if result.correct:
            content_lines = [
                str(result.correct_answer),
                f"{result.time_ms / 1000:.1f}s",
            ]

            # Add streak message if applicable
            streak_idx = min(4, result.current_streak // 3)
            if streak_msg := self.STREAK_MESSAGES[streak_idx]:
                content_lines.append(streak_msg)

            return Slide(
                title="Correct!",
                content='\n'.join(content_lines),
                color_scheme=ColorScheme.RESULT_GREEN,
                centered=True,
                large_font=True,
                voice_narration=f"Correct! {result.correct_answer}"
            )
        else:
            return Slide(
                title="Incorrect",
                content=f"You: {result.user_answer}\nAnswer: {result.correct_answer}",
                color_scheme=ColorScheme.RESULT_RED,
                centered=True,
                large_font=False,
                voice_narration=f"The answer was {result.correct_answer}"
            )

    def build_timer_slide(
        self,
        time_remaining_ms: int,
        time_target_ms: int,
        problem: str
    ) -> Slide:
        """Build a timer update slide."""
        # Calculate progress
        progress = 1.0 - (time_remaining_ms / time_target_ms)

        # Color based on time remaining
        if progress < 0.5:
            color = ColorScheme.MATH
        elif progress < 0.8:
            color = ColorScheme.QUANT_TIMER
        else:
            color = ColorScheme.RESULT_RED

        seconds_left = max(0, time_remaining_ms / 1000)

        return Slide(
            title=problem,
            content=f"{seconds_left:.1f}s",
            color_scheme=color,
            centered=True,
            large_font=True
        )

    def _build_mental_math_slides(self, result: dict) -> List[Slide]:
        """Build slides for mental math problem result."""
        slides = []

        # Problem slide
        problem = result.get("problem", result.get("input", ""))
        if problem:
            slides.append(Slide(
                title="Problem",
                content=problem,
                color_scheme=ColorScheme.MATH,
                centered=True,
                large_font=True
            ))

        # Solution slide
        solution = result.get("solution", result.get("answer", ""))
        if solution:
            slides.append(Slide(
                title="Answer",
                content=str(solution),
                color_scheme=ColorScheme.RESULT_GREEN,
                centered=True,
                large_font=True,
                voice_narration=f"The answer is {solution}"
            ))

        # Explanation if present
        explanation = result.get("explanation", result.get("steps", ""))
        if explanation:
            slides.extend(self._create_content_slides(
                content=explanation,
                title="Method",
                color_scheme=ColorScheme.INFO
            ))

        return slides

    def _build_probability_slides(self, result: dict) -> List[Slide]:
        """Build slides for probability problems."""
        slides = []

        # Problem setup
        problem = result.get("problem", "")
        if problem:
            slides.extend(self._create_content_slides(
                content=problem,
                title="Problem",
                color_scheme=ColorScheme.MATH
            ))

        # Solution with probability notation
        solution = result.get("solution", "")
        probability = result.get("probability", result.get("expected_value", ""))

        if probability:
            formatted = self._format_probability(probability)
            slides.append(Slide(
                title="P(X)" if "probability" in result else "E[X]",
                content=formatted,
                color_scheme=ColorScheme.RESULT_GREEN,
                centered=True,
                large_font=True
            ))

        # Steps/explanation
        steps = result.get("steps", result.get("explanation", ""))
        if steps:
            slides.extend(self._create_content_slides(
                content=steps,
                title="Solution",
                color_scheme=ColorScheme.INFO
            ))

        return slides

    def _build_market_making_slides(self, result: dict) -> List[Slide]:
        """Build slides for market making problems."""
        slides = []

        # Scenario
        scenario = result.get("scenario", result.get("problem", ""))
        if scenario:
            slides.extend(self._create_content_slides(
                content=scenario,
                title="Scenario",
                color_scheme=ColorScheme.INFO
            ))

        # Bid-Ask spread
        bid = result.get("bid", "")
        ask = result.get("ask", "")
        if bid and ask:
            slides.append(Slide(
                title="Market",
                content=f"BID: {bid}\nASK: {ask}",
                color_scheme=ColorScheme.RESULT_GREEN,
                centered=True
            ))

        # Reasoning
        reasoning = result.get("reasoning", result.get("explanation", ""))
        if reasoning:
            slides.extend(self._create_content_slides(
                content=reasoning,
                title="Reasoning",
                color_scheme=ColorScheme.INFO
            ))

        return slides

    def _build_generic_quant_slides(self, result: dict) -> List[Slide]:
        """Build slides for generic quant content."""
        slides = []

        # Problem
        problem = result.get("problem", result.get("input", ""))
        if problem:
            slides.extend(self._create_content_slides(
                content=problem,
                title="Problem",
                color_scheme=ColorScheme.MATH
            ))

        # Solution
        solution = result.get("solution", "")
        if solution:
            slides.append(Slide(
                title="Answer",
                content=str(solution),
                color_scheme=ColorScheme.RESULT_GREEN,
                centered=True,
                large_font=len(str(solution)) < 20
            ))

        return slides

    def _format_probability(self, prob: Union[float, str]) -> str:
        """Format probability for display."""
        if isinstance(prob, str):
            return prob

        # Format as percentage and fraction if clean
        percentage = f"{prob * 100:.1f}%"

        # Try to find nice fraction representation
        for denom in [2, 3, 4, 5, 6, 8, 10, 12]:
            numer = round(prob * denom)
            if abs(numer / denom - prob) < 0.001:
                return f"{numer}/{denom}  ({percentage})"

        return percentage

    def _generate_summary(self, engine_result: dict) -> Optional[str]:
        """Generate summary for quant result."""
        solution = engine_result.get("solution", engine_result.get("answer", ""))
        if solution:
            return f"= {solution}"[:self.MAX_LINE_CHARS]
        return None


# Convenience functions for WebSocket handlers

def format_problem_for_ws(
    problem: str,
    difficulty: int,
    category: str = "arithmetic"
) -> dict:
    """Format a problem for WebSocket transmission."""
    return {
        "type": "problem",
        "problem": problem,
        "difficulty": difficulty,
        "category": category,
        "time_target_ms": QuantSlideBuilder.TIME_TARGETS.get(difficulty, 15000),
        "display": {
            "centered": True,
            "large_font": True,
            "color": "math"
        }
    }


def format_result_for_ws(
    correct: bool,
    user_answer: any,
    correct_answer: any,
    time_ms: int,
    streak: int = 0
) -> dict:
    """Format a result for WebSocket transmission."""
    return {
        "type": "result",
        "correct": correct,
        "user_answer": user_answer,
        "correct_answer": correct_answer,
        "time_ms": time_ms,
        "streak": streak,
        "display": {
            "centered": True,
            "large_font": True,
            "color": "result_green" if correct else "result_red"
        }
    }
