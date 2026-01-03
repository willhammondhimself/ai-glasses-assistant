"""
Response Transformer

Transforms backend PaginatedResponse into DisplaySlide objects
optimized for the Halo Frame display.
"""

import textwrap
from typing import List, Dict, Any, Optional

from glasses_client.core.display import DisplaySlide, HALO_COLORS
from .colors import get_rgb_color


class ResponseTransformer:
    """
    Transforms backend API responses into display-ready slides.

    Handles:
    - Text wrapping for 45-char line limit
    - Color scheme mapping to RGB
    - Slide pagination
    - Large font detection
    """

    MAX_CHARS_PER_LINE = 45
    MAX_CONTENT_LINES = 3  # Title takes 1 of 4 lines

    def __init__(self):
        pass

    def transform(self, response: Dict[str, Any]) -> List[DisplaySlide]:
        """
        Transform a PaginatedResponse dict into DisplaySlides.

        Args:
            response: Backend PaginatedResponse as dictionary

        Returns:
            List of DisplaySlide objects ready for rendering
        """
        slides = []
        backend_slides = response.get("slides", [])
        total_slides = response.get("total_slides", len(backend_slides))

        for i, slide_data in enumerate(backend_slides):
            display_slide = self._transform_slide(
                slide_data,
                slide_index=i + 1,
                total_slides=total_slides
            )
            slides.append(display_slide)

        return slides

    def _transform_slide(
        self,
        slide_data: Dict[str, Any],
        slide_index: int,
        total_slides: int
    ) -> DisplaySlide:
        """Transform a single slide."""
        title = slide_data.get("title", "")
        content = slide_data.get("content", "")
        color_scheme = slide_data.get("color_scheme", "info")
        centered = slide_data.get("centered", False)
        large_font = slide_data.get("large_font", False)

        # Get RGB color
        color = get_rgb_color(color_scheme)

        # Build lines: title + wrapped content
        lines = []

        if title:
            lines.append(self._truncate_line(title, self.MAX_CHARS_PER_LINE))

        # Wrap content
        content_lines = self._wrap_content(content)
        lines.extend(content_lines[:self.MAX_CONTENT_LINES])

        return DisplaySlide(
            lines=lines,
            color=color,
            slide_index=slide_index,
            total_slides=total_slides,
            centered=centered,
            large_font=large_font
        )

    def _wrap_content(self, content: str) -> List[str]:
        """
        Wrap content text into lines.

        Args:
            content: Text content to wrap

        Returns:
            List of wrapped lines
        """
        if not content:
            return []

        # Handle existing newlines
        lines = []
        for paragraph in content.split('\n'):
            if not paragraph.strip():
                continue

            wrapped = textwrap.wrap(
                paragraph,
                width=self.MAX_CHARS_PER_LINE,
                break_long_words=True,
                break_on_hyphens=True
            )
            lines.extend(wrapped)

        return lines

    def _truncate_line(self, text: str, max_chars: int) -> str:
        """Truncate a line to max characters."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."

    # Convenience methods for specific response types

    def transform_math(self, response: Dict[str, Any]) -> List[DisplaySlide]:
        """
        Transform a math solve response.

        Ensures proper color coding for problem/steps/solution.
        """
        return self.transform(response)

    def transform_mental_math_problem(
        self,
        problem: str,
        difficulty: int = 2,
        category: str = "arithmetic"
    ) -> DisplaySlide:
        """
        Transform a mental math problem for display.

        Optimizes for large, centered display.
        """
        difficulty_label = f"D{difficulty}"
        title = f"[{category.upper()}]  {difficulty_label}"

        return DisplaySlide(
            lines=[title, problem],
            color=HALO_COLORS["math"],
            slide_index=1,
            total_slides=1,
            centered=True,
            large_font=True
        )

    def transform_mental_math_result(
        self,
        correct: bool,
        answer: Any,
        time_ms: int,
        streak: int = 0
    ) -> DisplaySlide:
        """
        Transform a mental math result for display.

        Shows green for correct, red for incorrect.
        """
        color_key = "result_green" if correct else "result_red"
        title = "Correct!" if correct else "Incorrect"

        lines = [title, str(answer), f"{time_ms / 1000:.1f}s"]

        if streak > 2:
            streak_messages = ["", "", "", "Nice!", "On fire!",
                              "On fire!", "On fire!", "On fire!",
                              "Unstoppable!", "Unstoppable!", "Unstoppable!",
                              "LEGENDARY!"]
            msg = streak_messages[min(streak, len(streak_messages) - 1)]
            if msg:
                lines.append(msg)

        return DisplaySlide(
            lines=lines,
            color=HALO_COLORS[color_key],
            slide_index=1,
            total_slides=1,
            centered=True,
            large_font=True
        )

    def transform_timer_update(
        self,
        problem: str,
        remaining_ms: int,
        target_ms: int
    ) -> DisplaySlide:
        """
        Transform a timer update for display.

        Color changes as time runs out.
        """
        progress = 1.0 - (remaining_ms / max(target_ms, 1))

        if progress < 0.5:
            color = HALO_COLORS["math"]
        elif progress < 0.8:
            color = HALO_COLORS["quant_timer"]
        else:
            color = HALO_COLORS["result_red"]

        seconds = remaining_ms / 1000

        return DisplaySlide(
            lines=[problem, f"{seconds:.1f}s"],
            color=color,
            slide_index=1,
            total_slides=1,
            centered=True,
            large_font=True
        )

    def transform_error(self, error_message: str) -> DisplaySlide:
        """Transform an error for display."""
        wrapped = self._wrap_content(error_message)

        return DisplaySlide(
            lines=["Error"] + wrapped[:3],
            color=HALO_COLORS["result_red"],
            slide_index=1,
            total_slides=1,
            centered=True,
            large_font=False
        )

    def transform_loading(self, message: str = "Loading...") -> DisplaySlide:
        """Transform a loading state for display."""
        return DisplaySlide(
            lines=[message],
            color=HALO_COLORS["info"],
            slide_index=1,
            total_slides=1,
            centered=True,
            large_font=False
        )

    def transform_code_error(
        self,
        code_line: str,
        error_message: str,
        line_number: Optional[int] = None
    ) -> DisplaySlide:
        """Transform a code error for display."""
        title = f"Error @ Line {line_number}" if line_number else "Error Found"

        lines = [title]
        lines.append(self._truncate_line(code_line, self.MAX_CHARS_PER_LINE))

        error_lines = self._wrap_content(error_message)
        lines.extend(error_lines[:2])

        return DisplaySlide(
            lines=lines,
            color=HALO_COLORS["code_error"],
            slide_index=1,
            total_slides=1,
            centered=False,
            large_font=False
        )

    def transform_code_fix(
        self,
        old_code: str,
        new_code: str,
        explanation: Optional[str] = None
    ) -> List[DisplaySlide]:
        """Transform a code fix for display."""
        slides = []

        # Slide 1: The fix
        slides.append(DisplaySlide(
            lines=[
                "Fix",
                f"- {self._truncate_line(old_code, 40)}",
                f"+ {self._truncate_line(new_code, 40)}"
            ],
            color=HALO_COLORS["code_fix"],
            slide_index=1,
            total_slides=2 if explanation else 1,
            centered=False,
            large_font=False
        ))

        # Slide 2: Explanation (if provided)
        if explanation:
            exp_lines = self._wrap_content(explanation)
            slides.append(DisplaySlide(
                lines=["Why"] + exp_lines[:3],
                color=HALO_COLORS["info"],
                slide_index=2,
                total_slides=2,
                centered=False,
                large_font=False
            ))

        return slides
