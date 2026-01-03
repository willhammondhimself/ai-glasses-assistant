"""
Base Slide Builder

Abstract base class for all domain-specific slide builders.
Handles common text wrapping and slide creation logic.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import textwrap

from backend.models.ar_response import Slide, PaginatedResponse, ColorScheme


class BaseSlideBuilder(ABC):
    """
    Base class for building slides from engine results.

    Display constraints (640x400 OLED @ 20 deg FOV):
    - Max 45 characters per line
    - Max 4 lines per slide (1 title + 3 content)
    - Title max 30 chars
    """

    MAX_LINE_CHARS = 45
    MAX_CONTENT_LINES = 3  # Title takes 1 line
    MAX_TITLE_CHARS = 30

    def __init__(self):
        self.response_type: str = "generic"

    @abstractmethod
    def build_slides(self, engine_result: dict) -> List[Slide]:
        """
        Build slides from an engine result.

        Args:
            engine_result: Dictionary from engine.solve() or similar

        Returns:
            List of Slide objects
        """
        pass

    def build_response(
        self,
        engine_result: dict,
        method: str = "",
        cached: bool = False,
        latency_ms: Optional[float] = None
    ) -> PaginatedResponse:
        """
        Build complete paginated response from engine result.

        Args:
            engine_result: Dictionary from engine
            method: Method used (sympy, claude, etc)
            cached: Whether result was from cache
            latency_ms: Processing latency

        Returns:
            PaginatedResponse with slides
        """
        if engine_result.get("error"):
            return PaginatedResponse.from_error(
                engine_result["error"],
                response_type=self.response_type
            )

        slides = self.build_slides(engine_result)

        return PaginatedResponse(
            type=self.response_type,
            total_slides=len(slides),
            slides=slides,
            summary=self._generate_summary(engine_result),
            raw_solution=engine_result.get("solution"),
            method=method,
            cached=cached,
            latency_ms=latency_ms
        )

    def _wrap_text(self, text: str, max_chars: int = None) -> List[str]:
        """
        Word-wrap text into lines that fit the display.

        Args:
            text: Text to wrap
            max_chars: Max characters per line (default: MAX_LINE_CHARS)

        Returns:
            List of wrapped lines
        """
        max_chars = max_chars or self.MAX_LINE_CHARS

        # Handle multiline input
        lines = []
        for paragraph in text.split('\n'):
            wrapped = textwrap.wrap(
                paragraph,
                width=max_chars,
                break_long_words=True,
                break_on_hyphens=True
            )
            lines.extend(wrapped if wrapped else [''])

        return lines

    def _truncate_title(self, title: str) -> str:
        """Truncate title to fit display constraints."""
        if len(title) <= self.MAX_TITLE_CHARS:
            return title
        return title[:self.MAX_TITLE_CHARS - 3] + "..."

    def _create_content_slides(
        self,
        content: str,
        title: str,
        color_scheme: ColorScheme = ColorScheme.INFO,
        voice_prefix: str = ""
    ) -> List[Slide]:
        """
        Create multiple slides from content that's too long for one slide.

        Args:
            content: Full content text
            title: Slide title
            color_scheme: Color scheme to use
            voice_prefix: Prefix for voice narration

        Returns:
            List of slides (may be multiple if content is long)
        """
        lines = self._wrap_text(content)
        slides = []
        slide_num = 1

        for i in range(0, len(lines), self.MAX_CONTENT_LINES):
            chunk_lines = lines[i:i + self.MAX_CONTENT_LINES]
            chunk_content = '\n'.join(chunk_lines)

            # Add continuation indicator to title if multiple slides
            slide_title = title
            if len(lines) > self.MAX_CONTENT_LINES:
                slide_title = f"{self._truncate_title(title)} ({slide_num})"
                slide_num += 1

            slides.append(Slide(
                title=slide_title,
                content=chunk_content,
                color_scheme=color_scheme,
                voice_narration=f"{voice_prefix}{chunk_content}" if voice_prefix else None
            ))

        return slides if slides else [Slide(
            title=title,
            content="",
            color_scheme=color_scheme
        )]

    def _generate_summary(self, engine_result: dict) -> Optional[str]:
        """
        Generate one-line summary for quick display mode.

        Override in subclasses for domain-specific summaries.
        """
        solution = engine_result.get("solution", "")
        if solution:
            # First line, truncated
            first_line = solution.split('\n')[0]
            return first_line[:self.MAX_LINE_CHARS] if len(first_line) > self.MAX_LINE_CHARS else first_line
        return None
