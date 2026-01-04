"""
Display Controller for Halo Frame

Manages rendering to the 640x400 OLED display with:
- Paginated slide navigation
- Color scheme application
- Font size management
- Progress indicators
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Awaitable
from enum import Enum


# Halo Frame display colors (RGB tuples)
HALO_COLORS = {
    # Math & Physics
    "math": (76, 201, 240),           # Blue
    "result_green": (0, 255, 136),    # Green
    "result_red": (255, 68, 68),      # Red

    # Code
    "code": (212, 212, 212),          # Light gray
    "code_error": (255, 68, 68),      # Red
    "code_fix": (0, 255, 136),        # Green

    # Quant
    "quant_timer": (255, 140, 0),     # Orange
    "quant_streak": (255, 215, 0),    # Gold

    # Meeting Mode (WHAM)
    "meeting": (78, 205, 196),        # Teal/cyan - calm, professional
    "suggestion": (149, 165, 166),    # Silver - subtle suggestion
    "recording": (255, 107, 107),     # Coral - recording indicator
    "processing": (255, 230, 109),    # Yellow - thinking
    "negotiation": (255, 177, 66),    # Orange - negotiation alert
    "alert": (255, 107, 129),         # Pink-red - important alert

    # General
    "info": (200, 200, 200),          # Gray
    "warning": (255, 217, 61),        # Yellow
    "success": (0, 255, 136),         # Green

    # Default
    "default": (255, 255, 255),       # White
}


class InputAction(Enum):
    """Actions from user input."""
    NEXT = "next"
    BACK = "back"
    EXIT = "exit"
    SELECT = "select"
    NONE = "none"


@dataclass
class DisplaySlide:
    """
    A single slide to render on the Halo Frame display.

    Constraints:
    - Max 4 lines (1 title + 3 content)
    - Max 45 chars per line
    - 640x400 pixel display
    """
    lines: List[str]
    color: tuple = (255, 255, 255)
    slide_index: int = 1
    total_slides: int = 1
    centered: bool = False
    large_font: bool = False

    def __post_init__(self):
        # Ensure lines don't exceed limits
        self.lines = [line[:45] for line in self.lines[:4]]


class DisplayController:
    """
    Controls the Halo Frame OLED display.

    Display specs:
    - Resolution: 640x400 pixels
    - FOV: 20 degrees
    - Max chars per line: 45
    - Max lines: 4
    """

    # Display constraints
    WIDTH = 640
    HEIGHT = 400
    MAX_CHARS = 45
    MAX_LINES = 4

    # Layout constants
    MARGIN_LEFT = 20
    MARGIN_TOP = 20
    LINE_HEIGHT = 80
    NAV_Y = 360
    NAV_X = 580

    def __init__(self, frame=None):
        """
        Initialize display controller.

        Args:
            frame: Frame SDK instance. If None, uses mock for testing.
        """
        self.frame = frame
        self._mock_mode = frame is None
        self._current_slide_index = 0
        self._input_callback: Optional[Callable[[], Awaitable[InputAction]]] = None

    def set_input_callback(self, callback: Callable[[], Awaitable[InputAction]]):
        """Set callback for receiving user input."""
        self._input_callback = callback

    async def clear(self):
        """Clear the display."""
        if self._mock_mode:
            print("[DISPLAY] Cleared")
            return

        await self.frame.display.clear()

    async def render_text(
        self,
        text: str,
        x: int,
        y: int,
        color: tuple = (255, 255, 255),
        large: bool = False
    ):
        """
        Render text at position.

        Args:
            text: Text to render (will be truncated to MAX_CHARS)
            x: X position
            y: Y position
            color: RGB tuple
            large: Use large font
        """
        text = text[:self.MAX_CHARS]

        if self._mock_mode:
            print(f"[DISPLAY] ({x},{y}) {text} color={color} large={large}")
            return

        # Frame SDK text rendering
        await self.frame.display.text(
            text, x, y,
            color=color,
            font_size=48 if large else 32
        )

    async def render_slide(self, slide: DisplaySlide):
        """
        Render a single slide to the display.

        Args:
            slide: The DisplaySlide to render
        """
        await self.clear()

        y = self.MARGIN_TOP

        for i, line in enumerate(slide.lines[:self.MAX_LINES]):
            x = self.MARGIN_LEFT

            # Center if requested
            if slide.centered:
                char_width = 14 if slide.large_font else 10
                text_width = len(line) * char_width
                x = (self.WIDTH - text_width) // 2

            await self.render_text(
                line,
                x, y,
                color=slide.color,
                large=slide.large_font and i == 0  # Large font for first line only
            )

            y += self.LINE_HEIGHT

        # Navigation indicator (slide_index/total_slides)
        if slide.total_slides > 1:
            nav_text = f"{slide.slide_index}/{slide.total_slides}"
            await self.render_text(
                nav_text,
                self.NAV_X, self.NAV_Y,
                color=(128, 128, 128)
            )

        if not self._mock_mode:
            await self.frame.display.show()

    async def render_paginated(
        self,
        slides: List[DisplaySlide],
        auto_advance_ms: int = 0
    ) -> int:
        """
        Render slides with tap-to-advance navigation.

        Args:
            slides: List of slides to display
            auto_advance_ms: Auto-advance after N ms (0 = manual only)

        Returns:
            Index of last slide viewed
        """
        if not slides:
            return 0

        idx = 0

        while idx < len(slides):
            await self.render_slide(slides[idx])

            # Wait for input
            action = await self._wait_for_input(timeout_ms=auto_advance_ms)

            if action == InputAction.NEXT:
                if idx < len(slides) - 1:
                    idx += 1
                else:
                    break  # At last slide, exit
            elif action == InputAction.BACK:
                if idx > 0:
                    idx -= 1
            elif action == InputAction.EXIT:
                break
            elif action == InputAction.NONE and auto_advance_ms > 0:
                # Auto-advance on timeout
                if idx < len(slides) - 1:
                    idx += 1
                else:
                    break

        self._current_slide_index = idx
        return idx

    async def _wait_for_input(self, timeout_ms: int = 0) -> InputAction:
        """
        Wait for user input.

        Args:
            timeout_ms: Timeout in milliseconds (0 = wait forever)

        Returns:
            The input action received
        """
        if self._input_callback:
            try:
                if timeout_ms > 0:
                    return await asyncio.wait_for(
                        self._input_callback(),
                        timeout=timeout_ms / 1000
                    )
                else:
                    return await self._input_callback()
            except asyncio.TimeoutError:
                return InputAction.NONE

        # Mock mode: simulate tap after brief delay
        if self._mock_mode:
            await asyncio.sleep(0.5)
            return InputAction.NEXT

        return InputAction.NONE

    # Convenience methods for common display patterns

    async def show_loading(self, message: str = "Loading..."):
        """Show a loading indicator."""
        slide = DisplaySlide(
            lines=[message],
            color=HALO_COLORS["info"],
            centered=True
        )
        await self.render_slide(slide)

    async def show_error(self, message: str):
        """Show an error message."""
        slide = DisplaySlide(
            lines=["Error", message],
            color=HALO_COLORS["result_red"],
            centered=True
        )
        await self.render_slide(slide)

    async def show_success(self, message: str):
        """Show a success message."""
        slide = DisplaySlide(
            lines=["Success", message],
            color=HALO_COLORS["result_green"],
            centered=True
        )
        await self.render_slide(slide)

    async def show_problem(
        self,
        problem: str,
        difficulty: int = 2,
        category: str = "MATH"
    ):
        """
        Show a problem for mental math mode.

        Displays large, centered problem text.
        """
        difficulty_label = f"D{difficulty}"
        slide = DisplaySlide(
            lines=[f"[{category}]  {difficulty_label}", problem],
            color=HALO_COLORS["math"],
            centered=True,
            large_font=True
        )
        await self.render_slide(slide)

    async def show_result(
        self,
        correct: bool,
        answer: str,
        time_ms: int,
        streak: int = 0
    ):
        """
        Show a result for mental math mode.

        Shows answer with green (correct) or red (incorrect) color.
        """
        color = HALO_COLORS["result_green"] if correct else HALO_COLORS["result_red"]
        title = "Correct!" if correct else "Incorrect"

        lines = [title, str(answer), f"{time_ms / 1000:.1f}s"]
        if streak > 2:
            lines.append(f"Streak: {streak}")

        slide = DisplaySlide(
            lines=lines,
            color=color,
            centered=True,
            large_font=True
        )
        await self.render_slide(slide)

    async def show_timer(
        self,
        problem: str,
        remaining_ms: int,
        target_ms: int
    ):
        """
        Show timer update for mental math mode.

        Changes color as time runs out.
        """
        progress = 1.0 - (remaining_ms / target_ms)

        if progress < 0.5:
            color = HALO_COLORS["math"]
        elif progress < 0.8:
            color = HALO_COLORS["quant_timer"]
        else:
            color = HALO_COLORS["result_red"]

        seconds = remaining_ms / 1000
        slide = DisplaySlide(
            lines=[problem, f"{seconds:.1f}s"],
            color=color,
            centered=True,
            large_font=True
        )
        await self.render_slide(slide)

    async def show_viewfinder(self, message: str = "Tap to capture"):
        """Show camera viewfinder overlay."""
        slide = DisplaySlide(
            lines=["[CAMERA]", "", message],
            color=HALO_COLORS["info"],
            centered=True
        )
        await self.render_slide(slide)

    # Meeting Mode Display Methods

    async def show_meeting_status(
        self,
        status: str,
        message: str = "",
        meeting_type: str = ""
    ):
        """
        Show meeting mode status.

        Args:
            status: Current status (listening, recording, processing)
            message: Optional message
            meeting_type: Type of meeting (general, negotiation, etc.)
        """
        lines = [f"[WHAM - {status.upper()}]"]

        if meeting_type:
            lines.append(f"Mode: {meeting_type}")

        if message:
            lines.append("")
            lines.append(message)

        color_map = {
            "listening": HALO_COLORS["meeting"],
            "recording": HALO_COLORS["recording"],
            "processing": HALO_COLORS["processing"],
        }
        color = color_map.get(status.lower(), HALO_COLORS["meeting"])

        slide = DisplaySlide(
            lines=lines,
            color=color,
            centered=True
        )
        await self.render_slide(slide)

    async def show_meeting_suggestion(
        self,
        suggestion: str,
        trigger: str = "double_tap",
        suggestion_type: str = "quick_response",
        alternatives: List[str] = None,
        tactical_notes: str = None
    ):
        """
        Show WHAM's meeting suggestion.

        Args:
            suggestion: The main suggestion text
            trigger: How triggered (double_tap or voice_command)
            suggestion_type: Type of suggestion
            alternatives: Optional alternative suggestions
            tactical_notes: Optional tactical notes
        """
        # Build lines
        lines = []

        # Header
        if trigger == "double_tap":
            lines.append("[QUICK HELP]")
        else:
            lines.append("[WHAM SUGGESTS]")

        # Wrap suggestion to fit display
        words = suggestion.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= self.MAX_CHARS:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # Add tactical note if present and room available
        if tactical_notes and len(lines) < 4:
            lines.append(f"TIP: {tactical_notes[:35]}...")

        # Determine color based on type
        type_colors = {
            "negotiation": HALO_COLORS["negotiation"],
            "fact_check": HALO_COLORS["alert"],
            "tactical_advice": HALO_COLORS["warning"],
        }
        color = type_colors.get(suggestion_type, HALO_COLORS["suggestion"])

        slide = DisplaySlide(
            lines=lines[:4],  # Max 4 lines
            color=color,
            centered=False  # Left-align for readability
        )
        await self.render_slide(slide)

    async def show_meeting_transcript(
        self,
        speaker: str,
        text: str,
        is_user: bool = False
    ):
        """
        Show a transcript update (subtle overlay).

        Args:
            speaker: Speaker identifier
            text: Transcript text
            is_user: True if this is the user speaking
        """
        prefix = "You" if is_user else speaker
        truncated = text[:35] + "..." if len(text) > 35 else text

        slide = DisplaySlide(
            lines=[f"[{prefix}]:", truncated],
            color=HALO_COLORS["info"],
            centered=False
        )
        await self.render_slide(slide)

    async def show_meeting_summary(
        self,
        suggestions_given: int,
        duration_seconds: float,
        meeting_type: str = "general"
    ):
        """
        Show meeting summary at end of session.

        Args:
            suggestions_given: Number of suggestions provided
            duration_seconds: Meeting duration in seconds
            meeting_type: Type of meeting
        """
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)

        lines = [
            "[MEETING ENDED]",
            f"Duration: {minutes}m {seconds}s",
            f"Suggestions: {suggestions_given}",
            "Good work, Will."
        ]

        slide = DisplaySlide(
            lines=lines,
            color=HALO_COLORS["result_green"],
            centered=True
        )
        await self.render_slide(slide)
