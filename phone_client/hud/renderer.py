"""
HUD Renderer for Halo Glasses.
Generates Lua code for the Halo display (640x400 OLED).
"""
from dataclasses import dataclass
from typing import Optional, List
from .colors import RGB, COLORS, rgb_to_lua, get_timer_color, get_streak_color


# Halo display constants
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 400
CENTER_X = DISPLAY_WIDTH // 2
CENTER_Y = DISPLAY_HEIGHT // 2

# Font sizes (approximate character heights in pixels)
FONT_LARGE = 48
FONT_MEDIUM = 32
FONT_SMALL = 24
FONT_TINY = 16


@dataclass
class TextElement:
    """A text element to render."""
    text: str
    x: int
    y: int
    color: RGB = COLORS.text_primary
    size: int = FONT_MEDIUM
    align: str = "center"  # left, center, right


@dataclass
class RectElement:
    """A rectangle element."""
    x: int
    y: int
    width: int
    height: int
    color: RGB = COLORS.primary
    filled: bool = True


@dataclass
class ProgressBar:
    """A progress bar element."""
    x: int
    y: int
    width: int
    height: int
    progress: float  # 0.0 to 1.0
    color: RGB = COLORS.primary
    background: RGB = COLORS.background_overlay


class HUDRenderer:
    """
    Generates Lua code for Halo glasses display.

    The Halo uses a Lua scripting interface for graphics.
    This class builds Lua commands for common HUD elements.
    """

    def __init__(self, width: int = DISPLAY_WIDTH, height: int = DISPLAY_HEIGHT):
        self.width = width
        self.height = height
        self._lua_commands: List[str] = []
        self._pixel_shift_x = 0
        self._pixel_shift_y = 0

    def clear(self):
        """Clear the command buffer and display."""
        self._lua_commands = ["frame.display.clear()"]

    def set_pixel_shift(self, x: int, y: int):
        """Set pixel shift offset for burn-in prevention."""
        self._pixel_shift_x = x
        self._pixel_shift_y = y

    def _shifted(self, x: int, y: int) -> tuple:
        """Apply pixel shift to coordinates."""
        return (x + self._pixel_shift_x, y + self._pixel_shift_y)

    def text(
        self,
        text: str,
        x: int,
        y: int,
        color: RGB = COLORS.text_primary,
        size: int = FONT_MEDIUM,
        align: str = "center"
    ):
        """Add text to the display."""
        sx, sy = self._shifted(x, y)
        color_lua = rgb_to_lua(color)

        self._lua_commands.append(
            f'frame.display.text("{text}", {sx}, {sy}, '
            f'{{color={color_lua}, size={size}, align="{align}"}})'
        )

    def rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        color: RGB = COLORS.primary,
        filled: bool = True
    ):
        """Add a rectangle to the display."""
        sx, sy = self._shifted(x, y)
        color_lua = rgb_to_lua(color)
        fill_str = "true" if filled else "false"

        self._lua_commands.append(
            f'frame.display.rect({sx}, {sy}, {width}, {height}, '
            f'{{color={color_lua}, filled={fill_str}}})'
        )

    def progress_bar(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        progress: float,
        color: RGB = COLORS.primary,
        background: RGB = COLORS.background_overlay
    ):
        """Add a progress bar."""
        progress = max(0.0, min(1.0, progress))
        fill_width = int(width * progress)

        # Background
        self.rect(x, y, width, height, background, filled=True)

        # Fill
        if fill_width > 0:
            self.rect(x, y, fill_width, height, color, filled=True)

        # Border
        self.rect(x, y, width, height, color, filled=False)

    def show(self):
        """Finalize and return the Lua code."""
        self._lua_commands.append("frame.display.show()")

    def get_lua(self) -> str:
        """Get the complete Lua script."""
        return "\n".join(self._lua_commands)

    # ============================================================
    # Mental Math HUD Templates
    # ============================================================

    def render_math_problem(
        self,
        problem_text: str,
        difficulty: int,
        elapsed_ms: float,
        target_ms: float,
        streak: int
    ) -> str:
        """
        Render the mental math problem display.

        Layout:
            Top-left: Difficulty badge
            Top-right: Timer
            Center: Problem
            Bottom-left: Streak counter
        """
        self.clear()

        # Difficulty badge (top-left)
        diff_colors = [
            COLORS.success,      # D1
            COLORS.primary,      # D2
            COLORS.accent,       # D3
            (200, 100, 255),     # D4
            COLORS.error,        # D5
        ]
        diff_color = diff_colors[min(difficulty - 1, 4)]
        self.text(f"D{difficulty}", 40, 30, diff_color, FONT_SMALL, "left")

        # Timer (top-right) - color changes with time pressure
        time_color = get_timer_color(elapsed_ms, target_ms)
        time_str = f"{elapsed_ms / 1000:.1f}s"
        self.text(time_str, self.width - 40, 30, time_color, FONT_SMALL, "right")

        # Timer progress bar
        progress = min(1.0, elapsed_ms / target_ms)
        self.progress_bar(
            self.width - 150, 50,
            110, 8,
            progress,
            time_color
        )

        # Main problem (center)
        self.text(
            problem_text,
            CENTER_X, CENTER_Y,
            COLORS.text_primary,
            FONT_LARGE,
            "center"
        )

        # Streak counter (bottom-left)
        if streak > 0:
            streak_color = get_streak_color(streak)
            self.text(f"Streak: {streak}", 40, self.height - 40, streak_color, FONT_SMALL, "left")

        self.show()
        return self.get_lua()

    def render_result(
        self,
        correct: bool,
        time_ms: float,
        feedback: str,
        streak: int,
        answer: Optional[float] = None
    ) -> str:
        """
        Render the result screen after an answer.

        Shows:
            - Correct/Wrong indicator
            - Time taken
            - WHAM feedback
            - Correct answer (if wrong)
        """
        self.clear()

        # Result indicator (large, centered)
        if correct:
            result_text = "CORRECT"
            result_color = COLORS.success
        else:
            result_text = "WRONG"
            result_color = COLORS.error

        self.text(result_text, CENTER_X, CENTER_Y - 60, result_color, FONT_LARGE, "center")

        # Time taken
        time_str = f"{time_ms / 1000:.2f}s"
        self.text(time_str, CENTER_X, CENTER_Y, COLORS.text_secondary, FONT_MEDIUM, "center")

        # Show correct answer if wrong
        if not correct and answer is not None:
            ans_text = f"Answer: {answer:g}"
            self.text(ans_text, CENTER_X, CENTER_Y + 40, COLORS.warning, FONT_SMALL, "center")

        # WHAM feedback (bottom)
        # Truncate if too long
        if len(feedback) > 45:
            feedback = feedback[:42] + "..."
        self.text(feedback, CENTER_X, self.height - 60, COLORS.primary, FONT_SMALL, "center")

        # Streak (if any)
        if streak > 0:
            streak_color = get_streak_color(streak)
            self.text(f"Streak: {streak}", CENTER_X, self.height - 30, streak_color, FONT_TINY, "center")

        self.show()
        return self.get_lua()

    def render_session_summary(
        self,
        total: int,
        correct: int,
        accuracy: float,
        avg_time_ms: float,
        best_streak: int,
        grade: str
    ) -> str:
        """
        Render end-of-session summary.
        """
        self.clear()

        # Title
        self.text("SESSION COMPLETE", CENTER_X, 40, COLORS.primary, FONT_MEDIUM, "center")

        # Grade (large)
        grade_colors = {
            'S': COLORS.streak_legendary,
            'A': COLORS.streak_gold,
            'B': COLORS.success,
            'C': COLORS.primary,
            'D': COLORS.warning,
            'F': COLORS.error,
        }
        grade_color = grade_colors.get(grade[0], COLORS.text_primary)
        self.text(grade, CENTER_X, 100, grade_color, FONT_LARGE, "center")

        # Stats
        y_start = 160
        line_height = 35

        stats = [
            f"Problems: {correct}/{total}",
            f"Accuracy: {accuracy:.1f}%",
            f"Avg Time: {avg_time_ms / 1000:.2f}s",
            f"Best Streak: {best_streak}",
        ]

        for i, stat in enumerate(stats):
            self.text(stat, CENTER_X, y_start + i * line_height, COLORS.text_secondary, FONT_SMALL, "center")

        self.show()
        return self.get_lua()

    def render_idle(self, greeting: str, time_str: str) -> str:
        """
        Render idle/standby screen.
        """
        self.clear()

        # WHAM greeting
        self.text(greeting, CENTER_X, CENTER_Y - 20, COLORS.primary, FONT_MEDIUM, "center")

        # Time
        self.text(time_str, CENTER_X, CENTER_Y + 30, COLORS.text_dim, FONT_SMALL, "center")

        # Subtle "Ready" indicator
        self.text("Ready", CENTER_X, self.height - 40, COLORS.primary_dim, FONT_TINY, "center")

        self.show()
        return self.get_lua()

    def render_connecting(self) -> str:
        """Render connecting screen."""
        self.clear()
        self.text("WHAM", CENTER_X, CENTER_Y - 30, COLORS.primary, FONT_LARGE, "center")
        self.text("Connecting...", CENTER_X, CENTER_Y + 30, COLORS.text_dim, FONT_SMALL, "center")
        self.show()
        return self.get_lua()

    def render_error(self, message: str) -> str:
        """Render error screen."""
        self.clear()
        self.text("ERROR", CENTER_X, CENTER_Y - 30, COLORS.error, FONT_MEDIUM, "center")
        self.text(message[:40], CENTER_X, CENTER_Y + 20, COLORS.text_secondary, FONT_SMALL, "center")
        self.show()
        return self.get_lua()


# Test
if __name__ == "__main__":
    renderer = HUDRenderer()

    print("=== Mental Math Problem ===")
    lua = renderer.render_math_problem("47 x 83", 2, 1500, 4000, 5)
    print(lua)
    print()

    print("=== Correct Result ===")
    lua = renderer.render_result(True, 2340, "Well executed, sir. 2.34s.", 6)
    print(lua)
    print()

    print("=== Wrong Result ===")
    lua = renderer.render_result(False, 5200, "The correct answer was 3901.", 0, 3901)
    print(lua)
