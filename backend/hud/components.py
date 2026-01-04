"""
Iron Man HUD Components

Reusable display components for AR glasses HUD.
Each component generates render data for the glasses display.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from .colors import (
    RGB, RGBA,
    WHAM_COLORS,
    get_timer_color,
    get_streak_color,
    get_accuracy_color,
    get_speed_tier_color,
    get_difficulty_color,
    rgb_to_hex,
)


class Alignment(Enum):
    """Text/component alignment."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class AnimationType(Enum):
    """Animation types for HUD elements."""
    NONE = "none"
    FADE_IN = "fade_in"
    SLIDE_IN = "slide_in"
    PULSE = "pulse"
    GLOW = "glow"
    FLASH = "flash"
    SCALE_UP = "scale_up"


@dataclass
class Position:
    """Screen position for HUD element."""
    x: int  # Pixels from left
    y: int  # Pixels from top


@dataclass
class Size:
    """Size of HUD element."""
    width: int
    height: int


# Display specs for Brilliant Labs Halo
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 400


class HUDComponent(ABC):
    """Base class for all HUD components."""

    def __init__(
        self,
        position: Position,
        size: Optional[Size] = None,
        visible: bool = True,
        animation: AnimationType = AnimationType.NONE,
    ):
        self.position = position
        self.size = size
        self.visible = visible
        self.animation = animation

    @abstractmethod
    def render(self) -> Dict[str, Any]:
        """Generate render data for this component."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        if not self.visible:
            return {"visible": False}

        data = self.render()
        data["position"] = {"x": self.position.x, "y": self.position.y}
        if self.size:
            data["size"] = {"width": self.size.width, "height": self.size.height}
        if self.animation != AnimationType.NONE:
            data["animation"] = self.animation.value
        data["visible"] = True
        return data


class ProgressBar(HUDComponent):
    """
    Animated progress bar with Iron Man styling.

    Used for:
    - Timer countdown
    - Session progress
    - Accuracy meter
    """

    def __init__(
        self,
        position: Position,
        width: int = 200,
        height: int = 8,
        progress: float = 0.0,  # 0-1
        color: Optional[RGB] = None,
        bg_color: Optional[RGBA] = None,
        show_glow: bool = True,
        show_segments: bool = False,
        segment_count: int = 10,
    ):
        super().__init__(position, Size(width, height))
        self.progress = max(0, min(1, progress))
        self.color = color or WHAM_COLORS.primary
        self.bg_color = bg_color or WHAM_COLORS.panel_bg
        self.show_glow = show_glow
        self.show_segments = show_segments
        self.segment_count = segment_count

    def render(self) -> Dict[str, Any]:
        return {
            "type": "progress_bar",
            "progress": self.progress,
            "color": rgb_to_hex(self.color),
            "bg_color": rgb_to_hex((self.bg_color[0], self.bg_color[1], self.bg_color[2])),
            "glow": self.show_glow,
            "segments": self.show_segments,
            "segment_count": self.segment_count if self.show_segments else None,
        }


class TimerDisplay(HUDComponent):
    """
    Timer with color-coded urgency transitions.

    Features:
    - Smooth color transitions (cyan -> amber -> red)
    - Large readable digits
    - Optional arc/ring visual
    """

    def __init__(
        self,
        position: Position,
        elapsed_ms: int = 0,
        target_ms: int = 5000,
        show_arc: bool = True,
        show_milliseconds: bool = False,
        size_preset: str = "medium",  # small, medium, large
    ):
        # Size presets
        sizes = {
            "small": Size(60, 30),
            "medium": Size(100, 50),
            "large": Size(150, 75),
        }
        super().__init__(position, sizes.get(size_preset, sizes["medium"]))

        self.elapsed_ms = elapsed_ms
        self.target_ms = target_ms
        self.show_arc = show_arc
        self.show_milliseconds = show_milliseconds
        self.size_preset = size_preset

    @property
    def remaining_ms(self) -> int:
        return max(0, self.target_ms - self.elapsed_ms)

    @property
    def progress(self) -> float:
        if self.target_ms <= 0:
            return 0
        return min(1, self.elapsed_ms / self.target_ms)

    def render(self) -> Dict[str, Any]:
        color = get_timer_color(self.elapsed_ms, self.target_ms)
        remaining_sec = self.remaining_ms / 1000

        if self.show_milliseconds:
            time_text = f"{remaining_sec:.1f}s"
        else:
            time_text = f"{int(remaining_sec)}s" if remaining_sec >= 1 else f"{int(self.remaining_ms)}ms"

        return {
            "type": "timer",
            "time_text": time_text,
            "elapsed_ms": self.elapsed_ms,
            "remaining_ms": self.remaining_ms,
            "target_ms": self.target_ms,
            "progress": self.progress,
            "color": rgb_to_hex(color),
            "show_arc": self.show_arc,
            "urgency": self._get_urgency_level(),
        }

    def _get_urgency_level(self) -> str:
        """Get urgency level for UI styling."""
        ratio = self.elapsed_ms / self.target_ms if self.target_ms > 0 else 0
        if ratio <= 0.5:
            return "safe"
        elif ratio <= 0.75:
            return "caution"
        elif ratio <= 0.9:
            return "warning"
        else:
            return "danger"


class StreakCounter(HUDComponent):
    """
    Streak display with tier-based styling.

    Features:
    - Tier colors (bronze, silver, gold, etc.)
    - Celebration animations at milestones
    - Flame/fire effect for high streaks
    """

    MILESTONE_STREAKS = [3, 5, 7, 10, 15, 20, 25, 50]

    def __init__(
        self,
        position: Position,
        streak: int = 0,
        show_tier_name: bool = True,
        show_fire_effect: bool = True,
    ):
        super().__init__(position, Size(80, 40))
        self.streak = streak
        self.show_tier_name = show_tier_name
        self.show_fire_effect = show_fire_effect

    @property
    def is_milestone(self) -> bool:
        return self.streak in self.MILESTONE_STREAKS

    def render(self) -> Dict[str, Any]:
        color, tier = get_streak_color(self.streak)

        # Determine animation
        animation = AnimationType.NONE
        if self.is_milestone:
            animation = AnimationType.SCALE_UP
        elif self.streak >= 10 and self.show_fire_effect:
            animation = AnimationType.GLOW

        return {
            "type": "streak_counter",
            "streak": self.streak,
            "tier": tier,
            "color": rgb_to_hex(color),
            "tier_name": tier if self.show_tier_name else None,
            "is_milestone": self.is_milestone,
            "fire_effect": self.show_fire_effect and self.streak >= 10,
            "animation": animation.value,
        }


class AccuracyMeter(HUDComponent):
    """
    Accuracy percentage display with visual indicator.

    Shows current session accuracy with color-coded feedback.
    """

    def __init__(
        self,
        position: Position,
        correct: int = 0,
        total: int = 0,
        show_fraction: bool = True,
        show_bar: bool = True,
    ):
        super().__init__(position, Size(100, 30))
        self.correct = correct
        self.total = total
        self.show_fraction = show_fraction
        self.show_bar = show_bar

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0
        return (self.correct / self.total) * 100

    def render(self) -> Dict[str, Any]:
        color = get_accuracy_color(self.accuracy)

        return {
            "type": "accuracy_meter",
            "accuracy": round(self.accuracy, 1),
            "correct": self.correct,
            "total": self.total,
            "color": rgb_to_hex(color),
            "fraction_text": f"{self.correct}/{self.total}" if self.show_fraction else None,
            "percentage_text": f"{self.accuracy:.0f}%",
            "show_bar": self.show_bar,
            "bar_progress": self.accuracy / 100 if self.show_bar else None,
        }


class DifficultyIndicator(HUDComponent):
    """
    Visual difficulty level indicator.

    Shows current difficulty with colored dots/bars.
    """

    def __init__(
        self,
        position: Position,
        difficulty: int = 2,
        max_difficulty: int = 5,
        style: str = "dots",  # dots, bars, text
    ):
        super().__init__(position, Size(60, 20))
        self.difficulty = max(1, min(max_difficulty, difficulty))
        self.max_difficulty = max_difficulty
        self.style = style

    def render(self) -> Dict[str, Any]:
        color = get_difficulty_color(self.difficulty)

        # Generate difficulty level indicators
        indicators = []
        for i in range(1, self.max_difficulty + 1):
            indicators.append({
                "level": i,
                "active": i <= self.difficulty,
                "color": rgb_to_hex(get_difficulty_color(i)) if i <= self.difficulty else None,
            })

        return {
            "type": "difficulty_indicator",
            "difficulty": self.difficulty,
            "max_difficulty": self.max_difficulty,
            "color": rgb_to_hex(color),
            "style": self.style,
            "indicators": indicators,
            "label": f"D{self.difficulty}",
        }


class ProblemDisplay(HUDComponent):
    """
    Main problem text display.

    Large, centered display for math problems/questions.
    """

    def __init__(
        self,
        position: Position,
        problem_text: str = "",
        problem_number: int = 1,
        category: str = "arithmetic",
        font_size: str = "large",  # small, medium, large, xlarge
    ):
        super().__init__(position, Size(DISPLAY_WIDTH - 40, 80))
        self.problem_text = problem_text
        self.problem_number = problem_number
        self.category = category
        self.font_size = font_size

    def render(self) -> Dict[str, Any]:
        return {
            "type": "problem_display",
            "text": self.problem_text,
            "problem_number": self.problem_number,
            "category": self.category,
            "font_size": self.font_size,
            "color": rgb_to_hex(WHAM_COLORS.text_primary),
            "alignment": Alignment.CENTER.value,
        }


class ResultOverlay(HUDComponent):
    """
    Result feedback overlay.

    Shows correct/incorrect with animations.
    """

    def __init__(
        self,
        position: Position,
        correct: bool,
        time_ms: int = 0,
        speed_tier: str = "good",
        message: str = "",
        correct_answer: Optional[str] = None,
    ):
        super().__init__(
            position,
            Size(DISPLAY_WIDTH - 80, 120),
            animation=AnimationType.FADE_IN
        )
        self.correct = correct
        self.time_ms = time_ms
        self.speed_tier = speed_tier
        self.message = message
        self.correct_answer = correct_answer

    def render(self) -> Dict[str, Any]:
        if self.correct:
            icon = "✓"
            color = get_speed_tier_color(self.speed_tier)
            animation = AnimationType.SCALE_UP
        else:
            icon = "✗"
            color = WHAM_COLORS.error
            animation = AnimationType.FLASH

        return {
            "type": "result_overlay",
            "correct": self.correct,
            "icon": icon,
            "color": rgb_to_hex(color),
            "time_ms": self.time_ms,
            "time_text": f"{self.time_ms / 1000:.2f}s",
            "speed_tier": self.speed_tier,
            "message": self.message,
            "correct_answer": self.correct_answer if not self.correct else None,
            "animation": animation.value,
        }


class SessionSummary(HUDComponent):
    """
    End-of-session summary display.

    Shows comprehensive stats with WHAM commentary.
    """

    def __init__(
        self,
        position: Position,
        correct: int,
        total: int,
        best_streak: int,
        avg_time_ms: int,
        duration_seconds: int,
        grade: str = "B",
        wham_message: str = "",
    ):
        super().__init__(
            position,
            Size(DISPLAY_WIDTH - 60, 300),
            animation=AnimationType.SLIDE_IN
        )
        self.correct = correct
        self.total = total
        self.best_streak = best_streak
        self.avg_time_ms = avg_time_ms
        self.duration_seconds = duration_seconds
        self.grade = grade
        self.wham_message = wham_message

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0
        return (self.correct / self.total) * 100

    def render(self) -> Dict[str, Any]:
        # Grade colors
        grade_colors = {
            "S": (255, 215, 0),    # Gold
            "A": WHAM_COLORS.success,
            "B": WHAM_COLORS.primary,
            "C": WHAM_COLORS.warning,
            "D": WHAM_COLORS.error,
        }
        grade_color = grade_colors.get(self.grade, WHAM_COLORS.text_secondary)

        return {
            "type": "session_summary",
            "stats": {
                "correct": self.correct,
                "total": self.total,
                "accuracy": round(self.accuracy, 1),
                "best_streak": self.best_streak,
                "avg_time_ms": self.avg_time_ms,
                "avg_time_text": f"{self.avg_time_ms / 1000:.2f}s",
                "duration_seconds": self.duration_seconds,
                "duration_text": self._format_duration(),
            },
            "grade": self.grade,
            "grade_color": rgb_to_hex(grade_color),
            "wham_message": self.wham_message,
            "primary_color": rgb_to_hex(WHAM_COLORS.primary),
        }

    def _format_duration(self) -> str:
        """Format duration as MM:SS."""
        minutes = self.duration_seconds // 60
        seconds = self.duration_seconds % 60
        return f"{minutes}:{seconds:02d}"


class WHAMMessage(HUDComponent):
    """
    WHAM personality message display.

    Shows witty feedback with typewriter effect option.
    """

    def __init__(
        self,
        position: Position,
        message: str,
        importance: str = "normal",  # normal, milestone, warning
        typewriter: bool = False,
    ):
        super().__init__(position, Size(DISPLAY_WIDTH - 60, 40))
        self.message = message
        self.importance = importance
        self.typewriter = typewriter

    def render(self) -> Dict[str, Any]:
        # Importance colors
        colors = {
            "normal": WHAM_COLORS.text_secondary,
            "milestone": WHAM_COLORS.primary_bright,
            "warning": WHAM_COLORS.warning,
        }
        color = colors.get(self.importance, WHAM_COLORS.text_secondary)

        return {
            "type": "wham_message",
            "message": self.message,
            "color": rgb_to_hex(color),
            "importance": self.importance,
            "typewriter": self.typewriter,
            "font_style": "italic",
        }
