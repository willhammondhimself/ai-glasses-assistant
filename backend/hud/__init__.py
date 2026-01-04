"""
Iron Man HUD System

Full-color OLED display components for Brilliant Labs Halo AR glasses.
WHAM-integrated HUD display.

Display specs:
- Resolution: 640x400 pixels
- Color: Full RGB OLED
- FOV: 20 degrees diagonal
"""

from .colors import (
    WHAM_COLORS,
    HUDColorScheme,
    get_timer_color,
    get_streak_color,
    get_accuracy_color,
)
from .components import (
    HUDComponent,
    ProgressBar,
    TimerDisplay,
    StreakCounter,
    AccuracyMeter,
    DifficultyIndicator,
    ProblemDisplay,
    ResultOverlay,
    SessionSummary,
)
from .layouts import (
    HUDLayout,
    MentalMathLayout,
    PokerLayout,
    CodeDebugLayout,
)

__all__ = [
    # Colors
    "WHAM_COLORS",
    "HUDColorScheme",
    "get_timer_color",
    "get_streak_color",
    "get_accuracy_color",
    # Components
    "HUDComponent",
    "ProgressBar",
    "TimerDisplay",
    "StreakCounter",
    "AccuracyMeter",
    "DifficultyIndicator",
    "ProblemDisplay",
    "ResultOverlay",
    "SessionSummary",
    # Layouts
    "HUDLayout",
    "MentalMathLayout",
    "PokerLayout",
    "CodeDebugLayout",
]
