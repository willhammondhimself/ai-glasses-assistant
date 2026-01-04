"""
Iron Man HUD Color System

WHAM color palette for AR glasses display.
Optimized for OLED visibility and eye comfort.
"""

from dataclasses import dataclass
from typing import Tuple, Dict


# RGB color type
RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, float]


@dataclass(frozen=True)
class HUDColorScheme:
    """Complete color scheme for HUD elements."""
    # Primary colors
    primary: RGB
    primary_dim: RGB
    primary_bright: RGB

    # Accent colors
    accent: RGB
    accent_dim: RGB

    # Status colors
    success: RGB
    warning: RGB
    error: RGB

    # Text colors
    text_primary: RGB
    text_secondary: RGB
    text_dim: RGB

    # Background/overlay
    overlay: RGBA
    panel_bg: RGBA


# ============================================================================
# WHAM Color Palette
# ============================================================================

WHAM_COLORS = HUDColorScheme(
    # Primary cyan family (Iron Man HUD signature)
    primary=(0, 255, 255),           # #00FFFF - Bright cyan
    primary_dim=(0, 180, 200),       # Dimmed for less important elements
    primary_bright=(100, 255, 255),  # High emphasis

    # Accent blue
    accent=(50, 150, 255),           # Electric blue accent
    accent_dim=(30, 100, 180),       # Subtle accent

    # Status colors
    success=(0, 255, 128),           # Green-cyan for correct/success
    warning=(255, 191, 0),           # Amber/gold for warnings
    error=(255, 64, 64),             # Red for errors/timeout

    # Text colors
    text_primary=(255, 255, 255),    # Pure white for main text
    text_secondary=(200, 220, 255),  # Slight blue tint
    text_dim=(128, 160, 180),        # Dimmed text

    # Overlays (with alpha)
    overlay=(0, 20, 40, 0.7),        # Dark blue overlay
    panel_bg=(10, 30, 50, 0.8),      # Panel background
)


# Alternative color schemes
EDITH_COLORS = HUDColorScheme(
    # EDITH uses more red/amber tones
    primary=(255, 100, 100),         # Coral/salmon
    primary_dim=(180, 80, 80),
    primary_bright=(255, 150, 150),

    accent=(255, 180, 0),            # Gold
    accent_dim=(180, 130, 0),

    success=(100, 255, 150),
    warning=(255, 200, 50),
    error=(255, 50, 50),

    text_primary=(255, 255, 255),
    text_secondary=(255, 220, 200),
    text_dim=(180, 150, 140),

    overlay=(40, 10, 10, 0.7),
    panel_bg=(50, 20, 20, 0.8),
)


# ============================================================================
# Timer Color Transitions
# ============================================================================

# Time thresholds for color transitions (as percentage of target time)
TIMER_THRESHOLDS = {
    "safe": 0.5,      # 0-50% of time: cyan
    "caution": 0.75,  # 50-75%: transition to amber
    "warning": 0.9,   # 75-90%: amber
    "danger": 1.0,    # 90-100%: red
}

def get_timer_color(elapsed_ms: int, target_ms: int) -> RGB:
    """
    Get timer color based on elapsed time.

    Smooth transition:
    - 0-50%: Cyan (comfortable pace)
    - 50-75%: Cyan -> Amber (time pressure building)
    - 75-90%: Amber (hurry up)
    - 90-100%+: Red (danger/overtime)
    """
    if target_ms <= 0:
        return WHAM_COLORS.primary

    ratio = elapsed_ms / target_ms

    if ratio <= TIMER_THRESHOLDS["safe"]:
        # Pure cyan - plenty of time
        return WHAM_COLORS.primary

    elif ratio <= TIMER_THRESHOLDS["caution"]:
        # Transition cyan -> amber
        t = (ratio - 0.5) / 0.25  # 0 to 1 over this range
        return _interpolate_color(
            WHAM_COLORS.primary,
            WHAM_COLORS.warning,
            t
        )

    elif ratio <= TIMER_THRESHOLDS["warning"]:
        # Transition amber -> red (starting)
        t = (ratio - 0.75) / 0.15
        return _interpolate_color(
            WHAM_COLORS.warning,
            WHAM_COLORS.error,
            t * 0.5  # Only go halfway to red
        )

    else:
        # Red - time's up or overtime
        return WHAM_COLORS.error


def _interpolate_color(start: RGB, end: RGB, t: float) -> RGB:
    """Interpolate between two colors."""
    t = max(0, min(1, t))  # Clamp to 0-1
    return (
        int(start[0] + (end[0] - start[0]) * t),
        int(start[1] + (end[1] - start[1]) * t),
        int(start[2] + (end[2] - start[2]) * t),
    )


# ============================================================================
# Streak Color System
# ============================================================================

# Streak tier thresholds
STREAK_TIERS = {
    "bronze": (3, (205, 127, 50)),    # 3-4 streak
    "silver": (5, (192, 192, 192)),   # 5-6 streak
    "gold": (7, (255, 215, 0)),       # 7-9 streak
    "platinum": (10, (229, 228, 226)), # 10-14 streak
    "diamond": (15, (185, 242, 255)),  # 15-19 streak
    "legendary": (20, (255, 100, 255)), # 20+ streak (purple/magenta)
}

def get_streak_color(streak: int) -> Tuple[RGB, str]:
    """
    Get streak display color and tier name.

    Returns:
        Tuple of (color, tier_name)
    """
    if streak < 3:
        return WHAM_COLORS.text_secondary, "none"

    for tier_name, (threshold, color) in reversed(STREAK_TIERS.items()):
        if streak >= threshold:
            return color, tier_name

    return WHAM_COLORS.text_secondary, "none"


# ============================================================================
# Accuracy Color System
# ============================================================================

def get_accuracy_color(accuracy: float) -> RGB:
    """
    Get color for accuracy percentage display.

    Args:
        accuracy: 0-100 percentage

    Returns:
        RGB color tuple
    """
    if accuracy >= 95:
        return WHAM_COLORS.success
    elif accuracy >= 85:
        return WHAM_COLORS.primary
    elif accuracy >= 70:
        return WHAM_COLORS.warning
    else:
        return WHAM_COLORS.error


# ============================================================================
# Speed Tier Colors
# ============================================================================

SPEED_TIER_COLORS: Dict[str, RGB] = {
    "exceptional": (255, 215, 0),   # Gold - Jane Street caliber
    "excellent": WHAM_COLORS.success,  # Green
    "good": WHAM_COLORS.primary,       # Cyan
    "needs_work": WHAM_COLORS.warning, # Amber
}

def get_speed_tier_color(tier: str) -> RGB:
    """Get color for speed performance tier."""
    return SPEED_TIER_COLORS.get(tier, WHAM_COLORS.text_secondary)


# ============================================================================
# Difficulty Level Colors
# ============================================================================

DIFFICULTY_COLORS: Dict[int, RGB] = {
    1: (100, 200, 100),   # Easy - soft green
    2: (100, 200, 200),   # Medium - cyan
    3: (200, 200, 100),   # Hard - yellow
    4: (255, 150, 50),    # Expert - orange
    5: (255, 80, 80),     # Master - red
}

def get_difficulty_color(difficulty: int) -> RGB:
    """Get color for difficulty level indicator."""
    return DIFFICULTY_COLORS.get(difficulty, WHAM_COLORS.text_secondary)


# ============================================================================
# Utility Functions
# ============================================================================

def rgb_to_hex(color: RGB) -> str:
    """Convert RGB tuple to hex string."""
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def rgba_to_css(color: RGBA) -> str:
    """Convert RGBA tuple to CSS rgba() string."""
    return f"rgba({color[0]}, {color[1]}, {color[2]}, {color[3]})"


def apply_brightness(color: RGB, factor: float) -> RGB:
    """Apply brightness factor to color (1.0 = unchanged)."""
    return (
        min(255, int(color[0] * factor)),
        min(255, int(color[1] * factor)),
        min(255, int(color[2] * factor)),
    )


def apply_alpha(color: RGB, alpha: float) -> RGBA:
    """Add alpha channel to RGB color."""
    return (color[0], color[1], color[2], alpha)
