"""
JARVIS Color System for Halo HUD.
Iron Man inspired cyan/blue palette with semantic colors.
"""
from dataclasses import dataclass
from typing import Tuple

# Type alias for RGB color
RGB = Tuple[int, int, int]


@dataclass
class JarvisColors:
    """Iron Man / JARVIS color palette."""

    # Primary colors
    primary: RGB = (0, 255, 255)          # Cyan - signature JARVIS color
    primary_bright: RGB = (100, 255, 255)  # Bright cyan for emphasis
    primary_dim: RGB = (0, 180, 200)       # Dimmed cyan for backgrounds

    # Accent colors
    accent: RGB = (50, 150, 255)           # Electric blue
    accent_bright: RGB = (100, 180, 255)   # Bright blue
    accent_dim: RGB = (30, 100, 180)       # Dim blue

    # Semantic colors
    success: RGB = (0, 255, 128)           # Green-cyan for correct
    warning: RGB = (255, 191, 0)           # Amber/gold for warnings
    error: RGB = (255, 64, 64)             # Red for errors/wrong

    # Text colors
    text_primary: RGB = (255, 255, 255)    # White
    text_secondary: RGB = (200, 220, 255)  # Blue-tinted white
    text_dim: RGB = (128, 140, 160)        # Dimmed text

    # Background
    background: RGB = (0, 0, 0)            # Black (OLED off)
    background_overlay: RGB = (10, 20, 30) # Slight blue tint

    # Streak tier colors
    streak_bronze: RGB = (205, 127, 50)    # Bronze (3-4)
    streak_silver: RGB = (192, 192, 192)   # Silver (5-6)
    streak_gold: RGB = (255, 215, 0)       # Gold (7-9)
    streak_platinum: RGB = (229, 228, 226) # Platinum (10-14)
    streak_diamond: RGB = (185, 242, 255)  # Diamond (15-24)
    streak_legendary: RGB = (255, 0, 255)  # Magenta (25+)


# Default instance
COLORS = JarvisColors()


def rgb_to_hex(color: RGB) -> str:
    """Convert RGB tuple to hex string."""
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def rgb_to_lua(color: RGB) -> str:
    """Convert RGB tuple to Lua color string for Halo."""
    return f"0x{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def lerp_color(color1: RGB, color2: RGB, t: float) -> RGB:
    """
    Linear interpolation between two colors.

    Args:
        color1: Start color
        color2: End color
        t: Interpolation factor (0.0 to 1.0)
    """
    t = max(0.0, min(1.0, t))
    return (
        int(color1[0] + (color2[0] - color1[0]) * t),
        int(color1[1] + (color2[1] - color1[1]) * t),
        int(color1[2] + (color2[2] - color1[2]) * t),
    )


def get_timer_color(elapsed_ms: float, target_ms: float) -> RGB:
    """
    Get timer color based on time pressure.

    Color transitions:
        0-50%: Cyan (safe)
        50-75%: Cyan -> Amber (warning transition)
        75-90%: Amber (warning)
        90-100%+: Amber -> Red (danger)
    """
    if target_ms <= 0:
        return COLORS.primary

    ratio = elapsed_ms / target_ms

    if ratio <= 0.5:
        # Safe zone - pure cyan
        return COLORS.primary
    elif ratio <= 0.75:
        # Transition to warning
        t = (ratio - 0.5) / 0.25
        return lerp_color(COLORS.primary, COLORS.warning, t)
    elif ratio <= 0.9:
        # Warning zone - amber
        return COLORS.warning
    else:
        # Danger zone - transition to red
        t = min(1.0, (ratio - 0.9) / 0.1)
        return lerp_color(COLORS.warning, COLORS.error, t)


def get_streak_color(streak: int) -> RGB:
    """
    Get color for streak counter based on tier.

    Tiers:
        0-2: Primary (building)
        3-4: Bronze
        5-6: Silver
        7-9: Gold
        10-14: Platinum
        15-24: Diamond
        25+: Legendary
    """
    if streak < 3:
        return COLORS.primary
    elif streak < 5:
        return COLORS.streak_bronze
    elif streak < 7:
        return COLORS.streak_silver
    elif streak < 10:
        return COLORS.streak_gold
    elif streak < 15:
        return COLORS.streak_platinum
    elif streak < 25:
        return COLORS.streak_diamond
    else:
        return COLORS.streak_legendary


def get_accuracy_color(accuracy: float) -> RGB:
    """
    Get color based on accuracy percentage.

    95%+: Success green
    85-95%: Cyan
    70-85%: Warning amber
    <70%: Error red
    """
    if accuracy >= 95:
        return COLORS.success
    elif accuracy >= 85:
        return COLORS.primary
    elif accuracy >= 70:
        return COLORS.warning
    else:
        return COLORS.error


def get_speed_tier_color(time_ms: float, target_ms: float) -> RGB:
    """
    Get color based on speed performance tier.

    <50% of target: Gold (exceptional)
    50-75%: Green (excellent)
    75-100%: Cyan (good)
    >100%: Amber (needs work)
    """
    if target_ms <= 0:
        return COLORS.primary

    ratio = time_ms / target_ms

    if ratio < 0.5:
        return COLORS.streak_gold  # Exceptional
    elif ratio < 0.75:
        return COLORS.success      # Excellent
    elif ratio <= 1.0:
        return COLORS.primary      # Good
    else:
        return COLORS.warning      # Needs work


def get_difficulty_color(difficulty: int) -> RGB:
    """Get color for difficulty level indicator."""
    colors = {
        1: (100, 255, 100),  # Light green - easy
        2: COLORS.primary,    # Cyan - standard
        3: COLORS.accent,     # Blue - challenging
        4: (200, 100, 255),   # Purple - hard
        5: COLORS.error,      # Red - extreme
    }
    return colors.get(difficulty, COLORS.primary)


# Streak tier names for display
STREAK_TIERS = {
    0: ("", COLORS.primary),
    3: ("Bronze", COLORS.streak_bronze),
    5: ("Silver", COLORS.streak_silver),
    7: ("Gold", COLORS.streak_gold),
    10: ("Platinum", COLORS.streak_platinum),
    15: ("Diamond", COLORS.streak_diamond),
    25: ("Legendary", COLORS.streak_legendary),
    50: ("Mythic", (255, 100, 255)),
}


def get_streak_tier_name(streak: int) -> str:
    """Get the tier name for a streak count."""
    tier_name = ""
    for threshold, (name, _) in sorted(STREAK_TIERS.items(), reverse=True):
        if streak >= threshold:
            tier_name = name
            break
    return tier_name
