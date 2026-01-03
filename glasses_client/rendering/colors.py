"""
Color Scheme Definitions for Halo Frame

RGB color mappings for different content types.
"""

from enum import Enum
from typing import Tuple, Dict, Union


class ColorScheme(str, Enum):
    """Color schemes matching backend ColorScheme enum."""
    # Math & Physics
    MATH = "math"
    RESULT_GREEN = "result_green"
    RESULT_RED = "result_red"

    # Code
    CODE = "code"
    CODE_ERROR = "code_error"
    CODE_FIX = "code_fix"

    # Quant
    QUANT_TIMER = "quant_timer"
    QUANT_STREAK = "quant_streak"

    # General
    INFO = "info"
    WARNING = "warning"
    SUCCESS = "success"


# RGB color values (0-255)
COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    # Math & Physics
    "math": (76, 201, 240),           # Bright blue
    "result_green": (0, 255, 136),    # Neon green
    "result_red": (255, 68, 68),      # Soft red

    # Code
    "code": (212, 212, 212),          # Light gray
    "code_error": (255, 68, 68),      # Red
    "code_fix": (0, 255, 136),        # Green

    # Quant
    "quant_timer": (255, 140, 0),     # Orange
    "quant_streak": (255, 215, 0),    # Gold

    # General
    "info": (200, 200, 200),          # Gray
    "warning": (255, 217, 61),        # Yellow
    "success": (0, 255, 136),         # Green

    # Default
    "default": (255, 255, 255),       # White
}


def get_rgb_color(scheme: Union[str, ColorScheme]) -> Tuple[int, int, int]:
    """
    Get RGB tuple for a color scheme.

    Args:
        scheme: Color scheme name or enum value

    Returns:
        RGB tuple (r, g, b) with values 0-255
    """
    if isinstance(scheme, ColorScheme):
        scheme = scheme.value

    return COLOR_MAP.get(scheme, COLOR_MAP["default"])


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def blend_colors(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    ratio: float = 0.5
) -> Tuple[int, int, int]:
    """
    Blend two colors together.

    Args:
        color1: First RGB color
        color2: Second RGB color
        ratio: Blend ratio (0 = all color1, 1 = all color2)

    Returns:
        Blended RGB color
    """
    return tuple(
        int(c1 * (1 - ratio) + c2 * ratio)
        for c1, c2 in zip(color1, color2)
    )


def get_timer_color(progress: float) -> Tuple[int, int, int]:
    """
    Get color for timer based on progress (0-1).

    Colors transition: blue → orange → red as time runs out.
    """
    if progress < 0.5:
        return COLOR_MAP["math"]
    elif progress < 0.8:
        # Blend from math blue to timer orange
        blend_progress = (progress - 0.5) / 0.3
        return blend_colors(
            COLOR_MAP["math"],
            COLOR_MAP["quant_timer"],
            blend_progress
        )
    else:
        # Blend from timer orange to result red
        blend_progress = (progress - 0.8) / 0.2
        return blend_colors(
            COLOR_MAP["quant_timer"],
            COLOR_MAP["result_red"],
            blend_progress
        )
