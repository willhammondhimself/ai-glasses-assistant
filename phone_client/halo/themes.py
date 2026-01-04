"""
Themes - Contextual color themes for different WHAM modes.
Provides consistent color palettes optimized for each use case.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Optional


@dataclass
class ColorTheme:
    """
    Complete color palette for a mode.

    All colors are RGB tuples (0-255 per channel).
    Themes are OLED-optimized with true black backgrounds.
    """
    # Primary colors
    primary: Tuple[int, int, int]       # Main accent color
    accent: Tuple[int, int, int]        # Secondary accent
    background: Tuple[int, int, int]    # Background (usually black)

    # Semantic colors
    success: Tuple[int, int, int]       # Positive outcomes
    warning: Tuple[int, int, int]       # Caution/attention
    danger: Tuple[int, int, int]        # Errors/negative

    # Text colors
    text_primary: Tuple[int, int, int]   # Main text
    text_secondary: Tuple[int, int, int] # Dimmed text
    text_dim: Tuple[int, int, int]       # Very subtle text

    # UI colors
    border: Tuple[int, int, int]        # Borders/dividers
    overlay: Tuple[int, int, int]       # Overlay backgrounds

    # Mode-specific (optional)
    highlight: Optional[Tuple[int, int, int]] = None

    def get_color(self, name: str) -> Tuple[int, int, int]:
        """
        Get a color by name.

        Args:
            name: Color attribute name

        Returns:
            RGB tuple
        """
        return getattr(self, name, self.text_primary)


class ContextualThemes:
    """
    Predefined color themes for each WHAM mode.

    Each theme is carefully designed for:
    - Visual clarity in the specific context
    - OLED power efficiency (darker colors)
    - Psychological appropriateness (green for money, blue for focus, etc.)
    - Consistent visual language across modes
    """

    POKER = ColorTheme(
        # Green felt aesthetic with gold accents
        primary=(0, 180, 120),           # Poker felt green
        accent=(255, 215, 0),            # Gold (chips/money)
        background=(10, 25, 15),         # Very dark green tint

        success=(0, 220, 100),           # Win green
        warning=(255, 180, 60),          # Caution amber
        danger=(255, 80, 80),            # Lose/fold red

        text_primary=(255, 255, 255),
        text_secondary=(180, 200, 180),  # Greenish tint
        text_dim=(100, 120, 100),

        border=(60, 80, 60),
        overlay=(20, 40, 25),

        highlight=(255, 215, 0)          # Gold for important info
    )

    MENTAL_MATH = ColorTheme(
        # Clean cyan/electric blue - analytical, focused
        primary=(0, 255, 255),           # Cyan (WHAM signature)
        accent=(50, 150, 255),           # Electric blue
        background=(0, 0, 0),            # Pure black

        success=(0, 255, 128),           # Correct answer green
        warning=(255, 191, 0),           # Timer warning amber
        danger=(255, 64, 64),            # Wrong answer red

        text_primary=(255, 255, 255),
        text_secondary=(180, 180, 180),
        text_dim=(100, 100, 100),

        border=(60, 60, 70),
        overlay=(20, 20, 25),

        highlight=(0, 255, 255)          # Cyan highlights
    )

    HOMEWORK = ColorTheme(
        # Study blue - calm, focused, educational
        primary=(100, 150, 255),         # Study blue
        accent=(255, 180, 100),          # Highlight orange
        background=(5, 10, 20),          # Dark blue tint

        success=(0, 220, 150),           # Solved green
        warning=(255, 200, 80),          # Hint amber
        danger=(255, 100, 100),          # Error red

        text_primary=(255, 255, 255),
        text_secondary=(180, 190, 220),  # Bluish tint
        text_dim=(100, 110, 130),

        border=(60, 70, 90),
        overlay=(15, 20, 35),

        highlight=(255, 180, 100)        # Orange for key concepts
    )

    MEETING = ColorTheme(
        # Professional amber/warm - business context
        primary=(255, 160, 60),          # Professional amber
        accent=(100, 200, 255),          # Info blue
        background=(20, 15, 10),         # Warm dark

        success=(150, 220, 100),         # Agreement green
        warning=(255, 200, 100),         # Attention amber
        danger=(255, 120, 80),           # Alert orange

        text_primary=(255, 255, 255),
        text_secondary=(220, 200, 180),  # Warm tint
        text_dim=(140, 120, 100),

        border=(80, 70, 60),
        overlay=(35, 30, 25),

        highlight=(100, 200, 255)        # Blue for key points
    )

    CODE_DEBUG = ColorTheme(
        # Dark IDE aesthetic with syntax highlighting influence
        primary=(150, 255, 150),         # Green (success/pass)
        accent=(255, 150, 100),          # Orange (variables)
        background=(15, 15, 20),         # Dark IDE background

        success=(100, 255, 100),         # Test pass green
        warning=(255, 200, 100),         # Warning yellow
        danger=(255, 100, 100),          # Error red

        text_primary=(220, 220, 220),    # Code text
        text_secondary=(150, 150, 170),  # Comments
        text_dim=(90, 90, 100),

        border=(60, 60, 80),
        overlay=(25, 25, 35),

        highlight=(150, 150, 255)        # Purple for special
    )

    IDLE = ColorTheme(
        # Minimal, ambient - WHAM at rest
        primary=(0, 255, 255),           # Cyan accent
        accent=(80, 120, 160),           # Subtle blue
        background=(0, 0, 0),            # Pure black (OLED off)

        success=(0, 200, 128),           # Muted green
        warning=(200, 150, 0),           # Muted amber
        danger=(200, 80, 80),            # Muted red

        text_primary=(200, 200, 200),    # Slightly dimmed
        text_secondary=(120, 120, 120),
        text_dim=(60, 60, 60),

        border=(40, 40, 50),
        overlay=(15, 15, 20),

        highlight=(0, 200, 200)          # Dimmed cyan
    )

    # Mapping of mode names to themes
    _THEME_MAP: Dict[str, ColorTheme] = {
        "poker": POKER,
        "mental_math": MENTAL_MATH,
        "math": MENTAL_MATH,
        "homework": HOMEWORK,
        "study": HOMEWORK,
        "meeting": MEETING,
        "code_debug": CODE_DEBUG,
        "code": CODE_DEBUG,
        "debug": CODE_DEBUG,
        "idle": IDLE,
        "default": MENTAL_MATH,
    }

    @classmethod
    def get_theme(cls, mode: str) -> ColorTheme:
        """
        Get theme for a specific mode.

        Args:
            mode: Mode name (poker, mental_math, homework, meeting, code_debug, idle)

        Returns:
            ColorTheme for the mode (defaults to MENTAL_MATH)
        """
        return cls._THEME_MAP.get(mode.lower(), cls.MENTAL_MATH)

    @classmethod
    def list_themes(cls) -> list:
        """Get list of available theme names."""
        return list(set(cls._THEME_MAP.keys()) - {"default", "math", "study", "code", "debug"})


class ThemeManager:
    """
    Manages theme transitions and application.

    Provides smooth transitions between themes when modes change
    and integrates with the power manager for brightness adjustment.
    """

    def __init__(self, initial_mode: str = "idle", power_manager=None):
        """
        Initialize theme manager.

        Args:
            initial_mode: Starting mode
            power_manager: Optional PowerManager for brightness adjustment
        """
        self._current_mode = initial_mode
        self._current_theme = ContextualThemes.get_theme(initial_mode)
        self._power_manager = power_manager
        self._transition_in_progress = False

    @property
    def current_theme(self) -> ColorTheme:
        """Get current active theme."""
        return self._current_theme

    @property
    def current_mode(self) -> str:
        """Get current mode name."""
        return self._current_mode

    def set_mode(self, mode: str) -> Tuple[ColorTheme, ColorTheme]:
        """
        Switch to a new mode/theme.

        Args:
            mode: New mode name

        Returns:
            Tuple of (old_theme, new_theme)
        """
        old_theme = self._current_theme
        self._current_mode = mode
        self._current_theme = ContextualThemes.get_theme(mode)

        return (old_theme, self._current_theme)

    def get_color(self, color_name: str) -> Tuple[int, int, int]:
        """
        Get a color from the current theme.

        Args:
            color_name: Color attribute name

        Returns:
            RGB tuple, adjusted for power mode if applicable
        """
        color = self._current_theme.get_color(color_name)

        if self._power_manager:
            color = self._power_manager.apply_color_adjustment(color)

        return color

    def get_themed_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Get all colors from current theme as a dictionary.

        Returns:
            Dict mapping color names to RGB tuples
        """
        return {
            "primary": self._current_theme.primary,
            "accent": self._current_theme.accent,
            "background": self._current_theme.background,
            "success": self._current_theme.success,
            "warning": self._current_theme.warning,
            "danger": self._current_theme.danger,
            "text_primary": self._current_theme.text_primary,
            "text_secondary": self._current_theme.text_secondary,
            "text_dim": self._current_theme.text_dim,
            "border": self._current_theme.border,
            "overlay": self._current_theme.overlay,
            "highlight": self._current_theme.highlight or self._current_theme.primary,
        }

    def interpolate_themes(
        self,
        from_theme: ColorTheme,
        to_theme: ColorTheme,
        progress: float
    ) -> Dict[str, Tuple[int, int, int]]:
        """
        Interpolate between two themes for smooth transitions.

        Args:
            from_theme: Starting theme
            to_theme: Target theme
            progress: Transition progress (0.0-1.0)

        Returns:
            Dict of interpolated colors
        """
        progress = max(0.0, min(1.0, progress))

        def lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> Tuple[int, int, int]:
            return (
                int(c1[0] + (c2[0] - c1[0]) * progress),
                int(c1[1] + (c2[1] - c1[1]) * progress),
                int(c1[2] + (c2[2] - c1[2]) * progress)
            )

        return {
            "primary": lerp_color(from_theme.primary, to_theme.primary),
            "accent": lerp_color(from_theme.accent, to_theme.accent),
            "background": lerp_color(from_theme.background, to_theme.background),
            "success": lerp_color(from_theme.success, to_theme.success),
            "warning": lerp_color(from_theme.warning, to_theme.warning),
            "danger": lerp_color(from_theme.danger, to_theme.danger),
            "text_primary": lerp_color(from_theme.text_primary, to_theme.text_primary),
            "text_secondary": lerp_color(from_theme.text_secondary, to_theme.text_secondary),
            "text_dim": lerp_color(from_theme.text_dim, to_theme.text_dim),
            "border": lerp_color(from_theme.border, to_theme.border),
            "overlay": lerp_color(from_theme.overlay, to_theme.overlay),
        }


def rgb_to_hex(color: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex string."""
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


# Test
def test_themes():
    """Test theme system."""
    print("=== Theme System Test ===\n")

    # List available themes
    print("Available themes:", ContextualThemes.list_themes())
    print()

    # Show each theme
    themes = ["poker", "mental_math", "homework", "meeting", "code_debug", "idle"]

    for theme_name in themes:
        theme = ContextualThemes.get_theme(theme_name)
        print(f"{theme_name.upper()} Theme:")
        print(f"  Primary:     {rgb_to_hex(theme.primary)} {theme.primary}")
        print(f"  Accent:      {rgb_to_hex(theme.accent)} {theme.accent}")
        print(f"  Success:     {rgb_to_hex(theme.success)} {theme.success}")
        print(f"  Warning:     {rgb_to_hex(theme.warning)} {theme.warning}")
        print(f"  Danger:      {rgb_to_hex(theme.danger)} {theme.danger}")
        print(f"  Background:  {rgb_to_hex(theme.background)} {theme.background}")
        print()

    # Test theme manager
    print("Theme Manager Test:")
    manager = ThemeManager(initial_mode="idle")
    print(f"  Initial mode: {manager.current_mode}")

    old, new = manager.set_mode("poker")
    print(f"  Switched to poker")
    print(f"  Primary color: {rgb_to_hex(manager.get_color('primary'))}")

    # Test interpolation
    print("\nTheme Interpolation (50% poker → mental_math):")
    interpolated = manager.interpolate_themes(
        ContextualThemes.POKER,
        ContextualThemes.MENTAL_MATH,
        0.5
    )
    print(f"  Primary: {rgb_to_hex(interpolated['primary'])}")


if __name__ == "__main__":
    test_themes()
