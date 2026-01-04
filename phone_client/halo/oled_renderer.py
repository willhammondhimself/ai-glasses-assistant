"""
OLED Renderer - Optimized rendering engine for Halo's 640x400 OLED display.
Features burn-in prevention, power-aware color adjustment, and smooth rendering.
"""
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING

# Handle imports for both package and standalone usage
try:
    from ..core.power_manager import PowerManager, PowerMode
except ImportError:
    from core.power_manager import PowerManager, PowerMode


# Display constants
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 400
CENTER_X = DISPLAY_WIDTH // 2
CENTER_Y = DISPLAY_HEIGHT // 2

# Font sizes
FONT_LARGE = 48
FONT_MEDIUM = 32
FONT_SMALL = 24
FONT_TINY = 16

# Burn-in prevention defaults
DEFAULT_PIXEL_SHIFT_INTERVAL_MS = 30000  # 30 seconds
DEFAULT_PIXEL_SHIFT_AMOUNT = 2  # pixels


class TextAlign(Enum):
    """Text alignment options."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass
class DisplayConfig:
    """OLED display configuration."""
    width: int = DISPLAY_WIDTH
    height: int = DISPLAY_HEIGHT
    brightness: float = 1.0  # 0.0-1.0
    power_mode: str = "balanced"
    pixel_shift_enabled: bool = True
    pixel_shift_interval_ms: int = DEFAULT_PIXEL_SHIFT_INTERVAL_MS
    pixel_shift_amount: int = DEFAULT_PIXEL_SHIFT_AMOUNT
    dim_after_seconds: int = 60


@dataclass
class RenderElement:
    """Base render element."""
    x: int
    y: int
    layer: int = 0  # Higher layers render on top


@dataclass
class TextElement(RenderElement):
    """Text render element."""
    content: str = ""
    color: Tuple[int, int, int] = (255, 255, 255)
    size: int = FONT_MEDIUM
    align: TextAlign = TextAlign.CENTER
    spacing: int = 1


@dataclass
class RectElement(RenderElement):
    """Rectangle render element."""
    width: int = 100
    height: int = 50
    color: Tuple[int, int, int] = (0, 255, 255)
    filled: bool = True
    border_radius: int = 0


@dataclass
class ProgressBarElement(RenderElement):
    """Progress bar render element."""
    width: int = 200
    height: int = 10
    value: float = 0.0  # 0.0-1.0
    max_value: float = 1.0
    color: Tuple[int, int, int] = (0, 255, 255)
    background: Tuple[int, int, int] = (40, 40, 40)
    animated: bool = False


# OLED-optimized color palette
class OLEDColors:
    """OLED-optimized colors that are energy efficient and prevent burn-in."""

    # Primary palette (cyan-based - efficient on OLED)
    PRIMARY = (0, 255, 255)        # Cyan - main accent
    PRIMARY_DIM = (0, 160, 160)    # Dimmed cyan
    ACCENT = (50, 150, 255)        # Electric blue

    # Status colors
    SUCCESS = (0, 255, 128)        # Green-cyan
    WARNING = (255, 191, 0)        # Amber
    ERROR = (255, 64, 64)          # Red

    # Text colors (brightness optimized)
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (180, 180, 180)
    TEXT_DIM = (100, 100, 100)

    # UI colors
    BACKGROUND = (0, 0, 0)         # Pure black (OLED off)
    OVERLAY = (20, 20, 25)         # Subtle overlay
    BORDER = (60, 60, 70)

    # Poker card suits
    SUIT_HEARTS = (255, 68, 68)    # Red
    SUIT_DIAMONDS = (68, 136, 255) # Blue
    SUIT_CLUBS = (68, 204, 68)     # Green
    SUIT_SPADES = (255, 255, 255)  # White

    # Confidence colors
    CONF_HIGH = (0, 255, 128)      # Green
    CONF_MED = (255, 200, 0)       # Yellow
    CONF_LOW = (255, 80, 80)       # Red

    @classmethod
    def get_suit_color(cls, suit: str) -> Tuple[int, int, int]:
        """Get color for poker suit."""
        suit_map = {
            'h': cls.SUIT_HEARTS,
            'd': cls.SUIT_DIAMONDS,
            'c': cls.SUIT_CLUBS,
            's': cls.SUIT_SPADES,
        }
        return suit_map.get(suit.lower(), cls.TEXT_PRIMARY)

    @classmethod
    def get_confidence_color(cls, confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence level."""
        if confidence >= 0.8:
            return cls.CONF_HIGH
        elif confidence >= 0.6:
            return cls.CONF_MED
        else:
            return cls.CONF_LOW

    @classmethod
    def load_from_config(cls, config: dict) -> None:
        """
        Load custom colors from config.yaml.

        This allows overriding the default palette through configuration,
        making color changes require only config edits instead of code changes.

        Args:
            config: Configuration dict with 'colors' section

        Example config.yaml:
            colors:
              primary: [0, 255, 255]
              accent: [50, 150, 255]
              success: [0, 255, 128]
              warning: [255, 191, 0]
              error: [255, 64, 64]
              text: [255, 255, 255]
        """
        colors = config.get("colors", {})

        if not colors:
            return

        # Map config keys to class attributes
        config_to_attr = {
            "primary": "PRIMARY",
            "accent": "ACCENT",
            "success": "SUCCESS",
            "warning": "WARNING",
            "error": "ERROR",
            "text": "TEXT_PRIMARY",
        }

        for config_key, attr_name in config_to_attr.items():
            if config_key in colors:
                color_value = colors[config_key]
                if isinstance(color_value, list) and len(color_value) == 3:
                    setattr(cls, attr_name, tuple(color_value))

        # Also update derived colors
        if "primary" in colors:
            primary = tuple(colors["primary"])
            # Create dimmed version (60% brightness)
            cls.PRIMARY_DIM = (
                int(primary[0] * 0.6),
                int(primary[1] * 0.6),
                int(primary[2] * 0.6)
            )

    @classmethod
    def get_all_colors(cls) -> Dict[str, Tuple[int, int, int]]:
        """
        Get all colors as a dictionary.

        Returns:
            Dict mapping color names to RGB tuples
        """
        return {
            "primary": cls.PRIMARY,
            "primary_dim": cls.PRIMARY_DIM,
            "accent": cls.ACCENT,
            "success": cls.SUCCESS,
            "warning": cls.WARNING,
            "error": cls.ERROR,
            "text_primary": cls.TEXT_PRIMARY,
            "text_secondary": cls.TEXT_SECONDARY,
            "text_dim": cls.TEXT_DIM,
            "background": cls.BACKGROUND,
            "overlay": cls.OVERLAY,
            "border": cls.BORDER,
        }


def rgb_to_lua(color: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to Lua color string."""
    return f'"{color[0]:02x}{color[1]:02x}{color[2]:02x}"'


def lerp_color(
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    t: float
) -> Tuple[int, int, int]:
    """Linear interpolation between two colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t)
    )


class OLEDRenderer:
    """
    OLED-optimized Lua code generator for Halo glasses.

    Features:
    - Advanced burn-in prevention with pixel shifting
    - Power-aware color adjustment based on battery level
    - Efficient rendering with command batching
    - Support for animations and transitions
    """

    def __init__(
        self,
        config: DisplayConfig = None,
        power_manager: PowerManager = None
    ):
        """
        Initialize OLED renderer.

        Args:
            config: Display configuration
            power_manager: Optional power manager for battery-aware rendering
        """
        self.config = config or DisplayConfig()
        self.power_manager = power_manager

        # Display dimensions
        self.width = self.config.width
        self.height = self.config.height
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        # Burn-in prevention state
        self._pixel_offset = (0, 0)
        self._last_shift = time.time()
        self._shift_direction = 1  # Alternating direction

        # Frame buffer
        self._lua_commands: List[str] = []
        self._brightness = self.config.brightness

        # Statistics
        self._frame_count = 0
        self._last_render_time = 0

    def clear(self):
        """Clear the command buffer and display."""
        self._lua_commands = ["frame.display.clear()"]

    def apply_pixel_shift(self) -> Tuple[int, int]:
        """
        Apply pixel shift for burn-in prevention.

        Shifts display content by a few pixels periodically to prevent
        static elements from burning into the OLED panel.

        Returns:
            Current pixel offset (x, y)
        """
        if not self.config.pixel_shift_enabled:
            return (0, 0)

        now = time.time()
        elapsed_ms = (now - self._last_shift) * 1000

        if elapsed_ms >= self.config.pixel_shift_interval_ms:
            # Time to shift
            amount = self.config.pixel_shift_amount

            # Cycle through shift positions: center → right → center → left → repeat
            shifts = [
                (0, 0),
                (amount, 0),
                (amount, amount),
                (0, amount),
                (-amount, amount),
                (-amount, 0),
                (-amount, -amount),
                (0, -amount),
                (amount, -amount),
            ]

            current_idx = shifts.index(self._pixel_offset) if self._pixel_offset in shifts else 0
            next_idx = (current_idx + 1) % len(shifts)
            self._pixel_offset = shifts[next_idx]
            self._last_shift = now

        return self._pixel_offset

    def _shifted(self, x: int, y: int) -> Tuple[int, int]:
        """Apply pixel shift to coordinates."""
        offset = self.apply_pixel_shift()
        return (x + offset[0], y + offset[1])

    def _apply_power_adjustment(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Apply power-aware color adjustment."""
        if self.power_manager:
            return self.power_manager.apply_color_adjustment(color)
        return color

    def text(
        self,
        content: str,
        x: int,
        y: int,
        color: Tuple[int, int, int] = OLEDColors.TEXT_PRIMARY,
        size: int = FONT_MEDIUM,
        align: str = "center",
        spacing: int = 1
    ):
        """
        Add text to the display.

        Args:
            content: Text content
            x: X position
            y: Y position
            color: RGB color tuple
            size: Font size (FONT_LARGE, FONT_MEDIUM, FONT_SMALL, FONT_TINY)
            align: Text alignment (left, center, right)
            spacing: Letter spacing
        """
        sx, sy = self._shifted(x, y)
        adjusted_color = self._apply_power_adjustment(color)
        color_lua = rgb_to_lua(adjusted_color)

        # Escape quotes in content
        safe_content = content.replace('"', '\\"')

        self._lua_commands.append(
            f'frame.display.text("{safe_content}", {sx}, {sy}, '
            f'{{color = {color_lua}, spacing = {spacing}}})'
        )

    def rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        color: Tuple[int, int, int] = OLEDColors.PRIMARY,
        filled: bool = True
    ):
        """
        Add a rectangle to the display.

        Args:
            x: X position
            y: Y position
            width: Rectangle width
            height: Rectangle height
            color: RGB color tuple
            filled: Whether to fill the rectangle
        """
        sx, sy = self._shifted(x, y)
        adjusted_color = self._apply_power_adjustment(color)
        color_lua = rgb_to_lua(adjusted_color)
        fill_str = "true" if filled else "false"

        self._lua_commands.append(
            f'frame.display.rect({sx}, {sy}, {width}, {height}, '
            f'{{color = {color_lua}, filled = {fill_str}}})'
        )

    def progress_bar(
        self,
        x: int,
        y: int,
        width: int,
        height: int = 10,
        value: float = 0.0,
        max_value: float = 1.0,
        color: Tuple[int, int, int] = OLEDColors.PRIMARY,
        background: Tuple[int, int, int] = OLEDColors.OVERLAY
    ):
        """
        Add a progress bar.

        Args:
            x: X position
            y: Y position
            width: Bar width
            height: Bar height
            value: Current value
            max_value: Maximum value
            color: Fill color
            background: Background color
        """
        progress = max(0.0, min(1.0, value / max_value if max_value > 0 else 0))
        fill_width = int(width * progress)

        # Background
        self.rect(x, y, width, height, background, filled=True)

        # Fill
        if fill_width > 0:
            self.rect(x, y, fill_width, height, color, filled=True)

        # Border
        self.rect(x, y, width, height, color, filled=False)

    def confidence_bar(
        self,
        x: int,
        y: int,
        confidence: float,
        show_label: bool = True,
        width: int = 120
    ):
        """
        Render a confidence indicator bar.

        Args:
            x: X position
            y: Y position
            confidence: Confidence value (0.0-1.0)
            show_label: Whether to show percentage label
            width: Bar width
        """
        confidence = max(0.0, min(1.0, confidence))
        color = OLEDColors.get_confidence_color(confidence)

        # Bar segments
        bar_height = 8
        segment_width = width // 10
        segment_gap = 2

        for i in range(10):
            seg_x = x + i * (segment_width + segment_gap)
            filled = i < int(confidence * 10)
            seg_color = color if filled else OLEDColors.OVERLAY
            self.rect(seg_x, y, segment_width - segment_gap, bar_height, seg_color, filled=True)

        # Label
        if show_label:
            label = f"{int(confidence * 100)}%"
            self.text(label, x + width + 10, y, color, FONT_SMALL, "left")

    def card(
        self,
        card_str: str,
        x: int,
        y: int,
        size: int = FONT_LARGE
    ):
        """
        Render a poker card with suit-colored text.

        Args:
            card_str: Card string like "Ah", "Kc", "9s"
            x: X position
            y: Y position
            size: Font size
        """
        if len(card_str) < 2:
            self.text(card_str, x, y, OLEDColors.TEXT_PRIMARY, size)
            return

        rank = card_str[:-1]
        suit = card_str[-1]
        color = OLEDColors.get_suit_color(suit)

        self.text(card_str, x, y, color, size)

    def cards_row(
        self,
        cards: List[str],
        x: int,
        y: int,
        spacing: int = 50,
        size: int = FONT_LARGE
    ):
        """
        Render a row of poker cards.

        Args:
            cards: List of card strings
            x: Starting X position
            y: Y position
            spacing: Space between cards
            size: Font size
        """
        for i, card in enumerate(cards):
            self.card(card, x + i * spacing, y, size)

    def divider(
        self,
        y: int,
        color: Tuple[int, int, int] = OLEDColors.BORDER,
        width: int = None,
        margin: int = 40
    ):
        """
        Render a horizontal divider line.

        Args:
            y: Y position
            color: Line color
            width: Line width (default: display width - margins)
            margin: Side margins
        """
        line_width = width or (self.width - margin * 2)
        self.rect(margin, y, line_width, 1, color, filled=True)

    def icon(
        self,
        icon_type: str,
        x: int,
        y: int,
        color: Tuple[int, int, int] = OLEDColors.PRIMARY,
        size: int = FONT_MEDIUM
    ):
        """
        Render an icon/symbol.

        Args:
            icon_type: Icon identifier
            x: X position
            y: Y position
            color: Icon color
            size: Icon size
        """
        # Unicode symbols that work well on OLED
        icons = {
            "check": "✓",
            "cross": "✗",
            "arrow_right": "→",
            "arrow_left": "←",
            "arrow_up": "↑",
            "arrow_down": "↓",
            "bullet": "•",
            "star": "★",
            "heart": "♥",
            "diamond": "♦",
            "club": "♣",
            "spade": "♠",
            "warning": "⚠",
            "info": "ℹ",
            "battery_full": "🔋",
            "battery_low": "🪫",
        }

        symbol = icons.get(icon_type, icon_type)
        self.text(symbol, x, y, color, size)

    def show(self):
        """Finalize the frame and add show command."""
        self._lua_commands.append("frame.display.show()")
        self._frame_count += 1
        self._last_render_time = time.time()

    def get_lua(self) -> str:
        """
        Get the complete Lua script for current frame.

        Returns:
            Lua code string
        """
        return "\n".join(self._lua_commands)

    def set_brightness(self, brightness: float):
        """
        Set display brightness.

        Args:
            brightness: Brightness level (0.0-1.0)
        """
        self._brightness = max(0.0, min(1.0, brightness))
        self.config.brightness = self._brightness

    def get_stats(self) -> Dict[str, Any]:
        """Get renderer statistics."""
        return {
            "frame_count": self._frame_count,
            "pixel_offset": self._pixel_offset,
            "brightness": self._brightness,
            "power_mode": self.power_manager.get_mode().value if self.power_manager else "unknown",
            "last_render": self._last_render_time
        }

    # ============================================================
    # Screen Templates
    # ============================================================

    def render_thinking(
        self,
        title: str = "Analyzing...",
        elapsed_s: float = 0,
        context: str = ""
    ) -> str:
        """
        Render thinking/loading screen with pulsing indicator.

        Args:
            title: Title text
            elapsed_s: Seconds elapsed
            context: Optional context text

        Returns:
            Lua code string
        """
        self.clear()

        # Pulsing dots animation
        dots = "." * (int(elapsed_s * 2) % 4)
        self.text(f"{title}{dots}", self.center_x, self.center_y - 20, OLEDColors.PRIMARY, FONT_MEDIUM)

        # Elapsed time
        self.text(f"{elapsed_s:.1f}s", self.center_x, self.center_y + 30, OLEDColors.TEXT_DIM, FONT_SMALL)

        # Context if provided
        if context:
            self.text(context, self.center_x, self.height - 40, OLEDColors.TEXT_SECONDARY, FONT_SMALL)

        self.show()
        return self.get_lua()

    def render_toast(
        self,
        message: str,
        color: Tuple[int, int, int] = OLEDColors.PRIMARY,
        icon: str = None
    ) -> str:
        """
        Render a toast notification.

        Args:
            message: Toast message
            color: Message color
            icon: Optional icon

        Returns:
            Lua code string
        """
        self.clear()

        # Toast background (bottom of screen)
        toast_height = 60
        toast_y = self.height - toast_height

        self.rect(0, toast_y, self.width, toast_height, OLEDColors.OVERLAY, filled=True)

        # Icon if provided
        msg_x = self.center_x
        if icon:
            self.icon(icon, 40, toast_y + 20, color)
            msg_x = self.center_x + 20

        # Message
        self.text(message, msg_x, toast_y + 20, color, FONT_SMALL)

        self.show()
        return self.get_lua()

    def render_error(self, message: str, details: str = "") -> str:
        """
        Render error screen.

        Args:
            message: Error message
            details: Additional details

        Returns:
            Lua code string
        """
        self.clear()

        self.icon("warning", self.center_x, self.center_y - 50, OLEDColors.ERROR, FONT_LARGE)
        self.text("ERROR", self.center_x, self.center_y, OLEDColors.ERROR, FONT_MEDIUM)

        # Truncate message
        if len(message) > 50:
            message = message[:47] + "..."
        self.text(message, self.center_x, self.center_y + 40, OLEDColors.TEXT_SECONDARY, FONT_SMALL)

        if details:
            if len(details) > 60:
                details = details[:57] + "..."
            self.text(details, self.center_x, self.center_y + 70, OLEDColors.TEXT_DIM, FONT_TINY)

        self.show()
        return self.get_lua()

    def render_battery_warning(self, percent: int, mode: str) -> str:
        """
        Render battery warning overlay.

        Args:
            percent: Battery percentage
            mode: Current power mode

        Returns:
            Lua code string
        """
        self.clear()

        # Warning icon
        icon = "battery_low" if percent < 15 else "warning"
        color = OLEDColors.ERROR if percent < 10 else OLEDColors.WARNING

        self.icon(icon, self.center_x, self.center_y - 30, color, FONT_LARGE)
        self.text(f"Battery: {percent}%", self.center_x, self.center_y + 20, color, FONT_MEDIUM)
        self.text(f"Mode: {mode}", self.center_x, self.center_y + 60, OLEDColors.TEXT_DIM, FONT_SMALL)

        self.show()
        return self.get_lua()


# Factory function
def create_oled_renderer(
    config: dict = None,
    power_manager: PowerManager = None
) -> OLEDRenderer:
    """
    Factory function to create an OLED renderer.

    Args:
        config: Configuration dict with 'display' section
        power_manager: Optional power manager

    Returns:
        Configured OLEDRenderer instance
    """
    display_config = DisplayConfig()

    if config and "display" in config:
        dc = config["display"]
        display_config.pixel_shift_enabled = dc.get("pixel_shift_enabled", True)
        display_config.pixel_shift_interval_ms = dc.get("pixel_shift_interval_ms", 30000)
        display_config.pixel_shift_amount = dc.get("pixel_shift_amount", 2)
        display_config.dim_after_seconds = dc.get("dim_after_seconds", 60)

    return OLEDRenderer(config=display_config, power_manager=power_manager)


# Test
def test_oled_renderer():
    """Test OLED renderer functionality."""
    print("=== OLED Renderer Test ===\n")

    renderer = OLEDRenderer()

    # Test basic text
    print("Basic text:")
    renderer.clear()
    renderer.text("Hello WHAM", renderer.center_x, renderer.center_y, OLEDColors.PRIMARY)
    renderer.show()
    print(renderer.get_lua())
    print()

    # Test poker cards
    print("Poker cards:")
    renderer.clear()
    renderer.cards_row(["Ah", "Kc", "Qs", "Jd"], 100, 100)
    renderer.show()
    print(renderer.get_lua())
    print()

    # Test confidence bar
    print("Confidence bar:")
    renderer.clear()
    renderer.confidence_bar(200, 200, 0.75)
    renderer.show()
    print(renderer.get_lua())
    print()

    # Test thinking screen
    print("Thinking screen:")
    lua = renderer.render_thinking("Processing...", 2.5, "Poker analysis")
    print(lua)
    print()

    # Test pixel shift
    print("Pixel shift:")
    for i in range(3):
        offset = renderer.apply_pixel_shift()
        print(f"  Offset: {offset}")


if __name__ == "__main__":
    test_oled_renderer()
