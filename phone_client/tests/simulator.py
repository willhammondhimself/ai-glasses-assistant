#!/usr/bin/env python3
"""
Halo Display Simulator - Desktop testing tool for WHAM visual elements.

Test the OLED renderer without Halo hardware using a scaled pygame window.
Supports interactive mode and automated test scenarios.

Usage:
    python -m phone_client.tests.simulator --scenario poker_hand
    python -m phone_client.tests.simulator --scenario notification
    python -m phone_client.tests.simulator --scenario power_modes
    python -m phone_client.tests.simulator --interactive
"""
import argparse
import sys
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not installed. Install with: pip install pygame")

from halo.oled_renderer import OLEDRenderer, OLEDColors, DISPLAY_WIDTH, DISPLAY_HEIGHT
from halo.animations import AnimationEngine, Easing, AnimationTimings
from halo.themes import ContextualThemes, ColorTheme
from halo.haptics import HapticPatterns, HapticManager
from core.power_manager import PowerManager, PowerMode
from core.notifications import NotificationManager, NotificationPriority, Notification, NotificationRenderer


# Display settings
SCALE = 2  # 2x scale for visibility

# Interactive simulation modes
MODES = ["idle", "poker", "math", "homework", "meeting", "code_debug"]
BATTERY_LEVELS = [100, 75, 50, 30, 15, 5]
WINDOW_WIDTH = DISPLAY_WIDTH * SCALE   # 1280
WINDOW_HEIGHT = DISPLAY_HEIGHT * SCALE  # 800
FPS = 30


class HaloSimulator:
    """
    Desktop simulator for Halo AR glasses display.

    Renders OLED display elements in a scaled pygame window for testing
    visual layouts and animations without hardware.
    """

    def __init__(self, scale: int = SCALE):
        """
        Initialize the simulator.

        Args:
            scale: Display scale factor (default 2x)
        """
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame is required for simulator. Install with: pip install pygame")

        pygame.init()
        pygame.display.set_caption("WHAM Halo Simulator")

        self.scale = scale
        self.width = DISPLAY_WIDTH
        self.height = DISPLAY_HEIGHT
        self.window_width = DISPLAY_WIDTH * scale
        self.window_height = DISPLAY_HEIGHT * scale

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 48 * scale // 2)
        self.font_medium = pygame.font.SysFont("monospace", 32 * scale // 2)
        self.font_small = pygame.font.SysFont("monospace", 24 * scale // 2)
        self.font_tiny = pygame.font.SysFont("monospace", 16 * scale // 2)

        # Components
        self.renderer = OLEDRenderer()
        self.power_manager = PowerManager()
        self.animation_engine = AnimationEngine(renderer=self.renderer)
        self.notification_manager = NotificationManager()

        self.running = True
        self._screenshot_counter = 0

    def _scale_coord(self, x: int, y: int) -> Tuple[int, int]:
        """Scale coordinates for display."""
        return (x * self.scale, y * self.scale)

    def _scale_size(self, w: int, h: int) -> Tuple[int, int]:
        """Scale size for display."""
        return (w * self.scale, h * self.scale)

    def clear(self, color: Tuple[int, int, int] = (0, 0, 0)):
        """Clear the display."""
        self.screen.fill(color)

    def draw_text(
        self,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int] = (255, 255, 255),
        size: str = "medium",
        align: str = "center"
    ):
        """
        Draw text on the simulator display.

        Args:
            text: Text content
            x: X position (in 640x400 coordinates)
            y: Y position
            color: RGB color
            size: Font size (large, medium, small, tiny)
            align: Text alignment (left, center, right)
        """
        fonts = {
            "large": self.font_large,
            "medium": self.font_medium,
            "small": self.font_small,
            "tiny": self.font_tiny
        }
        font = fonts.get(size, self.font_medium)

        surface = font.render(text, True, color)
        rect = surface.get_rect()

        sx, sy = self._scale_coord(x, y)

        if align == "center":
            rect.centerx = sx
            rect.centery = sy
        elif align == "left":
            rect.left = sx
            rect.centery = sy
        elif align == "right":
            rect.right = sx
            rect.centery = sy

        self.screen.blit(surface, rect)

    def draw_rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        color: Tuple[int, int, int],
        filled: bool = True
    ):
        """Draw a rectangle."""
        sx, sy = self._scale_coord(x, y)
        sw, sh = self._scale_size(width, height)

        if filled:
            pygame.draw.rect(self.screen, color, (sx, sy, sw, sh))
        else:
            pygame.draw.rect(self.screen, color, (sx, sy, sw, sh), 2)

    def draw_progress_bar(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        value: float,
        color: Tuple[int, int, int] = OLEDColors.PRIMARY,
        background: Tuple[int, int, int] = OLEDColors.OVERLAY
    ):
        """Draw a progress bar."""
        # Background
        self.draw_rect(x, y, width, height, background, filled=True)

        # Fill
        fill_width = int(width * max(0, min(1, value)))
        if fill_width > 0:
            self.draw_rect(x, y, fill_width, height, color, filled=True)

        # Border
        self.draw_rect(x, y, width, height, color, filled=False)

    def draw_confidence_bar(
        self,
        x: int,
        y: int,
        confidence: float,
        width: int = 120
    ):
        """Draw a segmented confidence bar."""
        color = OLEDColors.get_confidence_color(confidence)
        segment_width = width // 10
        segment_gap = 2
        bar_height = 8

        for i in range(10):
            seg_x = x + i * (segment_width + segment_gap)
            filled = i < int(confidence * 10)
            seg_color = color if filled else OLEDColors.OVERLAY
            self.draw_rect(seg_x, y, segment_width - segment_gap, bar_height, seg_color, filled=True)

        # Label
        label = f"{int(confidence * 100)}%"
        self.draw_text(label, x + width + 20, y + 4, color, "small", "left")

    def draw_card(
        self,
        card_str: str,
        x: int,
        y: int,
        size: str = "large"
    ):
        """Draw a poker card with suit color."""
        if len(card_str) < 2:
            self.draw_text(card_str, x, y, OLEDColors.TEXT_PRIMARY, size)
            return

        suit = card_str[-1].lower()
        color = OLEDColors.get_suit_color(suit)
        self.draw_text(card_str, x, y, color, size, "left")

    def draw_cards_row(
        self,
        cards: List[str],
        x: int,
        y: int,
        spacing: int = 60
    ):
        """Draw a row of poker cards."""
        for i, card in enumerate(cards):
            self.draw_card(card, x + i * spacing, y)

    def capture_screenshot(self, filename: str = None) -> str:
        """
        Capture a screenshot of the current display.

        Args:
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved screenshot
        """
        if filename is None:
            self._screenshot_counter += 1
            filename = f"screenshot_{self._screenshot_counter:03d}.png"

        screenshots_dir = Path(__file__).parent / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)

        filepath = screenshots_dir / filename
        pygame.image.save(self.screen, str(filepath))

        return str(filepath)

    def update(self):
        """Update the display."""
        pygame.display.flip()

    def handle_events(self) -> bool:
        """
        Handle pygame events.

        Returns:
            True if should continue, False if should quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_s:
                    # Screenshot
                    path = self.capture_screenshot()
                    print(f"Screenshot saved: {path}")
        return True

    def run_frame(self):
        """Run a single frame."""
        self.clock.tick(FPS)
        return self.handle_events()

    def close(self):
        """Close the simulator."""
        pygame.quit()


# ============================================================
# Test Scenarios
# ============================================================

def scenario_poker_hand(sim: HaloSimulator):
    """Display a poker hand analysis screen."""
    sim.clear()

    # Title/Action recommendation
    sim.draw_text("RAISE $50", sim.width // 2, 50, OLEDColors.SUCCESS, "large")

    # Player cards
    sim.draw_text("Your Hand:", 70, 100, OLEDColors.TEXT_SECONDARY, "small", "left")
    sim.draw_cards_row(["Ah", "Kc"], 70, 130)

    # Community cards
    sim.draw_text("Board:", 70, 190, OLEDColors.TEXT_SECONDARY, "small", "left")
    sim.draw_cards_row(["Qs", "Jd", "Th", "7c", "2s"], 70, 220)

    # Divider
    sim.draw_rect(40, 280, sim.width - 80, 1, OLEDColors.BORDER, filled=True)

    # Stats
    sim.draw_text("Pot: $150", 150, 310, OLEDColors.TEXT_PRIMARY, "medium", "left")
    sim.draw_text("EV: +$45.20", 400, 310, OLEDColors.SUCCESS, "medium", "left")

    # Confidence bar
    sim.draw_text("Confidence:", 70, 360, OLEDColors.TEXT_SECONDARY, "small", "left")
    sim.draw_confidence_bar(170, 356, 0.85)

    sim.update()


def scenario_mental_math(sim: HaloSimulator):
    """Display a mental math problem screen."""
    sim.clear()

    # Difficulty badge (top-left)
    sim.draw_text("D3", 40, 30, OLEDColors.ACCENT, "small", "left")

    # Timer (top-right)
    sim.draw_text("2.4s", sim.width - 40, 30, OLEDColors.WARNING, "small", "right")
    sim.draw_progress_bar(sim.width - 150, 50, 110, 8, 0.6, OLEDColors.WARNING)

    # Main problem
    sim.draw_text("47 x 83 = ?", sim.width // 2, sim.height // 2, OLEDColors.TEXT_PRIMARY, "large")

    # Streak counter (bottom-left)
    sim.draw_text("Streak: 7", 40, sim.height - 40, OLEDColors.SUCCESS, "small", "left")

    sim.update()


def scenario_notification(sim: HaloSimulator):
    """Display notification examples."""
    sim.clear()

    # Main content (simulating underlying screen)
    sim.draw_text("WHAM", sim.width // 2, sim.height // 2 - 40, OLEDColors.PRIMARY, "large")
    sim.draw_text("Ready", sim.width // 2, sim.height // 2 + 20, OLEDColors.TEXT_DIM, "small")

    # Toast notification at bottom
    toast_height = 80
    toast_y = sim.height - toast_height

    # Toast background
    sim.draw_rect(0, toast_y, sim.width, toast_height, (30, 30, 35), filled=True)

    # Color accent bar
    sim.draw_rect(0, toast_y, 4, toast_height, OLEDColors.WARNING, filled=True)

    # Icon
    sim.draw_text("⚠️", 30, toast_y + 30, OLEDColors.WARNING, "medium", "left")

    # Title and message
    sim.draw_text("Battery Low: 15%", 60, toast_y + 20, OLEDColors.WARNING, "small", "left")
    sim.draw_text("Consider saving your session", 60, toast_y + 50, OLEDColors.TEXT_SECONDARY, "tiny", "left")

    sim.update()


def scenario_power_modes(sim: HaloSimulator):
    """Display power mode comparison."""
    modes = [
        (PowerMode.FULL, "FULL (100%)", 100),
        (PowerMode.BALANCED, "BALANCED (85%)", 50),
        (PowerMode.SAVER, "SAVER (60%)", 20),
        (PowerMode.CRITICAL, "CRITICAL (40%)", 5),
    ]

    sim.clear()
    sim.draw_text("Power Mode Comparison", sim.width // 2, 30, OLEDColors.PRIMARY, "medium")

    y_offset = 70
    row_height = 85

    test_color = (0, 255, 255)  # Cyan

    for mode, label, battery in modes:
        # Update power manager
        sim.power_manager.update_battery(battery)
        adjusted = sim.power_manager.apply_color_adjustment(test_color)

        # Mode label (left side)
        sim.draw_text(label, 40, y_offset + 15, OLEDColors.TEXT_PRIMARY, "small", "left")

        # Color sample (original) - centered area
        sim.draw_rect(240, y_offset, 45, 35, test_color, filled=True)
        sim.draw_text("→", 300, y_offset + 15, OLEDColors.TEXT_DIM, "small")

        # Color sample (adjusted)
        sim.draw_rect(330, y_offset, 45, 35, adjusted, filled=True)

        # RGB values (right side, fits within bounds)
        r, g, b = adjusted
        rgb_text = f"({r}, {g}, {b})"
        sim.draw_text(rgb_text, 400, y_offset + 15, OLEDColors.TEXT_SECONDARY, "tiny", "left")

        # Brightness indicator
        brightness = sim.power_manager.get_brightness()
        sim.draw_text(f"{int(brightness*100)}%", 550, y_offset + 15, OLEDColors.TEXT_DIM, "tiny", "left")

        y_offset += row_height

    sim.update()


def scenario_animations(sim: HaloSimulator):
    """Display animation timing demonstration."""
    sim.clear()
    sim.draw_text("Animation Demo", sim.width // 2, 30, OLEDColors.PRIMARY, "medium")

    # Show easing function curves
    from halo.animations import ease_linear, ease_in, ease_out, ease_in_out, ease_bounce

    easings = [
        ("Linear", ease_linear, OLEDColors.TEXT_SECONDARY),
        ("Ease-In", ease_in, OLEDColors.ERROR),
        ("Ease-Out", ease_out, OLEDColors.SUCCESS),
        ("Ease-In-Out", ease_in_out, OLEDColors.PRIMARY),
        ("Bounce", ease_bounce, OLEDColors.WARNING),
    ]

    graph_x = 100
    graph_y = 80
    graph_width = 200
    graph_height = 100

    for i, (name, func, color) in enumerate(easings):
        col = i % 2
        row = i // 2

        x_offset = col * 300
        y_offset = row * 120

        # Label
        sim.draw_text(name, graph_x + x_offset + graph_width // 2, graph_y + y_offset - 10, color, "small")

        # Graph background
        sim.draw_rect(graph_x + x_offset, graph_y + y_offset, graph_width, graph_height, OLEDColors.OVERLAY, filled=True)
        sim.draw_rect(graph_x + x_offset, graph_y + y_offset, graph_width, graph_height, OLEDColors.BORDER, filled=False)

        # Plot curve
        prev_x, prev_y = None, None
        for t in range(21):
            t_norm = t / 20
            value = func(t_norm)

            px = graph_x + x_offset + int(t_norm * graph_width)
            py = graph_y + y_offset + graph_height - int(value * graph_height)

            if prev_x is not None:
                # Can't draw lines directly, so draw small rects
                pass

            # Draw point
            sim.draw_rect(px - 1, py - 1, 3, 3, color, filled=True)

            prev_x, prev_y = px, py

    sim.update()


def scenario_session_summary(sim: HaloSimulator):
    """Display session summary screen."""
    sim.clear()

    # Title
    sim.draw_text("SESSION COMPLETE", sim.width // 2, 40, OLEDColors.PRIMARY, "medium")

    # Grade
    sim.draw_text("A+", sim.width // 2, 110, (255, 215, 0), "large")  # Gold

    # Divider
    sim.draw_rect(100, 160, sim.width - 200, 1, OLEDColors.BORDER, filled=True)

    # Stats
    stats = [
        ("Problems:", "24/25"),
        ("Accuracy:", "96%"),
        ("Avg Time:", "2.3s"),
        ("Best Streak:", "12"),
        ("API Cost:", "$0.45"),
    ]

    y = 190
    for label, value in stats:
        sim.draw_text(label, 180, y, OLEDColors.TEXT_SECONDARY, "small", "left")
        sim.draw_text(value, 460, y, OLEDColors.TEXT_PRIMARY, "small", "right")
        y += 35

    sim.update()


def scenario_homework(sim: HaloSimulator):
    """Display homework help screen with study-themed colors."""
    theme = ContextualThemes.HOMEWORK
    sim.clear()

    # Header with homework theme
    sim.draw_text("HOMEWORK HELP", sim.width // 2, 35, theme.primary, "medium")

    # Subject badge
    sim.draw_rect(40, 70, 80, 28, theme.accent, filled=True)
    sim.draw_text("PHYSICS", 80, 84, (0, 0, 0), "tiny")

    # Question display
    sim.draw_rect(40, 110, sim.width - 80, 100, (25, 30, 40), filled=True)
    sim.draw_rect(40, 110, sim.width - 80, 100, theme.border, filled=False)

    sim.draw_text("A ball is thrown upward at 20 m/s.", sim.width // 2, 135, theme.text_primary, "small")
    sim.draw_text("What is its maximum height?", sim.width // 2, 165, theme.text_primary, "small")
    sim.draw_text("(assume g = 10 m/s²)", sim.width // 2, 190, theme.text_dim, "tiny")

    # Solution steps
    sim.draw_text("SOLUTION", 60, 230, theme.accent, "small", "left")

    steps = [
        ("1.", "v² = u² - 2gh", theme.text_primary),
        ("2.", "0 = 400 - 20h", theme.text_primary),
        ("3.", "h = 20 meters", theme.success),
    ]

    y = 260
    for num, step, color in steps:
        sim.draw_text(num, 60, y, theme.text_dim, "small", "left")
        sim.draw_text(step, 90, y, color, "small", "left")
        y += 30

    # Confidence bar at bottom
    sim.draw_text("Answer confidence:", 60, 365, theme.text_dim, "tiny", "left")
    sim.draw_confidence_bar(200, 361, 0.92)

    sim.update()


def scenario_meeting(sim: HaloSimulator):
    """Display meeting mode screen with professional amber theme."""
    theme = ContextualThemes.MEETING
    sim.clear()

    # Header
    sim.draw_text("MEETING MODE", sim.width // 2, 35, theme.primary, "medium")

    # Meeting info card
    sim.draw_rect(40, 70, sim.width - 80, 80, (30, 25, 20), filled=True)
    sim.draw_rect(40, 70, 4, 80, theme.accent, filled=True)  # Accent bar

    sim.draw_text("Q3 Product Review", 60, 95, theme.text_primary, "small", "left")
    sim.draw_text("Sarah Chen, Mike Johnson + 3", 60, 120, theme.text_dim, "tiny", "left")
    sim.draw_text("25:14 remaining", sim.width - 60, 95, theme.warning, "small", "right")

    # Key points tracked
    sim.draw_text("KEY POINTS", 60, 175, theme.accent, "small", "left")

    points = [
        "• Launch date confirmed: March 15",
        "• Budget increased 15% for Q4",
        "• Action: Follow up with design team",
    ]

    y = 205
    for point in points:
        color = theme.success if "Action:" in point else theme.text_primary
        sim.draw_text(point, 60, y, color, "tiny", "left")
        y += 28

    # Speaker indicator
    sim.draw_rect(40, 300, sim.width - 80, 50, theme.overlay, filled=True)
    sim.draw_text("🎤 Sarah is speaking...", sim.width // 2, 325, theme.primary, "small")

    # Bottom status
    sim.draw_text("Transcribing", 60, 375, theme.text_dim, "tiny", "left")
    sim.draw_text("0.12 $/min", sim.width - 60, 375, theme.text_dim, "tiny", "right")

    sim.update()


def scenario_code_debug(sim: HaloSimulator):
    """Display code debug mode with IDE green theme."""
    theme = ContextualThemes.CODE_DEBUG
    sim.clear()

    # Header
    sim.draw_text("CODE DEBUG", sim.width // 2, 35, theme.primary, "medium")

    # File context
    sim.draw_text("auth_service.py:142", 60, 70, theme.accent, "small", "left")
    sim.draw_text("TypeError", sim.width - 60, 70, theme.danger, "small", "right")

    # Code snippet
    code_bg = (20, 25, 20)
    sim.draw_rect(40, 95, sim.width - 80, 120, code_bg, filled=True)
    sim.draw_rect(40, 95, sim.width - 80, 120, theme.border, filled=False)

    # Line numbers and code
    lines = [
        ("140", "def validate_token(token):", theme.text_dim),
        ("141", "    decoded = jwt.decode(token)", theme.text_primary),
        ("142", "    return decoded.user_id", theme.danger),  # Error line
        ("143", "                  ^^^^^^^", theme.danger),
    ]

    y = 115
    for num, code, color in lines:
        sim.draw_text(num, 55, y, (80, 80, 80), "tiny", "left")
        # Highlight error line
        if color == theme.danger:
            sim.draw_rect(85, y - 10, sim.width - 130, 22, (50, 20, 20), filled=True)
        sim.draw_text(code, 90, y, color, "tiny", "left")
        y += 25

    # Error explanation
    sim.draw_text("ANALYSIS", 60, 235, theme.accent, "small", "left")
    sim.draw_rect(40, 255, sim.width - 80, 70, theme.overlay, filled=True)

    sim.draw_text("'decoded' returns dict, not object.", 60, 275, theme.text_primary, "tiny", "left")
    sim.draw_text("Use decoded['user_id'] instead.", 60, 300, theme.success, "tiny", "left")

    # Quick fix button
    sim.draw_rect(200, 340, 240, 40, theme.primary, filled=True)
    sim.draw_text("Apply Fix (SPACE)", sim.width // 2, 360, (0, 0, 0), "small")

    sim.update()


def scenario_idle(sim: HaloSimulator):
    """Display idle/ready state with minimal WHAM branding."""
    theme = ContextualThemes.IDLE
    sim.clear()

    # Centered WHAM logo
    sim.draw_text("WHAM", sim.width // 2, sim.height // 2 - 30, theme.primary, "large")
    sim.draw_text("Will Hammond's Augmented Mind", sim.width // 2, sim.height // 2 + 20, theme.text_dim, "tiny")

    # Status bar at bottom
    sim.draw_rect(0, sim.height - 50, sim.width, 50, (15, 15, 20), filled=True)

    # Battery indicator
    battery = 75
    battery_color = OLEDColors.SUCCESS if battery > 30 else OLEDColors.WARNING
    sim.draw_text(f"🔋 {battery}%", 60, sim.height - 25, battery_color, "tiny", "left")

    # Time
    sim.draw_text("Ready", sim.width // 2, sim.height - 25, theme.text_dim, "tiny")

    # Mode hint
    sim.draw_text("Say 'Hey WHAM' or tap", sim.width - 60, sim.height - 25, theme.text_dim, "tiny", "right")

    sim.update()


def run_scenario(scenario_name: str, sim: HaloSimulator):
    """Run a specific test scenario."""
    scenarios = {
        "poker_hand": scenario_poker_hand,
        "mental_math": scenario_mental_math,
        "notification": scenario_notification,
        "power_modes": scenario_power_modes,
        "animations": scenario_animations,
        "session_summary": scenario_session_summary,
        "homework": scenario_homework,
        "meeting": scenario_meeting,
        "code_debug": scenario_code_debug,
        "idle": scenario_idle,
    }

    if scenario_name not in scenarios:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available: {', '.join(scenarios.keys())}")
        return

    scenarios[scenario_name](sim)

    # Wait for close
    print(f"Running scenario: {scenario_name}")
    print("Press S to screenshot, ESC to quit")

    while sim.run_frame():
        pass


def run_interactive(sim: HaloSimulator):
    """Run interactive mode with keyboard controls for testing without hardware."""
    # All scenarios mapped to number keys
    scenarios = [
        ("1", "idle", scenario_idle),
        ("2", "poker_hand", scenario_poker_hand),
        ("3", "mental_math", scenario_mental_math),
        ("4", "homework", scenario_homework),
        ("5", "meeting", scenario_meeting),
        ("6", "code_debug", scenario_code_debug),
        ("7", "notification", scenario_notification),
        ("8", "power_modes", scenario_power_modes),
        ("9", "session_summary", scenario_session_summary),
        ("0", "animations", scenario_animations),
    ]

    # Mode-specific scenarios for cycling
    mode_scenarios = {
        "idle": scenario_idle,
        "poker": scenario_poker_hand,
        "math": scenario_mental_math,
        "homework": scenario_homework,
        "meeting": scenario_meeting,
        "code_debug": scenario_code_debug,
    }

    current_idx = 0
    current_mode_idx = 0
    battery_idx = 0
    show_voice_prompt = False
    show_tap_feedback = False
    tap_feedback_time = 0

    print("\n" + "="*60)
    print("WHAM Halo Simulator - Interactive Mode")
    print("="*60)
    print("\nGLASSES SIMULATION CONTROLS:")
    print("  SPACE     - Tap gesture (confirm/select)")
    print("  V         - Voice command prompt")
    print("  M         - Cycle modes (idle→poker→math→homework→meeting→debug)")
    print("  B         - Cycle battery levels (100→75→50→30→15→5)")
    print("  N         - Trigger test notification")
    print("\nQUICK SCENARIO ACCESS:")
    print("  1-0       - Jump to specific scenario")
    print("  ←/→       - Navigate scenarios")
    print("\nUTILITY:")
    print("  S         - Save screenshot")
    print("  H         - Show this help")
    print("  ESC       - Quit")
    print("="*60 + "\n")

    scenarios[current_idx][2](sim)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return

                # Screenshot
                elif event.key == pygame.K_s:
                    path = sim.capture_screenshot(f"{scenarios[current_idx][1]}.png")
                    print(f"📸 Screenshot saved: {path}")

                # Navigate scenarios
                elif event.key == pygame.K_LEFT:
                    current_idx = (current_idx - 1) % len(scenarios)
                    scenarios[current_idx][2](sim)
                    print(f"◀ Scenario: {scenarios[current_idx][1]}")

                elif event.key == pygame.K_RIGHT:
                    current_idx = (current_idx + 1) % len(scenarios)
                    scenarios[current_idx][2](sim)
                    print(f"▶ Scenario: {scenarios[current_idx][1]}")

                # Number keys for quick scenario access
                elif event.key >= pygame.K_0 and event.key <= pygame.K_9:
                    if event.key == pygame.K_0:
                        idx = 9  # 0 maps to index 9
                    else:
                        idx = event.key - pygame.K_1  # 1-9 map to 0-8
                    if idx < len(scenarios):
                        current_idx = idx
                        scenarios[current_idx][2](sim)
                        print(f"⏩ Jumped to: {scenarios[current_idx][1]}")

                # SPACE - Tap gesture simulation
                elif event.key == pygame.K_SPACE:
                    show_tap_feedback = True
                    tap_feedback_time = time.time()
                    print("👆 TAP - Confirm/Select")
                    # Draw tap feedback overlay
                    _draw_tap_feedback(sim)

                # V - Voice command prompt
                elif event.key == pygame.K_v:
                    show_voice_prompt = not show_voice_prompt
                    if show_voice_prompt:
                        _draw_voice_prompt(sim)
                        print("🎤 Listening... (press V again to cancel)")
                    else:
                        scenarios[current_idx][2](sim)
                        print("🎤 Voice cancelled")

                # M - Cycle through modes
                elif event.key == pygame.K_m:
                    current_mode_idx = (current_mode_idx + 1) % len(MODES)
                    mode = MODES[current_mode_idx]
                    print(f"🔄 Mode: {mode.upper()}")

                    # Update to show the mode's scenario
                    if mode in mode_scenarios:
                        mode_scenarios[mode](sim)
                        # Also update current_idx to match
                        for i, (_, name, _) in enumerate(scenarios):
                            if name == mode or (mode == "poker" and name == "poker_hand") or \
                               (mode == "math" and name == "mental_math") or \
                               (mode == "debug" and name == "code_debug"):
                                current_idx = i
                                break

                # B - Cycle battery levels
                elif event.key == pygame.K_b:
                    battery_idx = (battery_idx + 1) % len(BATTERY_LEVELS)
                    battery = BATTERY_LEVELS[battery_idx]
                    sim.power_manager.update_battery(battery)
                    mode_name = sim.power_manager.get_mode().value
                    print(f"🔋 Battery: {battery}% (Power mode: {mode_name})")
                    # Refresh current scenario with new power settings
                    scenarios[current_idx][2](sim)

                # N - Trigger notification
                elif event.key == pygame.K_n:
                    _draw_notification_overlay(sim, scenarios[current_idx][2])
                    print("🔔 Notification triggered")

                # H - Show help
                elif event.key == pygame.K_h:
                    print("\n" + "="*60)
                    print("KEYBOARD CONTROLS:")
                    print("  SPACE - Tap gesture    M - Cycle modes")
                    print("  V - Voice prompt       B - Cycle battery")
                    print("  N - Notification       1-0 - Jump to scenario")
                    print("  ←/→ - Navigate         S - Screenshot")
                    print("  H - This help          ESC - Quit")
                    print("="*60 + "\n")

        # Clear tap feedback after 200ms
        if show_tap_feedback and time.time() - tap_feedback_time > 0.2:
            show_tap_feedback = False
            scenarios[current_idx][2](sim)

        sim.clock.tick(FPS)


def _draw_tap_feedback(sim: HaloSimulator):
    """Draw a tap confirmation overlay."""
    # Flash effect - brief cyan ring in center
    cx, cy = sim.width // 2, sim.height // 2
    sim.draw_rect(cx - 30, cy - 30, 60, 60, OLEDColors.PRIMARY, filled=False)
    sim.draw_rect(cx - 28, cy - 28, 56, 56, OLEDColors.PRIMARY, filled=False)
    sim.update()


def _draw_voice_prompt(sim: HaloSimulator):
    """Draw voice command prompt overlay."""
    sim.clear()

    # Listening indicator
    sim.draw_text("🎤", sim.width // 2, sim.height // 2 - 60, OLEDColors.PRIMARY, "large")
    sim.draw_text("Listening...", sim.width // 2, sim.height // 2, OLEDColors.TEXT_PRIMARY, "medium")
    sim.draw_text("Say a command", sim.width // 2, sim.height // 2 + 40, OLEDColors.TEXT_DIM, "small")

    # Example commands
    examples = [
        "\"What's my hand strength?\"",
        "\"Start mental math\"",
        "\"Help with this problem\"",
    ]
    y = sim.height // 2 + 100
    for ex in examples:
        sim.draw_text(ex, sim.width // 2, y, OLEDColors.TEXT_SECONDARY, "tiny")
        y += 25

    sim.update()


def _draw_notification_overlay(sim: HaloSimulator, restore_func):
    """Draw a notification overlay, then restore previous screen."""
    # Draw notification
    toast_height = 80
    toast_y = sim.height - toast_height

    # Toast background
    sim.draw_rect(0, toast_y, sim.width, toast_height, (30, 30, 35), filled=True)
    sim.draw_rect(0, toast_y, 4, toast_height, OLEDColors.SUCCESS, filled=True)

    # Content
    sim.draw_text("✓", 30, toast_y + 30, OLEDColors.SUCCESS, "medium", "left")
    sim.draw_text("Streak: 8!", 60, toast_y + 20, OLEDColors.SUCCESS, "small", "left")
    sim.draw_text("You're on fire! Keep it up.", 60, toast_y + 50, OLEDColors.TEXT_SECONDARY, "tiny", "left")

    sim.update()


def main():
    """Main entry point."""
    all_scenarios = [
        "idle", "poker_hand", "mental_math", "homework", "meeting",
        "code_debug", "notification", "power_modes", "animations", "session_summary"
    ]

    parser = argparse.ArgumentParser(
        description="WHAM Halo Display Simulator - Test AR glasses UI without hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive Mode Controls:
  SPACE     Tap gesture (confirm/select)
  V         Voice command prompt
  M         Cycle modes (idle→poker→math→homework→meeting→debug)
  B         Cycle battery levels
  N         Trigger notification
  1-0       Jump to scenario
  ←/→       Navigate scenarios
  S         Screenshot
  H         Help
  ESC       Quit

Examples:
  python simulator.py                    # Interactive mode (default)
  python simulator.py --scenario homework
  python simulator.py --screenshot       # Capture all scenarios
        """
    )
    parser.add_argument(
        "--scenario",
        choices=all_scenarios,
        help="Run a specific test scenario"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (default)"
    )
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="Capture screenshots of all scenarios and exit"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Display scale factor (default: 2)"
    )

    args = parser.parse_args()

    if not PYGAME_AVAILABLE:
        print("Error: pygame is required. Install with: pip install pygame")
        sys.exit(1)

    sim = HaloSimulator(scale=args.scale)

    try:
        if args.screenshot:
            # Capture all scenarios
            for scenario in all_scenarios:
                run_scenario(scenario, sim)
                path = sim.capture_screenshot(f"{scenario}.png")
                print(f"Captured: {path}")
        elif args.scenario:
            run_scenario(args.scenario, sim)
        elif args.interactive:
            run_interactive(sim)
        else:
            # Default to interactive
            run_interactive(sim)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
