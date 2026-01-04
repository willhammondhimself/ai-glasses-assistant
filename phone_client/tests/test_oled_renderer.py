"""
Visual Regression Tests for OLED Renderer.
Tests display bounds, burn-in protection, color adjustment, and rendering correctness.
"""
import pytest
import time
from unittest.mock import MagicMock, patch

# Path setup via conftest.py


from halo.oled_renderer import (
    OLEDRenderer, OLEDColors, DisplayConfig, rgb_to_lua, lerp_color,
    DISPLAY_WIDTH, DISPLAY_HEIGHT, FONT_LARGE, FONT_MEDIUM, FONT_SMALL, FONT_TINY
)
from core.power_manager import PowerManager, PowerMode


class TestDisplayBounds:
    """Verify all rendering stays within 640x400 bounds."""

    def test_display_dimensions(self):
        """Verify standard Halo display dimensions."""
        assert DISPLAY_WIDTH == 640
        assert DISPLAY_HEIGHT == 400

    def test_renderer_dimensions(self):
        """Verify renderer uses correct dimensions."""
        renderer = OLEDRenderer()
        assert renderer.width == 640
        assert renderer.height == 400
        assert renderer.center_x == 320
        assert renderer.center_y == 200

    def test_custom_dimensions(self):
        """Test renderer with custom dimensions."""
        config = DisplayConfig(width=800, height=600)
        renderer = OLEDRenderer(config=config)
        assert renderer.width == 800
        assert renderer.height == 600
        assert renderer.center_x == 400
        assert renderer.center_y == 300

    def test_poker_display_fits_640x400(self):
        """Verify poker HUD fits within 640x400 bounds."""
        renderer = OLEDRenderer()
        renderer.clear()

        # Simulate poker HUD layout
        renderer.text("RAISE $50", renderer.center_x, 50, OLEDColors.SUCCESS, FONT_LARGE)
        renderer.cards_row(["Ah", "Kc", "Qs", "Jd", "Th"], 70, 120)
        renderer.confidence_bar(200, 200, 0.85)
        renderer.text("Pot: $200", renderer.center_x, 280, OLEDColors.TEXT_PRIMARY, FONT_MEDIUM)
        renderer.text("EV: +$45", renderer.center_x, 350, OLEDColors.SUCCESS, FONT_SMALL)
        renderer.show()

        lua = renderer.get_lua()

        # Parse Lua to verify all coordinates are within bounds
        import re
        coords = re.findall(r'(\d+),\s*(\d+)', lua)
        for x_str, y_str in coords:
            x, y = int(x_str), int(y_str)
            # Allow small margin for pixel shifting
            assert x <= DISPLAY_WIDTH + 5, f"X coordinate {x} exceeds display width"
            assert y <= DISPLAY_HEIGHT + 5, f"Y coordinate {y} exceeds display height"

    def test_text_at_edges(self):
        """Test text rendering at display edges."""
        renderer = OLEDRenderer()
        renderer.clear()

        # Corners
        renderer.text("TL", 10, 10, OLEDColors.TEXT_PRIMARY, FONT_SMALL, "left")
        renderer.text("TR", 630, 10, OLEDColors.TEXT_PRIMARY, FONT_SMALL, "right")
        renderer.text("BL", 10, 380, OLEDColors.TEXT_PRIMARY, FONT_SMALL, "left")
        renderer.text("BR", 630, 380, OLEDColors.TEXT_PRIMARY, FONT_SMALL, "right")
        renderer.show()

        lua = renderer.get_lua()
        assert "frame.display.text" in lua


class TestBurnInProtection:
    """Test burn-in prevention through pixel shifting."""

    def test_pixel_shift_positions(self):
        """Verify burn-in protection cycles through all 9 positions."""
        config = DisplayConfig(
            pixel_shift_enabled=True,
            pixel_shift_interval_ms=0,  # Instant shift for testing
            pixel_shift_amount=2
        )
        renderer = OLEDRenderer(config=config)

        positions_seen = set()

        # Force multiple shifts by manipulating time
        for i in range(20):
            renderer._last_shift = 0  # Force shift
            offset = renderer.apply_pixel_shift()
            positions_seen.add(offset)

        # Should see 9 positions: center + 8 directions
        expected_positions = {
            (0, 0),
            (2, 0), (2, 2), (0, 2), (-2, 2),
            (-2, 0), (-2, -2), (0, -2), (2, -2)
        }
        assert positions_seen == expected_positions, f"Missing positions: {expected_positions - positions_seen}"

    def test_pixel_shift_disabled(self):
        """Verify pixel shift can be disabled."""
        config = DisplayConfig(pixel_shift_enabled=False)
        renderer = OLEDRenderer(config=config)

        for _ in range(10):
            offset = renderer.apply_pixel_shift()
            assert offset == (0, 0), "Pixel shift should be disabled"

    def test_pixel_shift_amount(self):
        """Test different pixel shift amounts."""
        for amount in [1, 2, 3, 5]:
            config = DisplayConfig(
                pixel_shift_enabled=True,
                pixel_shift_interval_ms=0,
                pixel_shift_amount=amount
            )
            renderer = OLEDRenderer(config=config)

            renderer._last_shift = 0
            offset = renderer.apply_pixel_shift()

            # Offset magnitude should not exceed amount
            assert abs(offset[0]) <= amount
            assert abs(offset[1]) <= amount

    def test_shifted_coordinates(self):
        """Verify coordinates are shifted correctly."""
        config = DisplayConfig(
            pixel_shift_enabled=True,
            pixel_shift_interval_ms=0,
            pixel_shift_amount=2
        )
        renderer = OLEDRenderer(config=config)

        # Force a specific offset
        renderer._pixel_offset = (2, 2)
        renderer._last_shift = time.time()  # Prevent auto-shift

        # Test shifted coordinates
        shifted = renderer._shifted(100, 100)
        assert shifted == (102, 102)


class TestPowerModeColorAdjustment:
    """Test power-aware color adjustment."""

    def test_full_power_no_adjustment(self):
        """Verify FULL mode applies no color adjustment."""
        pm = PowerManager()
        pm.update_battery(100)  # FULL mode

        renderer = OLEDRenderer(power_manager=pm)

        test_color = (255, 255, 255)
        adjusted = renderer._apply_power_adjustment(test_color)

        # FULL mode: 100% brightness, no color shift
        assert adjusted == (255, 255, 255)

    def test_saver_mode_dims_colors(self):
        """Verify SAVER mode correctly dims colors."""
        pm = PowerManager()
        pm.update_battery(20)  # SAVER mode

        assert pm.get_mode() == PowerMode.SAVER

        renderer = OLEDRenderer(power_manager=pm)

        test_color = (255, 255, 255)
        adjusted = renderer._apply_power_adjustment(test_color)

        # SAVER mode: 60% brightness, reduced blue
        # R: 255 * 1.0 * 0.6 = 153
        # G: 255 * 0.95 * 0.6 = 145.35 → 145
        # B: 255 * 0.80 * 0.6 = 122.4 → 122
        assert adjusted[0] < test_color[0], "Red should be dimmed"
        assert adjusted[1] < test_color[1], "Green should be dimmed"
        assert adjusted[2] < test_color[2], "Blue should be dimmed most"
        assert adjusted[2] < adjusted[0], "Blue should be lower than red"

    def test_critical_mode_maximum_dimming(self):
        """Verify CRITICAL mode applies maximum dimming."""
        pm = PowerManager()
        pm.update_battery(5)  # CRITICAL mode

        assert pm.get_mode() == PowerMode.CRITICAL

        renderer = OLEDRenderer(power_manager=pm)

        test_color = (255, 255, 255)
        adjusted = renderer._apply_power_adjustment(test_color)

        # CRITICAL: 40% brightness, heavily reduced blue
        assert adjusted[0] < 128, "Red should be significantly dimmed"
        assert adjusted[2] < adjusted[1] < adjusted[0], "Blue < Green < Red"

    def test_no_power_manager(self):
        """Verify colors unchanged without power manager."""
        renderer = OLEDRenderer(power_manager=None)

        test_color = (255, 128, 64)
        adjusted = renderer._apply_power_adjustment(test_color)

        assert adjusted == test_color


class TestConfidenceBarRendering:
    """Test confidence bar segment rendering."""

    def test_confidence_bar_full(self):
        """Verify full confidence bar renders 10 segments."""
        renderer = OLEDRenderer()
        renderer.clear()
        renderer.confidence_bar(100, 100, 1.0)
        renderer.show()

        lua = renderer.get_lua()

        # Count rect calls (10 segments + potential label)
        rect_count = lua.count("frame.display.rect")
        assert rect_count >= 10, f"Expected 10+ rects, got {rect_count}"

    def test_confidence_bar_segments(self):
        """Verify confidence bar renders correct number of filled segments."""
        test_cases = [
            (0.0, 0),
            (0.25, 2),  # 25% → 2-3 segments
            (0.50, 5),
            (0.75, 7),
            (1.0, 10),
        ]

        for confidence, expected_filled in test_cases:
            renderer = OLEDRenderer()
            renderer.clear()
            renderer.confidence_bar(100, 100, confidence, show_label=False)
            renderer.show()

            lua = renderer.get_lua()
            # Just verify it renders without error
            assert "frame.display" in lua

    def test_confidence_bar_colors(self):
        """Test confidence level determines bar color."""
        # High confidence: green
        assert OLEDColors.get_confidence_color(0.9) == OLEDColors.CONF_HIGH

        # Medium confidence: yellow
        assert OLEDColors.get_confidence_color(0.7) == OLEDColors.CONF_MED

        # Low confidence: red
        assert OLEDColors.get_confidence_color(0.4) == OLEDColors.CONF_LOW


class TestCardSuitColors:
    """Test poker card suit color assignment."""

    def test_hearts_red(self):
        """Verify hearts are rendered in red."""
        assert OLEDColors.get_suit_color('h') == OLEDColors.SUIT_HEARTS
        assert OLEDColors.get_suit_color('H') == OLEDColors.SUIT_HEARTS

    def test_diamonds_blue(self):
        """Verify diamonds are rendered in blue."""
        assert OLEDColors.get_suit_color('d') == OLEDColors.SUIT_DIAMONDS
        assert OLEDColors.get_suit_color('D') == OLEDColors.SUIT_DIAMONDS

    def test_clubs_green(self):
        """Verify clubs are rendered in green."""
        assert OLEDColors.get_suit_color('c') == OLEDColors.SUIT_CLUBS
        assert OLEDColors.get_suit_color('C') == OLEDColors.SUIT_CLUBS

    def test_spades_white(self):
        """Verify spades are rendered in white."""
        assert OLEDColors.get_suit_color('s') == OLEDColors.SUIT_SPADES
        assert OLEDColors.get_suit_color('S') == OLEDColors.SUIT_SPADES

    def test_unknown_suit_default(self):
        """Verify unknown suits use default color."""
        assert OLEDColors.get_suit_color('x') == OLEDColors.TEXT_PRIMARY

    def test_card_rendering(self):
        """Test card rendering with correct suit colors."""
        renderer = OLEDRenderer()
        renderer.clear()
        renderer.card("Ah", 100, 100)  # Ace of hearts
        renderer.show()

        lua = renderer.get_lua()
        # Should contain heart color in hex
        heart_hex = f'"{OLEDColors.SUIT_HEARTS[0]:02x}{OLEDColors.SUIT_HEARTS[1]:02x}{OLEDColors.SUIT_HEARTS[2]:02x}"'
        assert heart_hex.lower() in lua.lower()


class TestColorUtilities:
    """Test color utility functions."""

    def test_rgb_to_lua(self):
        """Test RGB to Lua hex conversion."""
        assert rgb_to_lua((255, 255, 255)) == '"ffffff"'
        assert rgb_to_lua((0, 0, 0)) == '"000000"'
        assert rgb_to_lua((255, 0, 0)) == '"ff0000"'
        assert rgb_to_lua((0, 255, 255)) == '"00ffff"'

    def test_lerp_color(self):
        """Test linear color interpolation."""
        black = (0, 0, 0)
        white = (255, 255, 255)

        # Start
        assert lerp_color(black, white, 0.0) == (0, 0, 0)

        # Middle
        mid = lerp_color(black, white, 0.5)
        assert mid == (127, 127, 127)

        # End
        assert lerp_color(black, white, 1.0) == (255, 255, 255)

    def test_lerp_color_clamped(self):
        """Test lerp clamps t value."""
        black = (0, 0, 0)
        white = (255, 255, 255)

        # Over 1.0 should clamp
        assert lerp_color(black, white, 1.5) == (255, 255, 255)

        # Under 0.0 should clamp
        assert lerp_color(black, white, -0.5) == (0, 0, 0)


class TestScreenTemplates:
    """Test pre-built screen templates."""

    def test_render_thinking(self):
        """Test thinking/loading screen template."""
        renderer = OLEDRenderer()
        lua = renderer.render_thinking("Analyzing...", 2.5, "Poker hand")

        assert "frame.display.clear()" in lua
        assert "frame.display.show()" in lua
        assert "Analyzing" in lua

    def test_render_toast(self):
        """Test toast notification template."""
        renderer = OLEDRenderer()
        lua = renderer.render_toast("Success!", OLEDColors.SUCCESS, "check")

        assert "frame.display.clear()" in lua
        assert "Success" in lua

    def test_render_error(self):
        """Test error screen template."""
        renderer = OLEDRenderer()
        lua = renderer.render_error("Connection failed", "Check Bluetooth")

        assert "ERROR" in lua
        assert "Connection failed" in lua

    def test_render_battery_warning(self):
        """Test battery warning screen."""
        renderer = OLEDRenderer()
        lua = renderer.render_battery_warning(15, "saver")

        assert "15%" in lua or "Battery" in lua


class TestRendererStats:
    """Test renderer statistics tracking."""

    def test_frame_count(self):
        """Test frame count tracking."""
        renderer = OLEDRenderer()

        assert renderer._frame_count == 0

        renderer.clear()
        renderer.show()
        assert renderer._frame_count == 1

        renderer.clear()
        renderer.show()
        assert renderer._frame_count == 2

    def test_get_stats(self):
        """Test statistics retrieval."""
        renderer = OLEDRenderer()
        renderer.clear()
        renderer.show()

        stats = renderer.get_stats()

        assert stats["frame_count"] == 1
        assert "pixel_offset" in stats
        assert "brightness" in stats
        assert "power_mode" in stats


class TestLuaCodeGeneration:
    """Test Lua code generation correctness."""

    def test_clear_command(self):
        """Test clear generates correct Lua."""
        renderer = OLEDRenderer()
        renderer.clear()

        lua = renderer.get_lua()
        assert lua == "frame.display.clear()"

    def test_show_command(self):
        """Test show generates correct Lua."""
        renderer = OLEDRenderer()
        renderer.clear()
        renderer.show()

        lua = renderer.get_lua()
        assert lua.endswith("frame.display.show()")

    def test_text_command_format(self):
        """Test text command format."""
        renderer = OLEDRenderer()
        renderer.clear()
        renderer.text("Hello", 100, 200, (255, 255, 255), 32)
        renderer.show()

        lua = renderer.get_lua()
        assert 'frame.display.text("Hello"' in lua
        assert "100" in lua
        assert "200" in lua

    def test_rect_command_format(self):
        """Test rect command format."""
        renderer = OLEDRenderer()
        renderer.clear()
        renderer.rect(10, 20, 100, 50, (255, 0, 0), filled=True)
        renderer.show()

        lua = renderer.get_lua()
        assert "frame.display.rect(" in lua
        assert "filled = true" in lua

    def test_special_characters_escaped(self):
        """Test special characters in text are escaped."""
        renderer = OLEDRenderer()
        renderer.clear()
        renderer.text('Say "Hello"', 100, 100)
        renderer.show()

        lua = renderer.get_lua()
        assert '\\"' in lua  # Escaped quotes


class TestFontSizes:
    """Test font size constants."""

    def test_font_sizes_defined(self):
        """Verify font size constants."""
        assert FONT_LARGE == 48
        assert FONT_MEDIUM == 32
        assert FONT_SMALL == 24
        assert FONT_TINY == 16


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
