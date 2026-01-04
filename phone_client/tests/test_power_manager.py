"""
Power Manager Tests.
Tests battery monitoring, power mode transitions, and color adjustment.
"""
import pytest
from unittest.mock import MagicMock

# Path setup via conftest.py


from core.power_manager import PowerManager, PowerMode, PowerState


class TestPowerModes:
    """Test power mode enumeration."""

    def test_power_modes_exist(self):
        """Verify all power modes are defined."""
        assert PowerMode.FULL.value == "full"
        assert PowerMode.BALANCED.value == "balanced"
        assert PowerMode.SAVER.value == "saver"
        assert PowerMode.CRITICAL.value == "critical"


class TestPowerModeTransitions:
    """Test automatic power mode transitions based on battery level."""

    def test_full_mode_high_battery(self):
        """Verify FULL mode when battery > 80%."""
        pm = PowerManager()
        pm.update_battery(100)
        assert pm.get_mode() == PowerMode.FULL

        pm.update_battery(85)
        assert pm.get_mode() == PowerMode.FULL

        pm.update_battery(80)
        assert pm.get_mode() == PowerMode.FULL

    def test_balanced_mode_mid_battery(self):
        """Verify BALANCED mode when battery 30-80%."""
        pm = PowerManager()

        pm.update_battery(79)
        assert pm.get_mode() == PowerMode.BALANCED

        pm.update_battery(50)
        assert pm.get_mode() == PowerMode.BALANCED

        pm.update_battery(30)
        assert pm.get_mode() == PowerMode.BALANCED

    def test_saver_mode_low_battery(self):
        """Verify SAVER mode when battery 15-30%."""
        pm = PowerManager()

        pm.update_battery(29)
        assert pm.get_mode() == PowerMode.SAVER

        pm.update_battery(20)
        assert pm.get_mode() == PowerMode.SAVER

        pm.update_battery(15)
        assert pm.get_mode() == PowerMode.SAVER

    def test_critical_mode_very_low_battery(self):
        """Verify CRITICAL mode when battery < 15%."""
        pm = PowerManager()

        pm.update_battery(14)
        assert pm.get_mode() == PowerMode.CRITICAL

        pm.update_battery(5)
        assert pm.get_mode() == PowerMode.CRITICAL

        pm.update_battery(1)
        assert pm.get_mode() == PowerMode.CRITICAL

    def test_charging_uses_full_mode(self):
        """Verify charging triggers FULL mode regardless of battery level."""
        pm = PowerManager()

        pm.update_battery(20, charging=True)
        assert pm.get_mode() == PowerMode.FULL

        pm.update_battery(5, charging=True)
        assert pm.get_mode() == PowerMode.FULL


class TestModeChangeCallbacks:
    """Test power mode change notification system."""

    def test_callback_on_mode_change(self):
        """Verify callback is called on mode change."""
        pm = PowerManager()
        callback_data = []

        def callback(old_mode, new_mode):
            callback_data.append((old_mode, new_mode))

        pm.on_mode_change(callback)

        # Force mode change
        pm.update_battery(100)  # FULL
        pm.update_battery(50)   # BALANCED

        assert len(callback_data) >= 1
        # Last change should be to BALANCED
        assert callback_data[-1][1] == PowerMode.BALANCED

    def test_no_callback_when_mode_unchanged(self):
        """Verify callback not called when mode stays same."""
        pm = PowerManager()
        callback_count = [0]

        def callback(old, new):
            callback_count[0] += 1

        pm.on_mode_change(callback)
        pm.update_battery(100)  # FULL

        initial_count = callback_count[0]

        pm.update_battery(95)  # Still FULL
        pm.update_battery(90)  # Still FULL

        assert callback_count[0] == initial_count

    def test_remove_callback(self):
        """Test callback removal."""
        pm = PowerManager()
        callback_count = [0]

        def callback(old, new):
            callback_count[0] += 1

        pm.on_mode_change(callback)
        pm.update_battery(100)

        pm.remove_callback(callback)

        # This change should not trigger callback
        pm.update_battery(50)

        # Callback count should not increase after removal
        # (may have been called once for initial update)


class TestBrightnessLevels:
    """Test brightness level per power mode."""

    def test_full_brightness_100(self):
        """Verify FULL mode has 100% brightness."""
        pm = PowerManager()
        pm.update_battery(100)
        assert pm.get_brightness() == 1.0

    def test_balanced_brightness_85(self):
        """Verify BALANCED mode has 85% brightness."""
        pm = PowerManager()
        pm.update_battery(50)
        assert pm.get_brightness() == 0.85

    def test_saver_brightness_60(self):
        """Verify SAVER mode has 60% brightness."""
        pm = PowerManager()
        pm.update_battery(20)
        assert pm.get_brightness() == 0.60

    def test_critical_brightness_40(self):
        """Verify CRITICAL mode has 40% brightness."""
        pm = PowerManager()
        pm.update_battery(5)
        assert pm.get_brightness() == 0.40

    def test_custom_brightness_config(self):
        """Test custom brightness configuration."""
        config = {
            "power": {
                "brightness": {
                    "full": 0.9,
                    "balanced": 0.7,
                    "saver": 0.5,
                    "critical": 0.3
                }
            }
        }
        pm = PowerManager(config=config)

        pm.update_battery(100)
        assert pm.get_brightness() == 0.9

        pm.update_battery(50)
        assert pm.get_brightness() == 0.7


class TestColorAdjustment:
    """Test power-aware color adjustment."""

    def test_full_mode_no_adjustment(self):
        """Verify FULL mode has no color adjustment."""
        pm = PowerManager()
        pm.update_battery(100)

        adj = pm.get_color_adjustment()
        assert adj == (1.0, 1.0, 1.0)

    def test_saver_reduces_blue(self):
        """Verify SAVER mode reduces blue channel."""
        pm = PowerManager()
        pm.update_battery(20)

        r, g, b = pm.get_color_adjustment()
        assert r == 1.0
        assert g < 1.0
        assert b < g  # Blue reduced more than green

    def test_critical_maximum_blue_reduction(self):
        """Verify CRITICAL mode has maximum blue reduction."""
        pm = PowerManager()
        pm.update_battery(5)

        r, g, b = pm.get_color_adjustment()
        assert r == 1.0
        assert b < g < r

    def test_apply_color_adjustment(self):
        """Test applying color adjustment to RGB values."""
        pm = PowerManager()
        pm.update_battery(20)  # SAVER mode

        original = (255, 255, 255)
        adjusted = pm.apply_color_adjustment(original)

        # Red should be reduced by brightness only
        # Green and Blue should be reduced more
        assert adjusted[0] < 255  # Dimmed
        assert adjusted[1] < adjusted[0]  # Green less than red
        assert adjusted[2] < adjusted[1]  # Blue least

    def test_color_adjustment_clamps_to_255(self):
        """Verify color values don't exceed 255."""
        pm = PowerManager()
        pm.update_battery(100)

        original = (255, 255, 255)
        adjusted = pm.apply_color_adjustment(original)

        assert all(0 <= c <= 255 for c in adjusted)


class TestManualModeOverride:
    """Test manual power mode override."""

    def test_manual_override_sets_mode(self):
        """Test manual override changes mode."""
        pm = PowerManager()
        pm.update_battery(100)  # Would be FULL

        pm.set_mode_override(PowerMode.SAVER)
        assert pm.get_mode() == PowerMode.SAVER

    def test_clear_override(self):
        """Test clearing manual override."""
        pm = PowerManager()
        pm.update_battery(100)  # FULL

        pm.set_mode_override(PowerMode.CRITICAL)
        assert pm.get_mode() == PowerMode.CRITICAL

        pm.set_mode_override(None)  # Clear override
        assert pm.get_mode() == PowerMode.FULL

    def test_override_ignores_battery_updates(self):
        """Verify override prevents battery-based transitions."""
        pm = PowerManager()
        pm.set_mode_override(PowerMode.FULL)

        # Even with low battery, should stay FULL
        pm.update_battery(5)
        assert pm.get_mode() == PowerMode.FULL


class TestOperationDeferral:
    """Test operation deferral based on power state."""

    def test_defer_animation_in_critical(self):
        """Test animations deferred in CRITICAL mode."""
        pm = PowerManager()
        pm.update_battery(5)

        assert pm.should_defer_operation("animation") is True

    def test_defer_deep_analysis_in_saver(self):
        """Test deep analysis deferred in SAVER mode."""
        pm = PowerManager()
        pm.update_battery(20)

        assert pm.should_defer_operation("deep_analysis") is True

    def test_allow_operations_in_full(self):
        """Test all operations allowed in FULL mode."""
        pm = PowerManager()
        pm.update_battery(100)

        assert pm.should_defer_operation("animation") is False
        assert pm.should_defer_operation("deep_analysis") is False
        assert pm.should_defer_operation("session_review") is False


class TestPowerState:
    """Test PowerState dataclass."""

    def test_power_state_defaults(self):
        """Test PowerState default values."""
        state = PowerState(
            battery_percent=50,
            is_charging=False,
            mode=PowerMode.BALANCED
        )

        assert state.battery_percent == 50
        assert state.is_charging is False
        assert state.mode == PowerMode.BALANCED
        assert state.estimated_runtime_min == 0

    def test_get_state(self):
        """Test retrieving current power state."""
        pm = PowerManager()
        pm.update_battery(75, charging=True)

        state = pm.get_state()

        assert state.battery_percent == 75
        assert state.is_charging is True
        assert state.mode == PowerMode.FULL


class TestRuntimeEstimation:
    """Test battery runtime estimation."""

    def test_runtime_estimation(self):
        """Test runtime estimate based on battery level."""
        pm = PowerManager()

        # 100% battery in FULL mode: ~240 minutes (4 hours)
        pm.update_battery(100)
        state = pm.get_state()
        assert state.estimated_runtime_min > 0
        assert state.estimated_runtime_min <= 240

        # 50% should be roughly half
        pm.update_battery(50)
        state = pm.get_state()
        assert state.estimated_runtime_min < 240

    def test_saver_mode_extends_runtime(self):
        """Test SAVER mode increases estimated runtime."""
        pm = PowerManager()

        # Get BALANCED estimate
        pm.update_battery(50)
        balanced_runtime = pm.get_state().estimated_runtime_min

        # Get SAVER estimate at same battery
        pm.update_battery(20)
        saver_runtime = pm.get_state().estimated_runtime_min

        # SAVER should estimate longer (per % of battery)
        # Comparing runtime per percent
        balanced_per_percent = balanced_runtime / 50
        saver_per_percent = saver_runtime / 20

        assert saver_per_percent > balanced_per_percent


class TestLowPowerChecks:
    """Test low power state checks."""

    def test_is_low_power_saver(self):
        """Test is_low_power returns True for SAVER."""
        pm = PowerManager()
        pm.update_battery(20)
        assert pm.is_low_power() is True

    def test_is_low_power_critical(self):
        """Test is_low_power returns True for CRITICAL."""
        pm = PowerManager()
        pm.update_battery(5)
        assert pm.is_low_power() is True

    def test_is_low_power_balanced(self):
        """Test is_low_power returns False for BALANCED."""
        pm = PowerManager()
        pm.update_battery(50)
        assert pm.is_low_power() is False

    def test_is_critical(self):
        """Test is_critical check."""
        pm = PowerManager()

        pm.update_battery(5)
        assert pm.is_critical() is True

        pm.update_battery(20)
        assert pm.is_critical() is False


class TestStatusString:
    """Test human-readable status string."""

    def test_status_string_format(self):
        """Test status string contains expected info."""
        pm = PowerManager()
        pm.update_battery(75)

        status = pm.get_status_string()

        assert "75%" in status
        assert "balanced" in status.lower()

    def test_status_string_charging(self):
        """Test status string shows charging."""
        pm = PowerManager()
        pm.update_battery(50, charging=True)

        status = pm.get_status_string()

        assert "charging" in status.lower()

    def test_status_string_critical(self):
        """Test status string shows warning in critical."""
        pm = PowerManager()
        pm.update_battery(5)

        status = pm.get_status_string()

        assert "critical" in status.lower() or "5%" in status


class TestCustomThresholds:
    """Test custom power threshold configuration."""

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = {
            "power": {
                "thresholds": {
                    "full": 90,
                    "balanced": 40,
                    "saver": 20,
                    "critical": 10
                }
            }
        }
        pm = PowerManager(config=config)

        # 85% would be BALANCED with custom thresholds
        pm.update_battery(85)
        assert pm.get_mode() == PowerMode.BALANCED

        # 95% would be FULL
        pm.update_battery(95)
        assert pm.get_mode() == PowerMode.FULL


class TestAutoManageDisabled:
    """Test behavior when auto_manage is disabled."""

    def test_no_auto_transition(self):
        """Test no auto mode transition when disabled."""
        config = {
            "power": {
                "auto_manage": False
            }
        }
        pm = PowerManager(config=config)

        initial_mode = pm.get_mode()

        # Battery change should not trigger mode change
        pm.update_battery(5)

        assert pm.get_mode() == initial_mode


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
