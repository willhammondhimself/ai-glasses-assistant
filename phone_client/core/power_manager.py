"""
Power Manager - Battery-aware display and processing management.
Automatically adjusts brightness and features based on battery level.
"""
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PowerMode(Enum):
    """Power modes with associated brightness levels."""
    FULL = "full"           # 100% - plugged in or >80% battery
    BALANCED = "balanced"   # 85% - 30-80% battery (default)
    SAVER = "saver"         # 60% - 15-30% battery
    CRITICAL = "critical"   # 40% - <15% battery


@dataclass
class PowerState:
    """Current power state information."""
    battery_percent: int
    is_charging: bool
    mode: PowerMode
    estimated_runtime_min: int = 0
    last_update: float = field(default_factory=time.time)


class PowerManager:
    """
    Battery-aware power management for Halo glasses.

    Automatically transitions between power modes based on battery level
    and provides brightness/color adjustments for power saving.
    """

    # Default thresholds (can be overridden by config)
    DEFAULT_THRESHOLDS = {
        "full": 80,
        "balanced": 30,
        "saver": 15,
        "critical": 5
    }

    # Default brightness levels per mode
    DEFAULT_BRIGHTNESS = {
        PowerMode.FULL: 1.0,
        PowerMode.BALANCED: 0.85,
        PowerMode.SAVER: 0.60,
        PowerMode.CRITICAL: 0.40
    }

    # RGB multipliers for power saving (reduce blue to save power on OLED)
    COLOR_ADJUSTMENTS = {
        PowerMode.FULL: (1.0, 1.0, 1.0),
        PowerMode.BALANCED: (1.0, 1.0, 0.95),
        PowerMode.SAVER: (1.0, 0.95, 0.80),
        PowerMode.CRITICAL: (1.0, 0.90, 0.70)
    }

    def __init__(self, config: dict = None):
        """
        Initialize power manager.

        Args:
            config: Configuration dict with 'power' section
        """
        self.config = config or {}
        power_config = self.config.get("power", {})

        # Load thresholds from config or use defaults
        self.thresholds = power_config.get("thresholds", self.DEFAULT_THRESHOLDS)

        # Load brightness levels from config or use defaults
        brightness_config = power_config.get("brightness", {})
        self.brightness_levels = {
            PowerMode.FULL: brightness_config.get("full", 1.0),
            PowerMode.BALANCED: brightness_config.get("balanced", 0.85),
            PowerMode.SAVER: brightness_config.get("saver", 0.60),
            PowerMode.CRITICAL: brightness_config.get("critical", 0.40)
        }

        # State
        self._mode = PowerMode.BALANCED
        self._state = PowerState(
            battery_percent=100,
            is_charging=False,
            mode=PowerMode.BALANCED
        )
        self._callbacks: List[Callable[[PowerMode, PowerMode], None]] = []
        self._manual_override: Optional[PowerMode] = None
        self._auto_manage = power_config.get("auto_manage", True)

        logger.info(f"PowerManager initialized (auto_manage={self._auto_manage})")

    def update_battery(self, percent: int, charging: bool = False) -> Optional[PowerMode]:
        """
        Update battery state and potentially trigger mode change.

        Args:
            percent: Battery percentage (0-100)
            charging: Whether device is charging

        Returns:
            New power mode if changed, None otherwise
        """
        old_mode = self._mode
        self._state = PowerState(
            battery_percent=percent,
            is_charging=charging,
            mode=self._mode,
            estimated_runtime_min=self._estimate_runtime(percent)
        )

        # Skip auto-management if disabled or manual override active
        if not self._auto_manage or self._manual_override:
            return None

        # Determine appropriate mode
        new_mode = self._determine_mode(percent, charging)

        if new_mode != old_mode:
            self._set_mode(new_mode, old_mode)
            return new_mode

        return None

    def _determine_mode(self, percent: int, charging: bool) -> PowerMode:
        """Determine power mode based on battery state."""
        # If charging, use FULL mode
        if charging:
            return PowerMode.FULL

        # Determine mode by thresholds
        if percent >= self.thresholds["full"]:
            return PowerMode.FULL
        elif percent >= self.thresholds["balanced"]:
            return PowerMode.BALANCED
        elif percent >= self.thresholds["saver"]:
            return PowerMode.SAVER
        else:
            return PowerMode.CRITICAL

    def _set_mode(self, new_mode: PowerMode, old_mode: PowerMode):
        """Set mode and notify callbacks."""
        self._mode = new_mode
        self._state.mode = new_mode

        logger.info(f"Power mode: {old_mode.value} → {new_mode.value}")

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(old_mode, new_mode)
            except Exception as e:
                logger.error(f"Power callback error: {e}")

    def _estimate_runtime(self, percent: int) -> int:
        """Estimate remaining runtime in minutes based on battery level."""
        # Rough estimate: Halo glasses ~4 hours at full brightness
        # Adjust based on current mode
        base_runtime_min = 240  # 4 hours at 100%

        mode_multipliers = {
            PowerMode.FULL: 1.0,
            PowerMode.BALANCED: 1.15,
            PowerMode.SAVER: 1.5,
            PowerMode.CRITICAL: 2.0
        }

        multiplier = mode_multipliers.get(self._mode, 1.0)
        return int((percent / 100) * base_runtime_min * multiplier)

    def get_mode(self) -> PowerMode:
        """Get current power mode."""
        return self._manual_override or self._mode

    def get_state(self) -> PowerState:
        """Get current power state."""
        return self._state

    def get_brightness(self) -> float:
        """
        Get recommended brightness level (0.0-1.0).

        Returns:
            Brightness multiplier for current power mode
        """
        mode = self.get_mode()
        return self.brightness_levels.get(mode, 0.85)

    def get_color_adjustment(self) -> Tuple[float, float, float]:
        """
        Get RGB multipliers for power-aware color adjustment.

        Reduces blue channel in lower power modes to save OLED power.

        Returns:
            Tuple of (R, G, B) multipliers (0.0-1.0)
        """
        mode = self.get_mode()
        return self.COLOR_ADJUSTMENTS.get(mode, (1.0, 1.0, 1.0))

    def apply_color_adjustment(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Apply power-aware adjustment to an RGB color.

        Args:
            color: RGB tuple (0-255 per channel)

        Returns:
            Adjusted RGB tuple
        """
        r_mult, g_mult, b_mult = self.get_color_adjustment()
        brightness = self.get_brightness()

        return (
            int(min(255, color[0] * r_mult * brightness)),
            int(min(255, color[1] * g_mult * brightness)),
            int(min(255, color[2] * b_mult * brightness))
        )

    def should_defer_operation(self, operation: str) -> bool:
        """
        Check if an operation should be deferred to save power.

        Args:
            operation: Operation type (e.g., 'animation', 'deep_analysis')

        Returns:
            True if operation should be deferred
        """
        mode = self.get_mode()

        # Operations that should be deferred in critical mode
        critical_defer = ["animation", "deep_analysis", "session_review"]
        if mode == PowerMode.CRITICAL and operation in critical_defer:
            return True

        # Operations that should be deferred in saver mode
        saver_defer = ["deep_analysis", "session_review"]
        if mode == PowerMode.SAVER and operation in saver_defer:
            return True

        return False

    def set_mode_override(self, mode: Optional[PowerMode]):
        """
        Manually override power mode.

        Args:
            mode: Power mode to force, or None to disable override
        """
        old_mode = self.get_mode()
        self._manual_override = mode

        if mode and mode != old_mode:
            logger.info(f"Manual power override: {mode.value}")
            self._set_mode(mode, old_mode)
        elif not mode:
            logger.info("Manual power override cleared")
            # Re-evaluate based on current battery
            self.update_battery(
                self._state.battery_percent,
                self._state.is_charging
            )

    def on_mode_change(self, callback: Callable[[PowerMode, PowerMode], None]):
        """
        Register callback for power mode changes.

        Args:
            callback: Function(old_mode, new_mode) called on mode change
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """Remove a registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def is_low_power(self) -> bool:
        """Check if in low power state (SAVER or CRITICAL)."""
        mode = self.get_mode()
        return mode in (PowerMode.SAVER, PowerMode.CRITICAL)

    def is_critical(self) -> bool:
        """Check if in critical power state."""
        return self.get_mode() == PowerMode.CRITICAL

    def auto_adjust_brightness(self) -> float:
        """
        Adjust brightness based on ambient light and time of day.

        Uses time-based heuristics until Halo SDK ambient light sensor
        is available. Combines with power mode for final brightness.

        Returns:
            Recommended brightness level (0.0-1.0)
        """
        from datetime import datetime

        # TODO: When Halo SDK available:
        # ambient_light = await frame.sensors.get_ambient_light()
        # return self._calculate_brightness_from_ambient(ambient_light)

        # Time-based heuristics
        hour = datetime.now().hour

        if 22 <= hour or hour < 6:      # Night (10pm-6am)
            time_brightness = 0.3
        elif 6 <= hour < 8:             # Early morning
            time_brightness = 0.5
        elif 8 <= hour < 18:            # Day (8am-6pm)
            time_brightness = 0.8
        else:                           # Evening (6pm-10pm)
            time_brightness = 0.6

        # Combine with power mode brightness
        power_brightness = self.get_brightness()

        # Use the lower of time-based and power-based brightness
        # This ensures power saving takes priority when battery is low
        final_brightness = min(time_brightness, power_brightness)

        logger.debug(f"Auto brightness: time={time_brightness:.1%}, "
                    f"power={power_brightness:.1%}, final={final_brightness:.1%}")

        return final_brightness

    def get_time_period(self) -> str:
        """
        Get current time period for display purposes.

        Returns:
            Time period name: 'night', 'morning', 'day', or 'evening'
        """
        from datetime import datetime
        hour = datetime.now().hour

        if 22 <= hour or hour < 6:
            return "night"
        elif 6 <= hour < 8:
            return "morning"
        elif 8 <= hour < 18:
            return "day"
        else:
            return "evening"

    def get_status_string(self) -> str:
        """Get human-readable power status."""
        state = self._state
        mode = self.get_mode()

        status = f"{state.battery_percent}%"
        if state.is_charging:
            status += " (charging)"

        mode_icons = {
            PowerMode.FULL: "🔋",
            PowerMode.BALANCED: "🔋",
            PowerMode.SAVER: "🪫",
            PowerMode.CRITICAL: "⚠️"
        }

        return f"{mode_icons.get(mode, '')} {status} ({mode.value})"


# Test
def test_power_manager():
    """Test power manager functionality."""
    print("=== Power Manager Test ===\n")

    pm = PowerManager()

    # Track mode changes
    mode_changes = []
    pm.on_mode_change(lambda old, new: mode_changes.append((old, new)))

    # Test battery updates
    test_levels = [100, 85, 75, 50, 25, 12, 5, 3]
    for level in test_levels:
        pm.update_battery(level)
        print(f"Battery {level}%: {pm.get_status_string()}")
        print(f"  Brightness: {pm.get_brightness():.0%}")
        print(f"  Color adj: {pm.get_color_adjustment()}")
        print()

    print(f"Mode changes: {len(mode_changes)}")
    for old, new in mode_changes:
        print(f"  {old.value} → {new.value}")

    # Test charging
    print("\nCharging:")
    pm.update_battery(20, charging=True)
    print(f"  {pm.get_status_string()}")

    # Test manual override
    print("\nManual override to SAVER:")
    pm.set_mode_override(PowerMode.SAVER)
    print(f"  {pm.get_status_string()}")

    # Test color application
    print("\nColor adjustment test:")
    test_color = (0, 255, 255)  # Cyan
    adjusted = pm.apply_color_adjustment(test_color)
    print(f"  Original: {test_color}")
    print(f"  Adjusted: {adjusted}")


if __name__ == "__main__":
    test_power_manager()
