"""
EDITH Power Management

Adaptive power management for AR glasses scanning.
Balances responsiveness with battery life.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PowerProfile(Enum):
    """Power consumption profiles."""
    PERFORMANCE = "performance"    # Maximum responsiveness
    BALANCED = "balanced"          # Default mode
    BATTERY_SAVER = "battery_saver"  # Extended battery
    ULTRA_SAVER = "ultra_saver"    # Minimum power


@dataclass
class AdaptiveFPS:
    """
    Adaptive frame rate controller.

    Adjusts scanning FPS based on:
    - Content presence (increase when content detected)
    - User activity (decrease during idle)
    - Battery level (reduce when low)
    - Time of day (reduce at night)
    """

    # FPS ranges by profile
    FPS_RANGES = {
        PowerProfile.PERFORMANCE: (5.0, 15.0),
        PowerProfile.BALANCED: (2.0, 10.0),
        PowerProfile.BATTERY_SAVER: (0.5, 5.0),
        PowerProfile.ULTRA_SAVER: (0.2, 2.0),
    }

    def __init__(self, profile: PowerProfile = PowerProfile.BALANCED):
        self.profile = profile
        self._current_fps = self.base_fps
        self._last_detection_time: Optional[datetime] = None
        self._last_activity_time: datetime = datetime.now()

    @property
    def base_fps(self) -> float:
        """Base FPS for current profile."""
        return self.FPS_RANGES[self.profile][0]

    @property
    def max_fps(self) -> float:
        """Maximum FPS for current profile."""
        return self.FPS_RANGES[self.profile][1]

    @property
    def current_fps(self) -> float:
        """Get current adaptive FPS."""
        return self._current_fps

    @property
    def frame_interval_ms(self) -> int:
        """Milliseconds between frames at current FPS."""
        return int(1000 / self._current_fps) if self._current_fps > 0 else 1000

    def on_detection(self) -> None:
        """Called when content is detected - increase FPS."""
        self._last_detection_time = datetime.now()
        self._current_fps = min(self.max_fps, self._current_fps * 1.5)
        logger.debug(f"Detection boost: FPS now {self._current_fps:.1f}")

    def on_no_detection(self) -> None:
        """Called when no content detected - decay FPS."""
        # Decay towards base FPS
        decay_factor = 0.95
        self._current_fps = max(
            self.base_fps,
            self._current_fps * decay_factor
        )

    def on_activity(self) -> None:
        """Called on user activity (head movement, tap, etc.)."""
        self._last_activity_time = datetime.now()

    def update(self) -> None:
        """
        Periodic update of FPS based on conditions.
        Should be called regularly (e.g., every second).
        """
        now = datetime.now()

        # Check idle timeout
        idle_duration = (now - self._last_activity_time).total_seconds()
        if idle_duration > 60:  # 1 minute idle
            self._current_fps = max(
                self.base_fps * 0.5,
                self._current_fps * 0.8
            )
            logger.debug(f"Idle reduction: FPS now {self._current_fps:.1f}")

        # Check if detection is stale
        if self._last_detection_time:
            since_detection = (now - self._last_detection_time).total_seconds()
            if since_detection > 5:  # 5 seconds since last detection
                self._current_fps = max(
                    self.base_fps,
                    self._current_fps * 0.9
                )

    def set_profile(self, profile: PowerProfile) -> None:
        """Change power profile."""
        old_profile = self.profile
        self.profile = profile

        # Adjust current FPS to new range
        min_fps, max_fps = self.FPS_RANGES[profile]
        self._current_fps = max(min_fps, min(max_fps, self._current_fps))

        logger.info(f"Profile changed: {old_profile.value} -> {profile.value}")


class PowerManager:
    """
    Central power management for EDITH system.

    Manages:
    - Scanning frame rate
    - Processing intensity
    - Network requests
    - Display brightness suggestions
    """

    # Battery thresholds
    BATTERY_THRESHOLDS = {
        "critical": 0.05,    # 5%
        "low": 0.15,         # 15%
        "medium": 0.30,      # 30%
        "good": 0.50,        # 50%
    }

    def __init__(self):
        self._battery_level: float = 1.0
        self._is_charging: bool = False
        self._profile: PowerProfile = PowerProfile.BALANCED
        self._adaptive_fps = AdaptiveFPS(self._profile)

        # Feature toggles based on power
        self._enable_continuous_scan = True
        self._enable_cloud_processing = True
        self._enable_proactive_suggestions = True

        # Stats
        self._last_profile_change: datetime = datetime.now()

    @property
    def battery_level(self) -> float:
        return self._battery_level

    @property
    def profile(self) -> PowerProfile:
        return self._profile

    @property
    def adaptive_fps(self) -> AdaptiveFPS:
        return self._adaptive_fps

    def update_battery(self, level: float, is_charging: bool = False) -> None:
        """
        Update battery status and adjust power profile if needed.

        Args:
            level: Battery level 0.0 to 1.0
            is_charging: Whether device is charging
        """
        self._battery_level = max(0, min(1, level))
        self._is_charging = is_charging

        # Auto-adjust profile based on battery
        if not is_charging:
            self._auto_adjust_profile()

    def _auto_adjust_profile(self) -> None:
        """Automatically adjust profile based on battery level."""
        if self._battery_level <= self.BATTERY_THRESHOLDS["critical"]:
            self._set_profile(PowerProfile.ULTRA_SAVER)
            self._enable_continuous_scan = False
            self._enable_cloud_processing = False
            self._enable_proactive_suggestions = False

        elif self._battery_level <= self.BATTERY_THRESHOLDS["low"]:
            self._set_profile(PowerProfile.BATTERY_SAVER)
            self._enable_continuous_scan = True
            self._enable_cloud_processing = True
            self._enable_proactive_suggestions = False

        elif self._battery_level <= self.BATTERY_THRESHOLDS["medium"]:
            if self._profile == PowerProfile.PERFORMANCE:
                self._set_profile(PowerProfile.BALANCED)
            self._enable_continuous_scan = True
            self._enable_cloud_processing = True
            self._enable_proactive_suggestions = True

        # If good battery and was in saver mode, restore balanced
        elif (self._battery_level > self.BATTERY_THRESHOLDS["good"] and
              self._profile in [PowerProfile.BATTERY_SAVER, PowerProfile.ULTRA_SAVER]):
            self._set_profile(PowerProfile.BALANCED)
            self._enable_all_features()

    def _set_profile(self, profile: PowerProfile) -> None:
        """Set power profile."""
        if profile != self._profile:
            self._profile = profile
            self._adaptive_fps.set_profile(profile)
            self._last_profile_change = datetime.now()
            logger.info(f"Power profile: {profile.value} (battery: {self._battery_level*100:.0f}%)")

    def _enable_all_features(self) -> None:
        """Enable all features."""
        self._enable_continuous_scan = True
        self._enable_cloud_processing = True
        self._enable_proactive_suggestions = True

    def set_manual_profile(self, profile: PowerProfile) -> None:
        """
        Manually set power profile (overrides auto-adjustment).

        User-initiated profile changes are respected until battery
        reaches critical levels.
        """
        if self._battery_level > self.BATTERY_THRESHOLDS["critical"]:
            self._set_profile(profile)
            if profile == PowerProfile.PERFORMANCE:
                self._enable_all_features()

    def can_scan(self) -> bool:
        """Check if scanning should continue."""
        return self._enable_continuous_scan

    def can_use_cloud(self) -> bool:
        """Check if cloud processing is allowed."""
        return self._enable_cloud_processing

    def can_suggest(self) -> bool:
        """Check if proactive suggestions are enabled."""
        return self._enable_proactive_suggestions

    def get_status(self) -> dict:
        """Get current power management status."""
        return {
            "battery_level": self._battery_level,
            "battery_percent": int(self._battery_level * 100),
            "is_charging": self._is_charging,
            "profile": self._profile.value,
            "current_fps": self._adaptive_fps.current_fps,
            "frame_interval_ms": self._adaptive_fps.frame_interval_ms,
            "features": {
                "continuous_scan": self._enable_continuous_scan,
                "cloud_processing": self._enable_cloud_processing,
                "proactive_suggestions": self._enable_proactive_suggestions,
            },
        }

    def estimate_remaining_time(self) -> Optional[timedelta]:
        """
        Estimate remaining scanning time based on battery and FPS.

        Returns:
            Estimated time remaining, or None if unknown
        """
        if self._is_charging:
            return None

        # Rough estimates based on profile
        # (Would need actual power measurements in production)
        hours_per_percent = {
            PowerProfile.PERFORMANCE: 0.3,
            PowerProfile.BALANCED: 0.5,
            PowerProfile.BATTERY_SAVER: 0.8,
            PowerProfile.ULTRA_SAVER: 1.2,
        }

        hours = self._battery_level * 100 * hours_per_percent.get(self._profile, 0.5)
        return timedelta(hours=hours)
