"""
Haptics - Tactile feedback patterns for Halo glasses events.
Provides haptic feedback for user interactions and notifications.
"""
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class HapticIntensity(Enum):
    """Haptic vibration intensity levels."""
    LIGHT = 0.3      # Subtle, ambient feedback
    MEDIUM = 0.5     # Normal interaction feedback
    STRONG = 0.8     # Attention-grabbing
    MAX = 1.0        # Emergency/critical


@dataclass
class HapticPulse:
    """A single haptic pulse."""
    duration_ms: int
    intensity: float
    pause_after_ms: int = 0


@dataclass
class HapticPattern:
    """A haptic pattern consisting of multiple pulses."""
    name: str
    pulses: List[HapticPulse]
    repeat: int = 1


class HapticPatterns:
    """
    Pre-defined haptic feedback patterns for common events.

    Patterns are designed to be:
    - Distinguishable: Different events feel different
    - Non-intrusive: Subtle enough for continuous use
    - Meaningful: Pattern duration/intensity matches event importance
    """

    @staticmethod
    def tap_feedback() -> HapticPattern:
        """
        Light tap for button presses and selections.
        Duration: 10ms, Intensity: 30%
        """
        return HapticPattern(
            name="tap",
            pulses=[HapticPulse(duration_ms=10, intensity=0.3)]
        )

    @staticmethod
    def success_pulse() -> HapticPattern:
        """
        Double pulse for success/confirmation.
        Duration: 50ms-pause-50ms, Intensity: 50%
        """
        return HapticPattern(
            name="success",
            pulses=[
                HapticPulse(duration_ms=50, intensity=0.5, pause_after_ms=50),
                HapticPulse(duration_ms=50, intensity=0.5)
            ]
        )

    @staticmethod
    def error_buzz() -> HapticPattern:
        """
        Strong buzz for errors.
        Duration: 200ms, Intensity: 80%
        """
        return HapticPattern(
            name="error",
            pulses=[HapticPulse(duration_ms=200, intensity=0.8)]
        )

    @staticmethod
    def notification_nudge() -> HapticPattern:
        """
        Gentle nudge for notification arrival.
        Duration: 30ms, Intensity: 40%
        """
        return HapticPattern(
            name="notification",
            pulses=[HapticPulse(duration_ms=30, intensity=0.4)]
        )

    @staticmethod
    def streak_celebration(streak: int) -> HapticPattern:
        """
        Escalating pattern based on streak length.

        Args:
            streak: Current streak count

        Returns:
            Pattern with pulses scaling with streak
        """
        # Base pattern intensity and count scale with streak
        if streak >= 50:
            # Epic celebration
            return HapticPattern(
                name="streak_legendary",
                pulses=[
                    HapticPulse(duration_ms=80, intensity=1.0, pause_after_ms=60),
                    HapticPulse(duration_ms=80, intensity=1.0, pause_after_ms=60),
                    HapticPulse(duration_ms=80, intensity=1.0, pause_after_ms=60),
                    HapticPulse(duration_ms=150, intensity=1.0)
                ]
            )
        elif streak >= 20:
            # Major milestone
            return HapticPattern(
                name="streak_gold",
                pulses=[
                    HapticPulse(duration_ms=60, intensity=0.8, pause_after_ms=50),
                    HapticPulse(duration_ms=60, intensity=0.8, pause_after_ms=50),
                    HapticPulse(duration_ms=100, intensity=0.9)
                ]
            )
        elif streak >= 10:
            # Good streak
            return HapticPattern(
                name="streak_silver",
                pulses=[
                    HapticPulse(duration_ms=50, intensity=0.6, pause_after_ms=40),
                    HapticPulse(duration_ms=80, intensity=0.7)
                ]
            )
        elif streak >= 5:
            # Building momentum
            return HapticPattern(
                name="streak_bronze",
                pulses=[
                    HapticPulse(duration_ms=40, intensity=0.5, pause_after_ms=30),
                    HapticPulse(duration_ms=50, intensity=0.5)
                ]
            )
        else:
            # Starting streak
            return HapticPatterns.success_pulse()

    @staticmethod
    def power_warning() -> HapticPattern:
        """
        Triple pulse for low battery warning.
        Duration: 40ms x 3 with 60ms pauses
        """
        return HapticPattern(
            name="power_warning",
            pulses=[
                HapticPulse(duration_ms=40, intensity=0.6, pause_after_ms=60),
                HapticPulse(duration_ms=40, intensity=0.6, pause_after_ms=60),
                HapticPulse(duration_ms=40, intensity=0.6)
            ]
        )

    @staticmethod
    def critical_alert() -> HapticPattern:
        """
        Strong pulsing pattern for critical alerts.
        """
        return HapticPattern(
            name="critical",
            pulses=[
                HapticPulse(duration_ms=100, intensity=1.0, pause_after_ms=80),
                HapticPulse(duration_ms=100, intensity=1.0, pause_after_ms=80),
                HapticPulse(duration_ms=100, intensity=1.0, pause_after_ms=80),
                HapticPulse(duration_ms=200, intensity=1.0)
            ]
        )

    @staticmethod
    def thinking_pulse() -> HapticPattern:
        """
        Slow, subtle pulse while AI is thinking.
        """
        return HapticPattern(
            name="thinking",
            pulses=[HapticPulse(duration_ms=100, intensity=0.2, pause_after_ms=800)],
            repeat=3
        )

    @staticmethod
    def poker_raise() -> HapticPattern:
        """
        Confident double-tap for raise recommendations.
        """
        return HapticPattern(
            name="poker_raise",
            pulses=[
                HapticPulse(duration_ms=40, intensity=0.6, pause_after_ms=40),
                HapticPulse(duration_ms=60, intensity=0.7)
            ]
        )

    @staticmethod
    def poker_fold() -> HapticPattern:
        """
        Single soft tap for fold recommendations.
        """
        return HapticPattern(
            name="poker_fold",
            pulses=[HapticPulse(duration_ms=30, intensity=0.3)]
        )

    @staticmethod
    def poker_allin() -> HapticPattern:
        """
        Strong pattern for all-in recommendations.
        """
        return HapticPattern(
            name="poker_allin",
            pulses=[
                HapticPulse(duration_ms=60, intensity=0.8, pause_after_ms=30),
                HapticPulse(duration_ms=60, intensity=0.8, pause_after_ms=30),
                HapticPulse(duration_ms=100, intensity=1.0)
            ]
        )

    @staticmethod
    def mode_change() -> HapticPattern:
        """
        Distinctive pattern for mode changes (poker/math/meeting).
        """
        return HapticPattern(
            name="mode_change",
            pulses=[
                HapticPulse(duration_ms=30, intensity=0.4, pause_after_ms=100),
                HapticPulse(duration_ms=50, intensity=0.5)
            ]
        )

    @staticmethod
    def correct_answer() -> HapticPattern:
        """
        Quick happy pulse for correct answers.
        """
        return HapticPatterns.success_pulse()

    @staticmethod
    def wrong_answer() -> HapticPattern:
        """
        Short buzz for wrong answers (not too harsh).
        """
        return HapticPattern(
            name="wrong",
            pulses=[HapticPulse(duration_ms=100, intensity=0.5)]
        )

    @staticmethod
    def timer_warning() -> HapticPattern:
        """
        Quick pulses when timer is running low.
        """
        return HapticPattern(
            name="timer_warning",
            pulses=[
                HapticPulse(duration_ms=20, intensity=0.4, pause_after_ms=100),
                HapticPulse(duration_ms=20, intensity=0.4)
            ]
        )


class HapticManager:
    """
    Manages haptic feedback execution for Halo glasses.

    Handles pattern playback, power-aware intensity adjustment,
    and coordination with the Bluetooth connection.
    """

    def __init__(self, connection=None, config: dict = None):
        """
        Initialize haptic manager.

        Args:
            connection: HaloConnection instance for sending commands
            config: Configuration dict with 'notifications' section
        """
        self.connection = connection
        self.config = config or {}

        notif_config = self.config.get("notifications", {})
        self.enabled = notif_config.get("vibration", True)

        self._intensity_multiplier = 1.0
        self._is_playing = False

    def set_enabled(self, enabled: bool):
        """Enable or disable haptic feedback."""
        self.enabled = enabled
        logger.info(f"Haptics: {'enabled' if enabled else 'disabled'}")

    def set_intensity_multiplier(self, multiplier: float):
        """
        Set global intensity multiplier.

        Args:
            multiplier: Intensity scale (0.0-1.0)
        """
        self._intensity_multiplier = max(0.0, min(1.0, multiplier))

    def apply_power_adjustment(self, intensity: float, power_mode: str) -> float:
        """
        Adjust intensity based on power mode.

        Args:
            intensity: Base intensity (0.0-1.0)
            power_mode: Current power mode

        Returns:
            Adjusted intensity
        """
        # Reduce haptic intensity in power saver modes
        power_multipliers = {
            "full": 1.0,
            "balanced": 0.9,
            "saver": 0.7,
            "critical": 0.5
        }
        multiplier = power_multipliers.get(power_mode, 1.0)
        return intensity * multiplier * self._intensity_multiplier

    async def play(self, pattern: HapticPattern, power_mode: str = "full"):
        """
        Play a haptic pattern.

        Args:
            pattern: HapticPattern to play
            power_mode: Current power mode for intensity adjustment
        """
        if not self.enabled:
            return

        if self._is_playing:
            logger.debug("Haptic pattern already playing, skipping")
            return

        self._is_playing = True

        try:
            for _ in range(pattern.repeat):
                for pulse in pattern.pulses:
                    adjusted_intensity = self.apply_power_adjustment(
                        pulse.intensity, power_mode
                    )

                    # Send haptic command to glasses
                    await self._send_haptic_command(
                        pulse.duration_ms,
                        adjusted_intensity
                    )

                    # Wait for pulse duration plus pause
                    total_wait = pulse.duration_ms + pulse.pause_after_ms
                    await asyncio.sleep(total_wait / 1000)
        finally:
            self._is_playing = False

    async def _send_haptic_command(self, duration_ms: int, intensity: float):
        """
        Send haptic command to Halo glasses.

        Args:
            duration_ms: Vibration duration
            intensity: Vibration intensity (0.0-1.0)
        """
        if self.connection is None:
            logger.debug(f"Haptic: {duration_ms}ms @ {intensity:.0%} (no connection)")
            return

        # Generate Lua command for haptic feedback
        lua_command = f"frame.vibrate({duration_ms}, {intensity})"

        try:
            await self.connection.send_lua(lua_command)
        except Exception as e:
            logger.error(f"Failed to send haptic command: {e}")

    # ============================================================
    # Convenience Methods
    # ============================================================

    async def tap(self):
        """Quick tap feedback."""
        await self.play(HapticPatterns.tap_feedback())

    async def success(self):
        """Success feedback."""
        await self.play(HapticPatterns.success_pulse())

    async def error(self):
        """Error feedback."""
        await self.play(HapticPatterns.error_buzz())

    async def notification(self):
        """Notification arrival feedback."""
        await self.play(HapticPatterns.notification_nudge())

    async def streak(self, streak: int):
        """Streak celebration feedback."""
        await self.play(HapticPatterns.streak_celebration(streak))

    async def battery_warning(self):
        """Low battery warning feedback."""
        await self.play(HapticPatterns.power_warning())

    async def poker_recommendation(self, action: str):
        """
        Haptic feedback for poker recommendations.

        Args:
            action: Recommendation action (raise, fold, call, allin, check)
        """
        if action in ("raise", "bet"):
            await self.play(HapticPatterns.poker_raise())
        elif action == "fold":
            await self.play(HapticPatterns.poker_fold())
        elif action == "allin":
            await self.play(HapticPatterns.poker_allin())
        else:
            await self.play(HapticPatterns.tap_feedback())

    async def answer_feedback(self, correct: bool):
        """
        Haptic feedback for answer correctness.

        Args:
            correct: Whether answer was correct
        """
        if correct:
            await self.play(HapticPatterns.correct_answer())
        else:
            await self.play(HapticPatterns.wrong_answer())


# Test
def test_haptics():
    """Test haptic patterns."""
    print("=== Haptic Patterns Test ===\n")

    # Test all patterns
    patterns = [
        ("Tap", HapticPatterns.tap_feedback()),
        ("Success", HapticPatterns.success_pulse()),
        ("Error", HapticPatterns.error_buzz()),
        ("Notification", HapticPatterns.notification_nudge()),
        ("Power Warning", HapticPatterns.power_warning()),
        ("Streak 5", HapticPatterns.streak_celebration(5)),
        ("Streak 10", HapticPatterns.streak_celebration(10)),
        ("Streak 20", HapticPatterns.streak_celebration(20)),
        ("Streak 50", HapticPatterns.streak_celebration(50)),
        ("Critical", HapticPatterns.critical_alert()),
        ("Poker Raise", HapticPatterns.poker_raise()),
        ("Poker All-In", HapticPatterns.poker_allin()),
        ("Mode Change", HapticPatterns.mode_change()),
    ]

    for name, pattern in patterns:
        total_duration = sum(p.duration_ms + p.pause_after_ms for p in pattern.pulses)
        avg_intensity = sum(p.intensity for p in pattern.pulses) / len(pattern.pulses)
        print(f"{name}:")
        print(f"  Pulses: {len(pattern.pulses)}")
        print(f"  Duration: {total_duration}ms")
        print(f"  Avg Intensity: {avg_intensity:.0%}")
        print()


if __name__ == "__main__":
    test_haptics()
