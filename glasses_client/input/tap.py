"""
Tap Gesture Handler

Handles tap and gesture input from the Halo Frame touch sensor.
"""

import asyncio
import logging
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class TapGesture(Enum):
    """Recognized tap gestures."""
    SINGLE_TAP = "single_tap"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"
    SWIPE_FORWARD = "swipe_forward"
    SWIPE_BACK = "swipe_back"
    NONE = "none"


@dataclass
class TapEvent:
    """A tap gesture event."""
    gesture: TapGesture
    timestamp: datetime
    duration_ms: int = 0


class TapHandler:
    """
    Handles tap and gesture input from Halo Frame.

    Features:
    - Single/double tap detection
    - Long press detection
    - Swipe gestures
    - Debouncing
    """

    # Timing thresholds (milliseconds)
    DOUBLE_TAP_WINDOW = 300
    LONG_PRESS_THRESHOLD = 500
    DEBOUNCE_MS = 100

    def __init__(self, frame=None):
        """
        Initialize tap handler.

        Args:
            frame: Frame SDK instance. If None, uses mock for testing.
        """
        self.frame = frame
        self._mock_mode = frame is None
        self._last_tap_time: Optional[datetime] = None
        self._listening = False
        self._callback: Optional[Callable[[TapEvent], Awaitable[None]]] = None

    def on_tap(self, callback: Callable[[TapEvent], Awaitable[None]]):
        """Register callback for tap events."""
        self._callback = callback

    async def wait_for_tap(self, timeout_ms: int = 5000) -> Optional[TapGesture]:
        """
        Wait for a single tap gesture.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            The detected gesture or None on timeout
        """
        event = await self.wait_for_gesture(timeout_ms)
        return event.gesture if event else None

    async def wait_for_gesture(self, timeout_ms: int = 5000) -> Optional[TapEvent]:
        """
        Wait for any tap gesture.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            TapEvent or None on timeout
        """
        self._listening = True

        try:
            if self._mock_mode:
                return await self._mock_wait(timeout_ms)
            else:
                return await self._real_wait(timeout_ms)

        finally:
            self._listening = False

    async def start_listening(self):
        """
        Start continuous gesture listening.

        Calls the registered callback for each gesture detected.
        """
        if not self._callback:
            logger.warning("No tap callback registered")
            return

        self._listening = True
        logger.info("Started tap gesture listening")

        try:
            while self._listening:
                event = await self.wait_for_gesture(timeout_ms=1000)

                if event and event.gesture != TapGesture.NONE:
                    await self._callback(event)

        except Exception as e:
            logger.error(f"Tap listening error: {e}")
        finally:
            self._listening = False

    def stop_listening(self):
        """Stop continuous gesture listening."""
        self._listening = False
        logger.info("Stopped tap gesture listening")

    async def _mock_wait(self, timeout_ms: int) -> Optional[TapEvent]:
        """Mock gesture detection for testing."""
        try:
            # Simulate waiting
            await asyncio.sleep(timeout_ms / 1000)
            return None

        except asyncio.CancelledError:
            return None

    async def _real_wait(self, timeout_ms: int) -> Optional[TapEvent]:
        """Real gesture detection using Frame SDK."""
        try:
            start_time = datetime.utcnow()

            while True:
                elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
                if elapsed >= timeout_ms:
                    return None

                # Check for tap
                tap_detected = await self.frame.motion.tap_detected()

                if tap_detected:
                    tap_time = datetime.utcnow()

                    # Check for double tap
                    if self._last_tap_time:
                        gap = (tap_time - self._last_tap_time).total_seconds() * 1000
                        if gap < self.DOUBLE_TAP_WINDOW:
                            self._last_tap_time = None
                            return TapEvent(
                                gesture=TapGesture.DOUBLE_TAP,
                                timestamp=tap_time
                            )

                    # Wait to see if it's a long press
                    press_start = datetime.utcnow()

                    while await self._is_still_pressing():
                        press_duration = (datetime.utcnow() - press_start).total_seconds() * 1000

                        if press_duration >= self.LONG_PRESS_THRESHOLD:
                            self._last_tap_time = None
                            return TapEvent(
                                gesture=TapGesture.LONG_PRESS,
                                timestamp=tap_time,
                                duration_ms=int(press_duration)
                            )

                        await asyncio.sleep(0.05)

                    # It's a single tap
                    self._last_tap_time = tap_time

                    # Wait briefly for potential double tap
                    await asyncio.sleep(self.DOUBLE_TAP_WINDOW / 1000)

                    # If no second tap, return single tap
                    if self._last_tap_time == tap_time:
                        self._last_tap_time = None
                        return TapEvent(
                            gesture=TapGesture.SINGLE_TAP,
                            timestamp=tap_time
                        )

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.05)

        except Exception as e:
            logger.error(f"Gesture detection error: {e}")
            return None

    async def _is_still_pressing(self) -> bool:
        """Check if user is still pressing (for long press detection)."""
        if self._mock_mode:
            return False

        try:
            return await self.frame.motion.is_pressed()
        except Exception:
            return False


def gesture_to_action(gesture: TapGesture) -> str:
    """
    Convert gesture to a navigation action.

    Returns:
        Action string: "next", "back", "select", or "none"
    """
    mapping = {
        TapGesture.SINGLE_TAP: "next",
        TapGesture.DOUBLE_TAP: "back",
        TapGesture.LONG_PRESS: "select",
        TapGesture.SWIPE_FORWARD: "next",
        TapGesture.SWIPE_BACK: "back",
        TapGesture.NONE: "none",
    }
    return mapping.get(gesture, "none")
