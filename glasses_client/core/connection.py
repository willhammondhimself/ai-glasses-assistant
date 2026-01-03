"""
Bluetooth LE Connection Manager for Halo Frame

Manages the Bluetooth connection to Brilliant Labs Halo Frame glasses.
"""

import asyncio
import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states."""
    DISCONNECTED = "disconnected"
    SCANNING = "scanning"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class FrameInfo:
    """Information about a connected Frame device."""
    name: str
    address: str
    firmware_version: str = ""
    battery_level: int = 0


class ConnectionManager:
    """
    Manages Bluetooth LE connection to Halo Frame.

    Handles:
    - Device scanning and discovery
    - Connection establishment
    - Reconnection on disconnect
    - Battery monitoring
    """

    def __init__(self):
        self.frame = None
        self.state = ConnectionState.DISCONNECTED
        self.frame_info: Optional[FrameInfo] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._on_disconnect_callback: Optional[Callable] = None
        self._on_connect_callback: Optional[Callable] = None

    def on_disconnect(self, callback: Callable):
        """Register callback for disconnect events."""
        self._on_disconnect_callback = callback

    def on_connect(self, callback: Callable):
        """Register callback for connect events."""
        self._on_connect_callback = callback

    async def scan_for_frame(self, timeout: float = 10.0) -> Optional[FrameInfo]:
        """
        Scan for nearby Halo Frame devices.

        Args:
            timeout: Scan timeout in seconds

        Returns:
            FrameInfo if found, None otherwise
        """
        self.state = ConnectionState.SCANNING
        logger.info(f"Scanning for Halo Frame devices (timeout={timeout}s)...")

        try:
            # Import Frame SDK
            from frame_sdk import Frame

            # Scan for devices
            frame = await Frame.discover(timeout=timeout)

            if frame:
                self.frame_info = FrameInfo(
                    name=frame.name or "Halo Frame",
                    address=frame.address or "unknown"
                )
                logger.info(f"Found Frame: {self.frame_info.name}")
                return self.frame_info

            logger.warning("No Frame device found")
            return None

        except ImportError:
            logger.warning("Frame SDK not installed. Using mock mode.")
            # Mock device for testing
            self.frame_info = FrameInfo(
                name="Mock Halo Frame",
                address="00:00:00:00:00:00",
                battery_level=100
            )
            return self.frame_info

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            self.state = ConnectionState.ERROR
            return None

    async def connect(self, address: Optional[str] = None) -> bool:
        """
        Connect to a Halo Frame device.

        Args:
            address: Device address (scans if not provided)

        Returns:
            True if connected successfully
        """
        self.state = ConnectionState.CONNECTING
        logger.info("Connecting to Halo Frame...")

        try:
            from frame_sdk import Frame

            if address:
                self.frame = await Frame.connect(address)
            else:
                # Auto-connect to first available
                self.frame = await Frame.discover_and_connect()

            if self.frame:
                self.state = ConnectionState.CONNECTED
                self._reconnect_attempts = 0

                # Get device info
                self.frame_info = FrameInfo(
                    name=self.frame.name or "Halo Frame",
                    address=self.frame.address or "unknown",
                    firmware_version=await self._get_firmware_version(),
                    battery_level=await self.get_battery_level()
                )

                logger.info(f"Connected to {self.frame_info.name}")

                if self._on_connect_callback:
                    await self._on_connect_callback()

                return True

            return False

        except ImportError:
            # Mock mode
            logger.warning("Frame SDK not installed. Using mock mode.")
            self.state = ConnectionState.CONNECTED
            self.frame = MockFrame()
            self.frame_info = FrameInfo(
                name="Mock Halo Frame",
                address="00:00:00:00:00:00",
                battery_level=100
            )
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.state = ConnectionState.ERROR
            return False

    async def disconnect(self):
        """Disconnect from the Frame."""
        if self.frame:
            try:
                await self.frame.disconnect()
            except Exception as e:
                logger.warning(f"Disconnect error: {e}")

        self.frame = None
        self.state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from Frame")

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to the last connected device.

        Returns:
            True if reconnected successfully
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return False

        self._reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}")

        # Wait before retry (exponential backoff)
        await asyncio.sleep(min(2 ** self._reconnect_attempts, 30))

        if self.frame_info and self.frame_info.address:
            return await self.connect(self.frame_info.address)

        return await self.connect()

    async def get_battery_level(self) -> int:
        """Get current battery level (0-100)."""
        if not self.frame:
            return 0

        try:
            return await self.frame.battery.level()
        except Exception:
            return 0

    async def _get_firmware_version(self) -> str:
        """Get firmware version."""
        if not self.frame:
            return ""

        try:
            return await self.frame.system.firmware_version()
        except Exception:
            return ""

    @property
    def is_connected(self) -> bool:
        """Check if connected to a Frame."""
        return self.state == ConnectionState.CONNECTED and self.frame is not None


class MockFrame:
    """Mock Frame for testing without hardware."""

    def __init__(self):
        self.display = MockDisplay()
        self.camera = MockCamera()
        self.microphone = MockMicrophone()
        self.motion = MockMotion()
        self.name = "Mock Frame"
        self.address = "00:00:00:00:00:00"

    async def disconnect(self):
        pass


class MockDisplay:
    """Mock display for testing."""

    async def clear(self):
        print("[MOCK DISPLAY] Cleared")

    async def text(self, text: str, x: int, y: int, color=None, font_size=32):
        print(f"[MOCK DISPLAY] ({x},{y}) '{text}' size={font_size}")

    async def show(self):
        print("[MOCK DISPLAY] Shown")


class MockCamera:
    """Mock camera for testing."""

    async def capture(self, resolution: int = 512) -> bytes:
        # Return a small test image
        return b'\x00' * 1024


class MockMicrophone:
    """Mock microphone for testing."""

    async def record(self, duration_ms: int = 5000) -> bytes:
        return b'\x00' * 1024


class MockMotion:
    """Mock motion sensor for testing."""

    async def tap_detected(self) -> bool:
        return False
