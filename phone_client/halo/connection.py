"""
Bluetooth LE connection to Brilliant Labs Halo AR Glasses.
Handles display updates, TTS, and camera capture.
"""
import asyncio
import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

try:
    from bleak import BleakClient, BleakScanner
    from bleak.exc import BleakError
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    BleakClient = None
    BleakScanner = None
    BleakError = Exception

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


@dataclass
class HaloConfig:
    """Configuration for Halo connection."""
    mac_address: str
    display_width: int = 640
    display_height: int = 400
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0
    tts_rate: int = 180
    tts_voice: str = "british_male"


# Halo BLE Service UUIDs (Brilliant Labs specific)
HALO_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
HALO_TX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # Write to Halo
HALO_RX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # Read from Halo


class HaloConnection:
    """
    Manages Bluetooth LE connection to Halo AR glasses.

    Provides async methods for:
    - send_lua(): Push display updates via Lua scripts
    - speak(): Text-to-speech output
    - capture_image(): Take photo from glasses camera
    """

    def __init__(self, config: HaloConfig):
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self._client: Optional[BleakClient] = None
        self._tts_engine = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._notification_handlers: list[Callable] = []
        self._last_lua: str = ""

        # Initialize TTS
        if TTS_AVAILABLE:
            self._init_tts()

    def _init_tts(self):
        """Initialize text-to-speech engine."""
        try:
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty('rate', self.config.tts_rate)

            # Try to set British voice if available
            voices = self._tts_engine.getProperty('voices')
            for voice in voices:
                if 'british' in voice.name.lower() or 'daniel' in voice.name.lower():
                    self._tts_engine.setProperty('voice', voice.id)
                    break
        except Exception as e:
            logger.warning(f"TTS initialization failed: {e}")
            self._tts_engine = None

    async def connect(self) -> bool:
        """
        Connect to Halo glasses via Bluetooth LE.
        Returns True if connection successful.
        """
        if not BLEAK_AVAILABLE:
            logger.error("Bleak not installed. Run: pip install bleak")
            return False

        self.state = ConnectionState.CONNECTING
        logger.info(f"Connecting to Halo at {self.config.mac_address}...")

        try:
            self._client = BleakClient(
                self.config.mac_address,
                disconnected_callback=self._on_disconnect
            )
            await self._client.connect()

            # Subscribe to notifications from Halo
            await self._client.start_notify(
                HALO_RX_CHAR_UUID,
                self._on_notification
            )

            self.state = ConnectionState.CONNECTED
            logger.info("Connected to Halo successfully")
            return True

        except BleakError as e:
            logger.error(f"Bluetooth connection failed: {e}")
            self.state = ConnectionState.DISCONNECTED
            return False
        except Exception as e:
            logger.error(f"Unexpected connection error: {e}")
            self.state = ConnectionState.DISCONNECTED
            return False

    async def disconnect(self):
        """Disconnect from Halo glasses."""
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._client and self._client.is_connected:
            await self._client.disconnect()

        self.state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from Halo")

    def _on_disconnect(self, client: BleakClient):
        """Handle unexpected disconnection."""
        logger.warning("Halo disconnected unexpectedly")
        self.state = ConnectionState.DISCONNECTED

        # Start auto-reconnect
        if not self._reconnect_task:
            self._reconnect_task = asyncio.create_task(self._auto_reconnect())

    async def _auto_reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        self.state = ConnectionState.RECONNECTING
        delay = self.config.reconnect_delay

        for attempt in range(self.config.reconnect_attempts):
            logger.info(f"Reconnect attempt {attempt + 1}/{self.config.reconnect_attempts}")

            if await self.connect():
                self._reconnect_task = None
                return

            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 30.0)  # Max 30 second delay

        logger.error("Failed to reconnect to Halo after multiple attempts")
        self._reconnect_task = None

    def _on_notification(self, sender: int, data: bytearray):
        """Handle notifications from Halo."""
        message = data.decode('utf-8', errors='ignore')
        logger.debug(f"Halo notification: {message}")

        for handler in self._notification_handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Notification handler error: {e}")

    def add_notification_handler(self, handler: Callable[[str], None]):
        """Register a handler for Halo notifications."""
        self._notification_handlers.append(handler)

    async def send_lua(self, lua_code: str) -> bool:
        """
        Send Lua script to Halo for display rendering.

        Args:
            lua_code: Lua script to execute on Halo

        Returns:
            True if sent successfully
        """
        if self.state != ConnectionState.CONNECTED:
            logger.warning("Cannot send Lua: not connected")
            return False

        try:
            # Halo expects Lua code as UTF-8 bytes
            data = lua_code.encode('utf-8')

            # Send in chunks if necessary (BLE MTU limit ~512 bytes)
            chunk_size = 500
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                await self._client.write_gatt_char(
                    HALO_TX_CHAR_UUID,
                    chunk,
                    response=True
                )

            self._last_lua = lua_code
            return True

        except BleakError as e:
            logger.error(f"Failed to send Lua: {e}")
            return False

    async def speak(self, text: str, block: bool = False) -> bool:
        """
        Speak text via TTS.

        Args:
            text: Text to speak
            block: If True, wait for speech to complete

        Returns:
            True if speech initiated successfully
        """
        if not self._tts_engine:
            logger.warning("TTS not available")
            return False

        try:
            if block:
                # Run TTS in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self._tts_engine.say(text) or self._tts_engine.runAndWait()
                )
            else:
                # Non-blocking speech
                self._tts_engine.say(text)
                asyncio.get_event_loop().run_in_executor(
                    None,
                    self._tts_engine.runAndWait
                )
            return True

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False

    async def capture_image(self) -> Optional[bytes]:
        """
        Capture image from Halo camera.

        Returns:
            JPEG image bytes or None if capture failed
        """
        if self.state != ConnectionState.CONNECTED:
            logger.warning("Cannot capture: not connected")
            return None

        try:
            # Send capture command to Halo
            capture_lua = "camera.capture()"
            await self.send_lua(capture_lua)

            # Wait for image data (Halo sends as notification)
            # This is simplified - real implementation would use proper protocol
            await asyncio.sleep(0.5)

            # TODO: Implement actual image capture protocol
            logger.info("Image capture requested (protocol TBD)")
            return None

        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to Halo."""
        return self.state == ConnectionState.CONNECTED

    async def clear_display(self):
        """Clear the Halo display."""
        await self.send_lua("frame.display.clear()")

    async def set_brightness(self, level: int):
        """Set display brightness (0-100)."""
        level = max(0, min(100, level))
        await self.send_lua(f"frame.display.set_brightness({level})")


async def scan_for_halo(timeout: float = 10.0) -> list[dict]:
    """
    Scan for nearby Halo devices.

    Returns:
        List of found devices with name and address
    """
    if not BLEAK_AVAILABLE:
        logger.error("Bleak not installed")
        return []

    found_devices = []

    def detection_callback(device, advertisement_data):
        if device.name and ('halo' in device.name.lower() or 'brilliant' in device.name.lower()):
            found_devices.append({
                'name': device.name,
                'address': device.address,
                'rssi': advertisement_data.rssi
            })

    scanner = BleakScanner(detection_callback)
    await scanner.start()
    await asyncio.sleep(timeout)
    await scanner.stop()

    return found_devices


# Example usage
async def main():
    """Test connection to Halo."""
    # First, scan for devices
    print("Scanning for Halo devices...")
    devices = await scan_for_halo(5.0)

    if not devices:
        print("No Halo devices found. Make sure glasses are on and in range.")
        return

    print(f"Found devices: {devices}")

    # Connect to first found device
    config = HaloConfig(mac_address=devices[0]['address'])
    halo = HaloConnection(config)

    if await halo.connect():
        print("Connected!")

        # Test display
        await halo.send_lua('''
            frame.display.clear()
            frame.display.text("JARVIS Online", 320, 200, {color="cyan", align="center"})
            frame.display.show()
        ''')

        # Test TTS
        await halo.speak("Systems online, sir.", block=True)

        await asyncio.sleep(3)
        await halo.disconnect()
    else:
        print("Connection failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
