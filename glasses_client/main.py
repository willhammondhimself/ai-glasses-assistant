"""
Halo Frame AR Glasses Client - Main Entry Point

Main application for the Brilliant Labs Halo Frame AR glasses.
Provides voice-controlled access to educational content.
"""

import asyncio
import logging
import sys
from typing import Optional

from glasses_client.core.display import DisplayController
from glasses_client.core.connection import ConnectionManager
from glasses_client.core.state import SessionState, AppMode
from glasses_client.input.voice import VoiceHandler, VoiceCommand
from glasses_client.input.tap import TapHandler
from glasses_client.api.client import APIClient, APIConfig
from glasses_client.api.websocket import WebSocketClient, WebSocketConfig
from glasses_client.modes.mental_math import MentalMathMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HaloFrameApp:
    """
    Main application for Halo Frame glasses.

    Manages:
    - Bluetooth connection to glasses
    - Mode selection and execution
    - Voice command handling
    - State persistence
    """

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        ws_base_url: str = "ws://localhost:8000"
    ):
        """
        Initialize the application.

        Args:
            api_base_url: Backend API URL
            ws_base_url: Backend WebSocket URL
        """
        # Configuration
        self.api_config = APIConfig(base_url=api_base_url)
        self.ws_config = WebSocketConfig(base_url=ws_base_url)

        # Core components (initialized in setup)
        self.connection: Optional[ConnectionManager] = None
        self.display: Optional[DisplayController] = None
        self.voice: Optional[VoiceHandler] = None
        self.tap: Optional[TapHandler] = None
        self.api: Optional[APIClient] = None
        self.state: Optional[SessionState] = None

        # Frame reference
        self.frame = None

        # Running state
        self._running = False

    async def setup(self) -> bool:
        """
        Set up the application.

        Returns:
            True if setup succeeded
        """
        logger.info("Setting up Halo Frame application...")

        # Initialize connection manager
        self.connection = ConnectionManager()

        # Try to connect to glasses
        logger.info("Scanning for Halo Frame...")
        frame_info = await self.connection.scan_for_frame(timeout=10.0)

        if frame_info:
            logger.info(f"Found: {frame_info.name}")
            connected = await self.connection.connect()

            if connected:
                self.frame = self.connection.frame
                logger.info("Connected to Halo Frame!")
            else:
                logger.warning("Failed to connect. Using mock mode.")
        else:
            logger.warning("No Halo Frame found. Using mock mode.")

        # Initialize components
        self.display = DisplayController(self.frame)
        self.voice = VoiceHandler(self.frame)
        self.tap = TapHandler(self.frame)
        self.api = APIClient(self.api_config)
        self.state = SessionState()

        # Check backend health
        if await self.api.health_check():
            logger.info("Backend connected!")
            self.state.set_online(True)
        else:
            logger.warning("Backend not available. Offline mode.")
            self.state.set_online(False)

        return True

    async def run(self):
        """Run the main application loop."""
        if not await self.setup():
            logger.error("Setup failed")
            return

        self._running = True
        logger.info("Starting Halo Frame application...")

        try:
            # Show welcome screen
            await self._show_welcome()

            # Main mode selection loop
            while self._running:
                mode = await self._select_mode()

                if mode == AppMode.MENTAL_MATH:
                    await self._run_mental_math()
                elif mode == AppMode.IDLE:
                    # User chose to exit
                    break
                else:
                    await self.display.show_error(f"Mode not implemented: {mode.value}")
                    await asyncio.sleep(2)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
            await self.display.show_error(str(e))
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean up and shut down."""
        self._running = False
        logger.info("Shutting down...")

        # Close API client
        if self.api:
            await self.api.close()

        # Disconnect from glasses
        if self.connection:
            await self.connection.disconnect()

        logger.info("Goodbye!")

    async def _show_welcome(self):
        """Show welcome screen."""
        from glasses_client.core.display import DisplaySlide, HALO_COLORS

        slide = DisplaySlide(
            lines=[
                "AI Glasses Coach",
                "",
                "Say a mode name:",
                "'mental math', 'camera', 'debug'"
            ],
            color=HALO_COLORS["info"],
            centered=True
        )
        await self.display.render_slide(slide)

    async def _select_mode(self) -> AppMode:
        """Wait for user to select a mode."""
        while self._running:
            # Listen for voice command
            result = await self.voice.listen(timeout_ms=10000)

            if result:
                command = result.command

                if command == VoiceCommand.MENTAL_MATH:
                    return AppMode.MENTAL_MATH
                elif command == VoiceCommand.CAMERA:
                    return AppMode.CAMERA_SOLVE
                elif command == VoiceCommand.DEBUG:
                    return AppMode.CODE_DEBUG
                elif command == VoiceCommand.EXIT:
                    return AppMode.IDLE
                elif command != VoiceCommand.UNKNOWN:
                    logger.info(f"Unhandled command: {command}")

            # Check for tap (show help)
            tap = await self.tap.wait_for_tap(timeout_ms=100)
            if tap:
                await self._show_welcome()

        return AppMode.IDLE

    async def _run_mental_math(self):
        """Run mental math mode."""
        mode = MentalMathMode(
            display=self.display,
            voice=self.voice,
            tap=self.tap,
            api=self.api,
            state=self.state,
            frame=self.frame
        )

        # Get difficulty from preferences
        difficulty = self.state.preferences.default_difficulty

        try:
            await mode.start(difficulty=difficulty)
        except Exception as e:
            logger.error(f"Mental math mode error: {e}")
            await self.display.show_error(str(e))


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Halo Frame AR Glasses Client")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Backend API URL"
    )
    parser.add_argument(
        "--ws-url",
        default="ws://localhost:8000",
        help="Backend WebSocket URL"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    app = HaloFrameApp(
        api_base_url=args.api_url,
        ws_base_url=args.ws_url
    )

    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
