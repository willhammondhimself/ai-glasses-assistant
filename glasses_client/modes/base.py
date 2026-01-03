"""
Base Mode

Abstract base class for all user flow modes.
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional

from glasses_client.core.display import DisplayController
from glasses_client.core.state import SessionState, AppMode
from glasses_client.input.voice import VoiceHandler
from glasses_client.input.tap import TapHandler
from glasses_client.api.client import APIClient
from glasses_client.api.websocket import WebSocketClient
from glasses_client.rendering.transformer import ResponseTransformer

logger = logging.getLogger(__name__)


class BaseMode(ABC):
    """
    Base class for all user flow modes.

    Provides common functionality:
    - Display rendering
    - Voice input handling
    - Tap gesture handling
    - API communication
    - State management
    """

    # The AppMode enum value for this mode
    mode_type: AppMode = AppMode.IDLE

    def __init__(
        self,
        display: DisplayController,
        voice: VoiceHandler,
        tap: TapHandler,
        api: APIClient,
        state: SessionState,
        frame=None
    ):
        """
        Initialize the mode.

        Args:
            display: Display controller
            voice: Voice input handler
            tap: Tap gesture handler
            api: REST API client
            state: Session state manager
            frame: Frame SDK instance (optional)
        """
        self.display = display
        self.voice = voice
        self.tap = tap
        self.api = api
        self.state = state
        self.frame = frame
        self.transformer = ResponseTransformer()
        self._ws: Optional[WebSocketClient] = None
        self._running = False

    @abstractmethod
    async def run(self, **kwargs):
        """
        Run the mode.

        Override in subclasses to implement the mode's main loop.
        """
        pass

    async def start(self, **kwargs):
        """
        Start the mode.

        Sets state and calls run().
        """
        self.state.set_mode(self.mode_type)
        self._running = True

        logger.info(f"Starting mode: {self.mode_type.value}")

        try:
            await self.run(**kwargs)
        except Exception as e:
            logger.error(f"Mode error: {e}")
            await self.display.show_error(str(e))
            raise
        finally:
            await self.stop()

    async def stop(self):
        """
        Stop the mode.

        Cleans up resources and resets state.
        """
        self._running = False

        # Close WebSocket if open
        if self._ws:
            await self._ws.disconnect()
            self._ws = None

        self.state.set_mode(AppMode.IDLE)
        logger.info(f"Stopped mode: {self.mode_type.value}")

    async def connect_websocket(self, path: str) -> WebSocketClient:
        """
        Connect to a WebSocket endpoint.

        Args:
            path: WebSocket path (e.g., "/ws/mental-math/session1")

        Returns:
            Connected WebSocketClient
        """
        self._ws = WebSocketClient()
        success = await self._ws.connect(path)

        if not success:
            raise ConnectionError(f"Failed to connect to WebSocket: {path}")

        return self._ws

    async def wait_for_input(self, timeout_ms: int = 5000) -> str:
        """
        Wait for either voice command or tap gesture.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            Action string: "next", "back", "exit", or specific command
        """
        # Create tasks for both input types
        voice_task = asyncio.create_task(self.voice.listen(timeout_ms))
        tap_task = asyncio.create_task(self.tap.wait_for_tap(timeout_ms))

        try:
            # Wait for first to complete
            done, pending = await asyncio.wait(
                [voice_task, tap_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel the other
            for task in pending:
                task.cancel()

            # Get result from completed task
            for task in done:
                result = task.result()

                if result:
                    # Voice result
                    if hasattr(result, 'command'):
                        return result.command.value

                    # Tap result
                    if hasattr(result, 'value'):
                        from glasses_client.input.tap import gesture_to_action
                        return gesture_to_action(result)

            return "none"

        except Exception as e:
            logger.warning(f"Input wait error: {e}")
            return "none"


# Import asyncio for wait_for_input
import asyncio
