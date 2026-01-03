"""
WebSocket Client

WebSocket client for real-time backend communication.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass
import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConfig:
    """WebSocket client configuration."""
    base_url: str = "ws://localhost:8000"
    reconnect_attempts: int = 5
    reconnect_delay_seconds: float = 1.0
    ping_interval_seconds: float = 30.0
    ping_timeout_seconds: float = 10.0


class WebSocketClient:
    """
    WebSocket client for real-time communication.

    Features:
    - Automatic reconnection
    - Message handlers
    - Ping/pong keep-alive
    - Clean disconnect
    """

    def __init__(self, config: Optional[WebSocketConfig] = None):
        self.config = config or WebSocketConfig()
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._handlers: Dict[str, Callable[[dict], Awaitable[None]]] = {}
        self._receive_task: Optional[asyncio.Task] = None
        self._default_handler: Optional[Callable[[dict], Awaitable[None]]] = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    def on_message(self, message_type: str):
        """
        Decorator to register a handler for a message type.

        Usage:
            @client.on_message("problem")
            async def handle_problem(data):
                print(f"Got problem: {data}")
        """
        def decorator(handler: Callable[[dict], Awaitable[None]]):
            self._handlers[message_type] = handler
            return handler
        return decorator

    def set_default_handler(self, handler: Callable[[dict], Awaitable[None]]):
        """Set handler for messages without a specific type handler."""
        self._default_handler = handler

    async def connect(self, path: str) -> bool:
        """
        Connect to a WebSocket endpoint.

        Args:
            path: WebSocket path (e.g., "/ws/mental-math/session123")

        Returns:
            True if connected successfully
        """
        url = f"{self.config.base_url}{path}"
        last_error = None

        for attempt in range(self.config.reconnect_attempts):
            try:
                self._ws = await websockets.connect(
                    url,
                    ping_interval=self.config.ping_interval_seconds,
                    ping_timeout=self.config.ping_timeout_seconds
                )
                self._connected = True
                logger.info(f"Connected to WebSocket: {url}")

                # Start receive loop
                self._receive_task = asyncio.create_task(self._receive_loop())

                return True

            except Exception as e:
                last_error = e
                logger.warning(f"WebSocket connection failed (attempt {attempt + 1}): {e}")

                if attempt < self.config.reconnect_attempts - 1:
                    await asyncio.sleep(
                        self.config.reconnect_delay_seconds * (attempt + 1)
                    )

        logger.error(f"Failed to connect after {self.config.reconnect_attempts} attempts")
        return False

    async def disconnect(self):
        """Disconnect from the WebSocket."""
        self._connected = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        logger.info("Disconnected from WebSocket")

    async def send(self, data: Dict[str, Any]) -> bool:
        """
        Send a message.

        Args:
            data: Message data (will be JSON encoded)

        Returns:
            True if sent successfully
        """
        if not self.is_connected:
            logger.warning("Cannot send: not connected")
            return False

        try:
            await self._ws.send(json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False

    async def receive(self, timeout: float = None) -> Optional[Dict[str, Any]]:
        """
        Receive a single message.

        Args:
            timeout: Timeout in seconds (None = wait forever)

        Returns:
            Message data or None on timeout/error
        """
        if not self.is_connected:
            return None

        try:
            if timeout:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
            else:
                raw = await self._ws.recv()

            return json.loads(raw)

        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Receive error: {e}")
            return None

    async def _receive_loop(self):
        """Background loop to receive and dispatch messages."""
        try:
            while self._connected and self._ws:
                try:
                    raw = await self._ws.recv()
                    data = json.loads(raw)

                    # Dispatch to handler
                    message_type = data.get("type", "unknown")

                    if message_type in self._handlers:
                        await self._handlers[message_type](data)
                    elif self._default_handler:
                        await self._default_handler(data)
                    else:
                        logger.debug(f"Unhandled message type: {message_type}")

                except websockets.ConnectionClosed:
                    logger.info("WebSocket connection closed")
                    break
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON received: {e}")
                except Exception as e:
                    logger.error(f"Receive loop error: {e}")

        finally:
            self._connected = False

    # Convenience methods for mental math mode

    async def start_problem(self, difficulty: int = 2, category: str = "arithmetic") -> bool:
        """Request a new mental math problem."""
        return await self.send({
            "action": "start",
            "difficulty": difficulty,
            "category": category
        })

    async def submit_answer(self, answer: Any) -> bool:
        """Submit an answer to the current problem."""
        return await self.send({
            "action": "answer",
            "answer": answer
        })

    async def skip_problem(self) -> bool:
        """Skip the current problem."""
        return await self.send({"action": "skip"})

    async def get_stats(self) -> bool:
        """Request session statistics."""
        return await self.send({"action": "stats"})

    async def end_session(self) -> bool:
        """End the current session."""
        return await self.send({"action": "end"})
