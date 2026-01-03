"""
WebSocket Connection Manager

Manages WebSocket connections for real-time AR glasses communication.
Supports multiple topics/channels for different modes (mental math, etc.)
"""

import asyncio
import logging
from typing import Dict, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    websocket: WebSocket
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_ping: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectionManager:
    """
    Manages WebSocket connections across multiple topics.

    Features:
    - Topic-based subscription (e.g., "mental-math:session123")
    - Broadcast to topic subscribers
    - Connection health monitoring
    - Graceful disconnect handling
    """

    def __init__(self):
        # topic -> set of ConnectionInfo
        self._connections: Dict[str, Set[ConnectionInfo]] = {}
        # websocket -> (topic, ConnectionInfo) for reverse lookup
        self._websocket_map: Dict[WebSocket, tuple[str, ConnectionInfo]] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        topic: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConnectionInfo:
        """
        Accept a WebSocket connection and subscribe to a topic.

        Args:
            websocket: The WebSocket connection
            topic: Topic to subscribe to (e.g., "mental-math:session123")
            metadata: Optional metadata about the connection

        Returns:
            ConnectionInfo for the new connection
        """
        await websocket.accept()

        conn_info = ConnectionInfo(
            websocket=websocket,
            metadata=metadata or {}
        )

        async with self._lock:
            if topic not in self._connections:
                self._connections[topic] = set()
            self._connections[topic].add(conn_info)
            self._websocket_map[websocket] = (topic, conn_info)

        logger.info(f"WebSocket connected to topic: {topic}")
        return conn_info

    async def disconnect(self, websocket: WebSocket) -> Optional[str]:
        """
        Remove a WebSocket connection.

        Args:
            websocket: The WebSocket to disconnect

        Returns:
            The topic the connection was subscribed to, or None
        """
        async with self._lock:
            if websocket not in self._websocket_map:
                return None

            topic, conn_info = self._websocket_map[websocket]

            if topic in self._connections:
                self._connections[topic].discard(conn_info)
                # Clean up empty topics
                if not self._connections[topic]:
                    del self._connections[topic]

            del self._websocket_map[websocket]

        logger.info(f"WebSocket disconnected from topic: {topic}")
        return topic

    async def send_personal(self, websocket: WebSocket, message: dict) -> bool:
        """
        Send a message to a specific connection.

        Args:
            websocket: Target WebSocket
            message: Message to send

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            await self.disconnect(websocket)
            return False

    async def broadcast(self, topic: str, message: dict) -> int:
        """
        Broadcast a message to all connections on a topic.

        Args:
            topic: Topic to broadcast to
            message: Message to broadcast

        Returns:
            Number of connections that received the message
        """
        if topic not in self._connections:
            return 0

        sent_count = 0
        failed_connections = []

        for conn_info in self._connections.get(topic, set()).copy():
            try:
                await conn_info.websocket.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Broadcast failed for connection: {e}")
                failed_connections.append(conn_info.websocket)

        # Clean up failed connections
        for ws in failed_connections:
            await self.disconnect(ws)

        return sent_count

    async def broadcast_except(
        self,
        topic: str,
        message: dict,
        exclude: WebSocket
    ) -> int:
        """
        Broadcast to all connections on a topic except one.

        Args:
            topic: Topic to broadcast to
            message: Message to broadcast
            exclude: WebSocket to exclude

        Returns:
            Number of connections that received the message
        """
        if topic not in self._connections:
            return 0

        sent_count = 0
        for conn_info in self._connections.get(topic, set()).copy():
            if conn_info.websocket != exclude:
                try:
                    await conn_info.websocket.send_json(message)
                    sent_count += 1
                except Exception:
                    await self.disconnect(conn_info.websocket)

        return sent_count

    def get_connection_count(self, topic: Optional[str] = None) -> int:
        """
        Get the number of active connections.

        Args:
            topic: Specific topic, or None for all connections

        Returns:
            Number of connections
        """
        if topic:
            return len(self._connections.get(topic, set()))
        return len(self._websocket_map)

    def get_topics(self) -> list[str]:
        """Get list of active topics."""
        return list(self._connections.keys())

    async def ping_all(self) -> Dict[str, int]:
        """
        Send ping to all connections to check health.

        Returns:
            Dict of topic -> number of healthy connections
        """
        results = {}
        now = datetime.utcnow()

        for topic, connections in list(self._connections.items()):
            healthy = 0
            for conn_info in list(connections):
                try:
                    await conn_info.websocket.send_json({"type": "ping"})
                    conn_info.last_ping = now
                    healthy += 1
                except Exception:
                    await self.disconnect(conn_info.websocket)
            results[topic] = healthy

        return results


# Global connection manager instance
ws_manager = ConnectionManager()


class WebSocketHandler:
    """
    Base class for WebSocket message handlers.

    Subclass and implement handle_message for specific functionality.
    """

    def __init__(self, manager: ConnectionManager = None):
        self.manager = manager or ws_manager

    async def handle_connection(
        self,
        websocket: WebSocket,
        topic: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Handle a WebSocket connection lifecycle.

        Args:
            websocket: The WebSocket connection
            topic: Topic to subscribe to
            metadata: Optional connection metadata
        """
        conn_info = await self.manager.connect(websocket, topic, metadata)

        try:
            # Send welcome message
            await websocket.send_json({
                "type": "connected",
                "topic": topic,
                "timestamp": datetime.utcnow().isoformat()
            })

            # Message loop
            while True:
                data = await websocket.receive_json()
                await self.handle_message(websocket, topic, data, conn_info)

        except WebSocketDisconnect:
            logger.info(f"Client disconnected from {topic}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except Exception:
                pass
        finally:
            await self.manager.disconnect(websocket)
            await self.on_disconnect(websocket, topic, conn_info)

    async def handle_message(
        self,
        websocket: WebSocket,
        topic: str,
        data: dict,
        conn_info: ConnectionInfo
    ):
        """
        Handle an incoming message. Override in subclasses.

        Args:
            websocket: The WebSocket that sent the message
            topic: Topic the connection is subscribed to
            data: The message data
            conn_info: Connection info for this websocket
        """
        # Default: echo back
        await websocket.send_json({
            "type": "echo",
            "data": data
        })

    async def on_disconnect(
        self,
        websocket: WebSocket,
        topic: str,
        conn_info: ConnectionInfo
    ):
        """
        Called when a connection is closed. Override for cleanup.

        Args:
            websocket: The disconnected WebSocket
            topic: Topic it was subscribed to
            conn_info: Connection info
        """
        pass
