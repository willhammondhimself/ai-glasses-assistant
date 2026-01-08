"""Real-time voice agent status tracking and WebSocket broadcasting.

Tracks voice agent states and broadcasts updates to connected clients.
States: idle, listening, thinking, speaking, tool_executing
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.websocket.manager import ws_manager, WebSocketHandler, ConnectionInfo
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class VoiceAgentState(str, Enum):
    """Voice agent states."""
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    TOOL_EXECUTING = "tool_executing"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class VoiceStatusUpdate:
    """Voice status update event."""
    state: VoiceAgentState
    timestamp: datetime = field(default_factory=datetime.utcnow)
    room_name: Optional[str] = None
    participant: Optional[str] = None
    tool_name: Optional[str] = None  # When executing a tool
    transcript: Optional[str] = None  # User speech transcript
    message: Optional[str] = None  # Agent speech text
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "state": self.state.value,
            "timestamp": self.timestamp.isoformat(),
            "room_name": self.room_name,
            "participant": self.participant,
            "tool_name": self.tool_name,
            "transcript": self.transcript,
            "message": self.message,
            "metadata": self.metadata,
        }


class VoiceStatusManager:
    """Manages voice agent status and broadcasts updates."""

    def __init__(self):
        self._current_state = VoiceAgentState.DISCONNECTED
        self._room_name: Optional[str] = None
        self._participant: Optional[str] = None
        self._state_history: List[VoiceStatusUpdate] = []
        self._max_history = 100
        self._session_start: Optional[datetime] = None
        self._tool_executions: int = 0
        self._user_messages: int = 0
        self._agent_messages: int = 0

    @property
    def current_state(self) -> VoiceAgentState:
        """Get current voice agent state."""
        return self._current_state

    @property
    def is_active(self) -> bool:
        """Check if voice agent is active (not disconnected)."""
        return self._current_state != VoiceAgentState.DISCONNECTED

    async def set_state(
        self,
        state: VoiceAgentState,
        room_name: Optional[str] = None,
        participant: Optional[str] = None,
        tool_name: Optional[str] = None,
        transcript: Optional[str] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VoiceStatusUpdate:
        """Set voice agent state and broadcast update.

        Args:
            state: New voice agent state
            room_name: LiveKit room name
            participant: User participant name
            tool_name: Tool being executed (for TOOL_EXECUTING state)
            transcript: User speech transcript (for LISTENING state)
            message: Agent speech text (for SPEAKING state)
            metadata: Additional metadata

        Returns:
            The status update that was broadcast
        """
        self._current_state = state

        if room_name:
            self._room_name = room_name
        if participant:
            self._participant = participant

        # Track session stats
        if state == VoiceAgentState.CONNECTING:
            self._session_start = datetime.utcnow()
            self._tool_executions = 0
            self._user_messages = 0
            self._agent_messages = 0
        elif state == VoiceAgentState.TOOL_EXECUTING:
            self._tool_executions += 1
        elif state == VoiceAgentState.LISTENING and transcript:
            self._user_messages += 1
        elif state == VoiceAgentState.SPEAKING:
            self._agent_messages += 1

        # Create update
        update = VoiceStatusUpdate(
            state=state,
            room_name=self._room_name,
            participant=self._participant,
            tool_name=tool_name,
            transcript=transcript,
            message=message,
            metadata=metadata or {},
        )

        # Add to history
        self._state_history.append(update)
        if len(self._state_history) > self._max_history:
            self._state_history = self._state_history[-self._max_history:]

        # Broadcast to WebSocket subscribers
        await self._broadcast(update)

        logger.debug(f"Voice state: {state.value} (room={self._room_name})")
        return update

    async def _broadcast(self, update: VoiceStatusUpdate) -> int:
        """Broadcast status update to all voice status subscribers.

        Args:
            update: Status update to broadcast

        Returns:
            Number of clients that received the update
        """
        message = {
            "type": "voice_status",
            **update.to_dict()
        }

        count = await ws_manager.broadcast("voice:status", message)
        if count > 0:
            logger.debug(f"Broadcast voice status to {count} clients")
        return count

    def get_status(self) -> Dict[str, Any]:
        """Get current voice agent status summary.

        Returns:
            Status summary dictionary
        """
        session_duration = None
        if self._session_start:
            session_duration = (datetime.utcnow() - self._session_start).total_seconds()

        return {
            "state": self._current_state.value,
            "is_active": self.is_active,
            "room_name": self._room_name,
            "participant": self._participant,
            "session": {
                "start": self._session_start.isoformat() if self._session_start else None,
                "duration_seconds": session_duration,
                "tool_executions": self._tool_executions,
                "user_messages": self._user_messages,
                "agent_messages": self._agent_messages,
            },
            "recent_history": [u.to_dict() for u in self._state_history[-10:]],
        }

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get state change history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of status updates
        """
        return [u.to_dict() for u in self._state_history[-limit:]]


# Global voice status manager
voice_status = VoiceStatusManager()


class VoiceStatusHandler(WebSocketHandler):
    """WebSocket handler for voice status updates.

    Clients connect to receive real-time voice agent status updates.
    """

    async def handle_message(
        self,
        websocket: WebSocket,
        topic: str,
        data: dict,
        conn_info: ConnectionInfo
    ):
        """Handle incoming messages from voice status subscribers.

        Supports:
        - {"type": "get_status"} - Get current status
        - {"type": "get_history", "limit": N} - Get state history
        - {"type": "ping"} - Keep-alive
        """
        msg_type = data.get("type", "")

        if msg_type == "get_status":
            await websocket.send_json({
                "type": "status",
                **voice_status.get_status()
            })

        elif msg_type == "get_history":
            limit = data.get("limit", 50)
            await websocket.send_json({
                "type": "history",
                "updates": voice_status.get_history(limit)
            })

        elif msg_type == "ping":
            await websocket.send_json({
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat()
            })

        else:
            # Unknown message type
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            })

    async def on_disconnect(
        self,
        websocket: WebSocket,
        topic: str,
        conn_info: ConnectionInfo
    ):
        """Handle client disconnect."""
        logger.debug(f"Voice status client disconnected from {topic}")


# Global handler instance
voice_status_handler = VoiceStatusHandler()


# Convenience functions for setting common states

async def voice_connecting(room_name: str, participant: str) -> VoiceStatusUpdate:
    """Set state to connecting."""
    return await voice_status.set_state(
        VoiceAgentState.CONNECTING,
        room_name=room_name,
        participant=participant
    )


async def voice_idle() -> VoiceStatusUpdate:
    """Set state to idle."""
    return await voice_status.set_state(VoiceAgentState.IDLE)


async def voice_listening(transcript: Optional[str] = None) -> VoiceStatusUpdate:
    """Set state to listening."""
    return await voice_status.set_state(
        VoiceAgentState.LISTENING,
        transcript=transcript
    )


async def voice_thinking() -> VoiceStatusUpdate:
    """Set state to thinking."""
    return await voice_status.set_state(VoiceAgentState.THINKING)


async def voice_speaking(message: Optional[str] = None) -> VoiceStatusUpdate:
    """Set state to speaking."""
    return await voice_status.set_state(
        VoiceAgentState.SPEAKING,
        message=message
    )


async def voice_tool_executing(tool_name: str) -> VoiceStatusUpdate:
    """Set state to tool executing."""
    return await voice_status.set_state(
        VoiceAgentState.TOOL_EXECUTING,
        tool_name=tool_name
    )


async def voice_disconnected() -> VoiceStatusUpdate:
    """Set state to disconnected."""
    return await voice_status.set_state(VoiceAgentState.DISCONNECTED)


async def voice_error(message: str) -> VoiceStatusUpdate:
    """Set state to error."""
    return await voice_status.set_state(
        VoiceAgentState.ERROR,
        metadata={"error": message}
    )
