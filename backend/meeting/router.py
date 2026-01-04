"""
Meeting Mode WebSocket Router

Handles real-time communication between glasses client and meeting services.

Protocol:
    Client → Server:
    - meeting_start: Start a meeting session
    - meeting_end: End the meeting
    - audio_chunk: Raw audio for transcription (base64)
    - double_tap: Quick suggestion trigger
    - voice_command: Voice command with recorded query

    Server → Client:
    - transcript_update: New transcription segment
    - suggestion: WHAM's suggestion
    - status: Connection/processing status
    - error: Error message
"""

import asyncio
import base64
import logging
import time
from typing import Dict, Optional, Any
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from backend.websocket.manager import WebSocketHandler, ConnectionInfo, ws_manager

from .models import (
    MeetingSession,
    MeetingConfig,
    TranscriptSegment,
    AudioChunk,
    TriggerType,
    WSMessageType,
    SpeakerRole,
)
from .transcription import TranscriptionService, AudioBuffer
from .context import ContextManager, get_context_manager
from .suggestions import SuggestionEngine, get_suggestion_engine

logger = logging.getLogger(__name__)


class MeetingHandler(WebSocketHandler):
    """
    WebSocket handler for meeting mode.

    Manages:
    - Audio streaming and transcription
    - Double-tap quick suggestions
    - Voice command detailed suggestions
    - Real-time transcript updates
    """

    def __init__(self):
        super().__init__()
        self.transcription = TranscriptionService()
        self._sessions: Dict[str, MeetingSession] = {}
        self._audio_buffers: Dict[str, AudioBuffer] = {}
        self._transcription_tasks: Dict[str, asyncio.Task] = {}

    async def handle_message(
        self,
        websocket: WebSocket,
        topic: str,
        data: dict,
        conn_info: ConnectionInfo
    ):
        """Handle incoming meeting mode messages."""
        msg_type = data.get("type", "")
        session_id = topic.split(":")[-1] if ":" in topic else topic

        try:
            if msg_type == WSMessageType.MEETING_START.value:
                await self._handle_meeting_start(websocket, session_id, data)

            elif msg_type == WSMessageType.MEETING_END.value:
                await self._handle_meeting_end(websocket, session_id)

            elif msg_type == WSMessageType.AUDIO_CHUNK.value:
                await self._handle_audio_chunk(websocket, session_id, data)

            elif msg_type == WSMessageType.DOUBLE_TAP.value:
                await self._handle_double_tap(websocket, session_id)

            elif msg_type == WSMessageType.VOICE_COMMAND.value:
                await self._handle_voice_command(websocket, session_id, data)

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await self._send_error(websocket, f"Unknown message type: {msg_type}")

        except Exception as e:
            logger.error(f"Meeting handler error: {e}")
            await self._send_error(websocket, str(e))

    async def _handle_meeting_start(
        self,
        websocket: WebSocket,
        session_id: str,
        data: dict
    ):
        """Start a new meeting session."""
        # Parse config
        config_data = data.get("config", {})
        config = MeetingConfig(
            meeting_type=config_data.get("meeting_type", "general"),
            participants=config_data.get("participants", []),
            context=config_data.get("context", ""),
            quick_context_segments=config_data.get("quick_context_segments", 5),
            proactive_suggestions=config_data.get("proactive_suggestions", False),
        )

        # Create session
        ctx_manager = get_context_manager()
        session = ctx_manager.create_session(config)

        # Store locally for quick access
        self._sessions[session_id] = session
        self._audio_buffers[session_id] = AudioBuffer()

        logger.info(f"Meeting started: {session.id} (type: {config.meeting_type})")

        await websocket.send_json({
            "type": WSMessageType.STATUS.value,
            "status": "meeting_started",
            "session_id": session.id,
            "message": "WHAM is listening. Double-tap for quick help, or say 'WHAM' for detailed assistance.",
        })

    async def _handle_meeting_end(
        self,
        websocket: WebSocket,
        session_id: str
    ):
        """End the meeting session."""
        ctx_manager = get_context_manager()
        session = ctx_manager.end_session(session_id)

        # Cancel any pending transcription
        if session_id in self._transcription_tasks:
            self._transcription_tasks[session_id].cancel()
            del self._transcription_tasks[session_id]

        # Clean up
        if session_id in self._sessions:
            del self._sessions[session_id]
        if session_id in self._audio_buffers:
            del self._audio_buffers[session_id]

        # Get meeting summary
        summary = ctx_manager.get_meeting_summary(session_id) if session else {}

        await websocket.send_json({
            "type": WSMessageType.STATUS.value,
            "status": "meeting_ended",
            "summary": summary,
            "message": "Meeting ended. Good work, Will.",
        })

    async def _handle_audio_chunk(
        self,
        websocket: WebSocket,
        session_id: str,
        data: dict
    ):
        """Process an audio chunk for transcription."""
        session = self._sessions.get(session_id)
        if not session:
            await self._send_error(websocket, "No active session")
            return

        # Decode audio
        try:
            audio_bytes = base64.b64decode(data.get("audio", ""))
        except Exception as e:
            await self._send_error(websocket, f"Invalid audio data: {e}")
            return

        chunk = AudioChunk(
            data=audio_bytes,
            duration_ms=data.get("duration_ms", 1000),
            sample_rate=data.get("sample_rate", 16000),
        )

        # Add to buffer
        buffer = self._audio_buffers.get(session_id)
        if not buffer:
            buffer = AudioBuffer()
            self._audio_buffers[session_id] = buffer

        # Transcribe in background
        asyncio.create_task(
            self._transcribe_and_broadcast(websocket, session_id, chunk)
        )

    async def _transcribe_and_broadcast(
        self,
        websocket: WebSocket,
        session_id: str,
        chunk: AudioChunk
    ):
        """Transcribe audio and broadcast result."""
        try:
            segment = await self.transcription.transcribe(chunk)

            if segment and segment.text.strip():
                # Add to context
                ctx_manager = get_context_manager()
                ctx_manager.add_segment(session_id, segment)

                # Broadcast transcript update
                await websocket.send_json({
                    "type": WSMessageType.TRANSCRIPT_UPDATE.value,
                    "segment": segment.to_dict(),
                })

        except Exception as e:
            logger.error(f"Transcription error: {e}")

    async def _handle_double_tap(
        self,
        websocket: WebSocket,
        session_id: str
    ):
        """Handle double-tap for quick suggestion."""
        start_time = time.perf_counter()

        ctx_manager = get_context_manager()
        suggestion_engine = get_suggestion_engine()

        # Get quick context (last 5 segments)
        context = ctx_manager.get_context_window(
            session_id,
            TriggerType.DOUBLE_TAP
        )

        if not context:
            await self._send_error(websocket, "No active session")
            return

        # Send processing status
        await websocket.send_json({
            "type": WSMessageType.STATUS.value,
            "status": "processing",
            "message": "Thinking...",
        })

        # Get suggestion
        suggestion = await suggestion_engine.get_suggestion(
            context,
            TriggerType.DOUBLE_TAP
        )

        # Track suggestion count
        session = ctx_manager.get_session(session_id)
        if session:
            session.suggestions_given += 1

        total_latency = (time.perf_counter() - start_time) * 1000

        await websocket.send_json({
            "type": WSMessageType.SUGGESTION.value,
            "trigger": "double_tap",
            "suggestion": suggestion.to_dict(),
            "total_latency_ms": total_latency,
        })

        logger.info(f"Quick suggestion in {total_latency:.0f}ms: {suggestion.suggestion[:50]}...")

    async def _handle_voice_command(
        self,
        websocket: WebSocket,
        session_id: str,
        data: dict
    ):
        """Handle voice command for detailed suggestion."""
        start_time = time.perf_counter()

        user_query = data.get("query", "")
        audio_b64 = data.get("audio")

        # If audio provided, transcribe it first
        if audio_b64 and not user_query:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                chunk = AudioChunk(
                    data=audio_bytes,
                    duration_ms=data.get("duration_ms", 5000),
                )
                segment = await self.transcription.transcribe(chunk)
                if segment:
                    user_query = segment.text
            except Exception as e:
                logger.error(f"Voice command transcription error: {e}")

        if not user_query:
            await self._send_error(websocket, "No voice command detected")
            return

        ctx_manager = get_context_manager()
        suggestion_engine = get_suggestion_engine()

        # Get full context
        context = ctx_manager.get_context_window(
            session_id,
            TriggerType.VOICE_COMMAND,
            user_query
        )

        if not context:
            await self._send_error(websocket, "No active session")
            return

        # Send processing status
        await websocket.send_json({
            "type": WSMessageType.STATUS.value,
            "status": "processing",
            "message": f"Processing: {user_query[:30]}...",
        })

        # Get suggestion
        suggestion = await suggestion_engine.get_suggestion(
            context,
            TriggerType.VOICE_COMMAND,
            user_query
        )

        # Track suggestion count
        session = ctx_manager.get_session(session_id)
        if session:
            session.suggestions_given += 1

        total_latency = (time.perf_counter() - start_time) * 1000

        await websocket.send_json({
            "type": WSMessageType.SUGGESTION.value,
            "trigger": "voice_command",
            "query": user_query,
            "suggestion": suggestion.to_dict(),
            "total_latency_ms": total_latency,
        })

        logger.info(f"Full suggestion in {total_latency:.0f}ms for '{user_query[:30]}...'")

    async def _send_error(self, websocket: WebSocket, message: str):
        """Send error message to client."""
        await websocket.send_json({
            "type": WSMessageType.ERROR.value,
            "message": message,
        })

    async def on_disconnect(
        self,
        websocket: WebSocket,
        topic: str,
        conn_info: ConnectionInfo
    ):
        """Clean up on disconnect."""
        session_id = topic.split(":")[-1] if ":" in topic else topic

        # Cancel transcription task
        if session_id in self._transcription_tasks:
            self._transcription_tasks[session_id].cancel()
            del self._transcription_tasks[session_id]

        # Don't end the meeting session - it might reconnect
        # Just clean up local state
        if session_id in self._audio_buffers:
            del self._audio_buffers[session_id]

        logger.info(f"Meeting WebSocket disconnected: {session_id}")


# Singleton handler instance
meeting_handler = MeetingHandler()
