"""
Integration tests for WHAM Meeting Mode.

Tests:
1. WebSocket connection flow (start/end meeting)
2. Double-tap quick suggestion (target: 2-3s, last 5 segments)
3. Voice command detailed suggestion (target: 3-4s, full context)
4. Audio chunk processing and transcription
5. Error handling (invalid messages, missing session, malformed audio)

Run with:
    pytest backend/tests/test_meeting_mode.py -v
"""

import asyncio
import base64
import time
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from backend.meeting.router import MeetingHandler, meeting_handler
from backend.meeting.models import (
    WSMessageType,
    TranscriptSegment,
    SuggestionResponse,
    SuggestionType,
    SpeakerRole,
    TriggerType,
)


pytestmark = pytest.mark.asyncio


class TestMeetingConnectionFlow:
    """Test 1: WebSocket connection flow."""

    async def test_meeting_start_returns_session_id(
        self,
        mock_websocket,
        mock_context_manager,
        mock_connection_info,
    ):
        """Connect → meeting_start → receive session_id."""
        handler = MeetingHandler()
        mock_cm, sessions = mock_context_manager

        # Send meeting_start
        await handler.handle_message(
            mock_websocket,
            topic="meeting:test-session",
            data={
                "type": "meeting_start",
                "config": {
                    "meeting_type": "negotiation",
                    "participants": ["Will", "Counterpart"],
                },
            },
            conn_info=mock_connection_info,
        )

        # Verify response
        assert len(mock_websocket.sent_messages) == 1
        response = mock_websocket.sent_messages[0]
        assert response["type"] == WSMessageType.STATUS.value
        assert response["status"] == "meeting_started"
        assert "session_id" in response

    async def test_meeting_end_returns_summary(
        self,
        mock_websocket,
        mock_context_manager,
        mock_connection_info,
    ):
        """Start meeting → end meeting → receive summary."""
        handler = MeetingHandler()
        mock_cm, sessions = mock_context_manager

        # Start meeting
        await handler.handle_message(
            mock_websocket,
            topic="meeting:test-session",
            data={"type": "meeting_start", "config": {}},
            conn_info=mock_connection_info,
        )

        # Get session_id from response
        session_id = mock_websocket.sent_messages[0].get("session_id")

        # End meeting
        await handler.handle_message(
            mock_websocket,
            topic=f"meeting:{session_id}",
            data={"type": "meeting_end"},
            conn_info=mock_connection_info,
        )

        # Verify end response
        end_response = mock_websocket.sent_messages[-1]
        assert end_response["type"] == WSMessageType.STATUS.value
        assert end_response["status"] == "meeting_ended"
        assert "summary" in end_response

    async def test_ping_pong(
        self,
        mock_websocket,
        mock_connection_info,
    ):
        """Test ping/pong keepalive."""
        handler = MeetingHandler()

        await handler.handle_message(
            mock_websocket,
            topic="meeting:test",
            data={"type": "ping"},
            conn_info=mock_connection_info,
        )

        assert len(mock_websocket.sent_messages) == 1
        assert mock_websocket.sent_messages[0]["type"] == "pong"


class TestDoubleTapQuickSuggestion:
    """Test 2: Double-tap quick suggestion (2-3s target, 5 segments)."""

    async def test_double_tap_returns_suggestion(
        self,
        mock_websocket,
        mock_context_manager,
        mock_suggestion_engine,
        mock_connection_info,
    ):
        """Double-tap should return suggestion with processing status."""
        handler = MeetingHandler()
        mock_cm, sessions = mock_context_manager

        # Start meeting and add segments
        await handler.handle_message(
            mock_websocket,
            topic="meeting:test-session",
            data={"type": "meeting_start", "config": {}},
            conn_info=mock_connection_info,
        )
        session_id = mock_websocket.sent_messages[0]["session_id"]

        # Add 5 transcript segments to the session
        for i in range(5):
            mock_cm.add_segment(
                session_id,
                TranscriptSegment(
                    text=f"Test segment {i}",
                    speaker=SpeakerRole.COUNTERPART,
                ),
            )

        # Send double-tap
        await handler.handle_message(
            mock_websocket,
            topic=f"meeting:{session_id}",
            data={"type": "double_tap"},
            conn_info=mock_connection_info,
        )

        # Verify processing status was sent
        processing_msg = mock_websocket.sent_messages[1]
        assert processing_msg["type"] == WSMessageType.STATUS.value
        assert processing_msg["status"] == "processing"

        # Verify suggestion was sent
        suggestion_msg = mock_websocket.sent_messages[2]
        assert suggestion_msg["type"] == WSMessageType.SUGGESTION.value
        assert suggestion_msg["trigger"] == "double_tap"
        assert "suggestion" in suggestion_msg
        assert "total_latency_ms" in suggestion_msg

    async def test_double_tap_uses_last_5_segments(
        self,
        mock_websocket,
        mock_context_manager,
        mock_suggestion_engine,
        mock_connection_info,
    ):
        """Double-tap should use only last 5 segments for quick context."""
        handler = MeetingHandler()
        mock_cm, sessions = mock_context_manager

        # Start meeting
        await handler.handle_message(
            mock_websocket,
            topic="meeting:test-session",
            data={"type": "meeting_start", "config": {}},
            conn_info=mock_connection_info,
        )
        session_id = mock_websocket.sent_messages[0]["session_id"]

        # Add 10 segments
        for i in range(10):
            mock_cm.add_segment(
                session_id,
                TranscriptSegment(
                    text=f"Segment {i}",
                    speaker=SpeakerRole.COUNTERPART,
                ),
            )

        # Send double-tap
        await handler.handle_message(
            mock_websocket,
            topic=f"meeting:{session_id}",
            data={"type": "double_tap"},
            conn_info=mock_connection_info,
        )

        # Verify get_context_window was called with DOUBLE_TAP trigger
        mock_cm.get_context_window.assert_called()
        call_args = mock_cm.get_context_window.call_args[0]
        assert call_args[1] == TriggerType.DOUBLE_TAP

    async def test_double_tap_without_session_returns_error(
        self,
        mock_websocket,
        mock_context_manager,
        mock_connection_info,
    ):
        """Double-tap without active session should return error."""
        handler = MeetingHandler()
        mock_cm, sessions = mock_context_manager

        # Send double-tap without starting meeting
        await handler.handle_message(
            mock_websocket,
            topic="meeting:nonexistent",
            data={"type": "double_tap"},
            conn_info=mock_connection_info,
        )

        assert len(mock_websocket.sent_messages) >= 1
        error_msg = mock_websocket.sent_messages[-1]
        assert error_msg["type"] == WSMessageType.ERROR.value


class TestVoiceCommandDetailedSuggestion:
    """Test 3: Voice command detailed suggestion (3-4s target, full context)."""

    async def test_voice_command_with_text_query(
        self,
        mock_websocket,
        mock_context_manager,
        mock_suggestion_engine,
        mock_connection_info,
    ):
        """Voice command with text query returns detailed suggestion."""
        handler = MeetingHandler()
        mock_cm, sessions = mock_context_manager

        # Start meeting
        await handler.handle_message(
            mock_websocket,
            topic="meeting:test-session",
            data={"type": "meeting_start", "config": {}},
            conn_info=mock_connection_info,
        )
        session_id = mock_websocket.sent_messages[0]["session_id"]

        # Add some transcript
        for i in range(20):
            mock_cm.add_segment(
                session_id,
                TranscriptSegment(
                    text=f"Meeting discussion point {i}",
                    speaker=SpeakerRole.COUNTERPART if i % 2 else SpeakerRole.USER,
                ),
            )

        # Send voice command
        user_query = "What leverage do I have in this negotiation?"
        await handler.handle_message(
            mock_websocket,
            topic=f"meeting:{session_id}",
            data={
                "type": "voice_command",
                "query": user_query,
            },
            conn_info=mock_connection_info,
        )

        # Verify processing status
        processing_msg = mock_websocket.sent_messages[1]
        assert processing_msg["status"] == "processing"

        # Verify suggestion includes query echo
        suggestion_msg = mock_websocket.sent_messages[2]
        assert suggestion_msg["type"] == WSMessageType.SUGGESTION.value
        assert suggestion_msg["trigger"] == "voice_command"
        assert suggestion_msg["query"] == user_query

    async def test_voice_command_uses_full_context(
        self,
        mock_websocket,
        mock_context_manager,
        mock_suggestion_engine,
        mock_connection_info,
    ):
        """Voice command should use full transcript context."""
        handler = MeetingHandler()
        mock_cm, sessions = mock_context_manager

        # Start meeting
        await handler.handle_message(
            mock_websocket,
            topic="meeting:test-session",
            data={"type": "meeting_start", "config": {}},
            conn_info=mock_connection_info,
        )
        session_id = mock_websocket.sent_messages[0]["session_id"]

        # Add segments
        for i in range(20):
            mock_cm.add_segment(
                session_id,
                TranscriptSegment(text=f"Point {i}"),
            )

        # Send voice command
        await handler.handle_message(
            mock_websocket,
            topic=f"meeting:{session_id}",
            data={
                "type": "voice_command",
                "query": "Summarize the key points",
            },
            conn_info=mock_connection_info,
        )

        # Verify get_context_window was called with VOICE_COMMAND trigger
        mock_cm.get_context_window.assert_called()
        call_args = mock_cm.get_context_window.call_args[0]
        assert call_args[1] == TriggerType.VOICE_COMMAND

    async def test_voice_command_without_query_returns_error(
        self,
        mock_websocket,
        mock_context_manager,
        mock_connection_info,
    ):
        """Voice command without query should return error."""
        handler = MeetingHandler()
        mock_cm, sessions = mock_context_manager

        # Start meeting
        await handler.handle_message(
            mock_websocket,
            topic="meeting:test-session",
            data={"type": "meeting_start", "config": {}},
            conn_info=mock_connection_info,
        )
        session_id = mock_websocket.sent_messages[0]["session_id"]

        # Send voice command without query
        await handler.handle_message(
            mock_websocket,
            topic=f"meeting:{session_id}",
            data={
                "type": "voice_command",
                # No query provided
            },
            conn_info=mock_connection_info,
        )

        error_msg = mock_websocket.sent_messages[-1]
        assert error_msg["type"] == WSMessageType.ERROR.value


class TestAudioChunkProcessing:
    """Test 4: Audio chunk processing and transcription."""

    async def test_audio_chunk_triggers_transcription(
        self,
        mock_websocket,
        mock_context_manager,
        mock_transcription_service,
        mock_connection_info,
        sample_audio_b64,
    ):
        """Audio chunk should trigger transcription and broadcast result."""
        handler = MeetingHandler()
        mock_cm, sessions = mock_context_manager

        # Manually inject the mock transcription service
        handler.transcription = mock_transcription_service

        # Start meeting
        await handler.handle_message(
            mock_websocket,
            topic="meeting:test-session",
            data={"type": "meeting_start", "config": {}},
            conn_info=mock_connection_info,
        )
        session_id = mock_websocket.sent_messages[0]["session_id"]

        # Send audio chunk
        await handler.handle_message(
            mock_websocket,
            topic=f"meeting:{session_id}",
            data={
                "type": "audio_chunk",
                "audio": sample_audio_b64,
                "duration_ms": 1000,
                "sample_rate": 16000,
            },
            conn_info=mock_connection_info,
        )

        # Wait for async transcription task
        await asyncio.sleep(0.1)

        # Verify transcription service was called
        mock_transcription_service.transcribe.assert_called()

    async def test_audio_chunk_without_session_returns_error(
        self,
        mock_websocket,
        mock_connection_info,
        sample_audio_b64,
    ):
        """Audio chunk without session should return error."""
        handler = MeetingHandler()

        await handler.handle_message(
            mock_websocket,
            topic="meeting:nonexistent",
            data={
                "type": "audio_chunk",
                "audio": sample_audio_b64,
            },
            conn_info=mock_connection_info,
        )

        error_msg = mock_websocket.sent_messages[-1]
        assert error_msg["type"] == WSMessageType.ERROR.value
        assert "No active session" in error_msg["message"]

    async def test_invalid_audio_returns_error(
        self,
        mock_websocket,
        mock_context_manager,
        mock_connection_info,
    ):
        """Invalid base64 audio should return error."""
        handler = MeetingHandler()
        mock_cm, sessions = mock_context_manager

        # Start meeting
        await handler.handle_message(
            mock_websocket,
            topic="meeting:test-session",
            data={"type": "meeting_start", "config": {}},
            conn_info=mock_connection_info,
        )
        session_id = mock_websocket.sent_messages[0]["session_id"]

        # Send invalid audio
        await handler.handle_message(
            mock_websocket,
            topic=f"meeting:{session_id}",
            data={
                "type": "audio_chunk",
                "audio": "not-valid-base64!@#$%",
            },
            conn_info=mock_connection_info,
        )

        error_msg = mock_websocket.sent_messages[-1]
        assert error_msg["type"] == WSMessageType.ERROR.value
        assert "Invalid audio" in error_msg["message"]


class TestErrorHandling:
    """Test 5: Error handling for various failure modes."""

    async def test_unknown_message_type_returns_error(
        self,
        mock_websocket,
        mock_connection_info,
    ):
        """Unknown message type should return error."""
        handler = MeetingHandler()

        await handler.handle_message(
            mock_websocket,
            topic="meeting:test",
            data={"type": "invalid_message_type"},
            conn_info=mock_connection_info,
        )

        error_msg = mock_websocket.sent_messages[-1]
        assert error_msg["type"] == WSMessageType.ERROR.value
        assert "Unknown message type" in error_msg["message"]

    async def test_empty_message_type_returns_error(
        self,
        mock_websocket,
        mock_connection_info,
    ):
        """Empty message type should return error."""
        handler = MeetingHandler()

        await handler.handle_message(
            mock_websocket,
            topic="meeting:test",
            data={},  # No type field
            conn_info=mock_connection_info,
        )

        error_msg = mock_websocket.sent_messages[-1]
        assert error_msg["type"] == WSMessageType.ERROR.value

    async def test_meeting_end_without_session_handles_gracefully(
        self,
        mock_websocket,
        mock_context_manager,
        mock_connection_info,
    ):
        """Ending a non-existent meeting should handle gracefully."""
        handler = MeetingHandler()
        mock_cm, sessions = mock_context_manager

        await handler.handle_message(
            mock_websocket,
            topic="meeting:nonexistent",
            data={"type": "meeting_end"},
            conn_info=mock_connection_info,
        )

        # Should still send a response (not crash)
        assert len(mock_websocket.sent_messages) >= 1


class TestLatencyBenchmarks:
    """Latency benchmark tests (informational, not strict pass/fail)."""

    async def test_double_tap_latency_target(
        self,
        mock_websocket,
        mock_context_manager,
        mock_suggestion_engine,
        mock_connection_info,
    ):
        """Verify double-tap latency is tracked and reasonable."""
        handler = MeetingHandler()
        mock_cm, sessions = mock_context_manager

        # Start meeting
        await handler.handle_message(
            mock_websocket,
            topic="meeting:test-session",
            data={"type": "meeting_start", "config": {}},
            conn_info=mock_connection_info,
        )
        session_id = mock_websocket.sent_messages[0]["session_id"]

        # Add segments
        for i in range(5):
            mock_cm.add_segment(
                session_id,
                TranscriptSegment(text=f"Test {i}"),
            )

        # Time the double-tap
        start = time.perf_counter()
        await handler.handle_message(
            mock_websocket,
            topic=f"meeting:{session_id}",
            data={"type": "double_tap"},
            conn_info=mock_connection_info,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Get reported latency
        suggestion_msg = mock_websocket.sent_messages[-1]
        reported_latency = suggestion_msg.get("total_latency_ms", 0)

        # Log for visibility (with mocks, this should be very fast)
        print(f"\nDouble-tap latency: {reported_latency:.0f}ms (elapsed: {elapsed_ms:.0f}ms)")

        # With mocked services, latency should be minimal
        assert reported_latency < 1000  # 1s max with mocks
