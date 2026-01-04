"""
Shared test fixtures for meeting mode testing.

Provides:
- Mock WebSocket connections
- Mock transcription service (bypasses Whisper API)
- Mock suggestion engine (bypasses Gemini API)
- Sample audio data
"""

import asyncio
import base64
import pytest
import io
import wave
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from backend.meeting.models import (
    TranscriptSegment,
    SuggestionResponse,
    SuggestionType,
    SpeakerRole,
    MeetingConfig,
    MeetingSession,
    TriggerType,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Generate valid 1-second 16kHz mono WAV audio."""
    sample_rate = 16000
    duration_ms = 1000
    num_samples = int(sample_rate * duration_ms / 1000)

    # Create silent audio (zeros)
    audio_data = bytes(num_samples * 2)  # 16-bit = 2 bytes per sample

    # Wrap in WAV format
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data)

    return buffer.getvalue()


@pytest.fixture
def sample_audio_b64(sample_audio_bytes: bytes) -> str:
    """Base64-encoded sample audio."""
    return base64.b64encode(sample_audio_bytes).decode()


@pytest.fixture
def mock_transcript_segments() -> List[TranscriptSegment]:
    """Sample transcript segments for testing."""
    return [
        TranscriptSegment(
            id="seg1",
            text="Let's discuss the quarterly targets.",
            speaker=SpeakerRole.COUNTERPART,
            confidence=0.95,
            duration_ms=2000,
        ),
        TranscriptSegment(
            id="seg2",
            text="I think we can hit those numbers with the new strategy.",
            speaker=SpeakerRole.USER,
            confidence=0.92,
            duration_ms=2500,
        ),
        TranscriptSegment(
            id="seg3",
            text="What about the budget constraints?",
            speaker=SpeakerRole.COUNTERPART,
            confidence=0.93,
            duration_ms=1800,
        ),
        TranscriptSegment(
            id="seg4",
            text="We have some flexibility there.",
            speaker=SpeakerRole.USER,
            confidence=0.91,
            duration_ms=1500,
        ),
        TranscriptSegment(
            id="seg5",
            text="The timeline seems aggressive though.",
            speaker=SpeakerRole.COUNTERPART,
            confidence=0.94,
            duration_ms=2000,
        ),
    ]


@pytest.fixture
def mock_suggestion_response() -> SuggestionResponse:
    """Sample suggestion for testing."""
    return SuggestionResponse(
        suggestion="Consider proposing a phased approach to address timeline concerns.",
        suggestion_type=SuggestionType.TACTICAL_ADVICE,
        confidence=0.88,
        alternatives=[
            "Ask for clarification on their specific timeline concerns.",
            "Propose milestone-based delivery to reduce risk.",
        ],
        tactical_notes="They seem hesitant about timeline - this is leverage for negotiation.",
        latency_ms=1500,
        tokens_used=150,
        cost=0.0,
    )


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket that records sent messages."""
    ws = AsyncMock()
    ws.sent_messages: List[Dict[str, Any]] = []

    async def record_send(data: dict):
        ws.sent_messages.append(data)

    ws.send_json = AsyncMock(side_effect=record_send)
    return ws


@pytest.fixture
def mock_transcription_service(mock_transcript_segments):
    """Mock TranscriptionService that returns canned transcripts."""
    segment_iter = iter(mock_transcript_segments)

    async def mock_transcribe(chunk):
        try:
            return next(segment_iter)
        except StopIteration:
            return TranscriptSegment(text="", confidence=0.0)

    with patch("backend.meeting.router.TranscriptionService") as mock_class:
        mock_instance = MagicMock()
        mock_instance.transcribe = AsyncMock(side_effect=mock_transcribe)
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_suggestion_engine(mock_suggestion_response):
    """Mock SuggestionEngine that returns canned suggestions."""
    async def mock_get_suggestion(context, trigger_type, user_query=None):
        return mock_suggestion_response

    with patch("backend.meeting.router.get_suggestion_engine") as mock_getter:
        mock_engine = MagicMock()
        mock_engine.get_suggestion = AsyncMock(side_effect=mock_get_suggestion)
        mock_getter.return_value = mock_engine
        yield mock_engine


@pytest.fixture
def mock_context_manager(mock_transcript_segments):
    """Mock ContextManager for session management."""
    sessions: Dict[str, MeetingSession] = {}

    def create_session(config: MeetingConfig) -> MeetingSession:
        session = MeetingSession(config=config)
        sessions[session.id] = session
        return session

    def get_session(session_id: str):
        return sessions.get(session_id)

    def add_segment(session_id: str, segment: TranscriptSegment):
        if session_id in sessions:
            sessions[session_id].add_segment(segment)

    def get_context_window(session_id: str, trigger_type, user_query=None):
        session = sessions.get(session_id)
        if not session:
            return None

        if trigger_type == TriggerType.DOUBLE_TAP:
            segments = session.get_recent_segments(5)
        else:
            segments = session.transcript

        return {
            "session": session,
            "segments": segments,
            "transcript_text": session.transcript_text,
            "user_query": user_query,
        }

    def end_session(session_id: str):
        if session_id in sessions:
            session = sessions[session_id]
            session.is_active = False
            session.ended_at = datetime.utcnow()
            return session
        return None

    def get_meeting_summary(session_id: str):
        session = sessions.get(session_id)
        if session:
            return {
                "duration_seconds": session.duration_seconds,
                "segment_count": len(session.transcript),
                "suggestions_given": session.suggestions_given,
            }
        return {}

    with patch("backend.meeting.router.get_context_manager") as mock_getter:
        mock_cm = MagicMock()
        mock_cm.create_session = MagicMock(side_effect=create_session)
        mock_cm.get_session = MagicMock(side_effect=get_session)
        mock_cm.add_segment = MagicMock(side_effect=add_segment)
        mock_cm.get_context_window = MagicMock(side_effect=get_context_window)
        mock_cm.end_session = MagicMock(side_effect=end_session)
        mock_cm.get_meeting_summary = MagicMock(side_effect=get_meeting_summary)
        mock_getter.return_value = mock_cm
        yield mock_cm, sessions


class MockConnectionInfo:
    """Mock connection info for testing."""
    def __init__(self, client_id: str = "test-client"):
        self.client_id = client_id
        self.connected_at = datetime.utcnow()
        self.topics = set()


@pytest.fixture
def mock_connection_info():
    """Create mock connection info."""
    return MockConnectionInfo()
