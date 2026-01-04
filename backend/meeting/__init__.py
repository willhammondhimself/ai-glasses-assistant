"""
Meeting Mode - Real-time meeting assistant for WHAM.

Components:
- TranscriptionService: Whisper API for continuous transcription
- ContextManager: Manages transcript buffer and context windows
- SuggestionEngine: Gemini 2.5 Pro for tactical suggestions
- MeetingHandler: WebSocket handler for real-time communication
"""

from .models import (
    MeetingSession,
    TranscriptSegment,
    SuggestionRequest,
    SuggestionResponse,
    MeetingConfig,
)
from .transcription import TranscriptionService
from .context import ContextManager
from .suggestions import SuggestionEngine
from .router import MeetingHandler, meeting_handler

__all__ = [
    "MeetingSession",
    "TranscriptSegment",
    "SuggestionRequest",
    "SuggestionResponse",
    "MeetingConfig",
    "TranscriptionService",
    "ContextManager",
    "SuggestionEngine",
    "MeetingHandler",
    "meeting_handler",
]
