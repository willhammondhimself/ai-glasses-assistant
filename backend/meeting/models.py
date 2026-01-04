"""
Meeting Mode Data Models

Defines all data structures for the meeting assistant.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid


class TriggerType(Enum):
    """How the suggestion was triggered."""
    DOUBLE_TAP = "double_tap"       # Quick mode - last 5 segments
    VOICE_COMMAND = "voice_command" # Full context mode
    PROACTIVE = "proactive"         # WHAM detected an opportunity


class SuggestionType(Enum):
    """Type of suggestion WHAM can provide."""
    QUICK_RESPONSE = "quick_response"       # What to say right now
    TACTICAL_ADVICE = "tactical_advice"     # Strategic meeting advice
    FACT_CHECK = "fact_check"               # Verify a claim
    CONTEXT_FILL = "context_fill"           # Background on a topic
    NEGOTIATION = "negotiation"             # Leverage/positioning advice
    CLARIFICATION = "clarification"         # Suggest asking for clarity


class SpeakerRole(Enum):
    """Role of speaker in the meeting."""
    USER = "user"           # Will (the glasses wearer)
    COUNTERPART = "other"   # Other meeting participant(s)
    UNKNOWN = "unknown"


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    speaker: SpeakerRole = SpeakerRole.UNKNOWN
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0
    duration_ms: int = 0
    is_final: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "speaker": self.speaker.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "is_final": self.is_final,
        }


@dataclass
class MeetingConfig:
    """Configuration for a meeting session."""
    meeting_type: str = "general"  # general, negotiation, interview, sales
    participants: List[str] = field(default_factory=list)
    context: str = ""  # Pre-meeting context/goals
    quick_context_segments: int = 5  # Segments for double-tap mode
    full_context_tokens: int = 50000  # Token limit for voice command mode
    auto_detect_speakers: bool = True
    proactive_suggestions: bool = False  # WHAM speaks up on its own


@dataclass
class MeetingSession:
    """Active meeting session state."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: MeetingConfig = field(default_factory=MeetingConfig)
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    transcript: List[TranscriptSegment] = field(default_factory=list)
    suggestions_given: int = 0
    total_audio_ms: int = 0
    is_active: bool = True

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    @property
    def transcript_text(self) -> str:
        """Full transcript as plain text."""
        return "\n".join(
            f"[{seg.speaker.value}]: {seg.text}"
            for seg in self.transcript
            if seg.text.strip()
        )

    def add_segment(self, segment: TranscriptSegment):
        self.transcript.append(segment)
        self.total_audio_ms += segment.duration_ms

    def get_recent_segments(self, n: int) -> List[TranscriptSegment]:
        """Get the N most recent transcript segments."""
        return self.transcript[-n:] if self.transcript else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "segment_count": len(self.transcript),
            "suggestions_given": self.suggestions_given,
            "is_active": self.is_active,
        }


@dataclass
class SuggestionRequest:
    """Request for a meeting suggestion."""
    session_id: str
    trigger: TriggerType
    user_query: Optional[str] = None  # For voice command mode
    context_override: Optional[str] = None  # Additional context


@dataclass
class SuggestionResponse:
    """WHAM's response to a suggestion request."""
    suggestion: str
    suggestion_type: SuggestionType
    confidence: float
    alternatives: List[str] = field(default_factory=list)
    tactical_notes: Optional[str] = None  # Additional strategic insight
    latency_ms: float = 0
    tokens_used: int = 0
    cost: float = 0.0  # Free for Gemini student tier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion": self.suggestion,
            "type": self.suggestion_type.value,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "tactical_notes": self.tactical_notes,
            "latency_ms": self.latency_ms,
        }


@dataclass
class AudioChunk:
    """A chunk of audio data for transcription."""
    data: bytes
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: int = 1000  # 1 second chunks
    sample_rate: int = 16000
    channels: int = 1


# WebSocket Message Types
class WSMessageType(Enum):
    """WebSocket message types for meeting mode."""
    # Client → Server
    MEETING_START = "meeting_start"
    MEETING_END = "meeting_end"
    AUDIO_CHUNK = "audio_chunk"
    DOUBLE_TAP = "double_tap"
    VOICE_COMMAND = "voice_command"

    # Server → Client
    TRANSCRIPT_UPDATE = "transcript_update"
    SUGGESTION = "suggestion"
    STATUS = "status"
    ERROR = "error"
