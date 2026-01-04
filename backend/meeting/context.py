"""
Context Manager - Manages transcript buffer and context windows.

Handles:
- Transcript storage and retrieval
- Context window selection (quick vs full)
- Token counting for Gemini context limits
- Meeting goal tracking
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque

from .models import (
    MeetingSession,
    MeetingConfig,
    TranscriptSegment,
    SpeakerRole,
    TriggerType,
)

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """A window of context for AI processing."""
    segments: List[TranscriptSegment]
    meeting_context: str  # Pre-meeting context/goals
    total_tokens: int
    is_full_context: bool  # True if voice command, False if double-tap


class ContextManager:
    """
    Manages meeting context and transcript buffering.

    Features:
    - Maintains rolling transcript buffer
    - Provides quick context (last N segments) for double-tap
    - Provides full context (up to token limit) for voice commands
    - Tracks meeting goals and key moments
    """

    # Approximate tokens per character (conservative estimate)
    CHARS_PER_TOKEN = 4

    def __init__(self, max_segments: int = 10000):
        self.max_segments = max_segments
        self._sessions: Dict[str, MeetingSession] = {}

    def create_session(self, config: MeetingConfig = None) -> MeetingSession:
        """Create a new meeting session."""
        session = MeetingSession(config=config or MeetingConfig())
        self._sessions[session.id] = session
        logger.info(f"Created meeting session {session.id}")
        return session

    def get_session(self, session_id: str) -> Optional[MeetingSession]:
        """Get an existing session."""
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> Optional[MeetingSession]:
        """End a meeting session."""
        session = self._sessions.get(session_id)
        if session:
            session.ended_at = datetime.utcnow()
            session.is_active = False
            logger.info(
                f"Ended session {session_id}: "
                f"{len(session.transcript)} segments, "
                f"{session.duration_seconds:.0f}s"
            )
        return session

    def add_segment(self, session_id: str, segment: TranscriptSegment) -> bool:
        """
        Add a transcript segment to a session.

        Returns:
            True if added successfully
        """
        session = self._sessions.get(session_id)
        if not session or not session.is_active:
            return False

        session.add_segment(segment)

        # Trim if over limit
        if len(session.transcript) > self.max_segments:
            session.transcript = session.transcript[-self.max_segments:]

        return True

    def get_context_window(
        self,
        session_id: str,
        trigger: TriggerType,
        user_query: str = None,
    ) -> Optional[ContextWindow]:
        """
        Get appropriate context window for a suggestion request.

        Args:
            session_id: Meeting session ID
            trigger: How the suggestion was triggered
            user_query: Optional user question (for voice commands)

        Returns:
            ContextWindow with appropriate segments
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        if trigger == TriggerType.DOUBLE_TAP:
            return self._get_quick_context(session)
        else:
            return self._get_full_context(session, user_query)

    def _get_quick_context(self, session: MeetingSession) -> ContextWindow:
        """
        Get quick context for double-tap (last N segments).

        Optimized for speed - minimal context, fast response.
        """
        n = session.config.quick_context_segments
        segments = session.get_recent_segments(n)

        # Build context string
        context_text = self._format_segments(segments)
        tokens = self._estimate_tokens(context_text + session.config.context)

        return ContextWindow(
            segments=segments,
            meeting_context=session.config.context,
            total_tokens=tokens,
            is_full_context=False,
        )

    def _get_full_context(
        self,
        session: MeetingSession,
        user_query: str = None,
    ) -> ContextWindow:
        """
        Get full context for voice command.

        Includes as much transcript as fits in token limit.
        """
        token_limit = session.config.full_context_tokens

        # Start with meeting context and user query
        base_context = session.config.context
        if user_query:
            base_context += f"\n\nUser's question: {user_query}"

        base_tokens = self._estimate_tokens(base_context)
        available_tokens = token_limit - base_tokens

        # Add segments from most recent, going back
        segments = []
        total_tokens = 0

        for segment in reversed(session.transcript):
            segment_tokens = self._estimate_tokens(
                f"[{segment.speaker.value}]: {segment.text}"
            )

            if total_tokens + segment_tokens > available_tokens:
                break

            segments.insert(0, segment)
            total_tokens += segment_tokens

        return ContextWindow(
            segments=segments,
            meeting_context=base_context,
            total_tokens=base_tokens + total_tokens,
            is_full_context=True,
        )

    def _format_segments(self, segments: List[TranscriptSegment]) -> str:
        """Format segments into readable transcript."""
        if not segments:
            return "[No transcript yet]"

        lines = []
        for seg in segments:
            speaker = "You" if seg.speaker == SpeakerRole.USER else "Other"
            lines.append(f"[{speaker}]: {seg.text}")

        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.CHARS_PER_TOKEN

    def get_meeting_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the meeting for review."""
        session = self._sessions.get(session_id)
        if not session:
            return {}

        # Count speaker contributions
        user_segments = sum(
            1 for s in session.transcript
            if s.speaker == SpeakerRole.USER
        )
        other_segments = len(session.transcript) - user_segments

        # Estimate speaking time
        user_time_ms = sum(
            s.duration_ms for s in session.transcript
            if s.speaker == SpeakerRole.USER
        )
        total_time_ms = sum(s.duration_ms for s in session.transcript)

        return {
            "session_id": session_id,
            "duration_seconds": session.duration_seconds,
            "total_segments": len(session.transcript),
            "user_segments": user_segments,
            "other_segments": other_segments,
            "user_speaking_ratio": user_time_ms / total_time_ms if total_time_ms > 0 else 0,
            "suggestions_given": session.suggestions_given,
            "meeting_type": session.config.meeting_type,
        }

    def get_key_moments(
        self,
        session_id: str,
        moment_type: str = "all"
    ) -> List[Dict[str, Any]]:
        """
        Identify key moments in the meeting.

        Types: question, commitment, objection, all
        """
        session = self._sessions.get(session_id)
        if not session:
            return []

        moments = []

        for i, segment in enumerate(session.transcript):
            text_lower = segment.text.lower()

            # Detect questions
            if "?" in segment.text and (moment_type in ["question", "all"]):
                moments.append({
                    "type": "question",
                    "segment_index": i,
                    "text": segment.text,
                    "speaker": segment.speaker.value,
                    "timestamp": segment.timestamp.isoformat(),
                })

            # Detect commitments
            commitment_phrases = [
                "i will", "i'll", "we will", "we'll", "let's",
                "i can", "i promise", "you have my word"
            ]
            if any(p in text_lower for p in commitment_phrases):
                if moment_type in ["commitment", "all"]:
                    moments.append({
                        "type": "commitment",
                        "segment_index": i,
                        "text": segment.text,
                        "speaker": segment.speaker.value,
                        "timestamp": segment.timestamp.isoformat(),
                    })

            # Detect objections/concerns
            objection_phrases = [
                "but", "however", "concern", "problem", "issue",
                "not sure", "worried", "don't think"
            ]
            if any(p in text_lower for p in objection_phrases):
                if moment_type in ["objection", "all"]:
                    moments.append({
                        "type": "objection",
                        "segment_index": i,
                        "text": segment.text,
                        "speaker": segment.speaker.value,
                        "timestamp": segment.timestamp.isoformat(),
                    })

        return moments

    def search_transcript(
        self,
        session_id: str,
        query: str,
        limit: int = 10
    ) -> List[TranscriptSegment]:
        """Search transcript for relevant segments."""
        session = self._sessions.get(session_id)
        if not session:
            return []

        query_lower = query.lower()
        matches = []

        for segment in session.transcript:
            if query_lower in segment.text.lower():
                matches.append(segment)
                if len(matches) >= limit:
                    break

        return matches

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove old inactive sessions."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        to_remove = [
            sid for sid, session in self._sessions.items()
            if not session.is_active and session.ended_at
            and session.ended_at < cutoff
        ]

        for sid in to_remove:
            del self._sessions[sid]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old meeting sessions")


# Singleton instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get the global context manager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager
