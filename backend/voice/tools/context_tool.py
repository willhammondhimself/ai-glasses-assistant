"""Context awareness voice tool - manage and query user context."""
import logging
import re
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class ContextVoiceTool(VoiceTool):
    """Voice-controlled context awareness management."""

    name = "context"
    description = "Manage context awareness - focus mode, availability, status"

    keywords = [
        r"\bfocus\s+mode\b",
        r"\bdo\s+not\s+disturb\b",
        r"\bbusy\b",
        r"\bavailable\b",
        r"\bstatus\b",
        r"\bin\s+a\s+meeting\b",
        r"\bdon'?t\s+interrupt\b",
        r"\bquiet\s+mode\b",
        r"\bback\s+to\s+normal\b",
        r"\bwhat('s| is)\s+my\s+(status|mode|context)\b",
    ]

    priority = 7

    def __init__(self):
        self._context_engine = None

    def _get_engine(self):
        """Get context engine (lazy load)."""
        if self._context_engine is None:
            from backend.services.context_engine import get_context_engine
            self._context_engine = get_context_engine()
        return self._context_engine

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute context command.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult with context info
        """
        query_lower = query.lower()
        engine = self._get_engine()

        try:
            # Enable focus mode
            if self._is_focus_request(query_lower):
                duration = self._extract_duration(query_lower)
                return await self._handle_focus_mode(engine, True, duration)

            # Disable focus mode / back to normal
            if self._is_normal_request(query_lower):
                return await self._handle_focus_mode(engine, False)

            # Set busy/unavailable
            if self._is_busy_request(query_lower):
                return await self._handle_busy(engine, True)

            # Set available
            if self._is_available_request(query_lower):
                return await self._handle_busy(engine, False)

            # Meeting mode
            if self._is_meeting_request(query_lower):
                return await self._handle_meeting(engine, query_lower)

            # Query current status
            if self._is_status_query(query_lower):
                return await self._handle_status_query(engine)

            # Default: show status
            return await self._handle_status_query(engine)

        except Exception as e:
            logger.error(f"Context tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble with that context command.",
                data={"error": str(e)}
            )

    def _is_focus_request(self, query: str) -> bool:
        patterns = [
            "focus mode", "enable focus", "start focus",
            "don't disturb", "do not disturb", "quiet mode",
            "don't interrupt", "need to focus"
        ]
        return any(p in query for p in patterns) and "off" not in query and "disable" not in query

    def _is_normal_request(self, query: str) -> bool:
        patterns = [
            "back to normal", "normal mode", "disable focus",
            "focus off", "stop focus", "end focus",
            "turn off focus", "I'm done focusing"
        ]
        return any(p in query for p in patterns)

    def _is_busy_request(self, query: str) -> bool:
        patterns = ["i'm busy", "set busy", "unavailable", "mark busy"]
        return any(p in query for p in patterns)

    def _is_available_request(self, query: str) -> bool:
        patterns = ["i'm available", "available now", "set available", "I'm free"]
        return any(p in query for p in patterns)

    def _is_meeting_request(self, query: str) -> bool:
        patterns = ["in a meeting", "meeting mode", "start meeting", "joining a call"]
        return any(p in query for p in patterns)

    def _is_status_query(self, query: str) -> bool:
        patterns = ["what's my status", "my mode", "my context", "am I busy", "what mode"]
        return any(p in query for p in patterns)

    def _extract_duration(self, query: str) -> int:
        """Extract duration in minutes from query."""
        # Look for patterns like "30 minutes", "1 hour", "2 hours"
        hour_match = re.search(r'(\d+)\s*hours?', query)
        if hour_match:
            return int(hour_match.group(1)) * 60

        min_match = re.search(r'(\d+)\s*min', query)
        if min_match:
            return int(min_match.group(1))

        # Default 60 minutes
        return 60

    async def _handle_focus_mode(
        self,
        engine,
        enabled: bool,
        duration: int = 60
    ) -> VoiceToolResult:
        """Handle focus mode toggle."""
        engine.set_focus_mode(enabled, duration)

        if enabled:
            message = f"Focus mode on for {duration} minutes. I'll only interrupt for urgent matters."
        else:
            message = "Focus mode off. Back to normal notifications."

        ctx = engine.get_context()
        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "action": "focus_mode",
                "enabled": enabled,
                "duration_minutes": duration if enabled else 0,
                "context": ctx.to_dict()
            }
        )

    async def _handle_busy(self, engine, busy: bool) -> VoiceToolResult:
        """Handle busy/available toggle."""
        ctx = engine.get_context()
        ctx.available = not busy

        if busy:
            ctx.notification_level = "urgent"
            message = "Marked as busy. Only urgent notifications will come through."
        else:
            ctx.notification_level = "all"
            message = "Marked as available."

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "action": "availability",
                "available": not busy,
                "context": ctx.to_dict()
            }
        )

    async def _handle_meeting(self, engine, query: str) -> VoiceToolResult:
        """Handle meeting mode."""
        # Check if ending meeting
        if "end" in query or "over" in query or "done" in query:
            engine.set_meeting_status(False)
            return VoiceToolResult(
                success=True,
                message="Meeting mode off.",
                data={"action": "meeting_end"}
            )

        # Extract duration
        duration = self._extract_duration(query)
        from datetime import datetime, timedelta
        ends_at = datetime.now() + timedelta(minutes=duration)

        engine.set_meeting_status(True, ends_at)

        return VoiceToolResult(
            success=True,
            message=f"Meeting mode on. I'll keep responses brief and only urgent alerts.",
            data={
                "action": "meeting_start",
                "duration_minutes": duration,
                "ends_at": ends_at.isoformat()
            }
        )

    async def _handle_status_query(self, engine) -> VoiceToolResult:
        """Handle status query."""
        ctx = engine.get_context()

        # Build natural status message
        parts = []

        # Mode
        mode_names = {
            "work": "work mode",
            "personal": "personal mode",
            "focus": "focus mode",
            "meeting": "in a meeting",
            "commute": "commuting",
            "exercise": "exercising",
            "leisure": "leisure time"
        }
        parts.append(f"You're in {mode_names.get(ctx.mode.value, ctx.mode.value)}")

        # Availability
        if not ctx.available:
            parts.append("and set to unavailable")
        else:
            parts.append("and available")

        # Focus level
        if ctx.focus_level >= 8:
            parts.append("with high focus")
        elif ctx.focus_level <= 3:
            parts.append("in relaxed mode")

        message = " ".join(parts) + "."

        return VoiceToolResult(
            success=True,
            message=message,
            data={"context": ctx.to_dict()}
        )
