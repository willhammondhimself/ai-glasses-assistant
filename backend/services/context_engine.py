"""Context Awareness Engine - infer user context from multiple signals."""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UserMode(Enum):
    """User activity modes."""
    WORK = "work"
    PERSONAL = "personal"
    COMMUTE = "commute"
    EXERCISE = "exercise"
    SLEEP = "sleep"
    FOCUS = "focus"
    MEETING = "meeting"
    LEISURE = "leisure"


class ActivityType(Enum):
    """Specific activity types."""
    CODING = "coding"
    IN_MEETING = "in_meeting"
    WALKING = "walking"
    DRIVING = "driving"
    EATING = "eating"
    READING = "reading"
    SHOPPING = "shopping"
    UNKNOWN = "unknown"


@dataclass
class UserContext:
    """Current user context inferred from signals."""
    mode: UserMode = UserMode.PERSONAL
    activity: ActivityType = ActivityType.UNKNOWN
    focus_level: int = 5  # 0-10, higher = more focused
    available: bool = True  # Can receive notifications?
    location_type: str = "unknown"  # home, office, outdoors, transit
    time_context: str = "day"  # morning, afternoon, evening, night

    # Active signals
    in_meeting: bool = False
    meeting_ends_at: Optional[datetime] = None

    # Response preferences
    verbosity: str = "normal"  # brief, normal, detailed
    notification_level: str = "all"  # all, important, urgent, none

    # Recent activity
    last_tool_used: Optional[str] = None
    tools_used_recently: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "activity": self.activity.value,
            "focus_level": self.focus_level,
            "available": self.available,
            "location_type": self.location_type,
            "time_context": self.time_context,
            "in_meeting": self.in_meeting,
            "meeting_ends_at": self.meeting_ends_at.isoformat() if self.meeting_ends_at else None,
            "verbosity": self.verbosity,
            "notification_level": self.notification_level,
            "last_tool_used": self.last_tool_used,
        }


@dataclass
class ContextSignal:
    """A signal that contributes to context inference."""
    source: str
    signal_type: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0


class ContextEngine:
    """Infer user context from multiple signals."""

    def __init__(self):
        self._current_context = UserContext()
        self._signals: List[ContextSignal] = []
        self._last_update = datetime.now()

        # Context history for pattern detection
        self._context_history: List[Dict] = []
        self._max_history = 100

    def get_context(self) -> UserContext:
        """Get current user context.

        Returns:
            Current UserContext
        """
        # Re-infer if stale (older than 1 minute)
        if datetime.now() - self._last_update > timedelta(minutes=1):
            self._infer_context()

        return self._current_context

    def add_signal(self, source: str, signal_type: str, value: Any, confidence: float = 1.0):
        """Add a context signal.

        Args:
            source: Signal source (calendar, location, voice, etc.)
            signal_type: Type of signal
            value: Signal value
            confidence: Confidence in signal accuracy
        """
        signal = ContextSignal(
            source=source,
            signal_type=signal_type,
            value=value,
            confidence=confidence
        )
        self._signals.append(signal)

        # Keep only recent signals (last 10 minutes)
        cutoff = datetime.now() - timedelta(minutes=10)
        self._signals = [s for s in self._signals if s.timestamp > cutoff]

        # Trigger re-inference
        self._infer_context()

    def record_tool_use(self, tool_name: str):
        """Record that a tool was used.

        Args:
            tool_name: Name of the tool used
        """
        self._current_context.last_tool_used = tool_name
        self._current_context.tools_used_recently.append(tool_name)

        # Keep last 10 tools
        if len(self._current_context.tools_used_recently) > 10:
            self._current_context.tools_used_recently = \
                self._current_context.tools_used_recently[-10:]

    def set_meeting_status(self, in_meeting: bool, ends_at: Optional[datetime] = None):
        """Set meeting status directly.

        Args:
            in_meeting: Whether user is in a meeting
            ends_at: When the meeting ends
        """
        self._current_context.in_meeting = in_meeting
        self._current_context.meeting_ends_at = ends_at

        if in_meeting:
            self._current_context.mode = UserMode.MEETING
            self._current_context.available = False
            self._current_context.focus_level = 8
            self._current_context.verbosity = "brief"
            self._current_context.notification_level = "urgent"
        else:
            # Reset to normal
            self._current_context.available = True
            self._infer_context()

    def set_focus_mode(self, enabled: bool, duration_minutes: int = 60):
        """Enable/disable focus mode.

        Args:
            enabled: Whether focus mode is on
            duration_minutes: How long focus mode lasts
        """
        if enabled:
            self._current_context.mode = UserMode.FOCUS
            self._current_context.focus_level = 10
            self._current_context.available = False
            self._current_context.verbosity = "brief"
            self._current_context.notification_level = "urgent"
        else:
            self._current_context.focus_level = 5
            self._current_context.available = True
            self._infer_context()

    def _infer_context(self):
        """Infer context from all available signals."""
        ctx = self._current_context

        # Time-based context
        hour = datetime.now().hour
        if 5 <= hour < 9:
            ctx.time_context = "morning"
        elif 9 <= hour < 12:
            ctx.time_context = "late_morning"
        elif 12 <= hour < 17:
            ctx.time_context = "afternoon"
        elif 17 <= hour < 21:
            ctx.time_context = "evening"
        else:
            ctx.time_context = "night"

        # Time-based mode defaults
        if hour < 6 or hour >= 23:
            ctx.mode = UserMode.SLEEP if ctx.mode != UserMode.FOCUS else ctx.mode
            ctx.available = False
        elif 9 <= hour < 17 and ctx.time_context in ["late_morning", "afternoon"]:
            if ctx.mode not in [UserMode.FOCUS, UserMode.MEETING]:
                ctx.mode = UserMode.WORK
        else:
            if ctx.mode not in [UserMode.FOCUS, UserMode.MEETING]:
                ctx.mode = UserMode.PERSONAL

        # Process signals
        for signal in self._signals:
            self._process_signal(signal)

        # Adjust verbosity based on context
        if ctx.in_meeting or ctx.mode == UserMode.FOCUS:
            ctx.verbosity = "brief"
        elif ctx.mode == UserMode.LEISURE:
            ctx.verbosity = "normal"
        else:
            ctx.verbosity = "normal"

        self._last_update = datetime.now()

        # Record in history
        self._context_history.append({
            "context": ctx.to_dict(),
            "timestamp": datetime.now().isoformat()
        })
        if len(self._context_history) > self._max_history:
            self._context_history = self._context_history[-self._max_history:]

    def _process_signal(self, signal: ContextSignal):
        """Process a single signal to update context."""
        ctx = self._current_context

        if signal.signal_type == "calendar_event":
            # Check if in a meeting
            event = signal.value
            if event.get("is_now"):
                ctx.in_meeting = True
                ctx.mode = UserMode.MEETING
                ctx.available = False
                if event.get("end_time"):
                    ctx.meeting_ends_at = event["end_time"]

        elif signal.signal_type == "location":
            location = signal.value
            ctx.location_type = location.get("type", "unknown")

            # Infer activity from location
            if location.get("type") == "gym":
                ctx.mode = UserMode.EXERCISE
                ctx.activity = ActivityType.UNKNOWN
            elif location.get("type") == "office":
                ctx.mode = UserMode.WORK
            elif location.get("type") == "home":
                ctx.mode = UserMode.PERSONAL
            elif location.get("moving"):
                ctx.mode = UserMode.COMMUTE
                ctx.activity = ActivityType.DRIVING if location.get("speed", 0) > 10 else ActivityType.WALKING

        elif signal.signal_type == "activity":
            activity = signal.value
            if activity == "walking":
                ctx.activity = ActivityType.WALKING
            elif activity == "driving":
                ctx.activity = ActivityType.DRIVING
                ctx.available = True  # Can listen while driving
            elif activity == "coding":
                ctx.activity = ActivityType.CODING
                ctx.focus_level = 7

        elif signal.signal_type == "device":
            device = signal.value
            # Adjust based on device (glasses vs phone)
            if device == "glasses":
                ctx.verbosity = "brief"  # Shorter responses for glasses
            elif device == "phone":
                ctx.verbosity = "normal"

    def adapt_response(self, response: str) -> str:
        """Adapt a response based on current context.

        Args:
            response: Original response

        Returns:
            Adapted response
        """
        ctx = self._current_context

        if ctx.verbosity == "brief":
            # Truncate to first sentence or 100 chars
            if "." in response[:100]:
                return response[:response.index(".") + 1]
            return response[:100] + "..."

        return response

    def should_notify(self, notification_type: str, urgency: str = "normal") -> bool:
        """Check if a notification should be delivered.

        Args:
            notification_type: Type of notification
            urgency: Urgency level (low, normal, important, urgent)

        Returns:
            True if notification should be delivered
        """
        ctx = self._current_context

        # Map urgency to numeric level
        urgency_levels = {"low": 1, "normal": 2, "important": 3, "urgent": 4}
        urgency_num = urgency_levels.get(urgency, 2)

        # Map notification level to threshold
        thresholds = {"all": 1, "important": 3, "urgent": 4, "none": 5}
        threshold = thresholds.get(ctx.notification_level, 2)

        return urgency_num >= threshold

    def get_greeting_style(self) -> str:
        """Get appropriate greeting style based on context.

        Returns:
            Greeting style: formal, casual, brief
        """
        ctx = self._current_context

        if ctx.mode == UserMode.WORK or ctx.in_meeting:
            return "brief"
        elif ctx.time_context == "morning":
            return "casual"
        else:
            return "casual"


# Global instance
_context_engine: Optional[ContextEngine] = None


def get_context_engine() -> ContextEngine:
    """Get or create global context engine."""
    global _context_engine
    if _context_engine is None:
        _context_engine = ContextEngine()
    return _context_engine
