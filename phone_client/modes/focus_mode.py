"""
Focus Mode - Pomodoro-style focus sessions with distraction blocking.
Helps maintain concentration with timed work sessions and breaks.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Focus session states."""
    IDLE = "idle"               # No active session
    FOCUSING = "focusing"       # In a focus period
    SHORT_BREAK = "short_break" # Short break
    LONG_BREAK = "long_break"   # Long break (after multiple sessions)
    PAUSED = "paused"           # Session paused


class DistractionLevel(Enum):
    """Distraction blocking levels."""
    OFF = "off"                 # No blocking
    LOW = "low"                 # Block non-essential notifications
    MEDIUM = "medium"           # Block most notifications, allow urgent
    HIGH = "high"               # Block all except emergencies
    DND = "dnd"                 # Do not disturb - block everything


@dataclass
class FocusSession:
    """A single focus session."""
    id: str
    task: str                              # What the user is working on
    focus_duration_min: int                # Focus period length
    break_duration_min: int                # Break length
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    completed_pomodoros: int = 0           # Number of completed focus periods
    total_focus_time_s: int = 0            # Total focused time
    total_break_time_s: int = 0            # Total break time
    distractions_blocked: int = 0          # Number of blocked distractions
    state: SessionState = SessionState.IDLE
    notes: List[str] = field(default_factory=list)

    def get_duration(self) -> timedelta:
        """Get total session duration."""
        end = self.ended_at or datetime.now()
        return end - self.started_at

    def get_efficiency(self) -> float:
        """Calculate focus efficiency (focus time / total time)."""
        total = self.total_focus_time_s + self.total_break_time_s
        if total == 0:
            return 0.0
        return self.total_focus_time_s / total


@dataclass
class FocusStats:
    """Focus statistics summary."""
    total_sessions: int
    total_pomodoros: int
    total_focus_time_min: int
    avg_session_length_min: float
    avg_efficiency: float
    best_streak: int
    blocked_distractions: int


class FocusMode:
    """
    Focus Mode with Pomodoro technique support.

    Provides timed focus sessions with automatic breaks,
    distraction blocking, and productivity tracking.
    """

    # Default Pomodoro settings
    DEFAULT_FOCUS_MIN = 25
    DEFAULT_SHORT_BREAK_MIN = 5
    DEFAULT_LONG_BREAK_MIN = 15
    LONG_BREAK_INTERVAL = 4  # Long break after every 4 pomodoros

    # Distraction keywords to block
    DISTRACTION_KEYWORDS = [
        "notification", "message", "email", "social",
        "news", "update", "reminder"  # except scheduled
    ]

    # Allowed during focus (emergency)
    EMERGENCY_KEYWORDS = [
        "emergency", "urgent", "critical", "alarm", "timer"
    ]

    def __init__(self, config: dict):
        """
        Initialize Focus Mode.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Load settings
        focus_config = config.get("focus_mode", {})
        self.default_focus_min = focus_config.get("focus_duration_min", self.DEFAULT_FOCUS_MIN)
        self.default_short_break_min = focus_config.get("short_break_min", self.DEFAULT_SHORT_BREAK_MIN)
        self.default_long_break_min = focus_config.get("long_break_min", self.DEFAULT_LONG_BREAK_MIN)
        self.long_break_interval = focus_config.get("long_break_interval", self.LONG_BREAK_INTERVAL)
        self.auto_start_break = focus_config.get("auto_start_break", True)
        self.play_sounds = focus_config.get("play_sounds", True)

        # Session state
        self._current_session: Optional[FocusSession] = None
        self._timer_task: Optional[asyncio.Task] = None
        self._period_start_time: Optional[float] = None
        self._distraction_level = DistractionLevel.OFF
        self._sessions_history: List[FocusSession] = []

        # Callbacks
        self._on_state_change: List[Callable[[SessionState, SessionState], None]] = []
        self._on_period_complete: List[Callable[[str, int], None]] = []  # period_type, duration
        self._on_tick: List[Callable[[int, int], None]] = []  # elapsed, remaining

        logger.info("FocusMode initialized")

    async def start_session(
        self,
        task: str,
        focus_min: int = None,
        break_min: int = None,
        distraction_level: DistractionLevel = DistractionLevel.MEDIUM
    ) -> FocusSession:
        """
        Start a new focus session.

        Args:
            task: Description of what you're working on
            focus_min: Focus period length (minutes)
            break_min: Break period length (minutes)
            distraction_level: Level of distraction blocking

        Returns:
            The created FocusSession
        """
        if self._current_session and self._current_session.state != SessionState.IDLE:
            logger.warning("Session already in progress, ending it first")
            await self.end_session()

        import uuid

        focus_min = focus_min or self.default_focus_min
        break_min = break_min or self.default_short_break_min

        session = FocusSession(
            id=str(uuid.uuid4())[:8],
            task=task,
            focus_duration_min=focus_min,
            break_duration_min=break_min,
            state=SessionState.FOCUSING
        )

        self._current_session = session
        self._distraction_level = distraction_level
        self._period_start_time = time.time()

        # Start the timer
        self._timer_task = asyncio.create_task(
            self._run_timer(focus_min * 60, SessionState.FOCUSING)
        )

        logger.info(f"Focus session started: {task} ({focus_min}min focus)")
        self._notify_state_change(SessionState.IDLE, SessionState.FOCUSING)

        return session

    async def _run_timer(self, duration_s: int, period_type: SessionState):
        """Run the countdown timer for a period."""
        start = time.time()
        remaining = duration_s

        try:
            while remaining > 0:
                await asyncio.sleep(1)
                elapsed = int(time.time() - start)
                remaining = duration_s - elapsed

                # Notify tick callbacks
                for callback in self._on_tick:
                    try:
                        callback(elapsed, remaining)
                    except Exception as e:
                        logger.error(f"Tick callback error: {e}")

            # Period complete
            await self._on_period_end(period_type, duration_s)

        except asyncio.CancelledError:
            logger.debug("Timer cancelled")
            elapsed = int(time.time() - start)
            if period_type == SessionState.FOCUSING:
                self._current_session.total_focus_time_s += elapsed
            else:
                self._current_session.total_break_time_s += elapsed

    async def _on_period_end(self, period_type: SessionState, duration_s: int):
        """Handle period completion."""
        if not self._current_session:
            return

        # Update session stats
        if period_type == SessionState.FOCUSING:
            self._current_session.total_focus_time_s += duration_s
            self._current_session.completed_pomodoros += 1
            logger.info(f"Pomodoro #{self._current_session.completed_pomodoros} complete!")

            # Notify callbacks
            for callback in self._on_period_complete:
                try:
                    callback("focus", duration_s)
                except Exception as e:
                    logger.error(f"Period complete callback error: {e}")

            # Start break if auto-start enabled
            if self.auto_start_break:
                await self._start_break()

        else:  # Break ended
            self._current_session.total_break_time_s += duration_s

            for callback in self._on_period_complete:
                try:
                    callback("break", duration_s)
                except Exception as e:
                    logger.error(f"Period complete callback error: {e}")

            # Start next focus period
            await self._start_focus()

    async def _start_break(self):
        """Start a break period."""
        if not self._current_session:
            return

        # Determine break type
        if self._current_session.completed_pomodoros % self.long_break_interval == 0:
            break_type = SessionState.LONG_BREAK
            break_min = self.default_long_break_min
        else:
            break_type = SessionState.SHORT_BREAK
            break_min = self._current_session.break_duration_min

        old_state = self._current_session.state
        self._current_session.state = break_type
        self._period_start_time = time.time()

        # Start break timer
        self._timer_task = asyncio.create_task(
            self._run_timer(break_min * 60, break_type)
        )

        logger.info(f"Break started: {break_type.value} ({break_min}min)")
        self._notify_state_change(old_state, break_type)

    async def _start_focus(self):
        """Start a focus period."""
        if not self._current_session:
            return

        old_state = self._current_session.state
        self._current_session.state = SessionState.FOCUSING
        self._period_start_time = time.time()

        # Start focus timer
        self._timer_task = asyncio.create_task(
            self._run_timer(self._current_session.focus_duration_min * 60, SessionState.FOCUSING)
        )

        logger.info("Focus period started")
        self._notify_state_change(old_state, SessionState.FOCUSING)

    async def pause_session(self):
        """Pause the current session."""
        if not self._current_session:
            return

        if self._current_session.state == SessionState.PAUSED:
            return

        old_state = self._current_session.state

        # Cancel timer
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass

        self._current_session.state = SessionState.PAUSED
        logger.info("Session paused")
        self._notify_state_change(old_state, SessionState.PAUSED)

    async def resume_session(self):
        """Resume a paused session."""
        if not self._current_session:
            return

        if self._current_session.state != SessionState.PAUSED:
            return

        # Resume with focus period
        await self._start_focus()
        logger.info("Session resumed")

    async def end_session(self) -> Optional[FocusSession]:
        """End the current session."""
        if not self._current_session:
            return None

        # Cancel timer
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass

        old_state = self._current_session.state
        self._current_session.state = SessionState.IDLE
        self._current_session.ended_at = datetime.now()

        # Store in history
        session = self._current_session
        self._sessions_history.append(session)

        # Reset
        self._current_session = None
        self._distraction_level = DistractionLevel.OFF
        self._timer_task = None

        logger.info(f"Session ended: {session.completed_pomodoros} pomodoros, "
                   f"{session.total_focus_time_s // 60}min focus time")
        self._notify_state_change(old_state, SessionState.IDLE)

        return session

    async def skip_break(self):
        """Skip the current break and start focusing."""
        if not self._current_session:
            return

        if self._current_session.state not in (SessionState.SHORT_BREAK, SessionState.LONG_BREAK):
            return

        # Cancel break timer
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass

        await self._start_focus()
        logger.info("Break skipped")

    async def extend_focus(self, minutes: int = 5):
        """Extend the current focus period."""
        if not self._current_session:
            return

        if self._current_session.state != SessionState.FOCUSING:
            return

        # This is a simplified extension - in practice you'd
        # adjust the running timer
        self._current_session.focus_duration_min += minutes
        logger.info(f"Focus extended by {minutes} minutes")

    def add_note(self, note: str):
        """Add a note to the current session."""
        if self._current_session:
            self._current_session.notes.append(note)
            logger.debug(f"Note added: {note}")

    def should_block(self, notification_type: str, content: str = "") -> bool:
        """
        Check if a notification should be blocked.

        Args:
            notification_type: Type of notification
            content: Notification content

        Returns:
            True if notification should be blocked
        """
        if self._distraction_level == DistractionLevel.OFF:
            return False

        if not self._current_session:
            return False

        if self._current_session.state not in (SessionState.FOCUSING,):
            return False

        content_lower = content.lower()
        type_lower = notification_type.lower()

        # Always allow emergencies
        for keyword in self.EMERGENCY_KEYWORDS:
            if keyword in content_lower or keyword in type_lower:
                return False

        # Check distraction level
        if self._distraction_level == DistractionLevel.DND:
            self._current_session.distractions_blocked += 1
            return True

        if self._distraction_level == DistractionLevel.HIGH:
            self._current_session.distractions_blocked += 1
            return True

        if self._distraction_level == DistractionLevel.MEDIUM:
            for keyword in self.DISTRACTION_KEYWORDS:
                if keyword in content_lower or keyword in type_lower:
                    self._current_session.distractions_blocked += 1
                    return True

        if self._distraction_level == DistractionLevel.LOW:
            # Only block social/entertainment
            low_priority = ["social", "game", "entertainment", "news"]
            for keyword in low_priority:
                if keyword in type_lower:
                    self._current_session.distractions_blocked += 1
                    return True

        return False

    def get_current_session(self) -> Optional[FocusSession]:
        """Get the current session."""
        return self._current_session

    def get_time_remaining(self) -> int:
        """Get seconds remaining in current period."""
        if not self._current_session or not self._period_start_time:
            return 0

        if self._current_session.state == SessionState.FOCUSING:
            duration = self._current_session.focus_duration_min * 60
        elif self._current_session.state in (SessionState.SHORT_BREAK, SessionState.LONG_BREAK):
            if self._current_session.state == SessionState.LONG_BREAK:
                duration = self.default_long_break_min * 60
            else:
                duration = self._current_session.break_duration_min * 60
        else:
            return 0

        elapsed = time.time() - self._period_start_time
        return max(0, int(duration - elapsed))

    def get_stats(self, days: int = 7) -> FocusStats:
        """Get focus statistics for the past N days."""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [s for s in self._sessions_history if s.started_at > cutoff]

        if not recent:
            return FocusStats(
                total_sessions=0,
                total_pomodoros=0,
                total_focus_time_min=0,
                avg_session_length_min=0,
                avg_efficiency=0,
                best_streak=0,
                blocked_distractions=0
            )

        total_focus = sum(s.total_focus_time_s for s in recent)
        total_pomodoros = sum(s.completed_pomodoros for s in recent)
        blocked = sum(s.distractions_blocked for s in recent)

        avg_length = sum(s.get_duration().total_seconds() for s in recent) / len(recent) / 60
        efficiencies = [s.get_efficiency() for s in recent if s.get_efficiency() > 0]
        avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0

        return FocusStats(
            total_sessions=len(recent),
            total_pomodoros=total_pomodoros,
            total_focus_time_min=total_focus // 60,
            avg_session_length_min=avg_length,
            avg_efficiency=avg_efficiency,
            best_streak=max(s.completed_pomodoros for s in recent),
            blocked_distractions=blocked
        )

    def _notify_state_change(self, old_state: SessionState, new_state: SessionState):
        """Notify callbacks of state change."""
        for callback in self._on_state_change:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    def on_state_change(self, callback: Callable[[SessionState, SessionState], None]):
        """Register callback for state changes."""
        self._on_state_change.append(callback)

    def on_period_complete(self, callback: Callable[[str, int], None]):
        """Register callback for period completion."""
        self._on_period_complete.append(callback)

    def on_tick(self, callback: Callable[[int, int], None]):
        """Register callback for timer ticks (elapsed, remaining)."""
        self._on_tick.append(callback)

    def format_for_display(self) -> List[str]:
        """Format current state for HUD display."""
        lines = []

        if not self._current_session:
            lines.append("Focus Mode: Idle")
            lines.append("")
            lines.append("Say 'focus on [task]' to start")
            return lines

        session = self._current_session
        remaining = self.get_time_remaining()
        remaining_min = remaining // 60
        remaining_sec = remaining % 60

        # State indicator
        state_icons = {
            SessionState.FOCUSING: "🎯",
            SessionState.SHORT_BREAK: "☕",
            SessionState.LONG_BREAK: "🌴",
            SessionState.PAUSED: "⏸️"
        }
        icon = state_icons.get(session.state, "")

        lines.append(f"{icon} {session.state.value.upper()}")
        lines.append(f"Task: {session.task[:30]}{'...' if len(session.task) > 30 else ''}")
        lines.append("")
        lines.append(f"⏱️ {remaining_min:02d}:{remaining_sec:02d}")
        lines.append("")
        lines.append(f"Pomodoros: {session.completed_pomodoros}")

        if session.distractions_blocked > 0:
            lines.append(f"Blocked: {session.distractions_blocked} distractions")

        return lines

    def format_for_tts(self) -> str:
        """Format current state for TTS."""
        if not self._current_session:
            return "Focus mode is idle. Say 'focus on' followed by your task to start."

        session = self._current_session
        remaining = self.get_time_remaining()
        remaining_min = remaining // 60

        if session.state == SessionState.FOCUSING:
            return f"Focusing on {session.task}. {remaining_min} minutes remaining."
        elif session.state == SessionState.SHORT_BREAK:
            return f"Short break. {remaining_min} minutes remaining."
        elif session.state == SessionState.LONG_BREAK:
            return f"Long break after {session.completed_pomodoros} pomodoros. {remaining_min} minutes remaining."
        elif session.state == SessionState.PAUSED:
            return "Session paused. Say 'resume' to continue."

        return "Focus mode active."


# Test
def test_focus_mode():
    """Test focus mode functionality."""
    import asyncio

    print("=== Focus Mode Test ===\n")

    config = {
        "focus_mode": {
            "focus_duration_min": 1,  # Short for testing
            "short_break_min": 1,
            "auto_start_break": False,
            "play_sounds": False
        }
    }

    fm = FocusMode(config)

    # Track state changes
    state_changes = []
    fm.on_state_change(lambda old, new: state_changes.append((old.value, new.value)))

    async def run_test():
        print("1. Starting focus session...")
        session = await fm.start_session("Write documentation", focus_min=1)
        print(f"   Session ID: {session.id}")
        print(f"   Task: {session.task}")
        print(f"   State: {session.state.value}")
        print()

        # Display
        print("2. Display format:")
        for line in fm.format_for_display():
            print(f"   {line}")
        print()

        # Check blocking
        print("3. Distraction blocking:")
        print(f"   Block 'social notification': {fm.should_block('social', 'New message')}")
        print(f"   Block 'emergency alert': {fm.should_block('emergency', 'Critical alert')}")
        print()

        # Add note
        print("4. Adding note...")
        fm.add_note("Need to check API docs")
        print(f"   Notes: {session.notes}")
        print()

        # Wait a bit
        print("5. Waiting 2 seconds...")
        await asyncio.sleep(2)
        print(f"   Time remaining: {fm.get_time_remaining()}s")
        print()

        # End session
        print("6. Ending session...")
        final = await fm.end_session()
        print(f"   Focus time: {final.total_focus_time_s}s")
        print(f"   State: {final.state.value}")
        print()

        # State changes
        print("7. State changes:")
        for old, new in state_changes:
            print(f"   {old} → {new}")
        print()

        # TTS
        print("8. TTS format:")
        print(f"   {fm.format_for_tts()}")

    asyncio.run(run_test())


if __name__ == "__main__":
    test_focus_mode()
