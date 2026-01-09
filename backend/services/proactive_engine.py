"""Proactive Alert Engine - Context-aware background notifications."""
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of proactive alerts."""
    LEAVE_REMINDER = "leave_reminder"       # Traffic-based departure alerts
    FLIGHT_UPDATE = "flight_update"         # Flight delays/changes
    PACKAGE_ARRIVING = "package_arriving"   # Delivery notifications
    CALENDAR_REMINDER = "calendar_reminder" # Upcoming events
    WEATHER_ALERT = "weather_alert"         # Weather changes
    LEARNING_REMINDER = "learning_reminder" # Cards due for review
    GOAL_PROGRESS = "goal_progress"         # Fitness goal updates
    FOCUS_BREAK = "focus_break"             # Break reminders during focus
    CUSTOM = "custom"                       # Custom/user-defined alerts


@dataclass
class Alert:
    """A proactive alert."""
    id: str
    type: AlertType
    title: str
    message: str
    urgency: str = "normal"  # "low", "normal", "important", "urgent"
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    data: Dict[str, Any] = field(default_factory=dict)
    delivered: bool = False
    acknowledged: bool = False

    def to_voice_message(self) -> str:
        """Format for voice output."""
        if self.urgency == "urgent":
            return f"Urgent: {self.message}"
        elif self.urgency == "important":
            return f"Heads up: {self.message}"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "message": self.message,
            "urgency": self.urgency,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "data": self.data,
            "delivered": self.delivered,
            "acknowledged": self.acknowledged
        }

    def is_expired(self) -> bool:
        """Check if alert has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class AlertPreferences:
    """User preferences for alert types."""
    enabled_types: Set[AlertType] = field(default_factory=lambda: set(AlertType))
    quiet_hours_start: Optional[int] = None  # Hour (0-23)
    quiet_hours_end: Optional[int] = None
    learning_reminder_interval: int = 4  # Hours between learning reminders
    focus_break_interval: int = 90  # Minutes between break reminders
    min_urgency_during_focus: str = "important"

    def is_type_enabled(self, alert_type: AlertType) -> bool:
        return alert_type in self.enabled_types

    def is_quiet_hours(self) -> bool:
        if self.quiet_hours_start is None or self.quiet_hours_end is None:
            return False
        hour = datetime.now().hour
        if self.quiet_hours_start <= self.quiet_hours_end:
            return self.quiet_hours_start <= hour < self.quiet_hours_end
        else:
            return hour >= self.quiet_hours_start or hour < self.quiet_hours_end


class ProactiveEngine:
    """Background engine for context-aware proactive alerts.

    Features:
    - Periodic checking of multiple alert sources
    - Context-aware filtering (respects focus mode, meetings, etc.)
    - Delivery via WebSocket and optional voice
    - Alert history and acknowledgment tracking
    - User preferences for alert types and timing
    """

    def __init__(self):
        self._context_engine = None
        self._preferences = AlertPreferences()
        self._alert_queue: List[Alert] = []
        self._alert_history: List[Alert] = []
        self._check_interval = 60  # seconds
        self._last_checks: Dict[AlertType, datetime] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Callbacks for delivery
        self._websocket_callback: Optional[Callable] = None
        self._voice_callback: Optional[Callable] = None

        # Cooldowns to prevent spam
        self._cooldowns: Dict[AlertType, int] = {
            AlertType.LEARNING_REMINDER: 14400,  # 4 hours
            AlertType.FOCUS_BREAK: 5400,         # 90 minutes
            AlertType.GOAL_PROGRESS: 7200,       # 2 hours
            AlertType.WEATHER_ALERT: 3600,       # 1 hour
        }

    def _get_context_engine(self):
        """Lazy load context engine."""
        if self._context_engine is None:
            try:
                from backend.services.context_engine import get_context_engine
                self._context_engine = get_context_engine()
            except ImportError:
                logger.warning("Context engine not available")
        return self._context_engine

    def set_websocket_callback(self, callback: Callable):
        """Set callback for WebSocket delivery."""
        self._websocket_callback = callback

    def set_voice_callback(self, callback: Callable):
        """Set callback for voice delivery."""
        self._voice_callback = callback

    async def start(self):
        """Start the background alert checking loop."""
        if self._running:
            logger.warning("Proactive engine already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Proactive alert engine started")

    async def stop(self):
        """Stop the background loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Proactive alert engine stopped")

    async def _run_loop(self):
        """Main background loop."""
        while self._running:
            try:
                await self._check_all_sources()
                await self._process_queue()
            except Exception as e:
                logger.error(f"Error in proactive engine loop: {e}")

            await asyncio.sleep(self._check_interval)

    async def _check_all_sources(self):
        """Check all alert sources for new alerts."""
        alerts = []

        # Check each source with proper error handling
        checks = [
            (AlertType.CALENDAR_REMINDER, self._check_calendar),
            (AlertType.FLIGHT_UPDATE, self._check_flights),
            (AlertType.PACKAGE_ARRIVING, self._check_packages),
            (AlertType.LEARNING_REMINDER, self._check_learning),
            (AlertType.GOAL_PROGRESS, self._check_fitness_goals),
            (AlertType.FOCUS_BREAK, self._check_focus_breaks),
            (AlertType.CUSTOM, self._check_timers),  # Timer/reminder notifications
        ]

        for alert_type, check_func in checks:
            if not self._preferences.is_type_enabled(alert_type):
                continue

            if not self._is_cooldown_expired(alert_type):
                continue

            try:
                new_alerts = await check_func()
                alerts.extend(new_alerts)
            except Exception as e:
                logger.debug(f"Error checking {alert_type.value}: {e}")

        # Add to queue
        for alert in alerts:
            if not self._is_duplicate(alert):
                self._alert_queue.append(alert)
                self._last_checks[alert.type] = datetime.now()

    def _is_cooldown_expired(self, alert_type: AlertType) -> bool:
        """Check if cooldown period has expired for alert type."""
        last_check = self._last_checks.get(alert_type)
        if last_check is None:
            return True

        cooldown = self._cooldowns.get(alert_type, 300)  # Default 5 min
        return (datetime.now() - last_check).total_seconds() > cooldown

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is duplicate of recent alert."""
        # Check queue
        for existing in self._alert_queue:
            if existing.type == alert.type and existing.title == alert.title:
                return True

        # Check recent history (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        for existing in self._alert_history:
            if existing.created_at < cutoff:
                continue
            if existing.type == alert.type and existing.title == alert.title:
                return True

        return False

    async def _process_queue(self):
        """Process pending alerts in the queue."""
        if not self._alert_queue:
            return

        # Get context for filtering
        context = self._get_context_engine()

        alerts_to_remove = []

        for alert in self._alert_queue:
            # Check expiration
            if alert.is_expired():
                alerts_to_remove.append(alert)
                continue

            # Check if we should notify
            if not self._should_notify(alert, context):
                continue

            # Deliver alert
            await self._deliver_alert(alert, context)
            alert.delivered = True
            alerts_to_remove.append(alert)
            self._alert_history.append(alert)

        # Remove processed alerts
        for alert in alerts_to_remove:
            if alert in self._alert_queue:
                self._alert_queue.remove(alert)

        # Trim history
        if len(self._alert_history) > 100:
            self._alert_history = self._alert_history[-50:]

    def _should_notify(self, alert: Alert, context) -> bool:
        """Determine if alert should be delivered based on context."""
        # Check quiet hours
        if self._preferences.is_quiet_hours():
            if alert.urgency not in ["urgent"]:
                return False

        # Check context if available
        if context:
            try:
                ctx = context.get_current_context()
                return ctx.should_notify(alert.type.value, alert.urgency)
            except Exception:
                pass

        # Default: allow
        return True

    async def _deliver_alert(self, alert: Alert, context):
        """Deliver alert via configured channels."""
        logger.info(f"Delivering alert: {alert.title}")

        # WebSocket delivery
        if self._websocket_callback:
            try:
                await self._websocket_callback("alerts", alert.to_dict())
            except Exception as e:
                logger.error(f"WebSocket delivery failed: {e}")

        # Voice delivery (only for important/urgent when appropriate)
        if self._voice_callback and alert.urgency in ["important", "urgent"]:
            # Check if voice is appropriate
            should_speak = True
            if context:
                try:
                    ctx = context.get_current_context()
                    # Don't speak during meetings or focus mode
                    if ctx.current_activity in ["in_meeting", "focus"]:
                        should_speak = False
                except Exception:
                    pass

            if should_speak:
                try:
                    await self._voice_callback(alert.to_voice_message())
                except Exception as e:
                    logger.error(f"Voice delivery failed: {e}")

    # Alert source check methods

    async def _check_calendar(self) -> List[Alert]:
        """Check for upcoming calendar events."""
        alerts = []
        try:
            # Would integrate with calendar service
            # For now, placeholder
            pass
        except Exception as e:
            logger.debug(f"Calendar check error: {e}")
        return alerts

    async def _check_flights(self) -> List[Alert]:
        """Check for flight updates."""
        alerts = []
        try:
            from backend.services.travel_service import get_travel_service
            service = get_travel_service()

            if not service.is_configured():
                return []

            delayed = await service.check_flight_delays()
            for flight in delayed:
                alerts.append(Alert(
                    id=str(uuid.uuid4()),
                    type=AlertType.FLIGHT_UPDATE,
                    title=f"Flight {flight.flight_number} Delayed",
                    message=f"Your flight {flight.airline} {flight.flight_number} is delayed {flight.delay_minutes} minutes.",
                    urgency="important",
                    expires_at=flight.departure_time,
                    data={"flight": flight.to_dict()}
                ))
        except Exception as e:
            logger.debug(f"Flight check error: {e}")
        return alerts

    async def _check_packages(self) -> List[Alert]:
        """Check for package delivery updates."""
        alerts = []
        try:
            from backend.services.package_tracker import get_package_tracker
            service = get_package_tracker()

            packages = await service.get_expected_today()
            for pkg in packages:
                if pkg.get("status") == "out_for_delivery":
                    alerts.append(Alert(
                        id=str(uuid.uuid4()),
                        type=AlertType.PACKAGE_ARRIVING,
                        title="Package Arriving",
                        message=f"Your {pkg.get('carrier', 'package')} delivery is out for delivery.",
                        urgency="normal",
                        expires_at=datetime.now() + timedelta(hours=12),
                        data={"package": pkg}
                    ))
        except Exception as e:
            logger.debug(f"Package check error: {e}")
        return alerts

    async def _check_learning(self) -> List[Alert]:
        """Check for due flashcards."""
        alerts = []
        try:
            from backend.services.learning_service import get_learning_service
            service = get_learning_service()

            stats = await service.get_stats()
            if stats.due_today >= 5:  # Only alert if meaningful amount
                alerts.append(Alert(
                    id=str(uuid.uuid4()),
                    type=AlertType.LEARNING_REMINDER,
                    title="Cards Due for Review",
                    message=f"You have {stats.due_today} flashcards ready for review. Say 'quiz me' to start.",
                    urgency="low",
                    expires_at=datetime.now() + timedelta(hours=4),
                    data={"due_count": stats.due_today}
                ))
        except Exception as e:
            logger.debug(f"Learning check error: {e}")
        return alerts

    async def _check_fitness_goals(self) -> List[Alert]:
        """Check fitness goal progress."""
        alerts = []
        try:
            from backend.services.healthkit_service import get_healthkit_service
            service = get_healthkit_service()

            goals = await service.get_goal_progress()
            steps = goals.get("steps", {})

            # Alert when close to goal
            if 80 <= steps.get("percent", 0) < 100:
                remaining = steps.get("remaining", 0)
                alerts.append(Alert(
                    id=str(uuid.uuid4()),
                    type=AlertType.GOAL_PROGRESS,
                    title="Almost There!",
                    message=f"Just {remaining:,} more steps to hit your goal!",
                    urgency="low",
                    expires_at=datetime.now() + timedelta(hours=2),
                    data={"goal": "steps", "progress": steps}
                ))
        except Exception as e:
            logger.debug(f"Fitness check error: {e}")
        return alerts

    async def _check_focus_breaks(self) -> List[Alert]:
        """Check if user needs a break during focus mode."""
        alerts = []
        try:
            context = self._get_context_engine()
            if not context:
                return []

            ctx = context.get_current_context()

            # Check if in focus mode for extended time
            if ctx.mode == "focus" and ctx.focus_level >= 8:
                focus_duration = context.get_focus_duration()
                if focus_duration and focus_duration.total_seconds() > 5400:  # 90 min
                    alerts.append(Alert(
                        id=str(uuid.uuid4()),
                        type=AlertType.FOCUS_BREAK,
                        title="Time for a Break",
                        message="You've been focused for 90 minutes. A short break might help!",
                        urgency="low",
                        expires_at=datetime.now() + timedelta(minutes=30),
                        data={"focus_minutes": focus_duration.total_seconds() / 60}
                    ))
        except Exception as e:
            logger.debug(f"Focus break check error: {e}")
        return alerts

    async def _check_timers(self) -> List[Alert]:
        """Check for expired timers and reminders."""
        alerts = []
        try:
            from backend.voice.tools.reminders import RemindersTool
            tool = RemindersTool()
            expired = await tool.get_expired_reminders()

            for reminder in expired:
                alert_type = AlertType.CUSTOM
                title = "Timer Done" if reminder["is_timer"] else "Reminder"
                urgency = "important"  # Timers should be delivered promptly

                alerts.append(Alert(
                    id=str(uuid.uuid4()),
                    type=alert_type,
                    title=title,
                    message=reminder["message"],
                    urgency=urgency,
                    expires_at=datetime.now() + timedelta(hours=1),
                    data={"reminder_id": reminder["id"], "is_timer": reminder["is_timer"]}
                ))

                # Mark as completed so we don't deliver again
                await tool.mark_completed(reminder["id"])

            if expired:
                logger.info(f"Found {len(expired)} expired timer/reminders")

        except Exception as e:
            logger.debug(f"Timer check error: {e}")
        return alerts

    # Public API

    def get_pending_alerts(self) -> List[Alert]:
        """Get pending alerts in queue."""
        return [a for a in self._alert_queue if not a.is_expired()]

    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recently delivered alerts."""
        return self._alert_history[-limit:]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self._alert_history:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def enable_alert_type(self, alert_type: AlertType):
        """Enable an alert type."""
        self._preferences.enabled_types.add(alert_type)

    def disable_alert_type(self, alert_type: AlertType):
        """Disable an alert type."""
        self._preferences.enabled_types.discard(alert_type)

    def is_alert_type_enabled(self, alert_type: AlertType) -> bool:
        """Check if alert type is enabled."""
        return self._preferences.is_type_enabled(alert_type)

    def set_quiet_hours(self, start_hour: Optional[int], end_hour: Optional[int]):
        """Set quiet hours (no non-urgent alerts)."""
        self._preferences.quiet_hours_start = start_hour
        self._preferences.quiet_hours_end = end_hour

    def silence_alerts(self, duration_minutes: int):
        """Temporarily silence non-urgent alerts."""
        # Set quiet hours for specified duration
        now = datetime.now()
        end = now + timedelta(minutes=duration_minutes)
        self._preferences.quiet_hours_start = now.hour
        self._preferences.quiet_hours_end = end.hour

    def get_preferences(self) -> Dict[str, Any]:
        """Get current alert preferences."""
        return {
            "enabled_types": [t.value for t in self._preferences.enabled_types],
            "quiet_hours": {
                "start": self._preferences.quiet_hours_start,
                "end": self._preferences.quiet_hours_end
            },
            "learning_reminder_interval": self._preferences.learning_reminder_interval,
            "focus_break_interval": self._preferences.focus_break_interval
        }


# Global instance
_proactive_engine: Optional[ProactiveEngine] = None


def get_proactive_engine() -> ProactiveEngine:
    """Get or create global ProactiveEngine instance."""
    global _proactive_engine
    if _proactive_engine is None:
        _proactive_engine = ProactiveEngine()
    return _proactive_engine
