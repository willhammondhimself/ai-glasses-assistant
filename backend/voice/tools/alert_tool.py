"""Alert Management voice tool - Control proactive alert preferences."""
import logging
import re
from typing import Optional
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class AlertVoiceTool(VoiceTool):
    """Voice-controlled alert management.

    Allows users to:
    - Check pending alerts
    - Silence alerts temporarily
    - Enable/disable specific alert types
    - View alert preferences
    """

    name = "alerts"
    description = "Manage proactive alerts and notifications"

    keywords = [
        r"\balerts?\b",
        r"\bnotifications?\b",
        r"\b(turn\s+)?(on|off)\s+alerts?\b",
        r"\bsilence\s+(alerts?|notifications?)\b",
        r"\b(any|pending)\s+(alerts?|notifications?)\b",
        r"\b(what|show)\s+(are\s+)?my\s+alerts?\b",
        r"\balert\s+(settings?|preferences?)\b",
        r"\b(enable|disable)\s+.+\s*(alerts?|reminders?)\b",
        r"\bdo\s+not\s+disturb\b",
        r"\bquiet\s+(mode|hours?)\b",
    ]

    priority = 5

    def __init__(self):
        self._proactive_engine = None

    def _get_engine(self):
        """Get proactive engine (lazy load)."""
        if self._proactive_engine is None:
            from backend.services.proactive_engine import get_proactive_engine
            self._proactive_engine = get_proactive_engine()
        return self._proactive_engine

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute alert management command.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult with alert info
        """
        query_lower = query.lower()
        engine = self._get_engine()

        try:
            # Check pending alerts
            if self._is_check_alerts(query_lower):
                return await self._handle_check_alerts(engine)

            # Silence alerts
            if self._is_silence_request(query_lower):
                duration = self._extract_duration(query_lower)
                return await self._handle_silence(engine, duration)

            # Enable alert type
            if self._is_enable_request(query_lower):
                alert_type = self._extract_alert_type(query_lower)
                return await self._handle_enable(engine, alert_type)

            # Disable alert type
            if self._is_disable_request(query_lower):
                alert_type = self._extract_alert_type(query_lower)
                return await self._handle_disable(engine, alert_type)

            # Show preferences
            if self._is_preferences_query(query_lower):
                return await self._handle_preferences(engine)

            # Default: show pending alerts
            return await self._handle_check_alerts(engine)

        except Exception as e:
            logger.error(f"Alert tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble with alert management.",
                data={"error": str(e)}
            )

    def _is_check_alerts(self, query: str) -> bool:
        patterns = [
            "any alerts", "pending alerts", "what alerts",
            "show alerts", "check alerts", "my alerts"
        ]
        return any(p in query for p in patterns)

    def _is_silence_request(self, query: str) -> bool:
        patterns = [
            "silence", "quiet", "do not disturb",
            "mute alerts", "pause alerts", "stop alerts"
        ]
        return any(p in query for p in patterns)

    def _is_enable_request(self, query: str) -> bool:
        patterns = ["turn on", "enable", "start"]
        return any(p in query for p in patterns) and "alert" in query or "reminder" in query

    def _is_disable_request(self, query: str) -> bool:
        patterns = ["turn off", "disable", "stop", "no more"]
        return any(p in query for p in patterns) and ("alert" in query or "reminder" in query)

    def _is_preferences_query(self, query: str) -> bool:
        patterns = ["alert settings", "alert preferences", "notification settings"]
        return any(p in query for p in patterns)

    def _extract_duration(self, query: str) -> int:
        """Extract duration in minutes from query."""
        # "silence for an hour" -> 60
        # "silence for 30 minutes" -> 30
        # "do not disturb for 2 hours" -> 120

        # Check for hour patterns
        hour_match = re.search(r"(\d+)\s*hours?", query)
        if hour_match:
            return int(hour_match.group(1)) * 60

        if "an hour" in query or "one hour" in query:
            return 60

        # Check for minute patterns
        min_match = re.search(r"(\d+)\s*min", query)
        if min_match:
            return int(min_match.group(1))

        # Default: 1 hour
        return 60

    def _extract_alert_type(self, query: str) -> Optional[str]:
        """Extract alert type from query."""
        type_keywords = {
            "learning": "learning_reminder",
            "flashcard": "learning_reminder",
            "study": "learning_reminder",
            "flight": "flight_update",
            "travel": "flight_update",
            "package": "package_arriving",
            "delivery": "package_arriving",
            "calendar": "calendar_reminder",
            "meeting": "calendar_reminder",
            "event": "calendar_reminder",
            "weather": "weather_alert",
            "fitness": "goal_progress",
            "goal": "goal_progress",
            "step": "goal_progress",
            "break": "focus_break",
            "focus": "focus_break",
        }

        for keyword, alert_type in type_keywords.items():
            if keyword in query:
                return alert_type

        return None

    async def _handle_check_alerts(self, engine) -> VoiceToolResult:
        """Check for pending alerts."""
        pending = engine.get_pending_alerts()

        if not pending:
            return VoiceToolResult(
                success=True,
                message="No pending alerts.",
                data={"pending_count": 0}
            )

        # Read first few alerts
        messages = []
        for alert in pending[:3]:
            messages.append(alert.to_voice_message())

        if len(pending) > 3:
            count_more = len(pending) - 3
            messages.append(f"And {count_more} more.")

        return VoiceToolResult(
            success=True,
            message=" ".join(messages),
            data={
                "pending_count": len(pending),
                "alerts": [a.to_dict() for a in pending[:5]]
            }
        )

    async def _handle_silence(self, engine, duration_minutes: int) -> VoiceToolResult:
        """Silence alerts for specified duration."""
        engine.silence_alerts(duration_minutes)

        if duration_minutes >= 60:
            hours = duration_minutes // 60
            time_str = f"{hours} hour{'s' if hours > 1 else ''}"
        else:
            time_str = f"{duration_minutes} minutes"

        return VoiceToolResult(
            success=True,
            message=f"Alerts silenced for {time_str}. Urgent alerts will still come through.",
            data={"silenced_minutes": duration_minutes}
        )

    async def _handle_enable(self, engine, alert_type: Optional[str]) -> VoiceToolResult:
        """Enable an alert type."""
        if not alert_type:
            return VoiceToolResult(
                success=False,
                message="What type of alerts would you like to enable? For example: learning reminders, flight updates, or fitness goals.",
                data={"error": "no_type_specified"}
            )

        from backend.services.proactive_engine import AlertType
        try:
            type_enum = AlertType(alert_type)
            engine.enable_alert_type(type_enum)

            type_name = alert_type.replace("_", " ")
            return VoiceToolResult(
                success=True,
                message=f"{type_name.title()} alerts enabled.",
                data={"enabled": alert_type}
            )
        except ValueError:
            return VoiceToolResult(
                success=False,
                message=f"Unknown alert type: {alert_type}",
                data={"error": "unknown_type"}
            )

    async def _handle_disable(self, engine, alert_type: Optional[str]) -> VoiceToolResult:
        """Disable an alert type."""
        if not alert_type:
            return VoiceToolResult(
                success=False,
                message="What type of alerts would you like to disable? For example: learning reminders, flight updates, or fitness goals.",
                data={"error": "no_type_specified"}
            )

        from backend.services.proactive_engine import AlertType
        try:
            type_enum = AlertType(alert_type)
            engine.disable_alert_type(type_enum)

            type_name = alert_type.replace("_", " ")
            return VoiceToolResult(
                success=True,
                message=f"{type_name.title()} alerts disabled.",
                data={"disabled": alert_type}
            )
        except ValueError:
            return VoiceToolResult(
                success=False,
                message=f"Unknown alert type: {alert_type}",
                data={"error": "unknown_type"}
            )

    async def _handle_preferences(self, engine) -> VoiceToolResult:
        """Get current alert preferences."""
        prefs = engine.get_preferences()

        enabled = prefs.get("enabled_types", [])
        quiet = prefs.get("quiet_hours", {})

        parts = []

        # Enabled types
        if enabled:
            enabled_names = [t.replace("_", " ") for t in enabled]
            parts.append(f"Enabled alerts: {', '.join(enabled_names)}")
        else:
            parts.append("No alert types enabled")

        # Quiet hours
        if quiet.get("start") is not None and quiet.get("end") is not None:
            parts.append(f"Quiet hours: {quiet['start']}:00 to {quiet['end']}:00")

        message = ". ".join(parts) + "."

        return VoiceToolResult(
            success=True,
            message=message,
            data={"preferences": prefs}
        )
