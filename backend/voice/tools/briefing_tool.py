"""Morning briefing voice tool."""
import logging
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class BriefingVoiceTool(VoiceTool):
    """Voice-controlled morning briefing."""

    name = "briefing"
    description = "Get morning briefing with weather, calendar, emails, and more"

    keywords = [
        r"\bmorning\s+briefing\b",
        r"\bbriefing\b",
        r"\bwhat('s| is)\s+my\s+day\s+look\s+like\b",
        r"\bgive\s+me\s+the\s+rundown\b",
        r"\brun\s+down\b",
        r"\btoday('s)?\s+overview\b",
        r"\bwhat('s| is)\s+(?:happening\s+)?today\b",
        r"\bstart\s+my\s+day\b",
        r"\bdaily\s+summary\b",
    ]

    priority = 7

    def __init__(self):
        self._service = None

    def _get_service(self):
        """Get briefing service (lazy load)."""
        if self._service is None:
            from backend.services.morning_briefing import get_briefing_service
            self._service = get_briefing_service()
        return self._service

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute briefing request.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult with briefing
        """
        query_lower = query.lower()
        service = self._get_service()

        try:
            # Check for specific section requests
            if "weather" in query_lower:
                return await self._handle_weather_only()

            if "calendar" in query_lower or "schedule" in query_lower:
                return await self._handle_calendar_only()

            if "email" in query_lower:
                return await self._handle_email_only()

            # Full briefing
            return await self._handle_full_briefing()

        except Exception as e:
            logger.error(f"Briefing tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble generating your briefing.",
                data={"error": str(e)}
            )

    async def _handle_full_briefing(self) -> VoiceToolResult:
        """Generate and deliver full briefing."""
        service = self._get_service()

        # Use cached if recent
        briefing = service.get_last_briefing()
        if briefing and not service.should_regenerate(cache_minutes=15):
            pass  # Use cached
        else:
            briefing = await service.generate_briefing()

        if not briefing.items:
            return VoiceToolResult(
                success=True,
                message=f"{briefing.greeting} Looks like a clear day with nothing urgent.",
                data={"briefing": briefing.to_dict()}
            )

        # Generate voice summary
        voice_text = briefing.to_voice_summary(max_items=5)

        return VoiceToolResult(
            success=True,
            message=voice_text,
            data={"briefing": briefing.to_dict()}
        )

    async def _handle_weather_only(self) -> VoiceToolResult:
        """Get just weather from briefing."""
        try:
            from backend.voice.tools.weather import WeatherTool
            weather = WeatherTool()
            return await weather.execute("what's the weather today")
        except Exception as e:
            return VoiceToolResult(
                success=False,
                message="Couldn't get weather information.",
                data={"error": str(e)}
            )

    async def _handle_calendar_only(self) -> VoiceToolResult:
        """Get just calendar from briefing."""
        try:
            from backend.voice.tools.calendar_tool import CalendarVoiceTool
            cal = CalendarVoiceTool()
            return await cal.execute("what's on my calendar today")
        except Exception as e:
            return VoiceToolResult(
                success=False,
                message="Couldn't get calendar information.",
                data={"error": str(e)}
            )

    async def _handle_email_only(self) -> VoiceToolResult:
        """Get just email summary from briefing."""
        try:
            from backend.voice.tools.email_tool import EmailVoiceTool
            email = EmailVoiceTool()
            return await email.execute("check my email")
        except Exception as e:
            return VoiceToolResult(
                success=False,
                message="Couldn't get email information.",
                data={"error": str(e)}
            )
