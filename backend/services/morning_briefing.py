"""Enhanced Morning Briefing service for WHAM backend.

Aggregates data from multiple sources: email, calendar, weather, packages, stocks.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import random

logger = logging.getLogger(__name__)


class BriefingItemType(Enum):
    """Types of briefing items with priority weights."""
    URGENT = "urgent"
    CALENDAR = "calendar"
    WEATHER = "weather"
    COMMUTE = "commute"
    EMAIL = "email"
    PACKAGE = "package"
    REMINDER = "reminder"
    NEWS = "news"
    MARKET = "market"
    HEALTH = "health"
    FOCUS = "focus"


@dataclass
class BriefingItem:
    """Single item in the morning briefing."""
    type: BriefingItemType
    title: str
    content: str
    priority: int = 50
    action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "title": self.title,
            "content": self.content,
            "priority": self.priority,
            "action": self.action,
            "metadata": self.metadata
        }


@dataclass
class MorningBriefing:
    """Complete morning briefing."""
    greeting: str
    items: List[BriefingItem]
    day_rating: str
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "greeting": self.greeting,
            "items": [item.to_dict() for item in self.items],
            "day_rating": self.day_rating,
            "generated_at": self.generated_at.isoformat()
        }

    def to_voice_summary(self, max_items: int = 5) -> str:
        """Generate voice-friendly summary."""
        parts = [self.greeting]

        # Day rating
        if self.day_rating == "light":
            parts.append("Light day ahead.")
        elif self.day_rating == "busy":
            parts.append("Heads up, busy day.")
        elif self.day_rating == "packed":
            parts.append("Packed schedule today.")

        # Top items by type
        priority_items = sorted(self.items, key=lambda x: x.priority, reverse=True)[:max_items]

        for item in priority_items:
            parts.append(f"{item.title}. {item.content}")

        return " ".join(parts)


class MorningBriefingService:
    """Service to generate enhanced morning briefings."""

    GREETINGS = {
        "early": [
            "Early start, {name}.",
            "Up before the sun, {name}.",
        ],
        "morning": [
            "Good morning, {name}.",
            "Morning, {name}. Here's your day.",
            "Rise and shine, {name}.",
        ],
        "late_morning": [
            "Good morning, {name}.",
            "{name}, here's what you've missed.",
        ],
        "afternoon": [
            "Good afternoon, {name}.",
            "{name}, here's your update.",
        ],
    }

    DAY_RATINGS = {
        "light": (0, 3),
        "normal": (4, 6),
        "busy": (7, 10),
        "packed": (11, 100),
    }

    def __init__(self, user_name: str = "Will"):
        self.user_name = user_name
        self._last_briefing: Optional[MorningBriefing] = None
        self._last_generation: Optional[datetime] = None

    async def generate_briefing(self) -> MorningBriefing:
        """Generate a complete morning briefing.

        Returns:
            MorningBriefing object
        """
        items: List[BriefingItem] = []

        # Gather from all sources
        items.extend(await self._get_weather_items())
        items.extend(await self._get_calendar_items())
        items.extend(await self._get_email_items())
        items.extend(await self._get_package_items())
        items.extend(await self._get_reminder_items())
        items.extend(await self._get_market_items())

        # Sort by priority
        items.sort(key=lambda x: x.priority, reverse=True)

        # Calculate day rating
        calendar_count = len([i for i in items if i.type == BriefingItemType.CALENDAR])
        day_rating = self._calculate_day_rating(calendar_count)

        briefing = MorningBriefing(
            greeting=self._generate_greeting(),
            items=items,
            day_rating=day_rating
        )

        self._last_briefing = briefing
        self._last_generation = datetime.now()

        return briefing

    def _generate_greeting(self) -> str:
        """Generate time-appropriate greeting."""
        hour = datetime.now().hour

        if 4 <= hour < 6:
            category = "early"
        elif 6 <= hour < 10:
            category = "morning"
        elif 10 <= hour < 12:
            category = "late_morning"
        else:
            category = "afternoon"

        templates = self.GREETINGS.get(category, self.GREETINGS["morning"])
        return random.choice(templates).format(name=self.user_name)

    def _calculate_day_rating(self, event_count: int) -> str:
        """Calculate day busyness rating."""
        for rating, (min_e, max_e) in self.DAY_RATINGS.items():
            if min_e <= event_count <= max_e:
                return rating
        return "normal"

    async def _get_weather_items(self) -> List[BriefingItem]:
        """Get weather briefing items."""
        try:
            from backend.voice.tools.weather import WeatherTool
            weather_tool = WeatherTool()
            result = await weather_tool.execute("what's the weather today")

            if result.success:
                return [BriefingItem(
                    type=BriefingItemType.WEATHER,
                    title="Weather",
                    content=result.message,
                    priority=65,
                    metadata=result.data or {}
                )]
        except Exception as e:
            logger.debug(f"Weather fetch failed: {e}")

        return []

    async def _get_calendar_items(self) -> List[BriefingItem]:
        """Get today's calendar items."""
        try:
            from backend.voice.tools.calendar_tool import CalendarVoiceTool
            cal_tool = CalendarVoiceTool()
            result = await cal_tool.execute("what's on my calendar today")

            if result.success and result.data:
                events = result.data.get("events", [])
                items = []

                for i, event in enumerate(events[:5]):
                    items.append(BriefingItem(
                        type=BriefingItemType.CALENDAR,
                        title=event.get("summary", "Event"),
                        content=event.get("start_time", ""),
                        priority=80 - i * 5,  # Earlier events higher priority
                        metadata=event
                    ))

                return items
        except Exception as e:
            logger.debug(f"Calendar fetch failed: {e}")

        return []

    async def _get_email_items(self) -> List[BriefingItem]:
        """Get urgent/unread email summary."""
        try:
            from backend.services.gmail_service import get_gmail_service
            gmail = get_gmail_service()

            if not gmail.is_authenticated():
                return []

            unread_count = await gmail.get_unread_count()

            if unread_count > 0:
                return [BriefingItem(
                    type=BriefingItemType.EMAIL,
                    title="Email",
                    content=f"{unread_count} unread email{'s' if unread_count > 1 else ''}",
                    priority=55 if unread_count < 5 else 70,
                    metadata={"unread_count": unread_count}
                )]
        except Exception as e:
            logger.debug(f"Email fetch failed: {e}")

        return []

    async def _get_package_items(self) -> List[BriefingItem]:
        """Get package delivery updates."""
        try:
            from backend.services.package_tracker import get_package_tracker
            tracker = get_package_tracker()

            if not tracker.is_configured():
                return []

            arriving = await tracker.get_packages_arriving_today()

            if arriving:
                names = [p.title or p.carrier for p in arriving[:3]]
                return [BriefingItem(
                    type=BriefingItemType.PACKAGE,
                    title="Deliveries",
                    content=f"{len(arriving)} package{'s' if len(arriving) > 1 else ''} arriving: {', '.join(names)}",
                    priority=60,
                    metadata={"count": len(arriving)}
                )]
        except Exception as e:
            logger.debug(f"Package fetch failed: {e}")

        return []

    async def _get_reminder_items(self) -> List[BriefingItem]:
        """Get today's reminders."""
        try:
            from backend.voice.tools.reminders import RemindersTool
            reminder_tool = RemindersTool()
            result = await reminder_tool.execute("what are my reminders today")

            if result.success and result.data:
                reminders = result.data.get("reminders", [])
                if reminders:
                    return [BriefingItem(
                        type=BriefingItemType.REMINDER,
                        title="Reminders",
                        content=f"{len(reminders)} reminder{'s' if len(reminders) > 1 else ''} for today",
                        priority=70,
                        metadata={"reminders": reminders}
                    )]
        except Exception as e:
            logger.debug(f"Reminder fetch failed: {e}")

        return []

    async def _get_market_items(self) -> List[BriefingItem]:
        """Get stock market summary."""
        try:
            from backend.voice.tools.stocks import StocksTool
            stocks_tool = StocksTool()

            # Just check if market is open and get a quick summary
            result = await stocks_tool.execute("how is the market doing")

            if result.success:
                return [BriefingItem(
                    type=BriefingItemType.MARKET,
                    title="Markets",
                    content=result.message[:100] if len(result.message) > 100 else result.message,
                    priority=40,
                    metadata=result.data or {}
                )]
        except Exception as e:
            logger.debug(f"Market fetch failed: {e}")

        return []

    def get_last_briefing(self) -> Optional[MorningBriefing]:
        """Get the most recently generated briefing."""
        return self._last_briefing

    def should_regenerate(self, cache_minutes: int = 30) -> bool:
        """Check if briefing should be regenerated.

        Args:
            cache_minutes: Minutes to cache briefing

        Returns:
            True if should regenerate
        """
        if not self._last_generation:
            return True

        from datetime import timedelta
        return datetime.now() - self._last_generation > timedelta(minutes=cache_minutes)


# Global instance
_briefing_service: Optional[MorningBriefingService] = None


def get_briefing_service(user_name: str = "Will") -> MorningBriefingService:
    """Get or create global briefing service."""
    global _briefing_service
    if _briefing_service is None:
        _briefing_service = MorningBriefingService(user_name)
    return _briefing_service
