"""
Morning Briefing Assistant - Daily context aggregation and personalized wake-up briefing.
Auto-triggers on wake-up context to deliver weather, calendar, commute, and messages.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BriefingItemType(Enum):
    """Types of briefing items with priority weights."""
    URGENT = "urgent"           # Critical items (meetings starting soon, emergencies)
    CALENDAR = "calendar"       # Today's schedule
    WEATHER = "weather"         # Current conditions and forecast
    COMMUTE = "commute"         # Traffic and transit info
    MESSAGE = "message"         # Overnight messages/notifications
    REMINDER = "reminder"       # User-set reminders
    NEWS = "news"               # Optional news headlines
    HABIT = "habit"             # Daily habit tracking
    CHALLENGE = "challenge"     # Daily challenge summary


@dataclass
class BriefingItem:
    """Single item in the morning briefing."""
    type: BriefingItemType
    title: str
    content: str
    priority: int = 50          # 0-100, higher = more important
    action: Optional[str] = None  # Optional action to take
    metadata: Dict[str, Any] = field(default_factory=dict)
    read: bool = False


@dataclass
class MorningBriefing:
    """Complete morning briefing package."""
    greeting: str
    items: List[BriefingItem]
    generated_at: datetime = field(default_factory=datetime.now)
    weather_summary: Optional[str] = None
    day_rating: str = "normal"  # light, normal, busy, packed
    estimated_read_time_s: int = 30

    def get_priority_items(self, limit: int = 5) -> List[BriefingItem]:
        """Get top priority items."""
        sorted_items = sorted(self.items, key=lambda x: x.priority, reverse=True)
        return sorted_items[:limit]

    def get_by_type(self, item_type: BriefingItemType) -> List[BriefingItem]:
        """Get all items of a specific type."""
        return [item for item in self.items if item.type == item_type]

    def mark_read(self, item: BriefingItem):
        """Mark an item as read."""
        item.read = True


class MorningAssistant:
    """
    Morning Briefing Assistant for WHAM.

    Aggregates information from multiple sources and delivers a personalized
    morning briefing when the user wakes up.
    """

    # Time-based greeting templates
    GREETINGS = {
        "early": [  # 4am-6am
            "Early start, {name}. Here's your briefing.",
            "Up before the sun, {name}. Let's plan your day.",
            "Good early morning, {name}.",
        ],
        "morning": [  # 6am-9am
            "Good morning, {name}. Here's your day.",
            "Morning, {name}. Ready for today?",
            "Rise and shine, {name}.",
        ],
        "late_morning": [  # 9am-12pm
            "Good morning, {name}. Better late than never.",
            "Morning, {name}. Day's already in motion.",
            "{name}, catching up on the morning.",
        ],
    }

    # Day rating thresholds
    DAY_RATINGS = {
        "light": (0, 3),      # 0-3 events
        "normal": (4, 6),     # 4-6 events
        "busy": (7, 10),      # 7-10 events
        "packed": (11, 100),  # 11+ events
    }

    def __init__(self, config: dict, user_name: str = "Will"):
        """
        Initialize Morning Assistant.

        Args:
            config: Configuration dictionary
            user_name: User's name for personalized greetings
        """
        self.config = config
        self.user_name = user_name
        self._data_providers: Dict[str, Callable] = {}
        self._last_briefing: Optional[MorningBriefing] = None
        self._briefing_delivered_today = False
        self._last_delivery_date: Optional[datetime] = None

        # Load settings
        briefing_config = config.get("morning_briefing", {})
        self.auto_trigger = briefing_config.get("auto_trigger", True)
        self.trigger_time_start = briefing_config.get("trigger_start", "06:00")
        self.trigger_time_end = briefing_config.get("trigger_end", "10:00")
        self.include_weather = briefing_config.get("include_weather", True)
        self.include_commute = briefing_config.get("include_commute", True)
        self.include_news = briefing_config.get("include_news", False)
        self.max_items = briefing_config.get("max_items", 10)

        logger.info(f"MorningAssistant initialized for {user_name}")

    def register_provider(self, name: str, provider: Callable):
        """
        Register a data provider for briefing items.

        Args:
            name: Provider name (e.g., 'calendar', 'weather')
            provider: Async callable that returns List[BriefingItem]
        """
        self._data_providers[name] = provider
        logger.debug(f"Registered data provider: {name}")

    async def generate_briefing(self) -> MorningBriefing:
        """
        Generate a complete morning briefing.

        Returns:
            MorningBriefing with aggregated items
        """
        logger.info("Generating morning briefing...")
        items: List[BriefingItem] = []

        # Gather items from all providers
        for name, provider in self._data_providers.items():
            try:
                provider_items = await provider()
                if provider_items:
                    items.extend(provider_items)
                    logger.debug(f"Provider '{name}' returned {len(provider_items)} items")
            except Exception as e:
                logger.error(f"Provider '{name}' failed: {e}")

        # Add default items if no providers
        if not items:
            items = await self._get_default_items()

        # Generate greeting
        greeting = self._generate_greeting()

        # Calculate day rating
        calendar_items = [i for i in items if i.type == BriefingItemType.CALENDAR]
        day_rating = self._calculate_day_rating(len(calendar_items))

        # Generate weather summary
        weather_items = [i for i in items if i.type == BriefingItemType.WEATHER]
        weather_summary = weather_items[0].content if weather_items else None

        # Estimate read time (3 seconds per item, minimum 15s)
        read_time = max(15, len(items) * 3)

        # Sort by priority and limit
        items.sort(key=lambda x: x.priority, reverse=True)
        items = items[:self.max_items]

        briefing = MorningBriefing(
            greeting=greeting,
            items=items,
            weather_summary=weather_summary,
            day_rating=day_rating,
            estimated_read_time_s=read_time
        )

        self._last_briefing = briefing
        return briefing

    def _generate_greeting(self) -> str:
        """Generate time-appropriate greeting."""
        import random
        hour = datetime.now().hour

        if 4 <= hour < 6:
            category = "early"
        elif 6 <= hour < 9:
            category = "morning"
        else:
            category = "late_morning"

        templates = self.GREETINGS.get(category, self.GREETINGS["morning"])
        template = random.choice(templates)
        return template.format(name=self.user_name)

    def _calculate_day_rating(self, event_count: int) -> str:
        """Calculate how busy the day looks."""
        for rating, (min_events, max_events) in self.DAY_RATINGS.items():
            if min_events <= event_count <= max_events:
                return rating
        return "normal"

    async def _get_default_items(self) -> List[BriefingItem]:
        """Get default items when no providers are registered."""
        items = []
        now = datetime.now()

        # Time-based greeting item
        items.append(BriefingItem(
            type=BriefingItemType.WEATHER,
            title="Weather",
            content=f"Weather data unavailable. Check your phone.",
            priority=60
        ))

        # Default calendar placeholder
        items.append(BriefingItem(
            type=BriefingItemType.CALENDAR,
            title="Today's Schedule",
            content="No calendar connected. Say 'connect calendar' to set up.",
            priority=70,
            action="connect_calendar"
        ))

        return items

    def should_trigger(self) -> bool:
        """
        Check if briefing should auto-trigger.

        Returns:
            True if conditions are met for auto-trigger
        """
        if not self.auto_trigger:
            return False

        now = datetime.now()

        # Check if already delivered today
        if self._last_delivery_date and self._last_delivery_date.date() == now.date():
            return False

        # Parse trigger time window
        try:
            start_hour, start_min = map(int, self.trigger_time_start.split(":"))
            end_hour, end_min = map(int, self.trigger_time_end.split(":"))

            start_time = now.replace(hour=start_hour, minute=start_min, second=0)
            end_time = now.replace(hour=end_hour, minute=end_min, second=0)

            if start_time <= now <= end_time:
                return True
        except ValueError:
            logger.warning("Invalid trigger time format in config")

        return False

    async def deliver_briefing(self, display_callback: Callable) -> MorningBriefing:
        """
        Generate and deliver briefing via callback.

        Args:
            display_callback: Async function to display briefing items

        Returns:
            The delivered MorningBriefing
        """
        briefing = await self.generate_briefing()

        # Mark as delivered
        self._last_delivery_date = datetime.now()

        # Deliver via callback
        try:
            await display_callback(briefing)
        except Exception as e:
            logger.error(f"Briefing delivery failed: {e}")

        return briefing

    def get_last_briefing(self) -> Optional[MorningBriefing]:
        """Get the most recently generated briefing."""
        return self._last_briefing

    def format_for_display(self, briefing: MorningBriefing) -> List[str]:
        """
        Format briefing for HUD display.

        Args:
            briefing: MorningBriefing to format

        Returns:
            List of formatted strings for display
        """
        lines = []

        # Greeting
        lines.append(briefing.greeting)
        lines.append("")

        # Day rating indicator
        rating_icons = {
            "light": "☀️ Light day",
            "normal": "📋 Normal day",
            "busy": "📊 Busy day",
            "packed": "🔥 Packed day"
        }
        lines.append(rating_icons.get(briefing.day_rating, "📋 Today"))
        lines.append("")

        # Weather if available
        if briefing.weather_summary:
            lines.append(f"🌤️ {briefing.weather_summary}")
            lines.append("")

        # Priority items
        lines.append("─── Top Items ───")
        for item in briefing.get_priority_items(5):
            icon = self._get_type_icon(item.type)
            lines.append(f"{icon} {item.title}")
            if item.content and len(item.content) < 50:
                lines.append(f"   {item.content}")

        return lines

    def _get_type_icon(self, item_type: BriefingItemType) -> str:
        """Get icon for briefing item type."""
        icons = {
            BriefingItemType.URGENT: "🚨",
            BriefingItemType.CALENDAR: "📅",
            BriefingItemType.WEATHER: "🌤️",
            BriefingItemType.COMMUTE: "🚗",
            BriefingItemType.MESSAGE: "💬",
            BriefingItemType.REMINDER: "⏰",
            BriefingItemType.NEWS: "📰",
            BriefingItemType.HABIT: "✅",
            BriefingItemType.CHALLENGE: "🎯",
        }
        return icons.get(item_type, "•")

    def format_for_tts(self, briefing: MorningBriefing) -> str:
        """
        Format briefing for text-to-speech.

        Args:
            briefing: MorningBriefing to format

        Returns:
            Speakable string
        """
        parts = [briefing.greeting]

        # Day summary
        if briefing.day_rating == "light":
            parts.append("You have a light day ahead.")
        elif briefing.day_rating == "busy":
            parts.append("Heads up, busy day ahead.")
        elif briefing.day_rating == "packed":
            parts.append("Packed schedule today. Let's prioritize.")

        # Weather
        if briefing.weather_summary:
            parts.append(briefing.weather_summary)

        # Top 3 items
        top_items = briefing.get_priority_items(3)
        if top_items:
            parts.append("Your top priorities:")
            for i, item in enumerate(top_items, 1):
                parts.append(f"{i}. {item.title}")

        return " ".join(parts)


# Example data providers
async def mock_weather_provider() -> List[BriefingItem]:
    """Mock weather provider for testing."""
    return [BriefingItem(
        type=BriefingItemType.WEATHER,
        title="Weather",
        content="72°F, Sunny. High of 78°F.",
        priority=65,
        metadata={"temp": 72, "high": 78, "condition": "sunny"}
    )]


async def mock_calendar_provider() -> List[BriefingItem]:
    """Mock calendar provider for testing."""
    now = datetime.now()
    return [
        BriefingItem(
            type=BriefingItemType.CALENDAR,
            title="Team Standup",
            content="9:30 AM - Zoom",
            priority=80,
            metadata={"time": "09:30", "location": "Zoom"}
        ),
        BriefingItem(
            type=BriefingItemType.CALENDAR,
            title="Lunch with Sarah",
            content="12:00 PM - Cafe Milano",
            priority=60,
            metadata={"time": "12:00", "location": "Cafe Milano"}
        ),
    ]


# Test
def test_morning_assistant():
    """Test morning assistant functionality."""
    import asyncio

    print("=== Morning Assistant Test ===\n")

    config = {
        "morning_briefing": {
            "auto_trigger": True,
            "trigger_start": "06:00",
            "trigger_end": "10:00",
            "max_items": 10
        }
    }

    assistant = MorningAssistant(config, user_name="Will")

    # Register mock providers
    assistant.register_provider("weather", mock_weather_provider)
    assistant.register_provider("calendar", mock_calendar_provider)

    async def run_test():
        # Generate briefing
        briefing = await assistant.generate_briefing()

        print(f"Greeting: {briefing.greeting}")
        print(f"Day Rating: {briefing.day_rating}")
        print(f"Items: {len(briefing.items)}")
        print(f"Read Time: {briefing.estimated_read_time_s}s")
        print()

        # Display format
        print("--- HUD Display ---")
        for line in assistant.format_for_display(briefing):
            print(line)
        print()

        # TTS format
        print("--- TTS ---")
        print(assistant.format_for_tts(briefing))
        print()

        # Should trigger check
        print(f"Should auto-trigger: {assistant.should_trigger()}")

    asyncio.run(run_test())


if __name__ == "__main__":
    test_morning_assistant()
