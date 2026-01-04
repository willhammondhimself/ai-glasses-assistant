"""
WHAM Integrations - External service connectors.

Available integrations:
- GoogleCalendarIntegration: Google Calendar events and scheduling
- OpenWeatherIntegration: Weather data for morning briefings
- NotionIntegration: Export captures and memories to Notion

Environment variables:
- GOOGLE_CALENDAR_CREDENTIALS: Path to Google OAuth credentials.json
- OPENWEATHER_API_KEY: OpenWeather API key
- NOTION_TOKEN: Notion integration token
- NOTION_CAPTURES_DB: Notion database ID for captures
- NOTION_MEMORIES_DB: Notion database ID for memories
- NOTION_SESSIONS_DB: Notion database ID for session summaries
"""
from .google_calendar import (
    GoogleCalendarIntegration,
    CalendarEvent,
)
from .openweather import (
    OpenWeatherIntegration,
    Weather,
    Forecast,
    WeatherCondition,
    get_weather,
)
from .notion import (
    NotionIntegration,
    NotionPage,
    NotionDatabase,
    NotionBlockType,
    WHAMNotionExporter,
    create_page,
)

__all__ = [
    # Google Calendar
    "GoogleCalendarIntegration",
    "CalendarEvent",
    # OpenWeather
    "OpenWeatherIntegration",
    "Weather",
    "Forecast",
    "WeatherCondition",
    "get_weather",
    # Notion
    "NotionIntegration",
    "NotionPage",
    "NotionDatabase",
    "NotionBlockType",
    "WHAMNotionExporter",
    "create_page",
]
