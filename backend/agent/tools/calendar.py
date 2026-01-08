"""Apple Calendar tool via CalDAV for agent."""
import os
from datetime import datetime, timedelta
from typing import Optional, List
from zoneinfo import ZoneInfo
from .base import BaseTool, ToolResult

CALDAV_URL = "https://caldav.icloud.com"
LOCAL_TZ = ZoneInfo("America/Los_Angeles")


class CalendarTool(BaseTool):
    """Manage Apple Calendar events via CalDAV."""

    name = "calendar"
    description = "Manage Apple Calendar events. Can list, create, update, and delete events."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "create", "delete", "clear_range"],
                "description": "Action to perform"
            },
            "title": {"type": "string", "description": "Event title (for create)"},
            "start_time": {"type": "string", "description": "ISO datetime or relative like '2pm tomorrow'"},
            "end_time": {"type": "string", "description": "ISO datetime or relative"},
            "date_range": {"type": "string", "description": "For list/clear: 'today', 'tomorrow', 'this week'"},
            "event_id": {"type": "string", "description": "Event UID for delete"}
        },
        "required": ["action"]
    }

    def __init__(self):
        self.client = None
        self.calendar = None
        self._init_caldav()

    def _init_caldav(self):
        """Initialize CalDAV connection to iCloud."""
        try:
            import caldav

            apple_id = os.environ.get("APPLE_ID")
            app_password = os.environ.get("APPLE_APP_PASSWORD")

            if not apple_id or not app_password:
                print("Calendar: APPLE_ID or APPLE_APP_PASSWORD not set - using mock mode")
                return

            self.client = caldav.DAVClient(
                url=CALDAV_URL,
                username=apple_id,
                password=app_password
            )

            # Get principal and default calendar
            principal = self.client.principal()
            calendars = principal.calendars()

            if calendars:
                # Use first calendar (usually default)
                self.calendar = calendars[0]
                print(f"Calendar: Connected to '{self.calendar.name}'")
            else:
                print("Calendar: No calendars found")

        except Exception as e:
            print(f"Calendar init error: {e}")
            self.client = None
            self.calendar = None

    def is_configured(self) -> bool:
        """Check if calendar is properly configured."""
        return self.calendar is not None

    async def execute(self, action: str, **kwargs) -> ToolResult:
        """Execute calendar action."""
        # If no real connection, use mock mode for testing
        if not self.calendar:
            return await self._mock_execute(action, **kwargs)

        try:
            if action == "list":
                return await self._list_events(kwargs.get("date_range", "today"))
            elif action == "create":
                return await self._create_event(
                    kwargs.get("title", "New Event"),
                    kwargs.get("start_time"),
                    kwargs.get("end_time")
                )
            elif action == "clear_range":
                return await self._clear_range(kwargs.get("date_range", "today afternoon"))
            elif action == "delete":
                return await self._delete_event(kwargs.get("event_id"))

            return ToolResult(False, None, f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(False, None, f"Calendar error: {str(e)}")

    async def _mock_execute(self, action: str, **kwargs) -> ToolResult:
        """Mock calendar operations for testing without credentials."""
        if action == "list":
            date_range = kwargs.get("date_range", "today")
            return ToolResult(
                True,
                [
                    {"uid": "mock1", "summary": "Team standup", "start": "9:00 AM"},
                    {"uid": "mock2", "summary": "Lunch with Alex", "start": "12:30 PM"},
                    {"uid": "mock3", "summary": "Code review", "start": "3:00 PM"}
                ],
                f"Found 3 events for {date_range} (mock mode - set APPLE_ID & APPLE_APP_PASSWORD for real calendar)"
            )
        elif action == "create":
            title = kwargs.get("title", "New Event")
            start_time = kwargs.get("start_time", "tomorrow 2pm")
            return ToolResult(
                True,
                {"uid": "mock_new", "summary": title, "start": start_time},
                f"Created '{title}' at {start_time} (mock mode)"
            )
        elif action == "clear_range":
            date_range = kwargs.get("date_range", "this afternoon")
            return ToolResult(
                True,
                {"deleted": 2},
                f"Cleared 2 events from {date_range} (mock mode)"
            )
        elif action == "delete":
            event_id = kwargs.get("event_id", "unknown")
            return ToolResult(
                True,
                {"deleted": event_id},
                f"Deleted event {event_id} (mock mode)"
            )

        return ToolResult(False, None, f"Unknown action: {action}")

    async def _list_events(self, date_range: str) -> ToolResult:
        """List calendar events for date range."""
        start, end = self._parse_date_range(date_range)

        events = self.calendar.search(
            start=start,
            end=end,
            event=True,
            expand=True
        )

        event_list = []
        for event in events:
            try:
                vevent = event.vobject_instance.vevent
                event_data = {
                    "uid": str(vevent.uid.value) if hasattr(vevent, 'uid') else str(event.url),
                    "summary": str(vevent.summary.value) if hasattr(vevent, 'summary') else "No title",
                    "start": vevent.dtstart.value.strftime("%I:%M %p") if hasattr(vevent, 'dtstart') else "Unknown",
                    "start_iso": vevent.dtstart.value.isoformat() if hasattr(vevent, 'dtstart') else None,
                }
                if hasattr(vevent, 'dtend'):
                    event_data["end"] = vevent.dtend.value.strftime("%I:%M %p")
                if hasattr(vevent, 'location') and vevent.location.value:
                    event_data["location"] = str(vevent.location.value)
                event_list.append(event_data)
            except Exception as e:
                print(f"Error parsing event: {e}")
                continue

        # Sort by start time
        event_list.sort(key=lambda x: x.get("start_iso", "") or "")

        return ToolResult(True, event_list, f"Found {len(event_list)} events for {date_range}")

    async def _create_event(self, title: str, start: str, end: Optional[str]) -> ToolResult:
        """Create a new calendar event."""
        start_dt, end_dt = self._parse_times(start, end)

        # Create iCalendar event
        vcal = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//WHAM Assistant//EN
BEGIN:VEVENT
DTSTART:{start_dt.strftime('%Y%m%dT%H%M%S')}
DTEND:{end_dt.strftime('%Y%m%dT%H%M%S')}
SUMMARY:{title}
END:VEVENT
END:VCALENDAR"""

        event = self.calendar.save_event(vcal)

        return ToolResult(
            True,
            {
                "uid": str(event.url),
                "summary": title,
                "start": start_dt.strftime("%I:%M %p"),
                "end": end_dt.strftime("%I:%M %p")
            },
            f"Created: {title} at {start_dt.strftime('%I:%M %p')}"
        )

    async def _clear_range(self, date_range: str) -> ToolResult:
        """Delete all events in a date range."""
        result = await self._list_events(date_range)
        if not result.success:
            return result

        deleted = 0
        start, end = self._parse_date_range(date_range)

        events = self.calendar.search(
            start=start,
            end=end,
            event=True,
            expand=True
        )

        for event in events:
            try:
                event.delete()
                deleted += 1
            except Exception as e:
                print(f"Error deleting event: {e}")

        return ToolResult(True, {"deleted": deleted}, f"Cleared {deleted} events from {date_range}")

    async def _delete_event(self, event_uid: str) -> ToolResult:
        """Delete a specific event by UID."""
        if not event_uid:
            return ToolResult(False, None, "Event UID required")

        # Search for the event across a wide range
        now = datetime.now(LOCAL_TZ)
        start = now - timedelta(days=30)
        end = now + timedelta(days=365)

        events = self.calendar.search(
            start=start,
            end=end,
            event=True
        )

        for event in events:
            try:
                vevent = event.vobject_instance.vevent
                uid = str(vevent.uid.value) if hasattr(vevent, 'uid') else str(event.url)
                if uid == event_uid or str(event.url) == event_uid:
                    event.delete()
                    return ToolResult(True, {"deleted": event_uid}, f"Deleted event {event_uid}")
            except Exception as e:
                continue

        return ToolResult(False, None, f"Event {event_uid} not found")

    def _parse_date_range(self, range_str: str):
        """Parse date range string to datetime bounds."""
        now = datetime.now(LOCAL_TZ)

        if "afternoon" in range_str.lower():
            start = now.replace(hour=12, minute=0, second=0, microsecond=0)
            end = now.replace(hour=18, minute=0, second=0, microsecond=0)
        elif "morning" in range_str.lower():
            start = now.replace(hour=6, minute=0, second=0, microsecond=0)
            end = now.replace(hour=12, minute=0, second=0, microsecond=0)
        elif "tomorrow" in range_str.lower():
            tomorrow = now + timedelta(days=1)
            start = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
            end = tomorrow.replace(hour=23, minute=59, second=59, microsecond=0)
        elif "week" in range_str.lower():
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now + timedelta(days=7)
        else:  # today
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now.replace(hour=23, minute=59, second=59, microsecond=0)

        return start, end

    def _parse_times(self, start: str, end: Optional[str]):
        """Parse time strings to datetime objects."""
        from dateutil import parser as date_parser

        # Parse with local timezone awareness
        start_dt = date_parser.parse(start)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=LOCAL_TZ)

        if end:
            end_dt = date_parser.parse(end)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=LOCAL_TZ)
        else:
            # Default to 1 hour duration
            end_dt = start_dt + timedelta(hours=1)

        return start_dt, end_dt
