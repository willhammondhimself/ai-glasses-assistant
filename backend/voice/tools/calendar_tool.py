"""Apple Calendar voice tool wrapper using CalDAV."""
import logging
import re
from typing import Optional, Tuple
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class CalendarVoiceTool(VoiceTool):
    """Voice tool for Apple Calendar via CalDAV.

    Wraps the existing CalendarTool for voice interaction.
    Requires APPLE_ID and APPLE_APP_PASSWORD environment variables.
    """

    name = "calendar"
    description = "Check and manage Apple Calendar events"

    keywords = [
        # Query schedule
        r"\b(what('s| is)?|any)\s+(on\s+)?(my\s+)?(schedule|calendar|events|meetings)\b",
        r"\bwhat\s+do\s+i\s+have\s+(today|tomorrow|this\s+week)\b",
        r"\b(my\s+)?(schedule|calendar)\s+(for\s+)?(today|tomorrow|this\s+week)\b",
        r"\bam\s+i\s+(free|busy)\b",
        r"\bfree\s+time\b",
        r"\bwhat('s| is)\s+coming\s+up\b",
        # Create events
        r"\bschedule\s+(a|an|my)\b",
        r"\badd\s+.+\s+to\s+(my\s+)?calendar\b",
        r"\badd\s+(to\s+)?(my\s+)?calendar\b",
        r"\bput\s+(on|in)\s+(my\s+)?calendar\b",
        r"\bcreate\s+(a|an)?\s*(meeting|event|appointment)\b",
        r"\bbook\s+(a|an)\b",
        # Clear/delete events
        r"\bclear\s+(my\s+)?(calendar|schedule|afternoon|morning|day)\b",
        r"\bcancel\s+(my\s+)?(meeting|event|appointment)\b",
        r"\bdelete\s+(the\s+)?(meeting|event|appointment)\b",
        r"\bremove\s+(from\s+)?(my\s+)?calendar\b",
    ]

    priority = 10  # Higher priority for calendar queries

    def __init__(self):
        # Lazy import to avoid circular deps and defer CalDAV init
        self._calendar = None

    def _get_calendar(self):
        """Lazy load the calendar tool."""
        if self._calendar is None:
            from backend.agent.tools.calendar import CalendarTool
            self._calendar = CalendarTool()
        return self._calendar

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute calendar operation based on voice query.

        Args:
            query: The user's voice query

        Returns:
            VoiceToolResult with voice-friendly response
        """
        try:
            action, params = self._parse_query(query)
            logger.info(f"Calendar action: {action}, params: {params}")

            calendar = self._get_calendar()
            result = await calendar.execute(action, **params)

            if not result.success:
                return VoiceToolResult(
                    success=False,
                    message=result.message or "Sorry, I couldn't access your calendar."
                )

            message = self._format_for_voice(result, action, params)
            return VoiceToolResult(
                success=True,
                message=message,
                data=result.data
            )

        except Exception as e:
            logger.error(f"Calendar voice tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble accessing your calendar."
            )

    def _parse_query(self, query: str) -> Tuple[str, dict]:
        """Parse voice query to determine action and parameters.

        Args:
            query: The user's voice query

        Returns:
            Tuple of (action, params dict)
        """
        query_lower = query.lower()

        # Check for list/query operations first (questions about schedule)
        # These patterns indicate the user wants to see their schedule
        list_patterns = [
            r"\bwhat('s| is| do i have)\b.*\b(schedule|calendar|event|meeting)",
            r"\b(my\s+)?(schedule|calendar)\b.*\b(today|tomorrow|this week|look like)",
            r"\bam\s+i\s+(free|busy)\b",
            r"\bfree\s+time\b",
            r"\bwhat('s| is)\s+coming\s+up\b",
            r"\bon\s+(my\s+)?(schedule|calendar)\b",
            r"\bany\s+(events|meetings|appointments)\b",
        ]
        for pattern in list_patterns:
            if re.search(pattern, query_lower):
                date_range = self._extract_date_range(query)
                return "list", {"date_range": date_range}

        # Check for clear/cancel operations
        if re.search(r"\b(clear|cancel|delete|remove)\b", query_lower):
            date_range = self._extract_date_range(query)
            if "afternoon" in query_lower:
                date_range = "today afternoon"
            elif "morning" in query_lower:
                date_range = "today morning"
            return "clear_range", {"date_range": date_range}

        # Check for create operations (verbs that indicate scheduling)
        create_patterns = [
            r"\bschedule\s+(a|an|my)\b",
            r"\badd\s+.+\s+to\s+(my\s+)?calendar\b",
            r"\bput\s+.+\s+(on|in)\s+(my\s+)?calendar\b",
            r"\bcreate\s+(a|an)?\s*(meeting|event|appointment)\b",
            r"\bbook\s+(a|an)\b",
            r"\bset\s+up\s+(a|an)\b",
        ]
        for pattern in create_patterns:
            if re.search(pattern, query_lower):
                title, start_time, end_time = self._extract_event_details(query)
                return "create", {
                    "title": title,
                    "start_time": start_time,
                    "end_time": end_time
                }

        # Default to list
        date_range = self._extract_date_range(query)
        return "list", {"date_range": date_range}

    def _extract_date_range(self, query: str) -> str:
        """Extract date range from query."""
        query_lower = query.lower()

        if "tomorrow" in query_lower:
            return "tomorrow"
        elif "this week" in query_lower or "week" in query_lower:
            return "this week"
        elif "afternoon" in query_lower:
            return "today afternoon"
        elif "morning" in query_lower:
            return "today morning"
        else:
            return "today"

    def _extract_event_details(self, query: str) -> Tuple[str, str, Optional[str]]:
        """Extract event title, start time, and optional end time from query.

        Args:
            query: The user's voice query

        Returns:
            Tuple of (title, start_time, end_time)
        """
        # Try to extract title after common patterns
        title_patterns = [
            r"schedule\s+(?:a|an|my)?\s*(.+?)\s+(?:at|for|on|tomorrow|today)",
            r"add\s+(.+?)\s+to\s+(?:my\s+)?calendar",
            r"create\s+(?:a|an)?\s*(?:meeting|event|appointment)\s+(?:called|named|for)?\s*(.+?)\s+(?:at|for|on)",
            r"book\s+(?:a|an)?\s*(.+?)\s+(?:at|for|on)",
            r"put\s+(.+?)\s+(?:on|in)\s+(?:my\s+)?calendar",
        ]

        title = "Meeting"  # Default
        for pattern in title_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Clean up common words
                title = re.sub(r"^(a|an|my)\s+", "", title, flags=re.IGNORECASE)
                break

        # Extract time
        time_patterns = [
            r"at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
            r"for\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
            r"(\d{1,2}(?::\d{2})?\s*(?:am|pm))",
        ]

        start_time = None
        for pattern in time_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                time_str = match.group(1).strip()
                # Add context if just a number
                if not re.search(r"(am|pm)", time_str, re.IGNORECASE):
                    hour = int(re.search(r"\d+", time_str).group())
                    if hour < 12 and hour >= 7:
                        time_str += " am"
                    else:
                        time_str += " pm"
                start_time = time_str
                break

        # Handle relative times
        if "tomorrow" in query.lower():
            start_time = f"tomorrow {start_time}" if start_time else "tomorrow 9am"
        elif start_time is None:
            start_time = "today 2pm"  # Default

        return title, start_time, None

    def _format_for_voice(self, result, action: str, params: dict) -> str:
        """Format calendar result for voice output.

        Args:
            result: The ToolResult from CalendarTool
            action: The action performed
            params: The parameters used

        Returns:
            Voice-friendly message
        """
        if action == "list":
            events = result.data or []
            if not events:
                date_range = params.get("date_range", "today")
                return f"You have no events {date_range}. Your schedule is clear."

            # Format events for voice
            date_range = params.get("date_range", "today")
            count = len(events)

            if count == 1:
                event = events[0]
                return f"You have one event {date_range}: {event.get('summary', 'Event')} at {event.get('start', 'unknown time')}."

            # Multiple events
            event_list = []
            for event in events[:5]:  # Limit to 5 for voice
                event_list.append(f"{event.get('summary', 'Event')} at {event.get('start', 'unknown')}")

            if count <= 3:
                return f"You have {count} events {date_range}: " + ", ".join(event_list) + "."
            else:
                return f"You have {count} events {date_range}. First few are: " + ", ".join(event_list[:3]) + "."

        elif action == "create":
            event = result.data
            if event:
                return f"Done. I've added {event.get('summary', 'your event')} at {event.get('start', 'the scheduled time')}."
            return "Event created."

        elif action == "clear_range":
            deleted = result.data.get("deleted", 0) if result.data else 0
            date_range = params.get("date_range", "today")
            if deleted == 0:
                return f"No events to clear for {date_range}."
            elif deleted == 1:
                return f"Cleared 1 event from your {date_range} schedule."
            return f"Cleared {deleted} events from your {date_range} schedule."

        return result.message or "Calendar updated."
