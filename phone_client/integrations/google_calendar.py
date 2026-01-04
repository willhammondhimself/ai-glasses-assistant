"""
Google Calendar Integration - Fetch calendar events for morning briefing.

Setup:
1. Go to Google Cloud Console
2. Create a project and enable Google Calendar API
3. Create OAuth 2.0 credentials (Desktop app)
4. Download credentials.json and place in this directory
5. Run authenticate() to complete OAuth flow

Dependencies:
    pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Google API imports (optional)
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    HAS_GOOGLE_API = True
except ImportError:
    HAS_GOOGLE_API = False
    logger.warning("Google API not installed. Run: pip install google-auth-oauthlib google-api-python-client")


# OAuth scopes
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']


@dataclass
class CalendarEvent:
    """Represents a calendar event."""
    id: str
    summary: str
    start: datetime
    end: datetime
    location: Optional[str] = None
    description: Optional[str] = None
    is_all_day: bool = False
    attendees: List[str] = None
    meeting_link: Optional[str] = None

    def __post_init__(self):
        if self.attendees is None:
            self.attendees = []

    @property
    def duration_minutes(self) -> int:
        """Get event duration in minutes."""
        delta = self.end - self.start
        return int(delta.total_seconds() / 60)

    @property
    def is_happening_soon(self) -> bool:
        """Check if event starts within 30 minutes."""
        now = datetime.now()
        return now <= self.start <= now + timedelta(minutes=30)

    def format_time(self) -> str:
        """Format start time for display."""
        if self.is_all_day:
            return "All day"
        return self.start.strftime("%I:%M %p")


class GoogleCalendarIntegration:
    """
    Google Calendar integration for WHAM.

    Provides methods to:
    - Authenticate with Google OAuth
    - Fetch today's events
    - Get upcoming events
    - Find next meeting
    """

    def __init__(self, credentials_dir: str = None):
        """
        Initialize Google Calendar integration.

        Args:
            credentials_dir: Directory containing credentials.json and token.json
        """
        if credentials_dir:
            self.credentials_dir = Path(credentials_dir)
        else:
            self.credentials_dir = Path(__file__).parent

        self.credentials_file = self.credentials_dir / "credentials.json"
        self.token_file = self.credentials_dir / "token.json"
        self._service = None
        self._credentials = None

    def is_available(self) -> bool:
        """Check if Google API is available."""
        return HAS_GOOGLE_API

    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        if not HAS_GOOGLE_API:
            return False

        if self.token_file.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_file), SCOPES)
                return creds and creds.valid
            except Exception:
                return False
        return False

    def authenticate(self) -> bool:
        """
        Run OAuth flow to authenticate with Google.

        Returns:
            True if authentication successful
        """
        if not HAS_GOOGLE_API:
            logger.error("Google API not installed")
            return False

        if not self.credentials_file.exists():
            logger.error(f"credentials.json not found at {self.credentials_file}")
            logger.error("Download from Google Cloud Console and place in integrations/")
            return False

        try:
            creds = None

            # Load existing token
            if self.token_file.exists():
                creds = Credentials.from_authorized_user_file(str(self.token_file), SCOPES)

            # Refresh or get new token
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_file), SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # Save token
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())

            self._credentials = creds
            logger.info("Google Calendar authentication successful")
            return True

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def _get_service(self):
        """Get or create Calendar API service."""
        if not HAS_GOOGLE_API:
            return None

        if self._service:
            return self._service

        if not self._credentials:
            if not self.authenticate():
                return None

        self._service = build('calendar', 'v3', credentials=self._credentials)
        return self._service

    def get_todays_events(self) -> List[CalendarEvent]:
        """
        Get all events for today.

        Returns:
            List of CalendarEvent objects
        """
        if not HAS_GOOGLE_API:
            return self._get_mock_events()

        service = self._get_service()
        if not service:
            return []

        try:
            now = datetime.now()
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)

            events_result = service.events().list(
                calendarId='primary',
                timeMin=start_of_day.isoformat() + 'Z',
                timeMax=end_of_day.isoformat() + 'Z',
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])
            return [self._parse_event(e) for e in events]

        except Exception as e:
            logger.error(f"Failed to fetch events: {e}")
            return []

    def get_next_event(self) -> Optional[CalendarEvent]:
        """
        Get the next upcoming event.

        Returns:
            Next CalendarEvent or None
        """
        if not HAS_GOOGLE_API:
            events = self._get_mock_events()
            return events[0] if events else None

        service = self._get_service()
        if not service:
            return None

        try:
            now = datetime.now()

            events_result = service.events().list(
                calendarId='primary',
                timeMin=now.isoformat() + 'Z',
                maxResults=1,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])
            if events:
                return self._parse_event(events[0])
            return None

        except Exception as e:
            logger.error(f"Failed to fetch next event: {e}")
            return None

    def get_upcoming_events(self, hours: int = 24) -> List[CalendarEvent]:
        """
        Get events for the next N hours.

        Args:
            hours: Number of hours to look ahead

        Returns:
            List of CalendarEvent objects
        """
        if not HAS_GOOGLE_API:
            return self._get_mock_events()

        service = self._get_service()
        if not service:
            return []

        try:
            now = datetime.now()
            end = now + timedelta(hours=hours)

            events_result = service.events().list(
                calendarId='primary',
                timeMin=now.isoformat() + 'Z',
                timeMax=end.isoformat() + 'Z',
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])
            return [self._parse_event(e) for e in events]

        except Exception as e:
            logger.error(f"Failed to fetch upcoming events: {e}")
            return []

    def _parse_event(self, event: dict) -> CalendarEvent:
        """Parse Google Calendar event to CalendarEvent."""
        # Handle all-day events
        start_data = event.get('start', {})
        end_data = event.get('end', {})

        if 'date' in start_data:  # All-day event
            start = datetime.fromisoformat(start_data['date'])
            end = datetime.fromisoformat(end_data['date'])
            is_all_day = True
        else:
            start = datetime.fromisoformat(start_data['dateTime'].replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_data['dateTime'].replace('Z', '+00:00'))
            is_all_day = False

        # Extract meeting link
        meeting_link = None
        if 'hangoutLink' in event:
            meeting_link = event['hangoutLink']
        elif 'conferenceData' in event:
            entry_points = event['conferenceData'].get('entryPoints', [])
            for ep in entry_points:
                if ep.get('entryPointType') == 'video':
                    meeting_link = ep.get('uri')
                    break

        # Extract attendees
        attendees = []
        for attendee in event.get('attendees', []):
            if not attendee.get('self'):
                attendees.append(attendee.get('email', ''))

        return CalendarEvent(
            id=event['id'],
            summary=event.get('summary', 'Untitled'),
            start=start,
            end=end,
            location=event.get('location'),
            description=event.get('description'),
            is_all_day=is_all_day,
            attendees=attendees,
            meeting_link=meeting_link
        )

    def _get_mock_events(self) -> List[CalendarEvent]:
        """Return mock events for testing without Google API."""
        now = datetime.now()
        return [
            CalendarEvent(
                id="mock-1",
                summary="Team Standup",
                start=now.replace(hour=9, minute=30),
                end=now.replace(hour=10, minute=0),
                location="Zoom",
                meeting_link="https://zoom.us/j/123456789"
            ),
            CalendarEvent(
                id="mock-2",
                summary="Lunch with Sarah",
                start=now.replace(hour=12, minute=0),
                end=now.replace(hour=13, minute=0),
                location="Cafe Milano"
            ),
            CalendarEvent(
                id="mock-3",
                summary="Project Review",
                start=now.replace(hour=15, minute=0),
                end=now.replace(hour=16, minute=0),
                attendees=["alice@company.com", "bob@company.com"]
            ),
        ]


# Test
def test_calendar():
    """Test calendar integration."""
    print("=== Google Calendar Integration Test ===\n")

    cal = GoogleCalendarIntegration()

    print(f"API Available: {cal.is_available()}")
    print(f"Authenticated: {cal.is_authenticated()}")
    print()

    print("Today's events (mock):")
    events = cal.get_todays_events()
    for event in events:
        print(f"  {event.format_time()} - {event.summary}")
        if event.location:
            print(f"    Location: {event.location}")
    print()

    print("Next event:")
    next_event = cal.get_next_event()
    if next_event:
        print(f"  {next_event.summary} at {next_event.format_time()}")


if __name__ == "__main__":
    test_calendar()
