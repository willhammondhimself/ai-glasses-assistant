"""Apple Maps tool using URL schemes and MapKit JS."""
import logging
import re
import urllib.parse
from typing import Optional, Tuple
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class MapsTool(VoiceTool):
    """Get directions and place info using Apple Maps."""

    name = "maps"
    description = "Get directions and find places using Apple Maps"

    keywords = [
        r"\bdirections?\s+to\b",
        r"\bhow\s+(do\s+I\s+)?get\s+to\b",
        r"\bnavigate\s+to\b",
        r"\btake\s+me\s+to\b",
        r"\bfind\s+(?:a\s+)?(?:nearby\s+)?(?:\w+\s+)?(?:near|around)\b",
        r"\bwhere\s+is\b",
        r"\bnearby\b",
        r"\brestaurants?\s+near\b",
        r"\bcoffee\s+(?:shop|near)\b",
        r"\bgas\s+station\b",
    ]

    priority = 10

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Get directions or search for places.

        Args:
            query: The user's voice query

        Returns:
            VoiceToolResult with map URL and instructions
        """
        query_lower = query.lower()

        # Determine if this is a directions request or place search
        is_directions = any(word in query_lower for word in [
            "directions", "get to", "navigate", "take me", "how to get"
        ])

        if is_directions:
            return await self._get_directions(query)
        else:
            return await self._search_places(query)

    async def _get_directions(self, query: str) -> VoiceToolResult:
        """Generate Apple Maps directions URL."""
        destination = self._extract_destination(query)

        if not destination:
            return VoiceToolResult(
                success=False,
                message="I couldn't understand where you want to go. Try saying 'directions to' followed by the place."
            )

        # Build Apple Maps URL
        # maps://?daddr=destination&dirflg=d (driving)
        encoded_dest = urllib.parse.quote(destination)
        maps_url = f"maps://?daddr={encoded_dest}&dirflg=d"

        # Also provide a web fallback
        web_url = f"https://maps.apple.com/?daddr={encoded_dest}&dirflg=d"

        return VoiceToolResult(
            success=True,
            message=f"I found directions to {destination}. Opening Apple Maps.",
            data={
                "destination": destination,
                "maps_url": maps_url,
                "web_url": web_url,
                "action": "directions"
            }
        )

    async def _search_places(self, query: str) -> VoiceToolResult:
        """Search for places using Apple Maps."""
        search_term = self._extract_search_term(query)

        if not search_term:
            return VoiceToolResult(
                success=False,
                message="I couldn't understand what place you're looking for."
            )

        # Build Apple Maps search URL
        encoded_search = urllib.parse.quote(search_term)
        maps_url = f"maps://?q={encoded_search}"
        web_url = f"https://maps.apple.com/?q={encoded_search}"

        return VoiceToolResult(
            success=True,
            message=f"Searching Apple Maps for {search_term}.",
            data={
                "search_term": search_term,
                "maps_url": maps_url,
                "web_url": web_url,
                "action": "search"
            }
        )

    def _extract_destination(self, query: str) -> Optional[str]:
        """Extract destination from directions query."""
        patterns = [
            r"directions?\s+to\s+(.+?)(?:\?|$|please|thanks)",
            r"(?:how\s+)?(?:do\s+I\s+)?get\s+to\s+(.+?)(?:\?|$|please|thanks)",
            r"navigate\s+to\s+(.+?)(?:\?|$|please|thanks)",
            r"take\s+me\s+to\s+(.+?)(?:\?|$|please|thanks)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                dest = match.group(1).strip()
                # Clean up common suffixes
                dest = re.sub(r"\s+(please|thanks|thank you)$", "", dest, flags=re.IGNORECASE)
                return dest

        return None

    def _extract_search_term(self, query: str) -> Optional[str]:
        """Extract place search term from query."""
        query_lower = query.lower()

        # Common place patterns
        place_patterns = [
            r"(?:find|search\s+for)\s+(?:a\s+)?(.+?)(?:\s+near|\s+around|\s+nearby|$)",
            r"where\s+is\s+(?:the\s+)?(.+?)(?:\?|$)",
            r"(?:nearby|near\s+me)\s+(.+?)(?:\?|$)",
            r"(.+?)\s+(?:nearby|near\s+me|around\s+here)",
        ]

        for pattern in place_patterns:
            match = re.search(pattern, query_lower)
            if match:
                term = match.group(1).strip()
                if term and term not in ["a", "the", "some"]:
                    return term

        # Check for specific place types
        place_types = [
            "restaurant", "coffee", "cafe", "gas station", "grocery",
            "pharmacy", "hospital", "hotel", "bar", "gym", "bank", "atm"
        ]

        for place_type in place_types:
            if place_type in query_lower:
                return place_type

        return None
