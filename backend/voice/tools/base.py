"""Base class for voice-activated tools."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional
import re


@dataclass
class VoiceToolResult:
    """Result from voice tool execution."""
    success: bool
    message: str  # Voice-friendly response (concise for TTS)
    data: Any = None  # Structured data for further processing
    speak: bool = True  # Whether to speak the response


class VoiceTool(ABC):
    """Base class for voice-activated tools."""

    # Tool identity
    name: str
    description: str

    # Keyword patterns for activation (regex patterns)
    keywords: List[str] = []

    # Priority (higher = checked first)
    priority: int = 0

    @abstractmethod
    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute the tool with the voice query.

        Args:
            query: The user's voice query (transcribed text)
            **kwargs: Additional context (e.g., user_id, session_id)

        Returns:
            VoiceToolResult with voice-friendly response
        """
        pass

    def matches(self, query: str) -> bool:
        """Check if this tool should handle the query.

        Args:
            query: The user's voice query

        Returns:
            True if any keyword pattern matches
        """
        query_lower = query.lower()
        for pattern in self.keywords:
            if re.search(pattern, query_lower):
                return True
        return False

    def calculate_confidence(self, query: str) -> float:
        """Calculate match confidence (0.0-1.0) for a query.

        Confidence is based on:
        - Number of keyword patterns that match
        - Priority boost for high-priority tools

        Args:
            query: The user's voice query

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not self.keywords:
            return 0.0

        query_lower = query.lower()
        matches = sum(1 for pattern in self.keywords if re.search(pattern, query_lower))

        if matches == 0:
            return 0.0

        # Base confidence from keyword match ratio (capped at 0.8)
        base_confidence = min(matches / len(self.keywords) * 1.5, 0.8)

        # Priority boost (up to 0.2 for priority 10)
        priority_boost = min(self.priority / 50.0, 0.2)

        return min(base_confidence + priority_boost, 1.0)

    def extract_entity(self, query: str, pattern: str) -> Optional[str]:
        """Extract an entity from the query using a pattern.

        Args:
            query: The user's voice query
            pattern: Regex pattern with a capture group

        Returns:
            Captured group or None
        """
        match = re.search(pattern, query, re.IGNORECASE)
        if match and match.groups():
            return match.group(1).strip()
        return None
