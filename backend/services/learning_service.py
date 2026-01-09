"""Learning Service - Async wrapper around FlashcardEngine for voice integration."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Flashcard:
    """A flashcard for spaced repetition learning."""
    id: int
    front: str
    back: str
    source_type: str
    tags: List[str] = field(default_factory=list)
    easiness_factor: float = 2.5
    interval_days: int = 1
    repetitions: int = 0
    due_date: int = 0
    created_at: int = 0

    def to_voice_question(self) -> str:
        """Format front for voice output."""
        # Clean up any markdown or special formatting
        question = self.front.strip()
        # Remove LaTeX markers if present
        question = question.replace("$", "")
        return question

    def to_voice_answer(self) -> str:
        """Format back for voice output."""
        answer = self.back.strip()
        answer = answer.replace("$", "")
        return answer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "front": self.front,
            "back": self.back,
            "source_type": self.source_type,
            "tags": self.tags,
            "easiness_factor": self.easiness_factor,
            "interval_days": self.interval_days,
            "repetitions": self.repetitions,
            "due_date": self.due_date
        }


@dataclass
class LearningStats:
    """Learning statistics."""
    total_cards: int = 0
    due_today: int = 0
    mastered: int = 0  # Cards with interval >= 21 days
    reviews_today: int = 0
    retention_rate: float = 0.0
    streak_days: int = 0

    def to_voice_summary(self) -> str:
        """Generate voice-friendly stats summary."""
        parts = []

        if self.due_today > 0:
            parts.append(f"You have {self.due_today} cards due for review")
        else:
            parts.append("No cards due right now")

        if self.reviews_today > 0:
            parts.append(f"You've reviewed {self.reviews_today} cards today")

        if self.retention_rate > 0:
            parts.append(f"Your retention rate is {self.retention_rate:.0f}%")

        if self.mastered > 0:
            parts.append(f"{self.mastered} cards mastered")

        return ". ".join(parts) + "."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cards": self.total_cards,
            "due_today": self.due_today,
            "mastered": self.mastered,
            "reviews_today": self.reviews_today,
            "retention_rate": self.retention_rate,
            "streak_days": self.streak_days
        }


class LearningService:
    """Async wrapper around FlashcardEngine for voice-first learning.

    Provides:
    - Async interface to existing FlashcardEngine
    - Voice-friendly formatting
    - Session state management for quiz mode
    - Statistics and progress tracking
    """

    def __init__(self):
        self._engine = None
        self._current_card: Optional[Flashcard] = None
        self._session_reviewed: int = 0
        self._session_correct: int = 0

    def _get_engine(self):
        """Lazy load FlashcardEngine."""
        if self._engine is None:
            from backend.flashcards.engine import FlashcardEngine
            self._engine = FlashcardEngine()
        return self._engine

    def _row_to_flashcard(self, row: Dict) -> Flashcard:
        """Convert database row to Flashcard dataclass."""
        tags = row.get("tags", [])
        if isinstance(tags, str):
            import json
            try:
                tags = json.loads(tags) if tags else []
            except (json.JSONDecodeError, TypeError):
                tags = []

        return Flashcard(
            id=row.get("id", 0),
            front=row.get("front", ""),
            back=row.get("back", ""),
            source_type=row.get("source_type", "manual"),
            tags=tags,
            easiness_factor=row.get("easiness_factor", 2.5),
            interval_days=row.get("interval_days", 1),
            repetitions=row.get("repetitions", 0),
            due_date=row.get("due_date", 0),
            created_at=row.get("created_at", 0)
        )

    async def get_due_cards(
        self,
        limit: int = 10,
        tags: Optional[List[str]] = None,
        source_type: Optional[str] = None
    ) -> List[Flashcard]:
        """Get cards due for review.

        Args:
            limit: Maximum cards to return
            tags: Optional tag filter
            source_type: Optional source type filter

        Returns:
            List of due flashcards
        """
        engine = self._get_engine()
        rows = engine.get_due_cards(limit=limit, source_type=source_type, tags=tags)
        return [self._row_to_flashcard(row) for row in rows]

    async def get_next_card(
        self,
        tags: Optional[List[str]] = None
    ) -> Optional[Flashcard]:
        """Get the next card to review.

        Args:
            tags: Optional tag filter

        Returns:
            Next due flashcard or None
        """
        cards = await self.get_due_cards(limit=1, tags=tags)
        if cards:
            self._current_card = cards[0]
            return self._current_card
        return None

    async def review_card(
        self,
        card_id: int,
        quality: int,
        time_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """Submit a review for a card.

        Args:
            card_id: ID of the card
            quality: Rating 0-5 (0-2 incorrect, 3-5 correct)
            time_ms: Optional response time

        Returns:
            Review result with next interval
        """
        engine = self._get_engine()
        result = engine.review_card(card_id, quality, time_ms)

        # Update session stats
        self._session_reviewed += 1
        if quality >= 3:
            self._session_correct += 1

        # Clear current card if it was the one reviewed
        if self._current_card and self._current_card.id == card_id:
            self._current_card = None

        return result

    async def rate_current_card(self, correct: bool) -> Dict[str, Any]:
        """Rate the current card in a quiz session.

        Args:
            correct: True if user got it right, False otherwise

        Returns:
            Review result with next interval info
        """
        if not self._current_card:
            return {"error": "No current card to rate"}

        # Map correct/incorrect to quality scores
        # Correct: 4 (good recall with some hesitation)
        # Incorrect: 1 (recognized answer when shown)
        quality = 4 if correct else 1

        result = await self.review_card(self._current_card.id, quality)

        # Add voice-friendly message
        if correct:
            interval = result.get("next_review_human", "soon")
            result["voice_message"] = f"Got it! Next review in {interval}."
        else:
            result["voice_message"] = "No problem, we'll practice this one more."

        return result

    async def create_card(
        self,
        front: str,
        back: str,
        tags: Optional[List[str]] = None,
        source_type: str = "voice"
    ) -> Flashcard:
        """Create a new flashcard.

        Args:
            front: Question/prompt
            back: Answer
            tags: Optional tags
            source_type: Source type (default "voice")

        Returns:
            Created flashcard
        """
        engine = self._get_engine()
        result = engine.create_card(
            front=front,
            back=back,
            source_type=source_type,
            tags=tags
        )

        logger.info(f"Created flashcard: {front[:50]}...")

        return Flashcard(
            id=result.get("id", 0),
            front=front,
            back=back,
            source_type=source_type,
            tags=tags or []
        )

    async def get_stats(self) -> LearningStats:
        """Get learning statistics.

        Returns:
            LearningStats with current metrics
        """
        engine = self._get_engine()
        stats = engine.get_stats()

        return LearningStats(
            total_cards=stats.get("total_cards", 0),
            due_today=stats.get("due_today", 0),
            mastered=stats.get("mastered", 0),
            reviews_today=stats.get("reviews_today", 0),
            retention_rate=stats.get("retention_rate", 0.0)
        )

    async def get_review_forecast(self, days: int = 7) -> List[Dict]:
        """Get forecast of cards due in upcoming days.

        Args:
            days: Number of days to forecast

        Returns:
            List of {date, count} for each day
        """
        engine = self._get_engine()
        return engine.get_review_forecast(days=days)

    async def get_due_summary(self) -> str:
        """Get a voice-friendly summary of due cards.

        Returns:
            Summary string like "3 cards due now, 5 more tomorrow"
        """
        stats = await self.get_stats()
        forecast = await self.get_review_forecast(days=2)

        parts = []

        if stats.due_today > 0:
            parts.append(f"{stats.due_today} cards due now")
        else:
            parts.append("No cards due right now")

        # Check tomorrow's count
        tomorrow_count = 0
        for item in forecast:
            if item.get("date") and "tomorrow" not in item.get("date", ""):
                continue
            tomorrow_count = item.get("count", 0)

        if tomorrow_count > 0:
            parts.append(f"{tomorrow_count} more tomorrow")

        return ", ".join(parts) + "."

    def get_current_card(self) -> Optional[Flashcard]:
        """Get the current card in the quiz session.

        Returns:
            Current flashcard or None
        """
        return self._current_card

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics.

        Returns:
            Dict with reviewed count, correct count, accuracy
        """
        accuracy = 0.0
        if self._session_reviewed > 0:
            accuracy = (self._session_correct / self._session_reviewed) * 100

        return {
            "reviewed": self._session_reviewed,
            "correct": self._session_correct,
            "accuracy": accuracy
        }

    def reset_session(self):
        """Reset session statistics."""
        self._current_card = None
        self._session_reviewed = 0
        self._session_correct = 0


# Global instance
_learning_service: Optional[LearningService] = None


def get_learning_service() -> LearningService:
    """Get or create global LearningService instance."""
    global _learning_service
    if _learning_service is None:
        _learning_service = LearningService()
    return _learning_service
