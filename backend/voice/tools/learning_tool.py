"""Learning Coach voice tool - Spaced repetition flashcard learning."""
import logging
import re
from enum import Enum
from typing import List, Optional
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class QuizState(Enum):
    """Quiz session states."""
    IDLE = "idle"
    QUESTIONING = "questioning"  # Card front shown, waiting for user to answer
    WAITING_RATING = "waiting_rating"  # User answered, waiting for correct/wrong


class LearningVoiceTool(VoiceTool):
    """Voice-controlled learning coach with spaced repetition.

    Supports stateful quiz sessions where:
    1. User says "Quiz me" to start
    2. System reads question (front of card)
    3. User thinks/says answer
    4. User rates themselves: "I got it" or "Wrong"
    5. System confirms, shows answer, schedules next review
    6. Continues until "Done studying" or no more cards
    """

    name = "learning"
    description = "Learn with spaced repetition flashcards"

    keywords = [
        r"\b(quiz|review|study)\s*(me|cards?|flashcards?)?\b",
        r"\bflashcards?\b",
        r"\blearn(ing)?\s*(stats?|summary)?\b",
        r"\bpractice\b",
        r"\badd\s+(to\s+)?flashcards?\b",
        r"\b(how\s+)?many\s+cards?\s+(due|to\s+review)\b",
        r"\bspaced\s+repetition\b",
        r"\b(i\s+)?(got\s+it|knew\s+it|correct|right)\b",
        r"\b(i\s+)?(didn'?t\s+know|wrong|incorrect|missed)\b",
        r"\b(done|stop|finish)\s*(studying|review|quiz)?\b",
        r"\bnext\s+card\b",
        r"\bshow\s+(the\s+)?answer\b",
    ]

    priority = 6

    def __init__(self):
        self._learning_service = None
        self._state = QuizState.IDLE
        self._current_tags: Optional[List[str]] = None

    def _get_service(self):
        """Get learning service (lazy load)."""
        if self._learning_service is None:
            from backend.services.learning_service import get_learning_service
            self._learning_service = get_learning_service()
        return self._learning_service

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute learning command.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult with learning info
        """
        query_lower = query.lower()
        service = self._get_service()

        try:
            # Check for rating responses first (highest priority when in quiz)
            if self._state in (QuizState.QUESTIONING, QuizState.WAITING_RATING):
                if self._is_correct_rating(query_lower):
                    return await self._handle_rating(service, correct=True)
                elif self._is_incorrect_rating(query_lower):
                    return await self._handle_rating(service, correct=False)
                elif self._is_show_answer(query_lower):
                    return await self._handle_show_answer(service)
                elif self._is_done_studying(query_lower):
                    return await self._handle_done_studying(service)
                elif self._is_next_card(query_lower):
                    # Skip current and get next
                    return await self._handle_next_card(service)

            # Start quiz / get next card
            if self._is_quiz_request(query_lower):
                tags = self._extract_tags(query_lower)
                return await self._handle_start_quiz(service, tags)

            # Add flashcard
            if self._is_add_card(query_lower):
                return await self._handle_add_card(service, query)

            # Stats query
            if self._is_stats_query(query_lower):
                return await self._handle_stats(service)

            # Due count query
            if self._is_due_query(query_lower):
                return await self._handle_due_count(service)

            # Done studying
            if self._is_done_studying(query_lower):
                return await self._handle_done_studying(service)

            # Default: show stats or due count
            return await self._handle_summary(service)

        except Exception as e:
            logger.error(f"Learning tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble with the learning tool.",
                data={"error": str(e)}
            )

    def _is_quiz_request(self, query: str) -> bool:
        patterns = ["quiz me", "quiz", "review", "study", "practice", "next card"]
        return any(p in query for p in patterns)

    def _is_correct_rating(self, query: str) -> bool:
        patterns = [
            "got it", "knew it", "correct", "right", "yes",
            "i know", "easy", "good", "nailed it"
        ]
        return any(p in query for p in patterns)

    def _is_incorrect_rating(self, query: str) -> bool:
        patterns = [
            "wrong", "incorrect", "missed", "didn't know",
            "no", "forgot", "hard", "don't know"
        ]
        return any(p in query for p in patterns)

    def _is_show_answer(self, query: str) -> bool:
        patterns = ["show answer", "what's the answer", "reveal", "tell me"]
        return any(p in query for p in patterns)

    def _is_next_card(self, query: str) -> bool:
        patterns = ["next card", "skip", "another one", "next question"]
        return any(p in query for p in patterns)

    def _is_done_studying(self, query: str) -> bool:
        patterns = [
            "done studying", "stop studying", "finish",
            "done quiz", "stop quiz", "that's enough", "end session"
        ]
        return any(p in query for p in patterns)

    def _is_add_card(self, query: str) -> bool:
        patterns = ["add flashcard", "add card", "create flashcard", "new flashcard"]
        return any(p in query for p in patterns)

    def _is_stats_query(self, query: str) -> bool:
        patterns = ["learning stats", "flashcard stats", "my stats", "progress"]
        return any(p in query for p in patterns)

    def _is_due_query(self, query: str) -> bool:
        patterns = ["how many cards", "cards due", "any cards", "review count"]
        return any(p in query for p in patterns)

    def _extract_tags(self, query: str) -> Optional[List[str]]:
        """Extract tags/topics from query like 'study poker'."""
        # Look for topic after study/quiz/review
        match = re.search(r"(?:study|quiz|review|practice)\s+(?:on\s+)?(\w+)", query)
        if match:
            topic = match.group(1)
            if topic not in ["me", "cards", "flashcards"]:
                return [topic]
        return None

    def _extract_card_content(self, query: str) -> tuple:
        """Extract front and back from 'add flashcard: X answer Y'."""
        # Pattern: add flashcard: [front] answer [back]
        match = re.search(
            r"(?:add\s+(?:flash)?card|create\s+(?:flash)?card)[\s:]+(.+?)\s+(?:answer|is|equals?)\s+(.+)",
            query,
            re.IGNORECASE
        )
        if match:
            return match.group(1).strip(), match.group(2).strip()

        # Simpler pattern: add flashcard [front] - [back]
        match = re.search(
            r"(?:add\s+(?:flash)?card|create\s+(?:flash)?card)[\s:]+(.+?)\s*[-–]\s*(.+)",
            query,
            re.IGNORECASE
        )
        if match:
            return match.group(1).strip(), match.group(2).strip()

        return None, None

    async def _handle_start_quiz(self, service, tags: Optional[List[str]]) -> VoiceToolResult:
        """Start a quiz session and get the first card."""
        self._current_tags = tags

        card = await service.get_next_card(tags=tags)

        if not card:
            self._state = QuizState.IDLE
            topic = f" on {tags[0]}" if tags else ""
            return VoiceToolResult(
                success=True,
                message=f"No cards due for review{topic}. Great job staying on top of your learning!",
                data={"has_cards": False}
            )

        self._state = QuizState.QUESTIONING

        topic = f" ({', '.join(card.tags)})" if card.tags else ""
        message = f"Here's your card{topic}: {card.to_voice_question()}"

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "state": "questioning",
                "card": card.to_dict()
            }
        )

    async def _handle_rating(self, service, correct: bool) -> VoiceToolResult:
        """Handle user's self-rating of their answer."""
        current = service.get_current_card()

        if not current:
            self._state = QuizState.IDLE
            return VoiceToolResult(
                success=False,
                message="No card to rate. Say 'quiz me' to start.",
                data={"error": "no_current_card"}
            )

        # Submit the rating
        result = await service.rate_current_card(correct)

        if "error" in result:
            return VoiceToolResult(
                success=False,
                message=result.get("voice_message", "Error rating card."),
                data=result
            )

        # Build response with answer
        answer = current.to_voice_answer()
        voice_message = result.get("voice_message", "")

        if correct:
            message = f"{voice_message} The answer was: {answer}"
        else:
            message = f"The answer is: {answer}. {voice_message}"

        # Try to get next card
        next_card = await service.get_next_card(tags=self._current_tags)

        if next_card:
            self._state = QuizState.QUESTIONING
            message += f" Next card: {next_card.to_voice_question()}"
            data = {
                "state": "questioning",
                "card": next_card.to_dict(),
                "previous_result": result
            }
        else:
            self._state = QuizState.IDLE
            session = service.get_session_stats()
            message += f" That's all for now! You reviewed {session['reviewed']} cards."
            if session['reviewed'] > 0:
                message += f" Accuracy: {session['accuracy']:.0f}%."
            data = {
                "state": "idle",
                "session_complete": True,
                "session_stats": session
            }
            service.reset_session()

        return VoiceToolResult(
            success=True,
            message=message,
            data=data
        )

    async def _handle_show_answer(self, service) -> VoiceToolResult:
        """Show the answer without rating."""
        current = service.get_current_card()

        if not current:
            return VoiceToolResult(
                success=False,
                message="No card to show. Say 'quiz me' to start.",
                data={"error": "no_current_card"}
            )

        self._state = QuizState.WAITING_RATING
        answer = current.to_voice_answer()

        return VoiceToolResult(
            success=True,
            message=f"The answer is: {answer}. Did you get it right?",
            data={
                "state": "waiting_rating",
                "answer": answer
            }
        )

    async def _handle_next_card(self, service) -> VoiceToolResult:
        """Skip current card and get next."""
        next_card = await service.get_next_card(tags=self._current_tags)

        if next_card:
            self._state = QuizState.QUESTIONING
            return VoiceToolResult(
                success=True,
                message=f"Next card: {next_card.to_voice_question()}",
                data={
                    "state": "questioning",
                    "card": next_card.to_dict()
                }
            )
        else:
            self._state = QuizState.IDLE
            return VoiceToolResult(
                success=True,
                message="No more cards due for review.",
                data={"state": "idle", "has_cards": False}
            )

    async def _handle_done_studying(self, service) -> VoiceToolResult:
        """End the study session."""
        session = service.get_session_stats()
        service.reset_session()
        self._state = QuizState.IDLE

        if session['reviewed'] > 0:
            message = (
                f"Great session! You reviewed {session['reviewed']} cards "
                f"with {session['accuracy']:.0f}% accuracy."
            )
        else:
            message = "Study session ended. Come back anytime!"

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "state": "idle",
                "session_stats": session
            }
        )

    async def _handle_add_card(self, service, query: str) -> VoiceToolResult:
        """Add a new flashcard."""
        front, back = self._extract_card_content(query)

        if not front or not back:
            return VoiceToolResult(
                success=False,
                message="Say 'add flashcard' followed by the question, then 'answer' and the answer. For example: 'Add flashcard what is the capital of France answer Paris'.",
                data={"error": "invalid_format"}
            )

        card = await service.create_card(front=front, back=back)

        return VoiceToolResult(
            success=True,
            message=f"Created flashcard: {front[:50]}... I'll remind you to review it.",
            data={"card": card.to_dict()}
        )

    async def _handle_stats(self, service) -> VoiceToolResult:
        """Get learning statistics."""
        stats = await service.get_stats()

        return VoiceToolResult(
            success=True,
            message=stats.to_voice_summary(),
            data={"stats": stats.to_dict()}
        )

    async def _handle_due_count(self, service) -> VoiceToolResult:
        """Get count of due cards."""
        summary = await service.get_due_summary()

        return VoiceToolResult(
            success=True,
            message=summary,
            data={"summary": summary}
        )

    async def _handle_summary(self, service) -> VoiceToolResult:
        """Handle general learning query."""
        stats = await service.get_stats()

        if stats.due_today > 0:
            message = f"You have {stats.due_today} cards ready for review. Say 'quiz me' to start."
        else:
            message = "No cards due right now. "
            if stats.total_cards > 0:
                message += f"You have {stats.total_cards} cards total with {stats.retention_rate:.0f}% retention."
            else:
                message += "Say 'add flashcard' to create your first card."

        return VoiceToolResult(
            success=True,
            message=message,
            data={"stats": stats.to_dict()}
        )
