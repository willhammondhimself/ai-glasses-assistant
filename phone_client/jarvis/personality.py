"""
JARVIS Personality Engine.
Context-aware AI companion with Tony Stark butler style.
"""
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

from .templates import Templates


class SessionPhase(Enum):
    """Current phase of the session."""
    WARMUP = "warmup"       # First 5 problems
    ACTIVE = "active"       # Normal operation
    FLOW = "flow"           # High streak, fast times
    STRUGGLING = "struggling"  # Multiple errors
    COOLDOWN = "cooldown"   # Session ending


@dataclass
class SessionStats:
    """Statistics for the current session."""
    total_problems: int = 0
    correct: int = 0
    incorrect: int = 0
    current_streak: int = 0
    best_streak: int = 0
    total_time_ms: float = 0
    times: List[float] = field(default_factory=list)
    error_streak: int = 0
    phase: SessionPhase = SessionPhase.WARMUP

    @property
    def accuracy(self) -> float:
        if self.total_problems == 0:
            return 0.0
        return (self.correct / self.total_problems) * 100

    @property
    def avg_time_ms(self) -> float:
        if not self.times:
            return 0.0
        return sum(self.times) / len(self.times)

    def record_correct(self, time_ms: float):
        """Record a correct answer."""
        self.total_problems += 1
        self.correct += 1
        self.current_streak += 1
        self.error_streak = 0
        self.times.append(time_ms)
        self.total_time_ms += time_ms

        if self.current_streak > self.best_streak:
            self.best_streak = self.current_streak

        self._update_phase()

    def record_incorrect(self):
        """Record an incorrect answer."""
        self.total_problems += 1
        self.incorrect += 1
        self.current_streak = 0
        self.error_streak += 1

        self._update_phase()

    def _update_phase(self):
        """Update session phase based on performance."""
        if self.total_problems <= 5:
            self.phase = SessionPhase.WARMUP
        elif self.error_streak >= 3:
            self.phase = SessionPhase.STRUGGLING
        elif self.current_streak >= 10 and self.avg_time_ms < 3000:
            self.phase = SessionPhase.FLOW
        else:
            self.phase = SessionPhase.ACTIVE


class JarvisPersonality:
    """
    JARVIS personality engine.

    Provides context-aware responses based on:
    - Session performance
    - Time of day
    - Location
    - Streak status
    - User history
    """

    def __init__(
        self,
        user_name: str = "Will",
        address: str = "sir",
        location: Optional[str] = None,
        humor_level: str = "high"
    ):
        self.user_name = user_name
        self.address = address
        self.location = location
        self.humor_level = humor_level

        self.stats = SessionStats()
        self._last_streak_milestone = 0
        self._session_start = time.time()
        self._last_feedback: str = ""

    def reset_session(self):
        """Reset for a new session."""
        self.stats = SessionStats()
        self._last_streak_milestone = 0
        self._session_start = time.time()

    def get_greeting(self) -> str:
        """Get contextual greeting."""
        greeting = Templates.get_greeting(self.user_name, self.address)

        # Add location comment occasionally
        if self.location and self.humor_level == "high":
            loc_comment = Templates.get_location_comment(self.location)
            if loc_comment:
                greeting = f"{greeting} {loc_comment}"

        return greeting

    def get_mode_activation(self, mode: str, difficulty: int = 2) -> str:
        """Get mode activation message."""
        return Templates.get_mode_activation(mode, difficulty)

    def process_answer(
        self,
        correct: bool,
        time_ms: float,
        target_ms: float,
        expected_answer: Optional[float] = None
    ) -> Dict[str, str]:
        """
        Process an answer and generate JARVIS response.

        Returns dict with:
            - feedback: Main feedback line
            - streak_message: Streak milestone (if any)
            - encouragement: Extra encouragement (if needed)
        """
        response = {
            "feedback": "",
            "streak_message": "",
            "encouragement": "",
        }

        if correct:
            self.stats.record_correct(time_ms)

            # Speed feedback
            response["feedback"] = Templates.get_speed_feedback(
                time_ms, target_ms, self.address
            )

            # Check for streak milestone
            streak = self.stats.current_streak
            if streak in [3, 5, 7, 10, 15, 20, 25, 50] and streak > self._last_streak_milestone:
                response["streak_message"] = Templates.get_streak_message(streak, self.address)
                self._last_streak_milestone = streak

        else:
            # Record the break before getting message
            had_streak = self.stats.current_streak > 2
            self.stats.record_incorrect()
            self._last_streak_milestone = 0

            # Wrong answer feedback
            if expected_answer is not None:
                response["feedback"] = Templates.get_wrong_answer_feedback(
                    expected_answer, self.address
                )
            else:
                response["feedback"] = "Incorrect."

            # Streak break message
            if had_streak:
                response["streak_message"] = Templates.get_streak_break_message()

            # Encouragement if struggling
            if self.stats.error_streak >= 2:
                response["encouragement"] = Templates.get_encouragement(self.address)

        self._last_feedback = response["feedback"]
        return response

    def get_full_feedback(
        self,
        correct: bool,
        time_ms: float,
        target_ms: float,
        expected_answer: Optional[float] = None
    ) -> str:
        """
        Get complete feedback string (combines all response parts).
        """
        response = self.process_answer(correct, time_ms, target_ms, expected_answer)

        parts = [response["feedback"]]
        if response["streak_message"]:
            parts.append(response["streak_message"])
        if response["encouragement"]:
            parts.append(response["encouragement"])

        return " ".join(parts)

    def get_session_summary(self) -> Dict:
        """
        Get session summary with stats and JARVIS commentary.
        """
        stats = self.stats

        # Calculate grade
        if stats.accuracy >= 95 and stats.avg_time_ms < 2500:
            grade = "S"
        elif stats.accuracy >= 90:
            grade = "A"
        elif stats.accuracy >= 80:
            grade = "B"
        elif stats.accuracy >= 70:
            grade = "C"
        elif stats.accuracy >= 60:
            grade = "D"
        else:
            grade = "F"

        # Session duration
        duration_s = time.time() - self._session_start

        # JARVIS commentary
        commentary = Templates.get_session_end(stats.accuracy, self.address)

        # Extra commentary based on performance
        extras = []
        if stats.best_streak >= 20:
            extras.append(f"Peak streak of {stats.best_streak}. Impressive.")
        if stats.avg_time_ms < 2000 and stats.total_problems >= 10:
            extras.append("Sub-2-second average. Trading floor speed.")

        return {
            "total": stats.total_problems,
            "correct": stats.correct,
            "incorrect": stats.incorrect,
            "accuracy": stats.accuracy,
            "avg_time_ms": stats.avg_time_ms,
            "best_streak": stats.best_streak,
            "duration_s": duration_s,
            "grade": grade,
            "commentary": commentary,
            "extras": extras,
        }

    def get_idle_message(self) -> str:
        """Get message for idle state."""
        hour = datetime.now().hour

        if self.stats.total_problems > 0:
            # Mid-session idle
            return f"Awaiting input, {self.address}."
        else:
            # Start of session
            return "Ready when you are."

    def get_comeback_message(self) -> str:
        """Message after recovering from errors."""
        if self.stats.current_streak == 3 and self.stats.error_streak == 0:
            return f"Back on track, {self.address}."
        return ""

    @property
    def current_streak(self) -> int:
        return self.stats.current_streak

    @property
    def session_accuracy(self) -> float:
        return self.stats.accuracy

    @property
    def is_in_flow(self) -> bool:
        return self.stats.phase == SessionPhase.FLOW

    @property
    def is_struggling(self) -> bool:
        return self.stats.phase == SessionPhase.STRUGGLING


# Test
if __name__ == "__main__":
    print("=== JARVIS Personality Test ===\n")

    jarvis = JarvisPersonality(
        user_name="Will",
        address="sir",
        location="Claremont, CA"
    )

    print(f"Greeting: {jarvis.get_greeting()}")
    print(f"Mode activation: {jarvis.get_mode_activation('mental_math', 2)}")
    print()

    # Simulate a session
    print("=== Simulated Session ===\n")

    test_results = [
        (True, 1800, 4000),   # Fast correct
        (True, 2500, 4000),   # Good correct
        (True, 3200, 4000),   # Correct
        (False, 5000, 4000),  # Wrong
        (True, 2000, 4000),   # Correct
        (True, 1500, 4000),   # Fast
        (True, 1700, 4000),   # Fast
        (True, 2100, 4000),   # Good
        (True, 1900, 4000),   # Good
        (True, 2200, 4000),   # Good - streak 10!
    ]

    for i, (correct, time_ms, target) in enumerate(test_results, 1):
        feedback = jarvis.get_full_feedback(correct, time_ms, target, 42 if not correct else None)
        print(f"Q{i}: {'Correct' if correct else 'Wrong'} in {time_ms}ms")
        print(f"    JARVIS: {feedback}")
        print()

    print("=== Session Summary ===")
    summary = jarvis.get_session_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
