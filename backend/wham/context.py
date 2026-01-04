"""
WHAM Context Management

Tracks session state, user performance, and conversation context
for personalized, contextual responses.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum


class SessionPhase(Enum):
    """Current phase of the training session."""
    WARMUP = "warmup"        # First 3 problems
    ACTIVE = "active"        # Main session
    FLOW = "flow"            # High performance streak
    STRUGGLING = "struggling" # Multiple consecutive errors
    COOLDOWN = "cooldown"    # Session winding down


@dataclass
class ProblemResult:
    """Single problem attempt record."""
    problem_id: str
    difficulty: int
    category: str
    correct: bool
    time_ms: int
    target_ms: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def speed_ratio(self) -> float:
        """Ratio of actual time to target time."""
        return self.time_ms / self.target_ms if self.target_ms > 0 else 1.0

    @property
    def was_fast(self) -> bool:
        """Whether answer was faster than target."""
        return self.time_ms < self.target_ms


@dataclass
class SessionStats:
    """Aggregated session statistics."""
    total_problems: int = 0
    correct_count: int = 0
    current_streak: int = 0
    best_streak: int = 0
    total_time_ms: int = 0
    fastest_time_ms: Optional[int] = None
    slowest_time_ms: Optional[int] = None

    # Difficulty breakdown
    by_difficulty: Dict[int, Dict[str, int]] = field(default_factory=dict)

    # Category breakdown
    by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Session accuracy as percentage."""
        if self.total_problems == 0:
            return 0.0
        return (self.correct_count / self.total_problems) * 100

    @property
    def average_time_ms(self) -> int:
        """Average time per problem in milliseconds."""
        if self.total_problems == 0:
            return 0
        return self.total_time_ms // self.total_problems

    def update(self, result: ProblemResult) -> None:
        """Update stats with new problem result."""
        self.total_problems += 1
        self.total_time_ms += result.time_ms

        # Update fastest/slowest
        if self.fastest_time_ms is None or result.time_ms < self.fastest_time_ms:
            self.fastest_time_ms = result.time_ms
        if self.slowest_time_ms is None or result.time_ms > self.slowest_time_ms:
            self.slowest_time_ms = result.time_ms

        # Update correctness
        if result.correct:
            self.correct_count += 1
            self.current_streak += 1
            if self.current_streak > self.best_streak:
                self.best_streak = self.current_streak
        else:
            self.current_streak = 0

        # Update difficulty breakdown
        diff = result.difficulty
        if diff not in self.by_difficulty:
            self.by_difficulty[diff] = {"total": 0, "correct": 0, "time_ms": 0}
        self.by_difficulty[diff]["total"] += 1
        self.by_difficulty[diff]["time_ms"] += result.time_ms
        if result.correct:
            self.by_difficulty[diff]["correct"] += 1

        # Update category breakdown
        cat = result.category
        if cat not in self.by_category:
            self.by_category[cat] = {"total": 0, "correct": 0, "time_ms": 0}
        self.by_category[cat]["total"] += 1
        self.by_category[cat]["time_ms"] += result.time_ms
        if result.correct:
            self.by_category[cat]["correct"] += 1


class WHAMContext:
    """
    Maintains conversation and session context for WHAM personality.

    Tracks:
    - User preferences and address style
    - Current session state and statistics
    - Recent problem history for contextual responses
    - Performance trends for adaptive difficulty suggestions
    """

    # Jane Street calibrated target times (ms) by difficulty
    TARGET_TIMES = {
        1: 2000,   # D1: 2 seconds
        2: 4000,   # D2: 4 seconds
        3: 8000,   # D3: 8 seconds
        4: 12000,  # D4: 12 seconds
        5: 20000,  # D5: 20 seconds (expert)
    }

    def __init__(
        self,
        user_name: str = "Will",
        preferred_address: str = "sir",
        session_id: Optional[str] = None,
    ):
        self.user_name = user_name
        self.preferred_address = preferred_address
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Session tracking
        self.session_start = datetime.now()
        self.stats = SessionStats()
        self.recent_results: List[ProblemResult] = []
        self.phase = SessionPhase.WARMUP

        # Active problem tracking
        self.current_problem: Optional[Dict[str, Any]] = None
        self.problem_start_time: Optional[datetime] = None

        # Mode tracking
        self.current_mode: str = "mental_math"
        self.current_difficulty: int = 2

        # Milestone tracking
        self.milestones_triggered: set = set()
        self.last_streak_milestone: int = 0

    @property
    def address(self) -> str:
        """Get current address style for user."""
        return self.preferred_address

    @property
    def session_duration_minutes(self) -> int:
        """How long the session has been active."""
        delta = datetime.now() - self.session_start
        return int(delta.total_seconds() / 60)

    @property
    def time_of_day(self) -> str:
        """Get current time period for greetings."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "late_night"

    def get_target_time(self, difficulty: int) -> int:
        """Get target time in ms for a difficulty level."""
        return self.TARGET_TIMES.get(difficulty, 8000)

    def start_problem(self, problem: Dict[str, Any]) -> None:
        """Mark the start of a new problem."""
        self.current_problem = problem
        self.problem_start_time = datetime.now()

    def record_answer(
        self,
        correct: bool,
        problem_id: Optional[str] = None,
        difficulty: Optional[int] = None,
        category: str = "mental_math",
    ) -> ProblemResult:
        """Record an answer and return the result."""
        # Calculate time
        if self.problem_start_time:
            elapsed = datetime.now() - self.problem_start_time
            time_ms = int(elapsed.total_seconds() * 1000)
        else:
            time_ms = 0

        # Use provided or current values
        diff = difficulty or self.current_difficulty
        target_ms = self.get_target_time(diff)

        result = ProblemResult(
            problem_id=problem_id or f"p_{self.stats.total_problems + 1}",
            difficulty=diff,
            category=category,
            correct=correct,
            time_ms=time_ms,
            target_ms=target_ms,
        )

        # Update stats
        self.stats.update(result)
        self.recent_results.append(result)

        # Keep only last 20 results for context
        if len(self.recent_results) > 20:
            self.recent_results.pop(0)

        # Update phase
        self._update_phase()

        # Clear current problem
        self.current_problem = None
        self.problem_start_time = None

        return result

    def _update_phase(self) -> None:
        """Update session phase based on recent performance."""
        if self.stats.total_problems < 3:
            self.phase = SessionPhase.WARMUP
        elif self.stats.current_streak >= 5:
            self.phase = SessionPhase.FLOW
        elif self._recent_error_rate() > 0.5:
            self.phase = SessionPhase.STRUGGLING
        elif self.session_duration_minutes > 20:
            self.phase = SessionPhase.COOLDOWN
        else:
            self.phase = SessionPhase.ACTIVE

    def _recent_error_rate(self, window: int = 5) -> float:
        """Calculate error rate over recent problems."""
        recent = self.recent_results[-window:] if self.recent_results else []
        if not recent:
            return 0.0
        errors = sum(1 for r in recent if not r.correct)
        return errors / len(recent)

    def check_milestone(self) -> Optional[str]:
        """Check if any milestone was just reached."""
        # First correct
        if (self.stats.correct_count == 1 and
            "first_correct" not in self.milestones_triggered):
            self.milestones_triggered.add("first_correct")
            return "first_correct"

        # Ten problems
        if (self.stats.total_problems == 10 and
            "ten_problems" not in self.milestones_triggered):
            self.milestones_triggered.add("ten_problems")
            return "ten_problems"

        # Perfect set of 10
        if (self.stats.total_problems % 10 == 0 and
            self.stats.total_problems > 0):
            last_ten = self.recent_results[-10:]
            if len(last_ten) == 10 and all(r.correct for r in last_ten):
                key = f"perfect_set_{self.stats.total_problems}"
                if key not in self.milestones_triggered:
                    self.milestones_triggered.add(key)
                    return "perfect_set"

        # Comeback after errors
        if (self.stats.current_streak == 3 and
            len(self.recent_results) >= 6):
            # Check if there were 3 errors before this streak
            before_streak = self.recent_results[-6:-3]
            if sum(1 for r in before_streak if not r.correct) >= 2:
                key = f"comeback_{self.stats.total_problems}"
                if key not in self.milestones_triggered:
                    self.milestones_triggered.add(key)
                    return "comeback"

        return None

    def check_streak_milestone(self) -> Optional[int]:
        """Check if current streak is at a milestone."""
        streak = self.stats.current_streak
        milestones = [3, 5, 7, 10, 15, 20, 25, 50]

        if streak in milestones and streak > self.last_streak_milestone:
            self.last_streak_milestone = streak
            return streak

        # High streaks divisible by 5
        if streak > 25 and streak % 5 == 0 and streak > self.last_streak_milestone:
            self.last_streak_milestone = streak
            return streak

        return None

    def should_suggest_break(self) -> bool:
        """Check if a break should be suggested."""
        # Suggest after 15+ minutes with declining performance
        if self.session_duration_minutes >= 15:
            if self._recent_error_rate() > 0.4:
                return True
        # Always suggest after 30 minutes
        return self.session_duration_minutes >= 30

    def should_suggest_difficulty_change(self) -> Optional[str]:
        """Check if difficulty adjustment should be suggested."""
        if len(self.recent_results) < 5:
            return None

        recent = self.recent_results[-5:]
        accuracy = sum(1 for r in recent if r.correct) / len(recent)
        avg_speed_ratio = sum(r.speed_ratio for r in recent) / len(recent)

        # Suggest increase if crushing it
        if accuracy >= 0.9 and avg_speed_ratio < 0.6:
            return "increase"

        # Suggest decrease if struggling
        if accuracy < 0.5 or avg_speed_ratio > 1.5:
            return "decrease"

        return None

    def get_performance_trend(self) -> str:
        """Analyze performance trend over session."""
        if len(self.recent_results) < 10:
            return "steady"

        first_half = self.recent_results[:len(self.recent_results)//2]
        second_half = self.recent_results[len(self.recent_results)//2:]

        first_accuracy = sum(1 for r in first_half if r.correct) / len(first_half)
        second_accuracy = sum(1 for r in second_half if r.correct) / len(second_half)

        diff = second_accuracy - first_accuracy
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        return "steady"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for storage."""
        return {
            "session_id": self.session_id,
            "user_name": self.user_name,
            "preferred_address": self.preferred_address,
            "session_start": self.session_start.isoformat(),
            "current_mode": self.current_mode,
            "current_difficulty": self.current_difficulty,
            "stats": {
                "total_problems": self.stats.total_problems,
                "correct_count": self.stats.correct_count,
                "current_streak": self.stats.current_streak,
                "best_streak": self.stats.best_streak,
                "accuracy": self.stats.accuracy,
                "average_time_ms": self.stats.average_time_ms,
            },
            "phase": self.phase.value,
        }
