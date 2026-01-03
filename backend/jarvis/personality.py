"""
JARVIS Personality Core

The main personality engine that transforms dry responses into
Tony Stark's AI assistant style interactions.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from .templates import ResponseTemplates
from .context import JarvisContext, ProblemResult, SessionPhase
from .performance import PerformanceAnalyzer, PerformanceInsight


class JarvisPersonality:
    """
    JARVIS Personality Engine

    Transforms standard responses into Tony Stark's AI assistant style.
    Maintains context awareness for personalized, witty interactions.

    Key behaviors:
    - Formal but warm address ("sir", "Will")
    - Dry wit and understated humor
    - Performance-focused feedback (Jane Street calibrated)
    - Contextual awareness of session state
    - Proactive suggestions based on performance patterns
    """

    def __init__(
        self,
        user_name: str = "Will",
        preferred_address: str = "sir",
    ):
        self.templates = ResponseTemplates()
        self.context = JarvisContext(
            user_name=user_name,
            preferred_address=preferred_address,
        )
        self.analyzer = PerformanceAnalyzer()

    @property
    def address(self) -> str:
        """Current address style for user."""
        return self.context.address

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    def greet(self) -> str:
        """Generate session greeting based on time of day."""
        return self.templates.get_greeting(
            self.context.time_of_day,
            self.address,
        )

    def start_mode(self, mode: str) -> str:
        """Announce mode activation."""
        self.context.current_mode = mode
        return self.templates.get_mode_activation(mode, self.address)

    def end_session(self) -> Dict[str, Any]:
        """Generate session end summary with analysis."""
        summary = self.analyzer.generate_session_summary(
            self.context.recent_results,
            self.context.stats,
        )

        # Get trend
        trend = self.context.get_performance_trend()

        # Generate JARVIS-style summary message
        message = self.templates.get_session_summary(
            address=self.address,
            correct=self.context.stats.correct_count,
            total=self.context.stats.total_problems,
            avg_time_ms=self.context.stats.average_time_ms,
            trend=trend,
        )

        return {
            "message": message,
            "summary": summary,
            "session_id": self.context.session_id,
            "duration_minutes": self.context.session_duration_minutes,
        }

    # =========================================================================
    # Problem Flow
    # =========================================================================

    def present_problem(
        self,
        problem: Dict[str, Any],
        difficulty: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Format problem presentation with JARVIS style.

        Returns dict with:
        - problem_text: The problem to display
        - intro: Optional intro phrase
        - difficulty: Difficulty level
        - target_time_ms: Target time for this difficulty
        """
        if difficulty:
            self.context.current_difficulty = difficulty

        self.context.start_problem(problem)

        target_ms = self.context.get_target_time(self.context.current_difficulty)

        return {
            "problem_text": problem.get("question", str(problem)),
            "answer": problem.get("answer"),
            "difficulty": self.context.current_difficulty,
            "target_time_ms": target_ms,
            "problem_number": self.context.stats.total_problems + 1,
        }

    def process_answer(
        self,
        user_answer: str,
        correct_answer: str,
        difficulty: Optional[int] = None,
        category: str = "mental_math",
    ) -> Dict[str, Any]:
        """
        Process user's answer and generate JARVIS-style feedback.

        Returns comprehensive feedback dict with:
        - correct: Whether answer was correct
        - feedback: Primary feedback message
        - speed_feedback: Speed-specific message
        - streak_message: Streak acknowledgment (if milestone)
        - milestone_message: Achievement message (if any)
        - suggestions: Any proactive suggestions
        - stats: Current session stats
        """
        # Determine correctness
        correct = self._check_answer(user_answer, correct_answer)

        # Record the result
        result = self.context.record_answer(
            correct=correct,
            difficulty=difficulty,
            category=category,
        )

        # Build response
        response = {
            "correct": correct,
            "time_ms": result.time_ms,
            "target_ms": result.target_ms,
            "speed_tier": self.analyzer.classify_speed(result.time_ms, result.target_ms),
        }

        # Primary feedback
        if correct:
            response["feedback"] = self.templates.get_speed_feedback(
                result.time_ms,
                result.target_ms,
                self.address,
            )
        else:
            response["feedback"] = self.templates.get_encouragement(
                correct_answer,
                self.address,
            )
            response["correct_answer"] = correct_answer

        # Check for streak milestone
        streak_milestone = self.context.check_streak_milestone()
        if streak_milestone:
            response["streak_message"] = self.templates.get_streak_message(
                streak_milestone,
                self.address,
            )

        # Check for other milestones
        milestone = self.context.check_milestone()
        if milestone:
            response["milestone_message"] = self.templates.get_milestone_message(
                milestone,
                self.address,
            )

        # Check for proactive suggestions
        suggestions = self._get_suggestions()
        if suggestions:
            response["suggestions"] = suggestions

        # Current stats
        response["stats"] = {
            "total": self.context.stats.total_problems,
            "correct": self.context.stats.correct_count,
            "streak": self.context.stats.current_streak,
            "best_streak": self.context.stats.best_streak,
            "accuracy": f"{self.context.stats.accuracy:.1f}%",
        }

        return response

    def _check_answer(self, user_answer: str, correct_answer: str) -> bool:
        """
        Check if user's answer matches correct answer.
        Handles numeric tolerance and formatting variations.
        """
        # Normalize strings
        user_clean = str(user_answer).strip().lower().replace(" ", "")
        correct_clean = str(correct_answer).strip().lower().replace(" ", "")

        # Exact match
        if user_clean == correct_clean:
            return True

        # Try numeric comparison
        try:
            user_num = float(user_clean.replace(",", ""))
            correct_num = float(correct_clean.replace(",", ""))

            # Allow small floating point tolerance
            if abs(user_num - correct_num) < 0.001:
                return True

            # For larger numbers, allow 0.01% tolerance
            if correct_num != 0:
                relative_diff = abs(user_num - correct_num) / abs(correct_num)
                if relative_diff < 0.0001:
                    return True
        except (ValueError, TypeError):
            pass

        return False

    def _get_suggestions(self) -> List[Dict[str, str]]:
        """Generate proactive suggestions based on current context."""
        suggestions = []

        # Break suggestion
        if self.context.should_suggest_break():
            msg = self.templates.get_suggestion(
                "break_recommended",
                self.address,
                duration=self.context.session_duration_minutes,
            )
            if msg:
                suggestions.append({
                    "type": "break",
                    "message": msg,
                })

        # Difficulty adjustment suggestion
        diff_suggestion = self.context.should_suggest_difficulty_change()
        if diff_suggestion == "increase":
            msg = self.templates.get_suggestion("difficulty_increase", self.address)
            if msg:
                suggestions.append({
                    "type": "difficulty_increase",
                    "message": msg,
                })
        elif diff_suggestion == "decrease":
            msg = self.templates.get_suggestion("difficulty_decrease", self.address)
            if msg:
                suggestions.append({
                    "type": "difficulty_decrease",
                    "message": msg,
                })

        return suggestions

    # =========================================================================
    # Contextual Responses
    # =========================================================================

    def get_phase_aware_message(self) -> Optional[str]:
        """Get message appropriate for current session phase."""
        phase = self.context.phase

        if phase == SessionPhase.WARMUP:
            return f"Warming up, {self.address}. Finding your rhythm."
        elif phase == SessionPhase.FLOW:
            return f"You're in the zone, {self.address}. Maintain this focus."
        elif phase == SessionPhase.STRUGGLING:
            return f"A challenging stretch. Breathe and refocus, {self.address}."
        elif phase == SessionPhase.COOLDOWN:
            return f"Solid session, {self.address}. Wrap up when ready."
        return None

    def get_encouragement_for_error(self, error_count: int) -> str:
        """Get encouraging message based on consecutive errors."""
        if error_count == 1:
            return f"Minor setback, {self.address}. Onward."
        elif error_count == 2:
            return f"Two in a row—let's reset. Take a breath."
        elif error_count >= 3:
            return f"A rough patch. Perhaps a brief pause, {self.address}?"
        return ""

    def acknowledge_comeback(self) -> str:
        """Acknowledge recovery from error streak."""
        return f"Back on track, {self.address}. That's the resilience they want to see."

    # =========================================================================
    # Performance Analysis
    # =========================================================================

    def get_performance_insights(self) -> List[Dict[str, Any]]:
        """Get current performance insights."""
        insights = self.analyzer.analyze_session(
            self.context.recent_results,
            self.context.stats,
        )

        return [
            {
                "category": i.category,
                "message": i.message,
                "severity": i.severity,
                "recommendation": i.recommendation,
            }
            for i in insights
        ]

    def get_difficulty_recommendation(self) -> Dict[str, Any]:
        """Get recommended difficulty with reasoning."""
        rec_diff, reason = self.analyzer.get_recommended_difficulty(
            self.context.recent_results,
            self.context.current_difficulty,
        )

        return {
            "current": self.context.current_difficulty,
            "recommended": rec_diff,
            "reason": reason,
            "should_change": rec_diff != self.context.current_difficulty,
        }

    # =========================================================================
    # Quick Responses
    # =========================================================================

    def quick_correct(self, time_ms: int) -> str:
        """Quick correct answer response."""
        target_ms = self.context.get_target_time(self.context.current_difficulty)
        return self.templates.get_speed_feedback(time_ms, target_ms, self.address)

    def quick_incorrect(self, correct_answer: str) -> str:
        """Quick incorrect answer response."""
        return self.templates.get_encouragement(correct_answer, self.address)

    def quick_streak(self, streak: int) -> Optional[str]:
        """Quick streak acknowledgment."""
        return self.templates.get_streak_message(streak, self.address)

    # =========================================================================
    # Special Occasions
    # =========================================================================

    def get_jane_street_motivation(self) -> str:
        """Special motivational message for interview prep context."""
        accuracy = self.context.stats.accuracy
        streak = self.context.stats.best_streak

        if accuracy >= 95 and streak >= 10:
            return (f"This performance would turn heads at Jane Street, {self.address}. "
                   "Maintain this standard.")
        elif accuracy >= 85:
            return (f"Interview caliber, {self.address}. "
                   "Jane Street values exactly this kind of precision under pressure.")
        else:
            return (f"Every session builds the foundation, {self.address}. "
                   "The trading floor rewards consistent practice.")

    def get_difficulty_unlocked_message(self, new_difficulty: int) -> str:
        """Message when user unlocks a new difficulty level."""
        messages = {
            2: f"Level 2 unlocked, {self.address}. The real training begins.",
            3: f"Level 3, {self.address}. Now we're getting serious.",
            4: f"Level 4 access granted. Elite territory, {self.address}.",
            5: f"Level 5 unlocked. You're operating at Citadel speed now, {self.address}.",
        }
        return messages.get(
            new_difficulty,
            f"Difficulty {new_difficulty} unlocked, {self.address}."
        )

    # =========================================================================
    # Context Export
    # =========================================================================

    def export_context(self) -> Dict[str, Any]:
        """Export current context for persistence."""
        return self.context.to_dict()

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        return {
            "session_id": self.context.session_id,
            "duration_minutes": self.context.session_duration_minutes,
            "mode": self.context.current_mode,
            "difficulty": self.context.current_difficulty,
            "phase": self.context.phase.value,
            "stats": {
                "total": self.context.stats.total_problems,
                "correct": self.context.stats.correct_count,
                "streak": self.context.stats.current_streak,
                "best_streak": self.context.stats.best_streak,
                "accuracy": self.context.stats.accuracy,
                "avg_time_ms": self.context.stats.average_time_ms,
            },
        }
