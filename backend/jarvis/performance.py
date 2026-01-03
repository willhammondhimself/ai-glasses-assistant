"""
JARVIS Performance Analyzer

Analyzes user performance patterns and provides insights
for adaptive difficulty and targeted feedback.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from .context import ProblemResult, SessionStats


@dataclass
class PerformanceInsight:
    """A single performance insight with recommendation."""
    category: str  # speed, accuracy, consistency, stamina
    message: str
    severity: str  # positive, neutral, needs_attention
    recommendation: Optional[str] = None


class PerformanceAnalyzer:
    """
    Analyzes performance data to identify patterns and provide
    actionable insights for improvement.

    Key metrics:
    - Speed consistency (variance in response times)
    - Accuracy by difficulty level
    - Fatigue detection (performance over time)
    - Weak categories identification
    """

    # Speed tier thresholds (ratio to target time)
    SPEED_TIERS = {
        "exceptional": 0.5,    # < 50% of target
        "excellent": 0.75,     # 50-75% of target
        "good": 1.0,           # 75-100% of target
        "needs_work": float("inf"),  # > 100% of target
    }

    def __init__(self):
        self.historical_sessions: List[Dict] = []

    def classify_speed(self, time_ms: int, target_ms: int) -> str:
        """Classify speed into performance tier."""
        ratio = time_ms / target_ms if target_ms > 0 else 1.0

        if ratio < self.SPEED_TIERS["exceptional"]:
            return "exceptional"
        elif ratio < self.SPEED_TIERS["excellent"]:
            return "excellent"
        elif ratio <= self.SPEED_TIERS["good"]:
            return "good"
        else:
            return "needs_work"

    def analyze_session(
        self,
        results: List[ProblemResult],
        stats: SessionStats,
    ) -> List[PerformanceInsight]:
        """Generate insights from session performance."""
        insights = []

        if not results:
            return insights

        # Speed analysis
        insights.extend(self._analyze_speed(results))

        # Accuracy analysis
        insights.extend(self._analyze_accuracy(results, stats))

        # Consistency analysis
        insights.extend(self._analyze_consistency(results))

        # Fatigue detection
        insights.extend(self._analyze_fatigue(results))

        # Category-specific analysis
        insights.extend(self._analyze_categories(results, stats))

        return insights

    def _analyze_speed(self, results: List[ProblemResult]) -> List[PerformanceInsight]:
        """Analyze speed patterns."""
        insights = []

        if len(results) < 3:
            return insights

        # Calculate speed tier distribution
        tiers = {"exceptional": 0, "excellent": 0, "good": 0, "needs_work": 0}
        for r in results:
            tier = self.classify_speed(r.time_ms, r.target_ms)
            tiers[tier] += 1

        total = len(results)
        exceptional_pct = tiers["exceptional"] / total

        if exceptional_pct > 0.3:
            insights.append(PerformanceInsight(
                category="speed",
                message=f"{int(exceptional_pct * 100)}% of answers at exceptional speed",
                severity="positive",
                recommendation="Consider increasing difficulty for continued growth",
            ))

        slow_pct = tiers["needs_work"] / total
        if slow_pct > 0.4:
            insights.append(PerformanceInsight(
                category="speed",
                message=f"{int(slow_pct * 100)}% of answers above target time",
                severity="needs_attention",
                recommendation="Focus on speed drills at current difficulty",
            ))

        return insights

    def _analyze_accuracy(
        self,
        results: List[ProblemResult],
        stats: SessionStats,
    ) -> List[PerformanceInsight]:
        """Analyze accuracy patterns."""
        insights = []

        if stats.total_problems < 5:
            return insights

        accuracy = stats.accuracy

        if accuracy >= 95:
            insights.append(PerformanceInsight(
                category="accuracy",
                message=f"{accuracy:.0f}% accuracy - exceptional precision",
                severity="positive",
            ))
        elif accuracy >= 85:
            insights.append(PerformanceInsight(
                category="accuracy",
                message=f"{accuracy:.0f}% accuracy - solid performance",
                severity="positive",
            ))
        elif accuracy < 70:
            insights.append(PerformanceInsight(
                category="accuracy",
                message=f"{accuracy:.0f}% accuracy needs improvement",
                severity="needs_attention",
                recommendation="Slow down slightly to improve accuracy",
            ))

        return insights

    def _analyze_consistency(
        self,
        results: List[ProblemResult],
    ) -> List[PerformanceInsight]:
        """Analyze response time consistency."""
        insights = []

        if len(results) < 5:
            return insights

        # Calculate coefficient of variation for speed
        times = [r.time_ms for r in results]
        mean_time = sum(times) / len(times)
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        std_dev = variance ** 0.5
        cv = std_dev / mean_time if mean_time > 0 else 0

        if cv < 0.2:
            insights.append(PerformanceInsight(
                category="consistency",
                message="Highly consistent response times",
                severity="positive",
                recommendation="Excellent discipline - maintain this rhythm",
            ))
        elif cv > 0.5:
            insights.append(PerformanceInsight(
                category="consistency",
                message="High variance in response times",
                severity="needs_attention",
                recommendation="Work on maintaining steady pace",
            ))

        return insights

    def _analyze_fatigue(
        self,
        results: List[ProblemResult],
    ) -> List[PerformanceInsight]:
        """Detect fatigue patterns over session."""
        insights = []

        if len(results) < 10:
            return insights

        # Compare first half to second half
        mid = len(results) // 2
        first_half = results[:mid]
        second_half = results[mid:]

        # Accuracy comparison
        first_accuracy = sum(1 for r in first_half if r.correct) / len(first_half)
        second_accuracy = sum(1 for r in second_half if r.correct) / len(second_half)

        # Speed comparison
        first_avg_time = sum(r.time_ms for r in first_half) / len(first_half)
        second_avg_time = sum(r.time_ms for r in second_half) / len(second_half)

        accuracy_drop = first_accuracy - second_accuracy
        time_increase = (second_avg_time - first_avg_time) / first_avg_time

        if accuracy_drop > 0.15 or time_increase > 0.2:
            insights.append(PerformanceInsight(
                category="stamina",
                message="Performance declining - possible fatigue",
                severity="needs_attention",
                recommendation="Consider taking a short break",
            ))
        elif accuracy_drop < -0.1:
            insights.append(PerformanceInsight(
                category="stamina",
                message="Performance improving throughout session",
                severity="positive",
                recommendation="Good warm-up pattern",
            ))

        return insights

    def _analyze_categories(
        self,
        results: List[ProblemResult],
        stats: SessionStats,
    ) -> List[PerformanceInsight]:
        """Analyze performance by problem category."""
        insights = []

        if not stats.by_difficulty:
            return insights

        # Find weakest difficulty level
        worst_diff = None
        worst_accuracy = 1.0

        for diff, data in stats.by_difficulty.items():
            if data["total"] >= 3:  # Minimum sample size
                acc = data["correct"] / data["total"]
                if acc < worst_accuracy:
                    worst_accuracy = acc
                    worst_diff = diff

        if worst_diff and worst_accuracy < 0.7:
            insights.append(PerformanceInsight(
                category="difficulty",
                message=f"Difficulty {worst_diff} showing {worst_accuracy*100:.0f}% accuracy",
                severity="needs_attention",
                recommendation=f"More practice needed at D{worst_diff}",
            ))

        return insights

    def get_recommended_difficulty(
        self,
        results: List[ProblemResult],
        current_difficulty: int,
    ) -> Tuple[int, str]:
        """
        Recommend optimal difficulty based on recent performance.

        Returns:
            Tuple of (recommended_difficulty, reason)
        """
        if len(results) < 5:
            return current_difficulty, "Insufficient data for recommendation"

        recent = results[-10:]  # Last 10 problems

        accuracy = sum(1 for r in recent if r.correct) / len(recent)
        avg_speed_ratio = sum(r.speed_ratio for r in recent) / len(recent)

        # Increase difficulty if crushing it
        if accuracy >= 0.9 and avg_speed_ratio < 0.6:
            new_diff = min(current_difficulty + 1, 5)
            if new_diff > current_difficulty:
                return new_diff, "Exceptional performance - ready for next level"
            return current_difficulty, "Already at maximum difficulty"

        # Decrease if struggling
        if accuracy < 0.6 or avg_speed_ratio > 1.3:
            new_diff = max(current_difficulty - 1, 1)
            if new_diff < current_difficulty:
                return new_diff, "Consolidate fundamentals at lower difficulty"
            return current_difficulty, "Already at minimum difficulty"

        # Fine-tuning based on speed
        if accuracy >= 0.8 and avg_speed_ratio < 0.75:
            return min(current_difficulty + 1, 5), "Speed suggests readiness for increase"

        return current_difficulty, "Current difficulty is optimal"

    def calculate_session_score(self, stats: SessionStats) -> int:
        """
        Calculate overall session score (0-100).

        Factors:
        - Accuracy (40%)
        - Speed performance (30%)
        - Streak achievement (20%)
        - Volume (10%)
        """
        if stats.total_problems == 0:
            return 0

        # Accuracy component (40 points max)
        accuracy_score = min(stats.accuracy / 100 * 40, 40)

        # Speed component (30 points max) - based on problems under target
        # We'd need speed data for this, approximate from average
        avg_time = stats.average_time_ms
        # Assume D2 target of 4000ms as baseline
        speed_ratio = avg_time / 4000 if avg_time > 0 else 1.0
        if speed_ratio < 0.5:
            speed_score = 30
        elif speed_ratio < 0.75:
            speed_score = 25
        elif speed_ratio < 1.0:
            speed_score = 20
        else:
            speed_score = max(0, 15 - (speed_ratio - 1.0) * 10)

        # Streak component (20 points max)
        streak_score = min(stats.best_streak * 2, 20)

        # Volume component (10 points max)
        volume_score = min(stats.total_problems / 2, 10)

        return int(accuracy_score + speed_score + streak_score + volume_score)

    def generate_session_summary(
        self,
        results: List[ProblemResult],
        stats: SessionStats,
    ) -> Dict:
        """Generate comprehensive session summary."""
        score = self.calculate_session_score(stats)
        insights = self.analyze_session(results, stats)

        # Grade based on score
        if score >= 90:
            grade = "S"
            grade_comment = "Jane Street caliber session"
        elif score >= 80:
            grade = "A"
            grade_comment = "Interview ready performance"
        elif score >= 70:
            grade = "B"
            grade_comment = "Solid fundamentals"
        elif score >= 60:
            grade = "C"
            grade_comment = "Room for improvement"
        else:
            grade = "D"
            grade_comment = "More practice needed"

        return {
            "score": score,
            "grade": grade,
            "grade_comment": grade_comment,
            "stats": {
                "total": stats.total_problems,
                "correct": stats.correct_count,
                "accuracy": f"{stats.accuracy:.1f}%",
                "best_streak": stats.best_streak,
                "avg_time_ms": stats.average_time_ms,
            },
            "insights": [
                {
                    "category": i.category,
                    "message": i.message,
                    "severity": i.severity,
                    "recommendation": i.recommendation,
                }
                for i in insights
            ],
        }
