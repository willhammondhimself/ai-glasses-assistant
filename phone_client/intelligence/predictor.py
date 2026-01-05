"""
Predictive Engine - Generate proactive suggestions based on usage patterns.

Suggests actions based on:
- Time patterns (it's 2pm, you usually study now)
- Unusual behavior (you haven't captured anything today)
- Context (poker session costs are high, enable budget mode?)
"""
from typing import List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

from .pattern_analyzer import PatternAnalyzer, TimePattern


@dataclass
class Suggestion:
    """Proactive suggestion for user."""
    type: str  # "focus_session", "poker_prep", "capture_reminder", etc.
    title: str
    message: str
    confidence: float  # 0.0-1.0
    action: Optional[str] = None  # Command to execute if accepted
    priority: str = "normal"  # "low", "normal", "high"


class PredictiveEngine:
    """
    Proactive suggestion engine.

    Suggests actions based on:
    - Time patterns (it's 2pm, you usually study now)
    - Unusual behavior (you haven't captured anything today)
    - Context (poker session costs are high, enable budget mode?)
    - Reminders (homework due tomorrow)
    """

    def __init__(self):
        self.analyzer = PatternAnalyzer()
        self.last_suggestions = []
        self.suggestion_cooldown = timedelta(hours=2)  # Don't spam

    async def get_suggestions(self) -> List[Suggestion]:
        """
        Get proactive suggestions based on current context.

        Returns list of suggestions sorted by priority/confidence.
        """
        suggestions = []

        # Analyze patterns
        patterns = await self.analyzer.analyze_all_patterns()

        if patterns['confidence'] < 0.3:
            return []  # Not enough data

        now = datetime.now()

        # Check time-based suggestions
        time_suggestions = await self._check_time_patterns(patterns['time_patterns'], now)
        suggestions.extend(time_suggestions)

        # Check usage anomalies
        anomaly_suggestions = await self._check_usage_anomalies(patterns['usage_patterns'], now)
        suggestions.extend(anomaly_suggestions)

        # Check cost patterns
        cost_suggestions = await self._check_cost_patterns(patterns['cost_patterns'])
        suggestions.extend(cost_suggestions)

        # Filter out recently shown suggestions
        suggestions = self._filter_recent(suggestions)

        # Sort by priority and confidence
        suggestions.sort(key=lambda s: (
            {'high': 3, 'normal': 2, 'low': 1}[s.priority],
            s.confidence
        ), reverse=True)

        # Store for cooldown tracking
        self.last_suggestions = [(s, now) for s in suggestions]

        return suggestions[:3]  # Return top 3

    async def _check_time_patterns(self, patterns: List[TimePattern], now: datetime) -> List[Suggestion]:
        """Check if current time matches typical activity patterns."""
        suggestions = []

        current_hour = now.hour
        current_day = now.weekday()

        for pattern in patterns:
            try:
                # Check if time matches
                pattern_hour = int(pattern.typical_time.split(':')[0])

                # Within 30-minute window
                if abs(pattern_hour - current_hour) <= 0:
                    # Check day of week (if pattern specifies)
                    if pattern.day_of_week is None or pattern.day_of_week == current_day:
                        suggestions.append(Suggestion(
                            type=f"{pattern.activity}_reminder",
                            title=f"{pattern.activity.title()} Time",
                            message=f"You typically {pattern.activity} around {pattern.typical_time}. Ready to start?",
                            confidence=pattern.confidence,
                            action=f"start_{pattern.activity}",
                            priority="normal"
                        ))
            except:
                continue

        return suggestions

    async def _check_usage_anomalies(self, patterns: List, now: datetime) -> List[Suggestion]:
        """Check for unusual behavior patterns."""
        suggestions = []

        try:
            # Check if user hasn't captured anything today
            from pathlib import Path
            import json

            captures_path = Path(__file__).parent.parent / "captures" / "captures.json"
            if captures_path.exists():
                with open(captures_path) as f:
                    captures = json.load(f)

                # Count today's captures
                today_captures = 0
                for capture_id, capture in captures.items():
                    if isinstance(capture, dict) and 'created_at' in capture:
                        try:
                            capture_date = datetime.fromisoformat(capture['created_at'])
                            if capture_date.date() == now.date():
                                today_captures += 1
                        except:
                            pass

                # Find typical capture usage
                capture_pattern = next((p for p in patterns if p.feature == 'quick_capture'), None)

                if capture_pattern and capture_pattern.avg_uses_per_day > 2:
                    # User normally captures things
                    if today_captures == 0 and now.hour >= 12:
                        suggestions.append(Suggestion(
                            type="capture_reminder",
                            title="Quiet Day",
                            message=f"You typically capture {capture_pattern.avg_uses_per_day:.0f} notes per day. Anything on your mind?",
                            confidence=0.7,
                            action="open_quick_capture",
                            priority="low"
                        ))
        except:
            pass  # Gracefully handle missing files

        return suggestions

    async def _check_cost_patterns(self, patterns: List) -> List[Suggestion]:
        """Check for cost-related suggestions."""
        suggestions = []

        try:
            from pathlib import Path
            import json

            # Load today's summary if available
            summaries_dir = Path(__file__).parent.parent.parent / "logs" / "summaries"
            today_file = summaries_dir / f"{datetime.now().strftime('%Y-%m-%d')}.json"

            if today_file.exists():
                with open(today_file) as f:
                    today_summary = json.load(f)
                    today_cost = today_summary.get('total_cost', 0)
            else:
                today_cost = 0

            for pattern in patterns:
                # Check if cost is trending up
                if pattern.daily_trend == "increasing" and pattern.avg_cost > 1.0:
                    suggestions.append(Suggestion(
                        type="cost_warning",
                        title="Cost Increasing",
                        message=f"{pattern.activity.title()} costs averaging ${pattern.avg_cost:.2f} lately. Consider budget mode?",
                        confidence=0.8,
                        action="enable_budget_mode",
                        priority="high"
                    ))

                # Check if today is high-cost day
                today_name = datetime.now().strftime('%A')
                if today_name in pattern.high_cost_days:
                    suggestions.append(Suggestion(
                        type="cost_warning",
                        title="High Cost Day",
                        message=f"{pattern.activity.title()} sessions cost more on {today_name}s. Budget mode recommended.",
                        confidence=0.7,
                        action="enable_budget_mode",
                        priority="normal"
                    ))
        except:
            pass  # Gracefully handle missing files

        return suggestions

    def _filter_recent(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Filter out suggestions shown recently."""
        now = datetime.now()

        # Remove old suggestions from tracking
        self.last_suggestions = [
            (s, t) for s, t in self.last_suggestions
            if now - t < self.suggestion_cooldown
        ]

        recent_types = {s.type for s, _ in self.last_suggestions}

        return [s for s in suggestions if s.type not in recent_types]
