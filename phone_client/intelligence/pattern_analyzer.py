"""
Pattern Analyzer - Detect usage patterns from historical session data.

Analyzes logs/summaries/{YYYY-MM-DD}.json files to detect:
- Time patterns: When user typically does activities
- Cost patterns: Spending trends and high-cost days
- Usage patterns: Feature usage frequency and peak hours
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from pathlib import Path
from collections import defaultdict


@dataclass
class TimePattern:
    """Time-based usage pattern."""
    activity: str
    typical_time: str  # "14:00" (hour:minute)
    day_of_week: Optional[int] = None  # 0=Monday, 6=Sunday
    frequency: int = 0  # Times observed
    confidence: float = 0.0  # 0.0-1.0


@dataclass
class CostPattern:
    """Cost usage pattern."""
    activity: str
    avg_cost: float
    daily_trend: str  # "increasing", "decreasing", "stable"
    high_cost_days: List[str] = field(default_factory=list)  # ["Sunday", "Saturday"]


@dataclass
class UsagePattern:
    """General usage pattern."""
    feature: str
    avg_uses_per_day: float
    peak_hours: List[int] = field(default_factory=list)  # [14, 15, 16] = 2pm-4pm
    preferred_mode: Optional[str] = None


class PatternAnalyzer:
    """
    Analyze user behavior patterns from historical data.

    Detects:
    - Time patterns: "User always studies 2-4pm weekdays"
    - Cost patterns: "User spends more on Sundays (poker)"
    - Feature patterns: "User prefers local engines for algebra"
    - Context patterns: "User captures ideas after focus sessions"
    """

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "logs" / "summaries"
        self.data_dir = data_dir
        self.min_observations = 3  # Minimum times to establish pattern

    async def analyze_all_patterns(self, days: int = 30) -> Dict:
        """
        Analyze all patterns from last N days.

        Returns:
        {
            "time_patterns": [TimePattern, ...],
            "cost_patterns": [CostPattern, ...],
            "usage_patterns": [UsagePattern, ...],
            "confidence": 0.85  # Overall confidence in patterns
        }
        """
        # Load session data from last N days
        sessions = self._load_sessions(days)

        if len(sessions) < 7:
            return {
                "time_patterns": [],
                "cost_patterns": [],
                "usage_patterns": [],
                "confidence": 0.0,
                "message": "Not enough data (need at least 7 days)"
            }

        time_patterns = self._analyze_time_patterns(sessions)
        cost_patterns = self._analyze_cost_patterns(sessions)
        usage_patterns = self._analyze_usage_patterns(sessions)

        # Calculate overall confidence
        confidence = self._calculate_confidence(sessions, time_patterns)

        return {
            "time_patterns": time_patterns,
            "cost_patterns": cost_patterns,
            "usage_patterns": usage_patterns,
            "confidence": confidence
        }

    def _load_sessions(self, days: int) -> List[Dict]:
        """Load session data from last N days."""
        sessions = []
        cutoff = datetime.now() - timedelta(days=days)

        if not self.data_dir.exists():
            return []

        for file in self.data_dir.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    session_date = datetime.fromisoformat(data.get('date', '2000-01-01'))

                    if session_date >= cutoff:
                        sessions.append(data)
            except Exception as e:
                # Skip malformed files
                continue

        return sorted(sessions, key=lambda x: x.get('date', ''))

    def _analyze_time_patterns(self, sessions: List[Dict]) -> List[TimePattern]:
        """Detect time-based activity patterns."""
        patterns = []

        # Group activities by hour and day of week
        activity_times = defaultdict(lambda: defaultdict(int))

        for session in sessions:
            try:
                date = datetime.fromisoformat(session.get('date', ''))
                day_of_week = date.weekday()

                # Check each activity type
                for activity in ['poker', 'homework', 'focus', 'meetings']:
                    if activity in session and session[activity].get('count', 0) > 0:
                        # Use start_hour if available, otherwise assume activity happened during the day
                        hour = session.get(f'{activity}_start_hour', 12)
                        activity_times[activity][(day_of_week, hour)] += 1
            except:
                continue

        # Find recurring patterns
        for activity, times in activity_times.items():
            if not times:
                continue

            # Find most common time
            most_common = max(times.items(), key=lambda x: x[1])
            (day, hour), frequency = most_common

            if frequency >= self.min_observations:
                confidence = min(frequency / len(sessions), 1.0)

                patterns.append(TimePattern(
                    activity=activity,
                    typical_time=f"{hour:02d}:00",
                    day_of_week=day,
                    frequency=frequency,
                    confidence=confidence
                ))

        return sorted(patterns, key=lambda x: x.confidence, reverse=True)

    def _analyze_cost_patterns(self, sessions: List[Dict]) -> List[CostPattern]:
        """Detect cost usage patterns."""
        patterns = []

        # Analyze cost by activity
        activity_costs = defaultdict(list)
        day_costs = defaultdict(lambda: defaultdict(list))

        for session in sessions:
            try:
                date = datetime.fromisoformat(session.get('date', ''))
                day_name = date.strftime('%A')

                # Per activity
                for activity in ['poker', 'homework']:
                    if activity in session:
                        cost = session[activity].get('cost', 0)
                        if cost > 0:
                            activity_costs[activity].append(cost)
                            day_costs[activity][day_name].append(cost)
            except:
                continue

        # Analyze each activity
        for activity, costs in activity_costs.items():
            if len(costs) < 3:
                continue

            avg_cost = sum(costs) / len(costs)

            # Determine trend
            if len(costs) >= 7:
                recent_avg = sum(costs[-7:]) / 7
                old_avg = sum(costs[:-7]) / len(costs[:-7]) if len(costs) > 7 else avg_cost

                if recent_avg > old_avg * 1.2:
                    trend = "increasing"
                elif recent_avg < old_avg * 0.8:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            # Find high-cost days
            day_avgs = {}
            for day, day_cost_list in day_costs[activity].items():
                if day_cost_list:
                    day_avgs[day] = sum(day_cost_list) / len(day_cost_list)

            high_cost_days = [day for day, avg in day_avgs.items() if avg > avg_cost * 1.5]

            patterns.append(CostPattern(
                activity=activity,
                avg_cost=avg_cost,
                daily_trend=trend,
                high_cost_days=high_cost_days[:3]  # Top 3
            ))

        return patterns

    def _analyze_usage_patterns(self, sessions: List[Dict]) -> List[UsagePattern]:
        """Detect feature usage patterns."""
        patterns = []

        # Track feature usage
        feature_usage = defaultdict(lambda: {'count': 0, 'hours': []})

        for session in sessions:
            try:
                date = datetime.fromisoformat(session.get('date', ''))

                # Count captures
                captures = session.get('captures', [])
                if captures:
                    feature_usage['quick_capture']['count'] += len(captures)
                    # Estimate hours from timestamps if available
                    for capture in captures:
                        if isinstance(capture, dict) and 'timestamp' in capture:
                            try:
                                hour = datetime.fromisoformat(capture['timestamp']).hour
                                feature_usage['quick_capture']['hours'].append(hour)
                            except:
                                pass

                # Focus sessions
                if 'focus' in session and session['focus'].get('count', 0) > 0:
                    feature_usage['focus_mode']['count'] += session['focus']['count']
            except:
                continue

        # Calculate patterns
        days = len(sessions) if sessions else 1

        for feature, data in feature_usage.items():
            if data['count'] < 3:
                continue

            avg_uses = data['count'] / days

            # Find peak hours
            if data['hours']:
                hour_counts = defaultdict(int)
                for hour in data['hours']:
                    hour_counts[hour] += 1

                peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                peak_hours = [h for h, _ in peak_hours]
            else:
                peak_hours = []

            patterns.append(UsagePattern(
                feature=feature,
                avg_uses_per_day=avg_uses,
                peak_hours=peak_hours
            ))

        return patterns

    def _calculate_confidence(self, sessions: List[Dict], patterns: List[TimePattern]) -> float:
        """Calculate overall confidence in patterns."""
        if not sessions or not patterns:
            return 0.0

        # More data = higher confidence
        data_confidence = min(len(sessions) / 30, 1.0)  # Max at 30 days

        # More patterns = higher confidence
        pattern_confidence = min(len(patterns) / 5, 1.0)  # Max at 5 patterns

        # Average pattern confidence
        avg_pattern_conf = sum(p.confidence for p in patterns) / len(patterns) if patterns else 0.0

        return (data_confidence + pattern_confidence + avg_pattern_conf) / 3

    def calculate_skill_metrics(self, sessions: List[Dict], days: int = 30) -> Dict[str, any]:
        """
        Calculate skill progression from EXISTING session data (Phase 4).

        Returns:
        {
            'poker': {
                'skill_level': 'beginner'|'intermediate'|'advanced',
                'trend': 'improving'|'stable'|'declining',
                'metrics': {...},
                'confidence': 0.85
            },
            'homework': {...},
            'focus': {...},
            'overall_consistency': 0.75
        }
        """
        if len(sessions) < 3:
            return {'error': 'Need at least 3 days of data', 'confidence': 0.0}

        metrics = {}

        # Poker skill from win rate and mistake patterns
        poker_sessions = [s for s in sessions if s.get('poker', {}).get('count', 0) > 0]
        if len(poker_sessions) >= 3:
            metrics['poker'] = self._calculate_poker_skill(poker_sessions)

        # Homework skill from local solve rate (mastery indicator)
        hw_sessions = [s for s in sessions if s.get('homework', {}).get('count', 0) > 0]
        if len(hw_sessions) >= 3:
            metrics['homework'] = self._calculate_homework_skill(hw_sessions)

        # Focus skill from completion rate and efficiency
        focus_sessions = [s for s in sessions if s.get('focus', {}).get('count', 0) > 0]
        if len(focus_sessions) >= 3:
            metrics['focus'] = self._calculate_focus_skill(focus_sessions)

        # Overall consistency (activity regularity)
        metrics['overall_consistency'] = self._calculate_consistency(sessions)
        metrics['confidence'] = self._calculate_overall_confidence(metrics, len(sessions))

        return metrics

    def _calculate_poker_skill(self, sessions: List[Dict]) -> Dict:
        """Compute poker skill from mistake rate and profit trend."""
        total_hands = sum(s['poker']['count'] for s in sessions)

        # Extract mistakes from details
        total_mistakes = sum(
            sum(s.get('poker', {}).get('details', {}).get('mistakes', {}).values())
            for s in sessions
        )

        mistake_rate = total_mistakes / total_hands if total_hands > 0 else 0

        # Extract profit trend
        profits = [
            s['poker']['details'].get('profit_bb', 0)
            for s in sessions
            if 'poker' in s and 'details' in s['poker']
        ]
        avg_profit_bb100 = (sum(profits) / total_hands * 100) if total_hands > 0 else 0

        # Skill level thresholds
        if mistake_rate < 0.05 and avg_profit_bb100 > 5:
            skill_level = 'advanced'
        elif mistake_rate < 0.15 and avg_profit_bb100 > 0:
            skill_level = 'intermediate'
        else:
            skill_level = 'beginner'

        # Trend: first half vs second half
        trend = 'stable'
        if len(profits) >= 6:
            mid = len(profits) // 2
            first_avg = sum(profits[:mid]) / mid
            second_avg = sum(profits[mid:]) / (len(profits) - mid)

            if second_avg > first_avg * 1.2:
                trend = 'improving'
            elif second_avg < first_avg * 0.8:
                trend = 'declining'

        # Confidence based on sample size
        confidence = min(total_hands / 100, 1.0) if total_hands > 0 else 0.0

        return {
            'skill_level': skill_level,
            'trend': trend,
            'metrics': {
                'mistake_rate': mistake_rate,
                'win_rate_bb_per_100': avg_profit_bb100,
                'sample_size': total_hands
            },
            'confidence': confidence
        }

    def _calculate_homework_skill(self, sessions: List[Dict]) -> Dict:
        """Compute homework skill from local solve rate (higher = better mastery)."""
        local_rates = [
            s.get('homework', {}).get('details', {}).get('local_percentage', 0) / 100
            for s in sessions
            if s.get('homework', {}).get('details', {}).get('local_percentage', 0) > 0
        ]

        avg_local_rate = sum(local_rates) / len(local_rates) if local_rates else 0

        # Skill thresholds
        if avg_local_rate > 0.75:
            skill_level = 'advanced'
        elif avg_local_rate > 0.5:
            skill_level = 'intermediate'
        else:
            skill_level = 'beginner'

        # Trend analysis
        trend = 'stable'
        if len(local_rates) >= 6:
            mid = len(local_rates) // 2
            recent = sum(local_rates[mid:]) / (len(local_rates) - mid)
            older = sum(local_rates[:mid]) / mid

            if recent > older * 1.1:
                trend = 'improving'
            elif recent < older * 0.9:
                trend = 'declining'

        confidence = min(len(local_rates) / 10, 1.0)

        return {
            'skill_level': skill_level,
            'trend': trend,
            'metrics': {
                'local_solve_rate': avg_local_rate,
                'avg_problems_per_day': sum(s['homework']['count'] for s in sessions) / len(sessions)
            },
            'confidence': confidence
        }

    def _calculate_focus_skill(self, sessions: List[Dict]) -> Dict:
        """Compute focus skill from completion rate and session consistency."""
        focus_stats = [s.get('focus', {}) for s in sessions if s.get('focus')]

        if not focus_stats:
            return {'skill_level': 'beginner', 'trend': 'stable', 'confidence': 0.0}

        # Calculate completion rate (sessions with >0 duration)
        completed = sum(1 for f in focus_stats if f.get('duration_seconds', 0) > 0)
        completion_rate = completed / len(focus_stats)

        # Average session duration
        avg_duration = sum(f.get('duration_seconds', 0) for f in focus_stats) / len(focus_stats)

        # Skill thresholds
        if completion_rate > 0.8 and avg_duration > 1800:  # >30min avg
            skill_level = 'advanced'
        elif completion_rate > 0.6 and avg_duration > 900:  # >15min avg
            skill_level = 'intermediate'
        else:
            skill_level = 'beginner'

        confidence = min(len(focus_stats) / 10, 1.0)

        return {
            'skill_level': skill_level,
            'trend': 'stable',  # TODO: Add trend detection
            'metrics': {
                'completion_rate': completion_rate,
                'avg_duration_minutes': avg_duration / 60
            },
            'confidence': confidence
        }

    def _calculate_consistency(self, sessions: List[Dict]) -> float:
        """Calculate activity consistency (0-1)."""
        # Check how many days have at least one activity
        active_days = sum(1 for s in sessions if s.get('total_cost', 0) > 0 or len(s.get('captures', [])) > 0)
        return active_days / len(sessions) if sessions else 0.0

    def _calculate_overall_confidence(self, metrics: Dict, session_count: int) -> float:
        """Calculate overall confidence in skill estimates."""
        # Base confidence on data volume
        data_confidence = min(session_count / 30, 1.0)

        # Average skill confidences
        skill_confidences = [
            m.get('confidence', 0)
            for m in metrics.values()
            if isinstance(m, dict) and 'confidence' in m
        ]
        avg_skill_conf = sum(skill_confidences) / len(skill_confidences) if skill_confidences else 0.0

        return (data_confidence + avg_skill_conf) / 2
