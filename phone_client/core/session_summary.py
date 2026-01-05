"""
Session Summary Manager - Tracks activity and generates end-of-day insights.
Provides daily recap with stats across all WHAM modes.
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    """Track stats for one session type."""
    count: int = 0
    duration_seconds: float = 0
    cost: float = 0
    details: Dict[str, Any] = field(default_factory=dict)
    research_queries: List[Dict] = field(default_factory=list)  # Phase 4: Research tracking


@dataclass
class DailySummary:
    """Complete day summary."""
    date: str
    poker: SessionStats
    homework: SessionStats
    meetings: SessionStats
    code_debug: SessionStats
    general: SessionStats  # Phase 4: For research queries and general tracking
    total_cost: float
    total_duration_seconds: float
    battery_time_seconds: float
    top_insight: str
    budget_remaining: float


class SessionSummaryManager:
    """
    Tracks and generates session summaries.

    Records activity across all WHAM modes:
    - Poker: hands played, mistakes detected
    - Homework: problems solved (local vs cloud)
    - Meetings: sessions, suggestions given
    - Code Debug: files checked, errors found
    """

    def __init__(self, config: dict):
        """
        Initialize session summary manager.

        Args:
            config: WHAM configuration dictionary
        """
        self.config = config
        self.daily_budget = config.get('cost_limits', {}).get('daily', 5.00)

        # Per-mode stats
        self.poker_stats = SessionStats()
        self.homework_stats = SessionStats()
        self.meeting_stats = SessionStats()
        self.debug_stats = SessionStats()
        self.general_stats = SessionStats()  # Phase 4: For research queries

        # Session tracking
        self.session_start = datetime.now()
        self.insights: List[Dict[str, Any]] = []

        # Summary save location
        self.summary_dir = Path(config.get('session_summary', {}).get(
            'save_dir', './logs/summaries'
        ))

        logger.info("Session summary manager initialized")

    def record_poker_hand(self, hand_result: dict):
        """
        Record poker hand stats.

        Args:
            hand_result: Dict with hand_num, cost, mistake_detected, mistake, etc.
        """
        self.poker_stats.count += 1
        self.poker_stats.cost += hand_result.get('cost', 0)

        # Track profit/loss
        if 'result_bb' not in self.poker_stats.details:
            self.poker_stats.details['total_profit_bb'] = 0
        self.poker_stats.details['total_profit_bb'] += hand_result.get('result_bb', 0)

        # Track mistakes for insights
        if hand_result.get('mistake_detected'):
            self.insights.append({
                'type': 'poker_mistake',
                'hand_num': hand_result.get('hand_num', 0),
                'mistake': hand_result.get('mistake', 'Unknown mistake'),
                'timestamp': datetime.now().isoformat()
            })

            # Count mistake types
            if 'mistakes' not in self.poker_stats.details:
                self.poker_stats.details['mistakes'] = {}

            mistake_type = hand_result.get('mistake', 'unknown').split(':')[0]
            self.poker_stats.details['mistakes'][mistake_type] = \
                self.poker_stats.details['mistakes'].get(mistake_type, 0) + 1

        logger.debug(f"Recorded poker hand #{hand_result.get('hand_num', 0)}")

    def record_homework(self, problem_result: dict):
        """
        Record homework problem stats.

        Args:
            problem_result: Dict with problem, answer, cost, solved_locally, etc.
        """
        self.homework_stats.count += 1
        self.homework_stats.cost += problem_result.get('cost', 0)

        # Track local vs cloud
        if 'local' not in self.homework_stats.details:
            self.homework_stats.details['local'] = 0
            self.homework_stats.details['cloud'] = 0

        if problem_result.get('solved_locally'):
            self.homework_stats.details['local'] += 1
        else:
            self.homework_stats.details['cloud'] += 1

        # Track problem types
        if 'types' not in self.homework_stats.details:
            self.homework_stats.details['types'] = {}

        problem_type = problem_result.get('type', 'unknown')
        self.homework_stats.details['types'][problem_type] = \
            self.homework_stats.details['types'].get(problem_type, 0) + 1

        logger.debug(f"Recorded homework problem: {problem_result.get('problem', '')[:30]}")

    def record_meeting(self, meeting_result: dict):
        """
        Record meeting session stats.

        Args:
            meeting_result: Dict with duration_seconds, cost, suggestions, etc.
        """
        self.meeting_stats.count += 1
        self.meeting_stats.duration_seconds += meeting_result.get('duration_seconds', 0)
        self.meeting_stats.cost += meeting_result.get('cost', 0)

        # Track suggestions given
        if 'suggestions_given' not in self.meeting_stats.details:
            self.meeting_stats.details['suggestions_given'] = 0

        self.meeting_stats.details['suggestions_given'] += \
            meeting_result.get('suggestions', 0)

        # Track meeting types
        if 'types' not in self.meeting_stats.details:
            self.meeting_stats.details['types'] = {}

        meeting_type = meeting_result.get('type', 'general')
        self.meeting_stats.details['types'][meeting_type] = \
            self.meeting_stats.details['types'].get(meeting_type, 0) + 1

        logger.debug(f"Recorded meeting: {meeting_result.get('duration_seconds', 0):.0f}s")

    def record_debug(self, debug_result: dict):
        """
        Record code debug session stats.

        Args:
            debug_result: Dict with language, errors_found, cost, etc.
        """
        self.debug_stats.count += 1
        self.debug_stats.cost += debug_result.get('cost', 0)

        # Track errors found
        if 'errors_found' not in self.debug_stats.details:
            self.debug_stats.details['errors_found'] = 0

        self.debug_stats.details['errors_found'] += \
            debug_result.get('errors_found', 0)

        # Track languages
        if 'languages' not in self.debug_stats.details:
            self.debug_stats.details['languages'] = {}

        language = debug_result.get('language', 'unknown')
        self.debug_stats.details['languages'][language] = \
            self.debug_stats.details['languages'].get(language, 0) + 1

        logger.debug(f"Recorded debug: {language}")

    def record_research(
        self,
        query: str,
        cost: float,
        citations: List,
        context: Dict[str, Any]
    ):
        """
        Record research query (Phase 4).

        Args:
            query: Research question
            cost: Query cost
            citations: List of citations/sources
            context: Mode and skill level context
        """
        self.general_stats.research_queries.append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'cost': cost,
            'citations': citations,
            'context': context
        })
        self.general_stats.cost += cost
        self.general_stats.count += 1

        logger.debug(f"Recorded research query: {query[:50]}...")

    def generate_top_insight(self) -> str:
        """
        Generate most important insight from session.

        Returns:
            String with the top insight/pattern detected
        """
        if not self.insights:
            # Check for patterns without explicit insights
            if self.poker_stats.count > 0:
                profit = self.poker_stats.details.get('total_profit_bb', 0)
                if profit > 0:
                    return f"Poker: +{profit:.1f}bb across {self.poker_stats.count} hands"
                elif profit < 0:
                    return f"Poker: {profit:.1f}bb - review session for leaks"

            if self.homework_stats.count > 0:
                local_pct = (
                    self.homework_stats.details.get('local', 0) /
                    max(self.homework_stats.count, 1)
                ) * 100
                return f"Homework: {local_pct:.0f}% solved locally (saving ${self.homework_stats.cost:.2f})"

            return "No significant patterns detected."

        # Priority: Poker mistakes > Homework struggles > Meeting patterns
        poker_mistakes = [i for i in self.insights if i['type'] == 'poker_mistake']

        if poker_mistakes:
            # Find most common mistake type
            mistake_types: Dict[str, int] = {}
            for m in poker_mistakes:
                mtype = m['mistake'].split(':')[0] if ':' in m['mistake'] else m['mistake']
                mistake_types[mtype] = mistake_types.get(mtype, 0) + 1

            most_common = max(mistake_types, key=mistake_types.get)
            return f"Poker: {most_common} ({mistake_types[most_common]}x this session)"

        return "Session completed successfully."

    def generate_summary(self) -> DailySummary:
        """
        Generate complete session summary.

        Returns:
            DailySummary dataclass with all stats
        """
        total_cost = (
            self.poker_stats.cost +
            self.homework_stats.cost +
            self.meeting_stats.cost +
            self.debug_stats.cost +
            self.general_stats.cost
        )

        total_duration = (datetime.now() - self.session_start).total_seconds()

        return DailySummary(
            date=datetime.now().strftime("%Y-%m-%d"),
            poker=self.poker_stats,
            homework=self.homework_stats,
            meetings=self.meeting_stats,
            code_debug=self.debug_stats,
            general=self.general_stats,
            total_cost=total_cost,
            total_duration_seconds=total_duration,
            battery_time_seconds=total_duration,  # TODO: Track actual battery usage
            top_insight=self.generate_top_insight(),
            budget_remaining=self.daily_budget - total_cost
        )

    def format_summary(self, summary: Optional[DailySummary] = None) -> str:
        """
        Format summary for HUD display.

        Args:
            summary: Optional pre-generated summary, generates if None

        Returns:
            Formatted string for display
        """
        if summary is None:
            summary = self.generate_summary()

        lines = [
            "WHAM SESSION SUMMARY",
            "-" * 40,
            ""
        ]

        # Poker stats
        if summary.poker.count > 0:
            profit = summary.poker.details.get('total_profit_bb', 0)
            profit_str = f"{profit:+.1f}bb" if profit != 0 else "0bb"
            lines.append(f"Poker: {summary.poker.count} hands, {profit_str}, ${summary.poker.cost:.2f}")

        # Homework stats
        if summary.homework.count > 0:
            local = summary.homework.details.get('local', 0)
            cloud = summary.homework.details.get('cloud', 0)
            lines.append(f"Homework: {summary.homework.count} problems ({local} local, {cloud} cloud)")

        # Meeting stats
        if summary.meetings.count > 0:
            suggestions = summary.meetings.details.get('suggestions_given', 0)
            duration_min = summary.meetings.duration_seconds / 60
            lines.append(f"Meetings: {summary.meetings.count} sessions, {duration_min:.0f}min, {suggestions} suggestions")

        # Debug stats
        if summary.code_debug.count > 0:
            errors = summary.code_debug.details.get('errors_found', 0)
            lines.append(f"Code Debug: {summary.code_debug.count} checks, {errors} errors found")

        # Cost summary
        lines.extend([
            "",
            f"Cost: ${summary.total_cost:.2f} / ${self.daily_budget:.2f} budget",
            f"Time: {summary.total_duration_seconds / 3600:.1f}h active",
            "",
            f"Insight: {summary.top_insight}"
        ])

        return "\n".join(lines)

    def save_summary(self, summary: Optional[DailySummary] = None) -> str:
        """
        Save summary to JSON file.

        Args:
            summary: Optional pre-generated summary

        Returns:
            Path to saved file
        """
        if summary is None:
            summary = self.generate_summary()

        # Ensure directory exists
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with date
        filename = f"{summary.date}.json"
        filepath = self.summary_dir / filename

        # Convert to dict (handle nested dataclasses)
        summary_dict = {
            'date': summary.date,
            'poker': asdict(summary.poker),
            'homework': asdict(summary.homework),
            'meetings': asdict(summary.meetings),
            'code_debug': asdict(summary.code_debug),
            'general': asdict(summary.general),
            'total_cost': summary.total_cost,
            'total_duration_seconds': summary.total_duration_seconds,
            'battery_time_seconds': summary.battery_time_seconds,
            'top_insight': summary.top_insight,
            'budget_remaining': summary.budget_remaining,
            'insights': self.insights
        }

        with open(filepath, 'w') as f:
            json.dump(summary_dict, f, indent=2)

        logger.info(f"Session summary saved to {filepath}")
        return str(filepath)

    def reset(self):
        """Reset all stats for a new session."""
        self.poker_stats = SessionStats()
        self.homework_stats = SessionStats()
        self.meeting_stats = SessionStats()
        self.debug_stats = SessionStats()
        self.general_stats = SessionStats()
        self.session_start = datetime.now()
        self.insights.clear()
        logger.info("Session stats reset")


# Test
def test_session_summary():
    """Test session summary manager."""
    print("=== Session Summary Test ===\n")

    config = {
        'cost_limits': {'daily': 5.00},
        'session_summary': {'save_dir': './test_logs/summaries'}
    }

    manager = SessionSummaryManager(config)

    # Simulate poker session
    for i in range(5):
        manager.record_poker_hand({
            'hand_num': i + 1,
            'cost': 0.009,
            'result_bb': 2.5 if i % 2 == 0 else -1.5,
            'mistake_detected': i == 2,
            'mistake': 'MISTAKE: Too passive on river'
        })

    # Simulate homework
    manager.record_homework({
        'problem': '2x + 5 = 15',
        'answer': 'x = 5',
        'cost': 0,
        'solved_locally': True,
        'type': 'algebra'
    })

    manager.record_homework({
        'problem': 'Integral of sin(x)',
        'answer': '-cos(x) + C',
        'cost': 0.007,
        'solved_locally': False,
        'type': 'calculus'
    })

    # Generate and display summary
    summary = manager.generate_summary()
    formatted = manager.format_summary(summary)

    print(formatted)
    print()

    # Save summary
    path = manager.save_summary(summary)
    print(f"Saved to: {path}")


if __name__ == "__main__":
    test_session_summary()
