"""
API Cost Tracker for WHAM.
Tracks costs per model, per session, with warnings.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# Cost per request by model (USD)
COSTS = {
    # Gemini
    "gemini_flash": 0.002,
    "gemini_2.0_flash": 0.002,
    "gemini_2_0_flash": 0.002,

    # Perplexity
    "perplexity_sonar": 0.001,
    "perplexity_sonar_reasoning": 0.005,
    "sonar": 0.001,
    "sonar_reasoning": 0.005,

    # DeepSeek
    "deepseek_v3.1": 0.007,
    "deepseek_chat": 0.007,
    "deepseek_v3.2": 0.014,
    "deepseek_reasoner": 0.014,

    # Claude
    "claude_sonnet": 0.028,
    "claude_sonnet_4": 0.028,

    # Local (free)
    "local": 0.0,
    "cached": 0.0,
}

# Warning thresholds
WARN_SESSION_COST = 3.00  # Warn if session exceeds $3
WARN_HAND_COST = 0.015    # Warn if single hand exceeds $0.015


@dataclass
class CostEntry:
    """Single API call cost record."""
    model: str
    cost: float
    timestamp: float
    context: str = ""  # e.g., "hand_analysis", "ocr", "review"


@dataclass
class CostSummary:
    """Summary of costs for a session."""
    total_cost: float
    hand_count: int
    avg_cost_per_hand: float
    model_breakdown: Dict[str, float]
    warnings: List[str]
    session_duration_s: float


class CostTracker:
    """
    Track API costs per session.

    Features:
    - Per-model cost tracking
    - Session totals with warnings
    - ROI calculation vs poker winnings
    - Export for analysis
    """

    def __init__(self, warn_threshold: float = WARN_SESSION_COST):
        """
        Initialize cost tracker.

        Args:
            warn_threshold: Session cost threshold for warnings (default $3)
        """
        self.warn_threshold = warn_threshold
        self.session_start = time.time()
        self.entries: List[CostEntry] = []
        self.warnings: List[str] = []
        self.hand_count = 0

        # Poker tracking
        self.poker_profit_bb = 0.0
        self.bb_value_usd = 0.0  # e.g., $0.50 for $0.25/$0.50

    def add(
        self,
        model: str,
        cost: Optional[float] = None,
        context: str = ""
    ) -> float:
        """
        Record an API call cost.

        Args:
            model: Model name (must be in COSTS dict)
            cost: Override cost (uses COSTS lookup if None)
            context: What this call was for

        Returns:
            The cost added
        """
        actual_cost = cost if cost is not None else COSTS.get(model, 0.0)

        entry = CostEntry(
            model=model,
            cost=actual_cost,
            timestamp=time.time(),
            context=context
        )
        self.entries.append(entry)

        # Check warnings
        session_total = self.session_cost
        if session_total > self.warn_threshold:
            warning = f"Session cost ${session_total:.2f} exceeds ${self.warn_threshold:.2f}"
            if warning not in self.warnings:
                self.warnings.append(warning)
                logger.warning(warning)

        return actual_cost

    def add_hand(self, models_used: List[str]) -> float:
        """
        Record costs for a complete hand analysis.

        Args:
            models_used: List of models used for this hand

        Returns:
            Total cost for the hand
        """
        self.hand_count += 1
        hand_cost = 0.0

        for model in models_used:
            cost = self.add(model, context=f"hand_{self.hand_count}")
            hand_cost += cost

        if hand_cost > WARN_HAND_COST:
            warning = f"Hand {self.hand_count} cost ${hand_cost:.3f} > ${WARN_HAND_COST}"
            self.warnings.append(warning)
            logger.warning(warning)

        return hand_cost

    def record_poker_result(self, profit_bb: float, bb_value: float = 0.50):
        """
        Record poker profit for ROI calculation.

        Args:
            profit_bb: Profit in big blinds (positive = won)
            bb_value: Value of 1 big blind in USD
        """
        self.poker_profit_bb += profit_bb
        self.bb_value_usd = bb_value

    @property
    def session_cost(self) -> float:
        """Total session cost."""
        return sum(e.cost for e in self.entries)

    @property
    def model_breakdown(self) -> Dict[str, float]:
        """Cost breakdown by model."""
        breakdown = {}
        for entry in self.entries:
            breakdown[entry.model] = breakdown.get(entry.model, 0) + entry.cost
        return breakdown

    @property
    def poker_profit_usd(self) -> float:
        """Poker profit in USD."""
        return self.poker_profit_bb * self.bb_value_usd

    @property
    def net_profit(self) -> float:
        """Net profit after AI costs."""
        return self.poker_profit_usd - self.session_cost

    @property
    def roi(self) -> float:
        """
        ROI percentage.
        Positive = AI costs less than poker profit.
        """
        if self.poker_profit_usd <= 0:
            return -100.0 if self.session_cost > 0 else 0.0
        return ((self.poker_profit_usd - self.session_cost) / self.poker_profit_usd) * 100

    def get_summary(self) -> CostSummary:
        """Get complete session cost summary."""
        duration = time.time() - self.session_start
        avg_per_hand = self.session_cost / self.hand_count if self.hand_count else 0

        return CostSummary(
            total_cost=self.session_cost,
            hand_count=self.hand_count,
            avg_cost_per_hand=avg_per_hand,
            model_breakdown=self.model_breakdown,
            warnings=self.warnings.copy(),
            session_duration_s=duration
        )

    def format_summary(self) -> str:
        """Get formatted summary string."""
        summary = self.get_summary()

        lines = [
            f"Session Cost: ${summary.total_cost:.2f}",
            f"Hands: {summary.hand_count}",
            f"Avg/Hand: ${summary.avg_cost_per_hand:.3f}",
        ]

        if summary.model_breakdown:
            lines.append("By Model:")
            for model, cost in sorted(summary.model_breakdown.items(), key=lambda x: -x[1]):
                lines.append(f"  {model}: ${cost:.3f}")

        if self.poker_profit_bb != 0:
            lines.extend([
                f"Poker Profit: {self.poker_profit_bb:+.1f}bb (${self.poker_profit_usd:+.2f})",
                f"Net Profit: ${self.net_profit:+.2f}",
                f"ROI: {self.roi:+.1f}%"
            ])

        if summary.warnings:
            lines.append("Warnings:")
            for w in summary.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)

    def reset(self):
        """Reset for new session."""
        self.session_start = time.time()
        self.entries.clear()
        self.warnings.clear()
        self.hand_count = 0
        self.poker_profit_bb = 0.0

    def to_dict(self) -> dict:
        """Export as dictionary for JSON serialization."""
        return {
            "session_start": datetime.fromtimestamp(self.session_start).isoformat(),
            "total_cost": self.session_cost,
            "hand_count": self.hand_count,
            "model_breakdown": self.model_breakdown,
            "poker_profit_bb": self.poker_profit_bb,
            "poker_profit_usd": self.poker_profit_usd,
            "net_profit": self.net_profit,
            "roi": self.roi,
            "warnings": self.warnings,
            "entries": [
                {
                    "model": e.model,
                    "cost": e.cost,
                    "context": e.context,
                    "timestamp": e.timestamp
                }
                for e in self.entries
            ]
        }


# Test
def test_cost_tracker():
    """Test cost tracker."""
    print("=== Cost Tracker Test ===\n")

    tracker = CostTracker()

    # Simulate 10-hand session
    for i in range(10):
        # Each hand: Gemini OCR + DeepSeek analysis
        tracker.add_hand(["gemini_flash", "deepseek_v3.1"])

    # Record poker result: won 15bb at $0.50/bb
    tracker.record_poker_result(profit_bb=15, bb_value=0.50)

    print(tracker.format_summary())
    print()

    # Expected:
    # 10 hands × ($0.002 + $0.007) = $0.09 total
    # Poker profit: 15bb × $0.50 = $7.50
    # Net: $7.50 - $0.09 = $7.41
    # ROI: 98.8%


if __name__ == "__main__":
    test_cost_tracker()
