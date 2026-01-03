"""
Cost Tracker

Budget-aware cost tracking with zone-based enforcement.
Provides real-time budget monitoring and cost optimization hints.
"""

import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum


class BudgetZone(Enum):
    """Budget zones for cost management."""
    GREEN = "green"      # 0-60% - Normal operation
    YELLOW = "yellow"    # 60-85% - Caution, prefer local engines
    RED = "red"          # 85-100% - Critical, force local engines
    CRITICAL = "critical"  # >100% - Emergency, block expensive calls


@dataclass
class CostMetrics:
    """Current cost metrics snapshot."""
    total_cost: float
    budget_used_pct: float
    zone: BudgetZone
    calls_today: int
    calls_this_month: int
    avg_cost_per_call: float
    remaining_budget: float
    days_remaining: int
    projected_monthly: float


class CostTracker:
    """
    Budget-aware cost tracker with zone-based enforcement.

    Features:
    - Monthly budget tracking with zones
    - Real token extraction from Anthropic responses
    - Cost projection and optimization hints
    - Per-engine and per-model breakdown
    """

    # Default monthly budget in dollars
    DEFAULT_MONTHLY_BUDGET = 30.0

    # Budget zone thresholds (percentage of monthly budget)
    ZONE_THRESHOLDS = {
        BudgetZone.GREEN: (0.0, 0.60),      # $0-18
        BudgetZone.YELLOW: (0.60, 0.85),    # $18-25.50
        BudgetZone.RED: (0.85, 1.0),        # $25.50-30
        BudgetZone.CRITICAL: (1.0, float('inf'))
    }

    # Anthropic pricing (per 1M tokens) - updated for current models
    PRICING = {
        # Claude 4 series
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        # Claude 3.5 series
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
        # Claude 3 series (legacy)
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        db_path: Optional[str] = None,
        monthly_budget: Optional[float] = None
    ):
        """
        Initialize cost tracker.

        Args:
            db_path: Path to SQLite database
            monthly_budget: Monthly budget in dollars
        """
        self.db_path = db_path or os.getenv("DATABASE_PATH", "query_history.db")
        self.monthly_budget = monthly_budget or float(os.getenv("MONTHLY_BUDGET", self.DEFAULT_MONTHLY_BUDGET))
        self._init_db()

    def _init_db(self):
        """Create the api_calls table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    engine TEXT NOT NULL,
                    model TEXT,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0.0,
                    latency_ms REAL DEFAULT 0.0,
                    success BOOLEAN DEFAULT TRUE,
                    cached BOOLEAN DEFAULT FALSE,
                    problem_type TEXT,
                    complexity TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_timestamp ON api_calls(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_engine ON api_calls(engine)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_model ON api_calls(model)")
            conn.commit()

    def extract_tokens(self, response: Dict[str, Any]) -> tuple[int, int]:
        """
        Extract token counts from Anthropic API response.

        Args:
            response: Raw API response dict

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        usage = response.get("usage", {})
        return (
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0)
        )

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost from token usage.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in dollars
        """
        pricing = self.PRICING.get(model, self.PRICING[self.DEFAULT_MODEL])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def record_call(
        self,
        engine: str,
        model: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        success: bool = True,
        cached: bool = False,
        problem_type: Optional[str] = None,
        complexity: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Record an API call with cost tracking.

        Args:
            engine: Name of the engine
            model: Model used (for cost calculation)
            input_tokens: Input token count
            output_tokens: Output token count
            latency_ms: Response latency
            success: Whether call succeeded
            cached: Whether result was cached
            problem_type: Type of problem
            complexity: Problem complexity level
            metadata: Additional metadata

        Returns:
            ID of the recorded call
        """
        # Calculate cost (0 if cached or no model)
        cost = 0.0
        if model and not cached:
            cost = self.calculate_cost(model, input_tokens, output_tokens)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO api_calls
                (engine, model, input_tokens, output_tokens, cost, latency_ms,
                 success, cached, problem_type, complexity, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    engine, model, input_tokens, output_tokens, cost, latency_ms,
                    success, cached, problem_type, complexity,
                    str(metadata) if metadata else None
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_current_zone(self) -> BudgetZone:
        """
        Get current budget zone based on monthly spending.

        Returns:
            Current BudgetZone
        """
        metrics = self.get_monthly_metrics()

        for zone, (low, high) in self.ZONE_THRESHOLDS.items():
            if low <= metrics.budget_used_pct < high:
                return zone

        return BudgetZone.CRITICAL

    def get_monthly_metrics(self) -> CostMetrics:
        """
        Get current month's cost metrics.

        Returns:
            CostMetrics with current spending data
        """
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        days_in_month = 30  # Approximation
        days_elapsed = (now - month_start).days + 1
        days_remaining = max(1, days_in_month - days_elapsed)

        with sqlite3.connect(self.db_path) as conn:
            # Monthly total
            cursor = conn.execute(
                """
                SELECT COALESCE(SUM(cost), 0), COUNT(*)
                FROM api_calls
                WHERE timestamp >= ?
                """,
                (month_start.isoformat(),)
            )
            total_cost, calls_this_month = cursor.fetchone()

            # Today's calls
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            cursor = conn.execute(
                """
                SELECT COUNT(*)
                FROM api_calls
                WHERE timestamp >= ?
                """,
                (today_start.isoformat(),)
            )
            calls_today = cursor.fetchone()[0]

        # Calculate metrics
        budget_used_pct = total_cost / self.monthly_budget if self.monthly_budget > 0 else 0
        avg_cost = total_cost / calls_this_month if calls_this_month > 0 else 0
        remaining = max(0, self.monthly_budget - total_cost)

        # Project monthly spend based on current rate
        daily_rate = total_cost / days_elapsed if days_elapsed > 0 else 0
        projected = daily_rate * days_in_month

        # Determine zone
        zone = BudgetZone.GREEN
        for z, (low, high) in self.ZONE_THRESHOLDS.items():
            if low <= budget_used_pct < high:
                zone = z
                break

        return CostMetrics(
            total_cost=round(total_cost, 4),
            budget_used_pct=round(budget_used_pct, 4),
            zone=zone,
            calls_today=calls_today,
            calls_this_month=calls_this_month,
            avg_cost_per_call=round(avg_cost, 6),
            remaining_budget=round(remaining, 2),
            days_remaining=days_remaining,
            projected_monthly=round(projected, 2)
        )

    def should_use_expensive_model(self) -> bool:
        """
        Check if expensive model usage is advisable.

        Returns:
            True if budget allows expensive model use
        """
        zone = self.get_current_zone()
        return zone in (BudgetZone.GREEN, BudgetZone.YELLOW)

    def should_block_paid_calls(self) -> bool:
        """
        Check if paid API calls should be blocked.

        Returns:
            True if in critical zone and should block
        """
        zone = self.get_current_zone()
        return zone == BudgetZone.CRITICAL

    def get_optimization_hints(self) -> List[str]:
        """
        Get cost optimization hints based on current state.

        Returns:
            List of optimization suggestions
        """
        hints = []
        metrics = self.get_monthly_metrics()

        if metrics.zone == BudgetZone.YELLOW:
            hints.append("Consider using local engines more aggressively")
            hints.append("Enable stricter caching policies")

        elif metrics.zone == BudgetZone.RED:
            hints.append("⚠️ Budget critical - forcing local engines")
            hints.append("Only complex problems should use Claude")

        elif metrics.zone == BudgetZone.CRITICAL:
            hints.append("🚨 BUDGET EXCEEDED - blocking paid API calls")
            hints.append("Only cached and local results available")

        if metrics.projected_monthly > self.monthly_budget * 1.2:
            hints.append(f"Projected spend ${metrics.projected_monthly:.2f} exceeds budget by {((metrics.projected_monthly / self.monthly_budget) - 1) * 100:.0f}%")

        return hints

    def get_cost_breakdown(self, days: int = 30) -> Dict[str, Any]:
        """
        Get detailed cost breakdown.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with cost breakdown by engine, model, and day
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # By engine
            cursor = conn.execute(
                """
                SELECT engine, COALESCE(SUM(cost), 0), COUNT(*),
                       COALESCE(AVG(latency_ms), 0)
                FROM api_calls
                WHERE timestamp >= ?
                GROUP BY engine
                ORDER BY SUM(cost) DESC
                """,
                (cutoff,)
            )
            by_engine = {
                row[0]: {
                    "cost": round(row[1], 4),
                    "calls": row[2],
                    "avg_latency_ms": round(row[3], 1)
                }
                for row in cursor.fetchall()
            }

            # By model
            cursor = conn.execute(
                """
                SELECT model, COALESCE(SUM(cost), 0), COUNT(*),
                       COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0)
                FROM api_calls
                WHERE timestamp >= ? AND model IS NOT NULL
                GROUP BY model
                ORDER BY SUM(cost) DESC
                """,
                (cutoff,)
            )
            by_model = {
                row[0]: {
                    "cost": round(row[1], 4),
                    "calls": row[2],
                    "input_tokens": row[3],
                    "output_tokens": row[4]
                }
                for row in cursor.fetchall()
            }

            # Daily costs
            cursor = conn.execute(
                """
                SELECT DATE(timestamp), COALESCE(SUM(cost), 0), COUNT(*)
                FROM api_calls
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY DATE(timestamp) DESC
                """,
                (cutoff,)
            )
            daily = {
                row[0]: {"cost": round(row[1], 4), "calls": row[2]}
                for row in cursor.fetchall()
            }

            # Cache stats
            cursor = conn.execute(
                """
                SELECT cached, COUNT(*)
                FROM api_calls
                WHERE timestamp >= ?
                GROUP BY cached
                """,
                (cutoff,)
            )
            cache_data = dict(cursor.fetchall())
            total_calls = sum(cache_data.values())
            cache_hits = cache_data.get(1, 0)

        return {
            "by_engine": by_engine,
            "by_model": by_model,
            "daily": daily,
            "cache_hit_rate": cache_hits / total_calls if total_calls > 0 else 0,
            "period_days": days
        }

    def estimate_cost(
        self,
        problem: str,
        problem_type: str,
        complexity: str
    ) -> float:
        """
        Estimate cost for a problem before execution.

        Args:
            problem: Problem text
            problem_type: Type of problem
            complexity: Complexity level

        Returns:
            Estimated cost in dollars
        """
        # Rough token estimation
        input_tokens = len(problem) // 4 + 500  # Base overhead
        output_tokens = 500  # Default output estimate

        # Adjust by complexity
        complexity_multipliers = {
            "trivial": 0.5,
            "simple": 0.8,
            "moderate": 1.0,
            "complex": 1.5,
            "expert": 2.0
        }
        multiplier = complexity_multipliers.get(complexity, 1.0)
        output_tokens = int(output_tokens * multiplier)

        return self.calculate_cost(self.DEFAULT_MODEL, input_tokens, output_tokens)


# Global instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
