"""Meta-aware EV advisor that adjusts poker EV calculations based on current meta trends.

Integrates Perplexity news/meta analysis to provide real-time adjustments to
fold equity, aggression factors, and EV calculations.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MetaTrend:
    """Represents a poker meta trend that affects EV calculations."""
    name: str
    description: str
    ev_modifier: float  # Multiplier for EV (1.0 = no change)
    fold_equity_adjust: float  # Adjustment to villain fold % (e.g., -5 = 5% less likely to fold)
    aggression_adjust: float  # Adjustment to villain aggression expectation
    confidence: float  # 0.0-1.0 how reliable this trend is
    expires_at: datetime
    source: str = "perplexity"

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "ev_modifier": self.ev_modifier,
            "fold_equity_adjust": self.fold_equity_adjust,
            "aggression_adjust": self.aggression_adjust,
            "confidence": self.confidence,
            "expires_at": self.expires_at.isoformat(),
            "source": self.source
        }


@dataclass
class MetaSnapshot:
    """Current meta state with all active trends."""
    trends: List[MetaTrend] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    raw_analysis: str = ""

    @property
    def combined_ev_modifier(self) -> float:
        """Get combined EV modifier from all active trends."""
        if not self.trends:
            return 1.0
        # Weighted average by confidence
        total_weight = sum(t.confidence for t in self.trends if not t.is_expired())
        if total_weight == 0:
            return 1.0
        weighted_sum = sum(t.ev_modifier * t.confidence for t in self.trends if not t.is_expired())
        return weighted_sum / total_weight

    @property
    def combined_fold_equity_adjust(self) -> float:
        """Get combined fold equity adjustment."""
        active = [t for t in self.trends if not t.is_expired()]
        if not active:
            return 0.0
        # Average adjustment weighted by confidence
        total_weight = sum(t.confidence for t in active)
        if total_weight == 0:
            return 0.0
        return sum(t.fold_equity_adjust * t.confidence for t in active) / total_weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trends": [t.to_dict() for t in self.trends if not t.is_expired()],
            "combined_ev_modifier": round(self.combined_ev_modifier, 3),
            "combined_fold_equity_adjust": round(self.combined_fold_equity_adjust, 1),
            "last_updated": self.last_updated.isoformat(),
            "trend_count": len([t for t in self.trends if not t.is_expired()])
        }


class MetaAdvisor:
    """Provides meta-aware EV adjustments based on current poker trends.

    Integrates with Perplexity news to analyze current poker meta and adjust
    EV calculations accordingly.
    """

    # Default meta trends (fallback when API unavailable)
    DEFAULT_TRENDS = {
        "aggressive_3bet": MetaTrend(
            name="High 3-bet frequency",
            description="Players 3-betting 12-15% in SB vs BTN",
            ev_modifier=0.95,  # Slightly lower EV for thin value bets
            fold_equity_adjust=-5,  # Villains fold less to 3-bets
            aggression_adjust=1.2,
            confidence=0.7,
            expires_at=datetime.utcnow() + timedelta(hours=24)
        ),
        "cbet_overfold": MetaTrend(
            name="C-bet overfold tendency",
            description="Players overfolding to flop c-bets",
            ev_modifier=1.1,  # Higher EV for c-bet bluffs
            fold_equity_adjust=10,  # More fold equity on c-bets
            aggression_adjust=0.9,
            confidence=0.6,
            expires_at=datetime.utcnow() + timedelta(hours=24)
        ),
        "river_underbluff": MetaTrend(
            name="River underbluffing",
            description="Population underbluffs rivers significantly",
            ev_modifier=1.05,
            fold_equity_adjust=-10,  # Don't expect many bluffs
            aggression_adjust=0.8,
            confidence=0.65,
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
    }

    # Keywords that indicate specific meta trends
    TREND_KEYWORDS = {
        "aggressive_3bet": ["3-bet", "3bet", "three-bet", "small blind", "sb vs btn"],
        "cbet_overfold": ["c-bet", "cbet", "continuation bet", "flop fold"],
        "river_underbluff": ["river", "bluff frequency", "value heavy", "underbluff"],
        "squeeze": ["squeeze", "4-bet", "cold 4bet"],
        "multiway": ["multiway", "multi-way", "3-way", "4-way"],
        "probe": ["probe bet", "probing", "check-raise"],
        "donk": ["donk bet", "donk lead", "leading into"],
    }

    def __init__(self):
        self._meta_snapshot = MetaSnapshot()
        self._news_tool = None
        self._cache_duration = timedelta(minutes=30)

    async def _get_news_tool(self):
        """Lazy load news tool."""
        if self._news_tool is None:
            try:
                from backend.voice.tools.news_tool import PerplexityNewsTool
                self._news_tool = PerplexityNewsTool()
            except ImportError:
                logger.warning("Could not import news tool for meta analysis")
        return self._news_tool

    async def refresh_meta(self, force: bool = False) -> MetaSnapshot:
        """Refresh meta trends from Perplexity.

        Args:
            force: Force refresh even if cache is valid

        Returns:
            Updated MetaSnapshot
        """
        # Check if cache is still valid
        cache_age = datetime.utcnow() - self._meta_snapshot.last_updated
        if not force and cache_age < self._cache_duration and self._meta_snapshot.trends:
            return self._meta_snapshot

        news_tool = await self._get_news_tool()
        if news_tool is None or not news_tool.api_key:
            # Use default trends
            logger.info("Using default meta trends (Perplexity unavailable)")
            self._meta_snapshot = MetaSnapshot(
                trends=list(self.DEFAULT_TRENDS.values()),
                raw_analysis="Using default meta assumptions"
            )
            return self._meta_snapshot

        try:
            # Get meta analysis from Perplexity
            result = await news_tool._get_meta_analysis("poker meta")

            if result.success:
                trends = self._parse_meta_response(result.message)
                self._meta_snapshot = MetaSnapshot(
                    trends=trends,
                    last_updated=datetime.utcnow(),
                    raw_analysis=result.message
                )
                logger.info(f"Meta refresh: {len(trends)} trends identified")
            else:
                logger.warning(f"Meta refresh failed: {result.message}")
                # Keep existing trends or use defaults

        except Exception as e:
            logger.error(f"Meta refresh error: {e}")

        return self._meta_snapshot

    def _parse_meta_response(self, analysis: str) -> List[MetaTrend]:
        """Parse Perplexity response into actionable trends."""
        trends = []
        analysis_lower = analysis.lower()

        # Check for each known trend pattern
        for trend_key, keywords in self.TREND_KEYWORDS.items():
            if any(kw in analysis_lower for kw in keywords):
                if trend_key in self.DEFAULT_TRENDS:
                    # Use default with fresh expiry
                    base_trend = self.DEFAULT_TRENDS[trend_key]
                    trends.append(MetaTrend(
                        name=base_trend.name,
                        description=base_trend.description,
                        ev_modifier=base_trend.ev_modifier,
                        fold_equity_adjust=base_trend.fold_equity_adjust,
                        aggression_adjust=base_trend.aggression_adjust,
                        confidence=base_trend.confidence,
                        expires_at=datetime.utcnow() + timedelta(hours=24),
                        source="perplexity"
                    ))

        # Add dynamic adjustments based on specific content
        if "tight" in analysis_lower and "fold" in analysis_lower:
            trends.append(MetaTrend(
                name="Tight population",
                description="Players folding too much overall",
                ev_modifier=1.15,
                fold_equity_adjust=15,
                aggression_adjust=0.85,
                confidence=0.6,
                expires_at=datetime.utcnow() + timedelta(hours=12),
                source="perplexity"
            ))

        if "loose" in analysis_lower and "call" in analysis_lower:
            trends.append(MetaTrend(
                name="Loose population",
                description="Players calling too wide",
                ev_modifier=0.9,
                fold_equity_adjust=-15,
                aggression_adjust=0.9,
                confidence=0.6,
                expires_at=datetime.utcnow() + timedelta(hours=12),
                source="perplexity"
            ))

        if not trends:
            # Return default trends if no specific trends detected
            return list(self.DEFAULT_TRENDS.values())[:2]

        return trends

    def get_current_meta(self) -> MetaSnapshot:
        """Get current meta snapshot without refresh."""
        return self._meta_snapshot

    def calculate_adjusted_ev(
        self,
        base_ev: float,
        pot: float,
        bet_size: float,
        street: str = "flop",
        action: str = "bet"
    ) -> Dict[str, Any]:
        """Calculate meta-adjusted EV.

        Args:
            base_ev: Original EV calculation
            pot: Current pot size
            bet_size: Bet amount
            street: Current street (preflop, flop, turn, river)
            action: Action type (bet, raise, call)

        Returns:
            Dict with adjusted EV and explanation
        """
        meta = self._meta_snapshot
        modifier = meta.combined_ev_modifier
        fold_adjust = meta.combined_fold_equity_adjust

        # Street-specific adjustments
        street_multipliers = {
            "preflop": 1.0,
            "flop": 1.0,
            "turn": 0.95,  # Less bluff equity on turn
            "river": 0.9   # Even less on river
        }
        street_mult = street_multipliers.get(street, 1.0)

        # Apply meta modifier
        adjusted_ev = base_ev * modifier * street_mult

        # Calculate effective fold equity adjustment
        effective_fold_adjust = fold_adjust * street_mult

        return {
            "base_ev": round(base_ev, 2),
            "adjusted_ev": round(adjusted_ev, 2),
            "ev_modifier": round(modifier, 3),
            "fold_equity_adjust": round(effective_fold_adjust, 1),
            "street_modifier": street_mult,
            "active_trends": len([t for t in meta.trends if not t.is_expired()]),
            "recommendation": self._get_recommendation(base_ev, adjusted_ev, effective_fold_adjust),
            "meta_summary": self._get_meta_summary()
        }

    def adjust_fold_equity(
        self,
        base_fold_pct: float,
        street: str = "flop"
    ) -> float:
        """Adjust villain fold percentage based on meta.

        Args:
            base_fold_pct: Baseline fold percentage (0-100)
            street: Current street

        Returns:
            Adjusted fold percentage
        """
        adjustment = self._meta_snapshot.combined_fold_equity_adjust

        # Street modifiers
        if street == "river":
            adjustment *= 0.7  # Less trust on river
        elif street == "turn":
            adjustment *= 0.85

        # Clamp to reasonable range
        adjusted = base_fold_pct + adjustment
        return max(5, min(85, adjusted))

    def _get_recommendation(
        self,
        base_ev: float,
        adjusted_ev: float,
        fold_adjust: float
    ) -> str:
        """Generate recommendation based on meta-adjusted EV."""
        ev_diff = adjusted_ev - base_ev
        pct_change = (ev_diff / abs(base_ev)) * 100 if base_ev != 0 else 0

        if pct_change > 10:
            return f"Meta favors this line (+{pct_change:.0f}% EV). More aggression profitable."
        elif pct_change < -10:
            return f"Meta disfavors this line ({pct_change:.0f}% EV). Consider tighter approach."
        elif fold_adjust > 10:
            return "High fold equity in current meta. Bluffs more profitable."
        elif fold_adjust < -10:
            return "Low fold equity in current meta. Value bet thinner, bluff less."
        else:
            return "Standard play in current meta."

    def _get_meta_summary(self) -> str:
        """Get short summary of current meta."""
        active = [t for t in self._meta_snapshot.trends if not t.is_expired()]
        if not active:
            return "No meta data available"

        trend_names = [t.name for t in active[:3]]
        return f"Active trends: {', '.join(trend_names)}"


# Global instance
meta_advisor = MetaAdvisor()
