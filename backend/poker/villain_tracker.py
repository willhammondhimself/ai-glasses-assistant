"""Villain pattern tracking for opponent profiling and range estimation."""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class PokerAction(Enum):
    """Poker actions for tracking."""
    FOLD = "F"
    CHECK = "X"
    CALL = "C"
    BET = "B"
    RAISE = "R"
    ALL_IN = "A"


class Street(Enum):
    """Poker streets."""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


@dataclass
class ActionRecord:
    """Single action record for a villain."""
    street: Street
    action: PokerAction
    amount: float
    pot_size: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VillainProfile:
    """Profile for tracking opponent play patterns.

    Attributes:
        seat: Seat identifier (e.g., "Seat 1", "Seat 2")
        vpip: Voluntarily Put $ In Pot percentage
        pfr: Preflop Raise percentage
        aggression: Aggression factor (raises + bets) / calls
        fold_to_cbet: Fold to continuation bet percentage
        three_bet: 3-bet percentage
        fold_to_three_bet: Fold to 3-bet percentage
        hands_observed: Total hands observed
        recent_actions: Last N actions taken
        position_stats: Stats broken down by position
    """
    seat: str
    vpip: float = 0.0
    pfr: float = 0.0
    aggression: float = 0.0
    fold_to_cbet: float = 0.0
    three_bet: float = 0.0
    fold_to_three_bet: float = 0.0
    hands_observed: int = 0
    recent_actions: List[str] = field(default_factory=list)

    # Detailed tracking
    _vpip_hands: int = field(default=0, repr=False)
    _pfr_hands: int = field(default=0, repr=False)
    _aggression_bets: int = field(default=0, repr=False)
    _aggression_calls: int = field(default=0, repr=False)
    _cbet_faced: int = field(default=0, repr=False)
    _cbet_folded: int = field(default=0, repr=False)
    _three_bet_opps: int = field(default=0, repr=False)
    _three_bet_made: int = field(default=0, repr=False)

    # Showdown tracking
    showdowns_observed: int = field(default=0, repr=False)
    hands_shown: List[str] = field(default_factory=list, repr=False)  # Actual cards shown

    def get_range_estimate(self) -> str:
        """Estimate villain range based on stats.

        Returns:
            String description of estimated range
        """
        if self.hands_observed < 10:
            return "Insufficient data"

        if self.vpip > 45:
            return "Very loose - extremely wide range (any two cards)"
        elif self.vpip > 35:
            return "Loose - wide range including speculative hands"
        elif self.vpip > 28:
            return "Loose-aggressive - expanded range, aggressive"
        elif self.vpip > 22:
            return "TAG - standard tight-aggressive range"
        elif self.vpip > 15:
            return "Tight - premium and strong hands"
        else:
            return "Nit - ultra-premium hands only"

    def get_player_type(self) -> str:
        """Classify player type based on VPIP/PFR.

        Returns:
            Player type classification
        """
        if self.hands_observed < 20:
            return "Unknown"

        # Based on VPIP and PFR gap
        gap = self.vpip - self.pfr

        if self.vpip > 30 and self.pfr > 20:
            return "LAG (Loose-Aggressive)"
        elif self.vpip > 30 and self.pfr < 15:
            return "Calling Station"
        elif self.vpip < 20 and self.pfr > 12:
            return "TAG (Tight-Aggressive)"
        elif self.vpip < 15 and self.pfr < 10:
            return "Nit"
        elif gap > 15:
            return "Passive Fish"
        else:
            return "Regular"

    def get_exploitation_tips(self) -> List[str]:
        """Get exploitation recommendations based on tendencies.

        Returns:
            List of strategic tips
        """
        tips = []

        if self.hands_observed < 20:
            tips.append("Need more hands for reliable reads")
            return tips

        # VPIP-based tips
        if self.vpip > 35:
            tips.append("Value bet thinner - they call too wide")
            tips.append("Don't bluff river - they call too much")
        elif self.vpip < 18:
            tips.append("Steal their blinds frequently")
            tips.append("Respect their raises - they have it")

        # Aggression-based tips
        if self.aggression > 2.5:
            tips.append("Let them bluff - check/call more")
            tips.append("Trap with strong hands")
        elif self.aggression < 1.0:
            tips.append("Bet for value relentlessly")
            tips.append("Their raises are strong - consider folding")

        # C-bet response
        if self.fold_to_cbet > 60:
            tips.append("C-bet 100% - they fold too much")
        elif self.fold_to_cbet < 35:
            tips.append("Only c-bet for value - they don't fold")

        # 3-bet response
        if self.fold_to_three_bet > 70:
            tips.append("3-bet light for value - they overfold")

        return tips if tips else ["Standard play - no major leaks detected"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "seat": self.seat,
            "vpip": round(self.vpip, 1),
            "pfr": round(self.pfr, 1),
            "aggression": round(self.aggression, 2),
            "fold_to_cbet": round(self.fold_to_cbet, 1),
            "three_bet": round(self.three_bet, 1),
            "hands_observed": self.hands_observed,
            "recent_actions": self.recent_actions[-10:],  # Last 10
            "range_estimate": self.get_range_estimate(),
            "player_type": self.get_player_type(),
            "tips": self.get_exploitation_tips()
        }

    def to_hud_format(self) -> Dict[str, Any]:
        """Format for HUD display overlay."""
        return {
            "seat": self.seat,
            "stats": f"VPIP:{self.vpip:.0f} PFR:{self.pfr:.0f}",
            "agg": f"AGG:{self.aggression:.1f}",
            "f2cb": f"F2CB:{self.fold_to_cbet:.0f}%",
            "type": self.get_player_type(),
            "range": self.get_range_estimate(),
            "actions": " ".join(self.recent_actions[-5:]),
            "hands": self.hands_observed,
            "showdowns": self.showdowns_observed,
            "hands_shown": self.hands_shown[-5:] if self.hands_shown else []
        }


class VillainTracker:
    """Tracks opponent patterns across sessions.

    Supports Ignition's anonymous players (Seat 1, Seat 2, etc.)
    by tracking within-session patterns only.
    """

    def __init__(self, max_history: int = 100):
        """Initialize villain tracker.

        Args:
            max_history: Maximum hands to track per villain
        """
        self._profiles: Dict[str, VillainProfile] = {}
        self._action_history: Dict[str, List[ActionRecord]] = defaultdict(list)
        self._max_history = max_history
        self._session_start = datetime.now()

    def get_or_create_profile(self, seat: str) -> VillainProfile:
        """Get existing profile or create new one.

        Args:
            seat: Seat identifier (e.g., "Seat 1")

        Returns:
            VillainProfile for the seat
        """
        if seat not in self._profiles:
            self._profiles[seat] = VillainProfile(seat=seat)
            logger.info(f"Created new villain profile for {seat}")
        return self._profiles[seat]

    def record_action(
        self,
        seat: str,
        street: Street,
        action: PokerAction,
        amount: float = 0.0,
        pot_size: float = 0.0,
        is_preflop_voluntary: bool = False,
        is_preflop_raise: bool = False,
        facing_cbet: bool = False,
        facing_3bet: bool = False,
        is_3bet: bool = False
    ) -> VillainProfile:
        """Record an action for a villain.

        Args:
            seat: Seat identifier
            street: Current street
            action: Action taken
            amount: Bet/raise amount
            pot_size: Current pot size
            is_preflop_voluntary: If this contributes to VPIP
            is_preflop_raise: If this is a preflop raise (PFR)
            facing_cbet: If facing a continuation bet
            facing_3bet: If facing a 3-bet
            is_3bet: If this action is a 3-bet

        Returns:
            Updated VillainProfile
        """
        profile = self.get_or_create_profile(seat)

        # Record the action
        record = ActionRecord(
            street=street,
            action=action,
            amount=amount,
            pot_size=pot_size
        )
        self._action_history[seat].append(record)

        # Trim history if needed
        if len(self._action_history[seat]) > self._max_history:
            self._action_history[seat] = self._action_history[seat][-self._max_history:]

        # Update recent actions display
        profile.recent_actions.append(action.value)
        if len(profile.recent_actions) > 20:
            profile.recent_actions = profile.recent_actions[-20:]

        # Update stats
        if is_preflop_voluntary:
            profile._vpip_hands += 1

        if is_preflop_raise:
            profile._pfr_hands += 1

        # Track aggression
        if action in (PokerAction.BET, PokerAction.RAISE, PokerAction.ALL_IN):
            profile._aggression_bets += 1
        elif action == PokerAction.CALL:
            profile._aggression_calls += 1

        # Track c-bet response
        if facing_cbet:
            profile._cbet_faced += 1
            if action == PokerAction.FOLD:
                profile._cbet_folded += 1

        # Track 3-bet
        if is_3bet:
            profile._three_bet_opps += 1
            profile._three_bet_made += 1
        elif facing_3bet and action == PokerAction.FOLD:
            profile._three_bet_opps += 1

        # Recalculate stats
        self._recalculate_stats(profile)

        return profile

    def record_new_hand(self, active_seats: List[str]) -> None:
        """Record start of a new hand.

        Args:
            active_seats: List of seats in this hand
        """
        for seat in active_seats:
            profile = self.get_or_create_profile(seat)
            profile.hands_observed += 1

    def _recalculate_stats(self, profile: VillainProfile) -> None:
        """Recalculate all statistics for a profile."""
        hands = max(profile.hands_observed, 1)

        # VPIP
        profile.vpip = (profile._vpip_hands / hands) * 100

        # PFR
        profile.pfr = (profile._pfr_hands / hands) * 100

        # Aggression factor
        if profile._aggression_calls > 0:
            profile.aggression = profile._aggression_bets / profile._aggression_calls
        else:
            profile.aggression = profile._aggression_bets if profile._aggression_bets > 0 else 0

        # Fold to c-bet
        if profile._cbet_faced > 0:
            profile.fold_to_cbet = (profile._cbet_folded / profile._cbet_faced) * 100
        else:
            profile.fold_to_cbet = 0

        # 3-bet percentage
        if profile._three_bet_opps > 0:
            profile.three_bet = (profile._three_bet_made / profile._three_bet_opps) * 100
        else:
            profile.three_bet = 0

    def record_showdown(
        self,
        seat: str,
        cards: Optional[str] = None,
        hand_description: Optional[str] = None
    ) -> VillainProfile:
        """Record a showdown for a villain.

        Args:
            seat: Seat identifier (e.g., "Seat 3")
            cards: Actual cards shown (e.g., "Ac Kh")
            hand_description: Hand description (e.g., "Two Pair")

        Returns:
            Updated VillainProfile
        """
        profile = self.get_or_create_profile(seat)
        profile.showdowns_observed += 1

        if cards:
            # Normalize card string
            cards = cards.strip().upper().replace(" ", " ")
            profile.hands_shown.append(cards)

            # Keep last 20 hands shown
            if len(profile.hands_shown) > 20:
                profile.hands_shown = profile.hands_shown[-20:]

            logger.info(f"Recorded showdown for {seat}: {cards} ({hand_description or 'unknown'})")

        return profile

    def get_profile(self, seat: str) -> Optional[VillainProfile]:
        """Get profile for a seat if it exists.

        Args:
            seat: Seat identifier

        Returns:
            VillainProfile or None
        """
        return self._profiles.get(seat)

    def get_all_profiles(self) -> Dict[str, VillainProfile]:
        """Get all tracked profiles.

        Returns:
            Dict mapping seat to profile
        """
        return self._profiles.copy()

    def get_hud_data(self) -> List[Dict[str, Any]]:
        """Get all profiles formatted for HUD display.

        Returns:
            List of HUD-formatted profile dicts
        """
        return [
            profile.to_hud_format()
            for profile in self._profiles.values()
            if profile.hands_observed >= 5  # Only show after 5 hands
        ]

    def get_villain_at_seat(self, seat: str) -> Optional[Dict[str, Any]]:
        """Get villain data for HUD display at a specific seat.

        Args:
            seat: Seat identifier

        Returns:
            HUD-formatted profile dict or None
        """
        profile = self._profiles.get(seat)
        if profile and profile.hands_observed >= 3:
            return profile.to_hud_format()
        return None

    def clear_session(self) -> None:
        """Clear all profiles for a new session."""
        self._profiles.clear()
        self._action_history.clear()
        self._session_start = datetime.now()
        logger.info("Villain tracker session cleared")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current tracking session.

        Returns:
            Session summary dict
        """
        total_hands = sum(p.hands_observed for p in self._profiles.values())
        player_types = defaultdict(int)
        for profile in self._profiles.values():
            player_types[profile.get_player_type()] += 1

        return {
            "session_start": self._session_start.isoformat(),
            "duration_minutes": (datetime.now() - self._session_start).seconds // 60,
            "villains_tracked": len(self._profiles),
            "total_hands_observed": total_hands,
            "player_type_breakdown": dict(player_types),
            "profiles": [p.to_dict() for p in self._profiles.values()]
        }

    def analyze_table_dynamics(self) -> Dict[str, Any]:
        """Analyze overall table dynamics.

        Returns:
            Table dynamics analysis
        """
        if not self._profiles:
            return {"status": "No data"}

        vpips = [p.vpip for p in self._profiles.values() if p.hands_observed >= 10]
        aggressions = [p.aggression for p in self._profiles.values() if p.hands_observed >= 10]

        if not vpips:
            return {"status": "Insufficient data"}

        avg_vpip = sum(vpips) / len(vpips)
        avg_aggression = sum(aggressions) / len(aggressions) if aggressions else 0

        # Classify table
        if avg_vpip > 35:
            table_type = "Very loose table - value bet thin"
        elif avg_vpip > 28:
            table_type = "Loose table - widen value range"
        elif avg_vpip < 20:
            table_type = "Tight table - steal more"
        else:
            table_type = "Standard table - play solid"

        return {
            "table_type": table_type,
            "avg_vpip": round(avg_vpip, 1),
            "avg_aggression": round(avg_aggression, 2),
            "num_players_tracked": len(vpips),
            "recommendation": self._get_table_recommendation(avg_vpip, avg_aggression)
        }

    def _get_table_recommendation(self, avg_vpip: float, avg_agg: float) -> str:
        """Get strategic recommendation based on table dynamics."""
        if avg_vpip > 35 and avg_agg < 1.5:
            return "Value bet relentlessly, avoid bluffing"
        elif avg_vpip > 35 and avg_agg > 2.0:
            return "Tighten up preflop, trap postflop"
        elif avg_vpip < 22 and avg_agg > 2.0:
            return "Steal blinds, 3-bet light in position"
        elif avg_vpip < 22 and avg_agg < 1.5:
            return "Play straightforward, value bet thin"
        else:
            return "Standard TAG strategy recommended"


# Global instance for the current session
villain_tracker = VillainTracker()
