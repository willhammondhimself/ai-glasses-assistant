"""
Opponent Tracker - Local villain stat tracking.
Tracks VPIP, aggression, fold frequencies for exploitative play.
Instant lookups, no API calls.
"""
import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class VillainType(Enum):
    """Villain archetype classifications."""
    UNKNOWN = "unknown"
    CALLING_STATION = "calling_station"  # Loose-passive: calls too much
    TIGHT_PASSIVE = "tight_passive"      # Nit: folds too much, rarely raises
    MANIAC = "maniac"                    # Loose-aggressive: bets/raises constantly
    TAG = "tag"                          # Tight-aggressive: standard solid player
    LAG = "lag"                          # Loose-aggressive but controlled


@dataclass
class VillainStats:
    """Statistics for a single villain."""
    # Core stats
    hands_observed: int = 0
    vpip_count: int = 0        # Voluntarily put money in pot
    pfr_count: int = 0         # Preflop raise
    aggro_actions: int = 0     # Bets and raises
    passive_actions: int = 0   # Checks and calls

    # Postflop stats
    cbet_opportunities: int = 0
    cbet_count: int = 0
    fold_to_cbet_opportunities: int = 0
    fold_to_cbet_count: int = 0

    # Showdown stats
    showdowns: int = 0
    showdown_wins: int = 0

    # Timing
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Metadata
    notes: List[str] = field(default_factory=list)
    name: str = ""

    @property
    def vpip(self) -> float:
        """Voluntarily Put money In Pot percentage."""
        if self.hands_observed == 0:
            return 0.25  # Default assumption
        return self.vpip_count / self.hands_observed

    @property
    def pfr(self) -> float:
        """Preflop Raise percentage."""
        if self.hands_observed == 0:
            return 0.15
        return self.pfr_count / self.hands_observed

    @property
    def aggression(self) -> float:
        """Aggression factor (bets+raises / checks+calls)."""
        total = self.aggro_actions + self.passive_actions
        if total == 0:
            return 0.40  # Default assumption
        return self.aggro_actions / total

    @property
    def cbet_freq(self) -> float:
        """Continuation bet frequency."""
        if self.cbet_opportunities == 0:
            return 0.65  # Default assumption
        return self.cbet_count / self.cbet_opportunities

    @property
    def fold_to_cbet(self) -> float:
        """Fold to continuation bet frequency."""
        if self.fold_to_cbet_opportunities == 0:
            return 0.50  # Default assumption
        return self.fold_to_cbet_count / self.fold_to_cbet_opportunities

    @property
    def showdown_win_rate(self) -> float:
        """Win rate at showdown."""
        if self.showdowns == 0:
            return 0.50
        return self.showdown_wins / self.showdowns

    def get_type(self) -> VillainType:
        """Classify villain based on stats."""
        if self.hands_observed < 5:
            return VillainType.UNKNOWN

        vpip = self.vpip
        aggression = self.aggression

        # Calling station: plays lots of hands, rarely aggressive
        if vpip > 0.40 and aggression < 0.30:
            return VillainType.CALLING_STATION

        # Tight-passive nit: plays few hands, rarely aggressive
        if vpip < 0.20 and aggression < 0.35:
            return VillainType.TIGHT_PASSIVE

        # Maniac: plays lots of hands, very aggressive
        if vpip > 0.45 and aggression > 0.55:
            return VillainType.MANIAC

        # LAG: somewhat loose, aggressive
        if vpip > 0.30 and aggression > 0.45:
            return VillainType.LAG

        # Default to TAG
        return VillainType.TAG

    def to_dict(self) -> dict:
        """Export stats as dictionary."""
        d = asdict(self)
        d["vpip"] = self.vpip
        d["pfr"] = self.pfr
        d["aggression"] = self.aggression
        d["cbet_freq"] = self.cbet_freq
        d["fold_to_cbet"] = self.fold_to_cbet
        d["type"] = self.get_type().value
        return d


class OpponentTracker:
    """
    Track villain stats locally for exploitative play.

    Features:
    - Per-villain stat tracking
    - Automatic archetype classification
    - Persistence across sessions
    - Instant lookups (no API)
    """

    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize opponent tracker.

        Args:
            persist_path: Path to save/load stats (optional)
        """
        self.villains: Dict[str, VillainStats] = {}
        self.current_villain: Optional[str] = None
        self.persist_path = persist_path

        if persist_path:
            self._load()

    def set_villain(self, villain_id: str, name: str = ""):
        """
        Set current villain being tracked.

        Args:
            villain_id: Unique identifier (seat number, username, etc.)
            name: Display name
        """
        self.current_villain = villain_id

        if villain_id not in self.villains:
            self.villains[villain_id] = VillainStats(name=name)
        elif name:
            self.villains[villain_id].name = name

    def update(
        self,
        action_type: str,
        street: str = "preflop",
        voluntarily_invested: bool = False,
        is_raise: bool = False
    ):
        """
        Update villain stats based on observed action.

        Args:
            action_type: fold/check/call/bet/raise
            street: preflop/flop/turn/river
            voluntarily_invested: Did they put money in voluntarily?
            is_raise: Was this a raise specifically?
        """
        if not self.current_villain:
            return

        stats = self.villains.get(self.current_villain)
        if not stats:
            return

        stats.last_seen = time.time()

        # Track VPIP
        if voluntarily_invested:
            stats.vpip_count += 1

        # Track PFR
        if street == "preflop" and is_raise:
            stats.pfr_count += 1

        # Track aggression
        if action_type in ["bet", "raise"]:
            stats.aggro_actions += 1
        elif action_type in ["check", "call"]:
            stats.passive_actions += 1

    def record_hand(self, voluntarily_invested: bool = False):
        """
        Record that a hand was observed for current villain.

        Args:
            voluntarily_invested: Did they VPIP this hand?
        """
        if not self.current_villain:
            return

        stats = self.villains.get(self.current_villain)
        if stats:
            stats.hands_observed += 1
            if voluntarily_invested:
                stats.vpip_count += 1
            stats.last_seen = time.time()

    def record_cbet(self, made_cbet: bool):
        """Record a c-bet opportunity and result."""
        if not self.current_villain:
            return

        stats = self.villains.get(self.current_villain)
        if stats:
            stats.cbet_opportunities += 1
            if made_cbet:
                stats.cbet_count += 1

    def record_fold_to_cbet(self, folded: bool):
        """Record fold to c-bet opportunity and result."""
        if not self.current_villain:
            return

        stats = self.villains.get(self.current_villain)
        if stats:
            stats.fold_to_cbet_opportunities += 1
            if folded:
                stats.fold_to_cbet_count += 1

    def record_showdown(self, won: bool):
        """Record showdown result."""
        if not self.current_villain:
            return

        stats = self.villains.get(self.current_villain)
        if stats:
            stats.showdowns += 1
            if won:
                stats.showdown_wins += 1

    def add_note(self, note: str, villain_id: Optional[str] = None):
        """Add a note about a villain."""
        vid = villain_id or self.current_villain
        if not vid:
            return

        stats = self.villains.get(vid)
        if stats:
            stats.notes.append(f"[{time.strftime('%H:%M')}] {note}")

    def get_stats(self, villain_id: Optional[str] = None) -> dict:
        """
        Get current villain stats.

        Args:
            villain_id: Specific villain (default: current)

        Returns:
            Stats dictionary for prompt building
        """
        vid = villain_id or self.current_villain

        if not vid or vid not in self.villains:
            return {
                "type": "unknown",
                "vpip": 0.25,
                "aggression": 0.40,
                "hands_observed": 0,
                "fold_to_cbet": 0.50
            }

        stats = self.villains[vid]
        return {
            "type": stats.get_type().value,
            "vpip": stats.vpip,
            "pfr": stats.pfr,
            "aggression": stats.aggression,
            "cbet_freq": stats.cbet_freq,
            "fold_to_cbet": stats.fold_to_cbet,
            "hands_observed": stats.hands_observed,
            "notes": stats.notes[-3:] if stats.notes else []  # Last 3 notes
        }

    def get_exploits(self, villain_id: Optional[str] = None) -> List[str]:
        """
        Get exploitative adjustments for villain.

        Returns:
            List of exploit recommendations
        """
        stats = self.get_stats(villain_id)
        villain_type = VillainType(stats["type"])
        exploits = []

        if villain_type == VillainType.CALLING_STATION:
            exploits.extend([
                "Value bet thin - they call too much",
                "Never bluff - they don't fold",
                "Size up value bets",
                "Don't slow play"
            ])

        elif villain_type == VillainType.TIGHT_PASSIVE:
            exploits.extend([
                "Steal blinds aggressively",
                "Respect their raises - they have it",
                "Bluff when they check",
                "Fold to their aggression"
            ])

        elif villain_type == VillainType.MANIAC:
            exploits.extend([
                "Call down lighter",
                "Trap with strong hands",
                "Don't bluff - let them bluff",
                "Check-raise more"
            ])

        elif villain_type == VillainType.TAG:
            exploits.extend([
                "Play straightforward",
                "3-bet bluff occasionally",
                "Respect postflop aggression"
            ])

        elif villain_type == VillainType.LAG:
            exploits.extend([
                "Widen calling range",
                "3-bet for value more",
                "Don't over-adjust"
            ])

        # Add specific stat-based exploits
        if stats["fold_to_cbet"] > 0.60:
            exploits.append(f"C-bet bluff freely - they fold {stats['fold_to_cbet']:.0%}")
        if stats["fold_to_cbet"] < 0.35:
            exploits.append(f"Only c-bet for value - they call {1-stats['fold_to_cbet']:.0%}")

        return exploits

    def format_hud(self, villain_id: Optional[str] = None) -> str:
        """
        Format stats for HUD display.

        Returns:
            Formatted string for display
        """
        stats = self.get_stats(villain_id)

        if stats["hands_observed"] == 0:
            return "No data"

        return (
            f"{stats['type'].upper()} ({stats['hands_observed']} hands)\n"
            f"VPIP: {stats['vpip']:.0%} | AGG: {stats['aggression']:.0%}\n"
            f"Fold to c-bet: {stats['fold_to_cbet']:.0%}"
        )

    def _load(self):
        """Load stats from file."""
        if not self.persist_path:
            return

        path = Path(self.persist_path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            for vid, vdata in data.items():
                stats = VillainStats(
                    hands_observed=vdata.get("hands_observed", 0),
                    vpip_count=vdata.get("vpip_count", 0),
                    pfr_count=vdata.get("pfr_count", 0),
                    aggro_actions=vdata.get("aggro_actions", 0),
                    passive_actions=vdata.get("passive_actions", 0),
                    cbet_opportunities=vdata.get("cbet_opportunities", 0),
                    cbet_count=vdata.get("cbet_count", 0),
                    fold_to_cbet_opportunities=vdata.get("fold_to_cbet_opportunities", 0),
                    fold_to_cbet_count=vdata.get("fold_to_cbet_count", 0),
                    showdowns=vdata.get("showdowns", 0),
                    showdown_wins=vdata.get("showdown_wins", 0),
                    notes=vdata.get("notes", []),
                    name=vdata.get("name", "")
                )
                self.villains[vid] = stats

            logger.info(f"Loaded {len(self.villains)} villain profiles")
        except Exception as e:
            logger.error(f"Failed to load opponent data: {e}")

    def save(self):
        """Save stats to file."""
        if not self.persist_path:
            return

        try:
            data = {vid: v.to_dict() for vid, v in self.villains.items()}
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.villains)} villain profiles")
        except Exception as e:
            logger.error(f"Failed to save opponent data: {e}")


# Test
def test_opponent_tracker():
    """Test opponent tracker."""
    print("=== Opponent Tracker Test ===\n")

    tracker = OpponentTracker()

    # Simulate tracking a calling station
    tracker.set_villain("seat_3", "FishyMcFish")

    # Simulate 20 hands
    for i in range(20):
        # Calling station VPIPs 50% of hands
        vpip = i % 2 == 0
        tracker.record_hand(voluntarily_invested=vpip)

        if vpip:
            # They mostly call, rarely raise
            if i % 10 == 0:
                tracker.update("raise", voluntarily_invested=True)
            else:
                tracker.update("call", voluntarily_invested=True)

    tracker.add_note("Never folds draws")
    tracker.add_note("Calls river with any pair")

    print(tracker.format_hud())
    print()
    print("Stats:", tracker.get_stats())
    print()
    print("Exploits:")
    for exploit in tracker.get_exploits():
        print(f"  - {exploit}")


if __name__ == "__main__":
    test_opponent_tracker()
