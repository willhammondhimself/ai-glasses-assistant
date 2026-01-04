"""
Poker Session - Track hands, stats, and session context.
"""
import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HandRecord:
    """Record of a single hand played."""
    hand_number: int
    timestamp: float

    # Cards
    hero_cards: List[str]
    board: List[str]

    # Context
    position: str
    stack_bb: float
    pot_bb: float

    # Villain
    villain_type: str
    villain_stats: Dict[str, Any]

    # Action
    action_taken: str  # FOLD, CALL, RAISE
    sizing: str = ""
    action_sequence: str = ""

    # AI recommendation
    ai_action: str = ""
    ai_reasoning: str = ""
    ai_equity: float = 0.0
    followed_ai: bool = True

    # Result
    result_bb: float = 0.0  # Won/lost in bb
    showdown: bool = False
    notes: str = ""

    # Costs
    api_cost: float = 0.0


@dataclass
class SessionStats:
    """Aggregated session statistics."""
    hands_played: int = 0
    hands_won: int = 0
    hands_lost: int = 0

    total_profit_bb: float = 0.0
    biggest_win_bb: float = 0.0
    biggest_loss_bb: float = 0.0

    vpip_count: int = 0
    pfr_count: int = 0
    showdowns: int = 0
    showdown_wins: int = 0

    ai_recommendations: int = 0
    ai_followed: int = 0
    ai_correct_when_followed: int = 0
    ai_correct_when_ignored: int = 0

    total_api_cost: float = 0.0

    @property
    def win_rate_bb_per_100(self) -> float:
        """Win rate in bb/100 hands."""
        if self.hands_played == 0:
            return 0.0
        return (self.total_profit_bb / self.hands_played) * 100

    @property
    def vpip(self) -> float:
        """VPIP percentage."""
        if self.hands_played == 0:
            return 0.0
        return self.vpip_count / self.hands_played

    @property
    def pfr(self) -> float:
        """PFR percentage."""
        if self.hands_played == 0:
            return 0.0
        return self.pfr_count / self.hands_played

    @property
    def showdown_win_rate(self) -> float:
        """Win rate at showdown."""
        if self.showdowns == 0:
            return 0.0
        return self.showdown_wins / self.showdowns

    @property
    def ai_follow_rate(self) -> float:
        """How often hero followed AI recommendation."""
        if self.ai_recommendations == 0:
            return 0.0
        return self.ai_followed / self.ai_recommendations


class PokerSession:
    """
    Track a poker session for analysis.

    Features:
    - Hand-by-hand recording
    - Session statistics
    - AI accuracy tracking
    - Export for post-session review
    """

    def __init__(
        self,
        stakes: str = "$0.25/$0.50",
        bb_value: float = 0.50,
        persist_dir: Optional[str] = None
    ):
        """
        Initialize poker session.

        Args:
            stakes: Stakes description
            bb_value: Value of 1 big blind in USD
            persist_dir: Directory to save session data
        """
        self.stakes = stakes
        self.bb_value = bb_value
        self.persist_dir = persist_dir

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.hands: List[HandRecord] = []
        self.stats = SessionStats()

        # Current hand state
        self.current_hand_number = 0
        self.position = "BTN"
        self.stack_bb = 100.0

    def start_hand(
        self,
        hero_cards: List[str],
        position: str,
        stack_bb: float
    ) -> int:
        """
        Start tracking a new hand.

        Args:
            hero_cards: Hero's hole cards
            position: Hero's position
            stack_bb: Hero's stack in big blinds

        Returns:
            Hand number
        """
        self.current_hand_number += 1
        self.position = position
        self.stack_bb = stack_bb

        self._current_hand = HandRecord(
            hand_number=self.current_hand_number,
            timestamp=time.time(),
            hero_cards=hero_cards,
            board=[],
            position=position,
            stack_bb=stack_bb,
            pot_bb=1.5,  # Default SB+BB
            villain_type="unknown",
            villain_stats={},
            action_taken=""
        )

        return self.current_hand_number

    def update_board(self, board: List[str]):
        """Update community cards."""
        if hasattr(self, '_current_hand'):
            self._current_hand.board = board

    def update_pot(self, pot_bb: float):
        """Update pot size."""
        if hasattr(self, '_current_hand'):
            self._current_hand.pot_bb = pot_bb

    def set_villain(self, villain_type: str, villain_stats: Dict):
        """Set villain info for current hand."""
        if hasattr(self, '_current_hand'):
            self._current_hand.villain_type = villain_type
            self._current_hand.villain_stats = villain_stats

    def record_ai_recommendation(
        self,
        action: str,
        reasoning: str,
        equity: float,
        api_cost: float
    ):
        """Record AI's recommendation for the hand."""
        if hasattr(self, '_current_hand'):
            self._current_hand.ai_action = action
            self._current_hand.ai_reasoning = reasoning
            self._current_hand.ai_equity = equity
            self._current_hand.api_cost = api_cost
            self.stats.ai_recommendations += 1
            self.stats.total_api_cost += api_cost

    def record_action(
        self,
        action: str,
        sizing: str = "",
        action_sequence: str = ""
    ):
        """Record hero's actual action."""
        if hasattr(self, '_current_hand'):
            self._current_hand.action_taken = action
            self._current_hand.sizing = sizing
            self._current_hand.action_sequence = action_sequence

            # Track if followed AI
            if self._current_hand.ai_action:
                followed = action.upper() == self._current_hand.ai_action.upper()
                self._current_hand.followed_ai = followed
                if followed:
                    self.stats.ai_followed += 1

            # Track VPIP/PFR
            if action.upper() in ["CALL", "RAISE"]:
                self.stats.vpip_count += 1
            if action.upper() == "RAISE":
                self.stats.pfr_count += 1

    def end_hand(
        self,
        result_bb: float,
        showdown: bool = False,
        notes: str = ""
    ):
        """
        End the current hand and record result.

        Args:
            result_bb: Won (+) or lost (-) in big blinds
            showdown: Whether hand went to showdown
            notes: Optional notes about the hand
        """
        if not hasattr(self, '_current_hand'):
            return

        hand = self._current_hand
        hand.result_bb = result_bb
        hand.showdown = showdown
        hand.notes = notes

        # Update stats
        self.stats.hands_played += 1
        self.stats.total_profit_bb += result_bb

        if result_bb > 0:
            self.stats.hands_won += 1
            if result_bb > self.stats.biggest_win_bb:
                self.stats.biggest_win_bb = result_bb
        elif result_bb < 0:
            self.stats.hands_lost += 1
            if result_bb < self.stats.biggest_loss_bb:
                self.stats.biggest_loss_bb = result_bb

        if showdown:
            self.stats.showdowns += 1
            if result_bb > 0:
                self.stats.showdown_wins += 1

        # Track AI accuracy
        if hand.ai_action:
            # Did AI predict correctly?
            ai_would_win = (
                (hand.ai_action.upper() != "FOLD" and result_bb > 0) or
                (hand.ai_action.upper() == "FOLD" and result_bb <= 0)
            )

            if hand.followed_ai and ai_would_win:
                self.stats.ai_correct_when_followed += 1
            elif not hand.followed_ai and not ai_would_win:
                self.stats.ai_correct_when_ignored += 1

        # Save hand
        self.hands.append(hand)
        delattr(self, '_current_hand')

        # Auto-save periodically
        if self.persist_dir and len(self.hands) % 10 == 0:
            self.save()

    def get_hands_for_review(
        self,
        filter_losses: bool = False,
        filter_big_pots: bool = False,
        filter_showdowns: bool = False,
        limit: int = 20
    ) -> List[Dict]:
        """
        Get hands formatted for AI review.

        Args:
            filter_losses: Only include losing hands
            filter_big_pots: Only include big pots (>20bb)
            filter_showdowns: Only include showdown hands
            limit: Maximum hands to return

        Returns:
            List of hand dictionaries for review prompt
        """
        hands = self.hands.copy()

        if filter_losses:
            hands = [h for h in hands if h.result_bb < 0]
        if filter_big_pots:
            hands = [h for h in hands if h.pot_bb > 20]
        if filter_showdowns:
            hands = [h for h in hands if h.showdown]

        # Take most recent
        hands = hands[-limit:]

        return [
            {
                "hand_number": h.hand_number,
                "hero_cards": " ".join(h.hero_cards),
                "board": " ".join(h.board) if h.board else "Preflop",
                "position": h.position,
                "pot_bb": h.pot_bb,
                "villain_type": h.villain_type,
                "action_taken": f"{h.action_taken} {h.sizing}".strip(),
                "ai_action": h.ai_action,
                "followed_ai": h.followed_ai,
                "result": f"{h.result_bb:+.1f}bb",
                "notes": h.notes
            }
            for h in hands
        ]

    def get_summary(self) -> Dict:
        """Get session summary."""
        duration = time.time() - self.start_time

        return {
            "session_id": self.session_id,
            "stakes": self.stakes,
            "duration_minutes": duration / 60,
            "hands_played": self.stats.hands_played,
            "profit_bb": self.stats.total_profit_bb,
            "profit_usd": self.stats.total_profit_bb * self.bb_value,
            "win_rate_bb_100": self.stats.win_rate_bb_per_100,
            "vpip": self.stats.vpip,
            "pfr": self.stats.pfr,
            "showdown_win_rate": self.stats.showdown_win_rate,
            "biggest_win_bb": self.stats.biggest_win_bb,
            "biggest_loss_bb": self.stats.biggest_loss_bb,
            "ai_follow_rate": self.stats.ai_follow_rate,
            "total_api_cost": self.stats.total_api_cost,
            "net_profit_usd": (self.stats.total_profit_bb * self.bb_value) - self.stats.total_api_cost
        }

    def format_summary(self) -> str:
        """Format summary for display."""
        s = self.get_summary()

        return f"""Session: {s['session_id']}
Stakes: {s['stakes']} | Duration: {s['duration_minutes']:.0f}min

Hands: {s['hands_played']}
Profit: {s['profit_bb']:+.1f}bb (${s['profit_usd']:+.2f})
Win Rate: {s['win_rate_bb_100']:+.1f}bb/100

VPIP: {s['vpip']:.0%} | PFR: {s['pfr']:.0%}
Showdown Win: {s['showdown_win_rate']:.0%}

AI Cost: ${s['total_api_cost']:.2f}
Net Profit: ${s['net_profit_usd']:+.2f}
AI Follow Rate: {s['ai_follow_rate']:.0%}"""

    def save(self):
        """Save session to file."""
        if not self.persist_dir:
            return

        try:
            path = Path(self.persist_dir)
            path.mkdir(parents=True, exist_ok=True)

            filename = path / f"session_{self.session_id}.json"

            data = {
                "session_id": self.session_id,
                "stakes": self.stakes,
                "bb_value": self.bb_value,
                "start_time": self.start_time,
                "stats": asdict(self.stats),
                "hands": [asdict(h) for h in self.hands],
                "summary": self.get_summary()
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Session saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    @classmethod
    def load(cls, filepath: str) -> "PokerSession":
        """Load session from file."""
        with open(filepath) as f:
            data = json.load(f)

        session = cls(
            stakes=data["stakes"],
            bb_value=data["bb_value"]
        )
        session.session_id = data["session_id"]
        session.start_time = data["start_time"]

        # Restore stats
        for key, value in data["stats"].items():
            if hasattr(session.stats, key):
                setattr(session.stats, key, value)

        # Restore hands
        for h in data["hands"]:
            session.hands.append(HandRecord(**h))

        return session


# Test
def test_session():
    """Test poker session."""
    print("=== Poker Session Test ===\n")

    session = PokerSession(stakes="$0.25/$0.50", bb_value=0.50)

    # Simulate 5 hands
    for i in range(5):
        session.start_hand(
            hero_cards=["Ah", "Kc"],
            position="BTN",
            stack_bb=100
        )
        session.update_board(["9s", "7h", "3d"])
        session.update_pot(12.0)
        session.set_villain("calling_station", {"vpip": 0.45})

        session.record_ai_recommendation(
            action="RAISE",
            reasoning="Value bet vs calling station",
            equity=0.65,
            api_cost=0.009
        )

        session.record_action("RAISE", sizing="3/4 pot")

        # Alternate wins and losses
        result = 15.0 if i % 2 == 0 else -8.0
        session.end_hand(result_bb=result, showdown=True)

    print(session.format_summary())
    print()
    print("Hands for review:")
    for h in session.get_hands_for_review():
        print(f"  #{h['hand_number']}: {h['hero_cards']} -> {h['action_taken']} = {h['result']}")


if __name__ == "__main__":
    test_session()
