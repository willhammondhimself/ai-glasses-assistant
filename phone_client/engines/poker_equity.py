"""
Monte Carlo poker equity calculator using treys library.
Target: <50ms for 10,000 simulations.
"""
import time
import random
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy import treys
treys = None


def _get_treys():
    """Lazy load treys library."""
    global treys
    if treys is None:
        try:
            import treys as _treys
            treys = _treys
        except ImportError:
            logger.warning("treys not installed. Run: pip install treys")
    return treys


@dataclass
class EquityResult:
    """Result of equity calculation."""
    equity: float           # 0.0 - 1.0
    win_probability: float
    tie_probability: float
    simulations: int
    latency_ms: float


class PokerEquityEngine:
    """
    Fast Monte Carlo equity calculator.

    Features:
    - 10,000 trial simulations
    - <50ms computation time
    - Supports any board state (preflop to river)
    - Range-based villain modeling
    """

    # Predefined villain ranges (top X% of hands)
    RANGES = {
        "random": 1.0,     # All hands
        "tight": 0.15,     # Top 15% - 77+, AK, AQ, AJs+, KQs
        "loose": 0.40,     # Top 40%
        "very_tight": 0.10,  # Top 10%
        "very_loose": 0.60,  # Top 60%
    }

    def __init__(self, default_simulations: int = 10000):
        """
        Initialize poker equity engine.

        Args:
            default_simulations: Default number of Monte Carlo simulations
        """
        self.default_simulations = default_simulations
        self._treys = _get_treys()
        self._evaluator = None

        if self._treys:
            self._evaluator = self._treys.Evaluator()

    @property
    def is_available(self) -> bool:
        """Check if treys library is available."""
        return self._treys is not None and self._evaluator is not None

    def parse_card(self, card_str: str) -> Optional[int]:
        """
        Parse card string like 'Ah' to treys format.

        Args:
            card_str: Card string (e.g., "Ah", "Kc", "9s")

        Returns:
            Treys card integer or None if invalid
        """
        if not self._treys:
            return None

        try:
            # Handle different formats
            card_str = card_str.strip()
            if len(card_str) == 2:
                return self._treys.Card.new(card_str)
            return None
        except Exception as e:
            logger.warning(f"Failed to parse card '{card_str}': {e}")
            return None

    def calculate_equity(
        self,
        hero_cards: List[str],
        board: List[str] = None,
        villain_range: str = "random",
        simulations: int = None
    ) -> EquityResult:
        """
        Calculate hero equity via Monte Carlo simulation.

        Args:
            hero_cards: ["Ah", "Kc"]
            board: ["9s", "7h", "3d"] or []
            villain_range: "random", "tight", "loose", or specific range
            simulations: Number of trials (default 10,000)

        Returns:
            EquityResult with equity percentage and stats
        """
        if not self.is_available:
            logger.error("treys not available")
            return EquityResult(
                equity=0.5,
                win_probability=0.5,
                tie_probability=0.0,
                simulations=0,
                latency_ms=0
            )

        simulations = simulations or self.default_simulations
        board = board or []
        start = time.perf_counter()

        # Parse hero cards
        hero = []
        for c in hero_cards:
            parsed = self.parse_card(c)
            if parsed is not None:
                hero.append(parsed)

        if len(hero) != 2:
            logger.error(f"Invalid hero cards: {hero_cards}")
            return EquityResult(0.5, 0.5, 0.0, 0, 0)

        # Parse board cards
        board_cards = []
        for c in board:
            parsed = self.parse_card(c)
            if parsed is not None:
                board_cards.append(parsed)

        # Run simulation
        wins, ties = self._run_simulation(
            hero, board_cards, villain_range, simulations
        )

        latency = (time.perf_counter() - start) * 1000

        return EquityResult(
            equity=(wins + ties * 0.5) / simulations,
            win_probability=wins / simulations,
            tie_probability=ties / simulations,
            simulations=simulations,
            latency_ms=latency
        )

    def _run_simulation(
        self,
        hero: List[int],
        board_cards: List[int],
        villain_range: str,
        simulations: int
    ) -> Tuple[int, int]:
        """Run Monte Carlo simulation."""
        wins = 0
        ties = 0

        # Create list of all cards except hero and board
        dead_cards = set(hero + board_cards)

        # Build deck
        deck = [c for c in self._treys.Deck().cards if c not in dead_cards]

        for _ in range(simulations):
            # Shuffle deck copy
            sim_deck = deck.copy()
            random.shuffle(sim_deck)

            # Deal villain cards based on range
            villain = self._deal_villain(sim_deck, villain_range)
            if not villain:
                continue

            # Remove villain cards from deck
            for v in villain:
                if v in sim_deck:
                    sim_deck.remove(v)

            # Complete the board
            remaining = 5 - len(board_cards)
            runout = board_cards + sim_deck[:remaining]

            # Evaluate hands
            hero_score = self._evaluator.evaluate(runout, hero)
            villain_score = self._evaluator.evaluate(runout, villain)

            if hero_score < villain_score:  # Lower is better in treys
                wins += 1
            elif hero_score == villain_score:
                ties += 1

        return wins, ties

    def _deal_villain(self, deck: List[int], range_type: str) -> List[int]:
        """
        Deal villain cards based on range.

        Args:
            deck: Available deck (shuffled)
            range_type: "random", "tight", "loose", etc.

        Returns:
            Two villain cards
        """
        if range_type == "random":
            return deck[:2]

        # For other ranges, we need to check if dealt cards are in range
        max_attempts = 100
        for _ in range(max_attempts):
            cards = deck[:2]
            if self._is_in_range(cards, range_type):
                return cards
            random.shuffle(deck)

        # Fallback to random if we can't find a hand in range
        return deck[:2]

    def _is_in_range(self, cards: List[int], range_type: str) -> bool:
        """
        Check if cards fall within specified range.

        Args:
            cards: Two villain cards
            range_type: "tight", "loose", etc.

        Returns:
            True if cards are in range
        """
        Card = self._treys.Card

        rank1 = Card.get_rank_int(cards[0])
        rank2 = Card.get_rank_int(cards[1])
        suited = Card.get_suit_int(cards[0]) == Card.get_suit_int(cards[1])

        # Sort ranks (higher first)
        high = max(rank1, rank2)
        low = min(rank1, rank2)
        is_pair = rank1 == rank2
        gap = high - low

        if range_type == "tight":
            # Top 15%: pairs 77+, AK, AQ, AJs+, KQs
            if is_pair and high >= 5:  # 7 = rank 5 (0-indexed, 2=0, A=12)
                return True
            if high == 12:  # Ace
                if low >= 10:  # Queen or better
                    return True
                if suited and low >= 9:  # ATs+
                    return True
            if high == 11 and low == 10 and suited:  # KQs
                return True
            return False

        elif range_type == "loose":
            # Top 40%: any pair, any suited connector, any broadway
            if is_pair:
                return True
            if suited and gap <= 1:  # Suited connectors
                return True
            if low >= 8:  # T or better (broadway)
                return True
            if high == 12:  # Any ace
                return True
            return False

        elif range_type == "very_tight":
            # Top 10%: pairs TT+, AK, AQs
            if is_pair and high >= 8:  # TT+
                return True
            if high == 12 and low == 11:  # AK
                return True
            if high == 12 and low == 10 and suited:  # AQs
                return True
            return False

        elif range_type == "very_loose":
            # Top 60%: almost any playable hand
            if is_pair:
                return True
            if high >= 6:  # Any 8+
                return True
            if suited:
                return True
            return False

        return True  # Default: accept any hand

    def quick_equity(
        self,
        hero: str,
        board: str = "",
        villain: str = "random"
    ) -> float:
        """
        Quick equity calculation from string input.

        Args:
            hero: "AhKc" or "Ah Kc"
            board: "9s7h3d" or "9s 7h 3d" or ""
            villain: Range string

        Returns:
            Equity as float (0.0 - 1.0)
        """
        # Parse hero cards
        hero_cards = self._parse_hand_string(hero)
        board_cards = self._parse_hand_string(board) if board else []

        result = self.calculate_equity(hero_cards, board_cards, villain)
        return result.equity

    def _parse_hand_string(self, hand_str: str) -> List[str]:
        """Parse hand string like 'AhKc' or 'Ah Kc' into list of cards."""
        hand_str = hand_str.replace(" ", "")
        cards = []
        i = 0
        while i < len(hand_str) - 1:
            cards.append(hand_str[i:i+2])
            i += 2
        return cards


# Test
if __name__ == "__main__":
    print("=== Poker Equity Engine Test ===\n")

    engine = PokerEquityEngine()

    if not engine.is_available:
        print("treys not available. Install with: pip install treys")
    else:
        # Test 1: AKs vs random on flop
        print("Test 1: AhKh vs random on 9s7h3d")
        result = engine.calculate_equity(
            hero_cards=["Ah", "Kh"],
            board=["9s", "7h", "3d"],
            villain_range="random",
            simulations=10000
        )
        print(f"  Equity: {result.equity:.1%}")
        print(f"  Win: {result.win_probability:.1%}")
        print(f"  Tie: {result.tie_probability:.1%}")
        print(f"  Latency: {result.latency_ms:.1f}ms")
        print()

        # Test 2: AA vs tight range preflop
        print("Test 2: AsAh vs tight preflop")
        result = engine.calculate_equity(
            hero_cards=["As", "Ah"],
            board=[],
            villain_range="tight",
            simulations=10000
        )
        print(f"  Equity: {result.equity:.1%}")
        print(f"  Latency: {result.latency_ms:.1f}ms")
        print()

        # Test 3: Quick equity
        print("Test 3: Quick equity - AhKc on Qs Jd Th")
        equity = engine.quick_equity("AhKc", "QsJdTh", "random")
        print(f"  Equity: {equity:.1%}")
