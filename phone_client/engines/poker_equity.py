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


@dataclass
class MultiOpponentResult:
    """Result of multi-opponent equity calculation with statistical confidence."""
    equity: float               # 0.0 - 1.0
    win_probability: float
    tie_probability: float
    ci_low: float              # Wilson CI lower bound
    ci_high: float             # Wilson CI upper bound
    std_error: float           # Standard error of equity estimate
    variance: float            # Sample variance
    convergence: float         # Convergence metric (lower = more stable)
    simulations: int
    opponents: int
    latency_ms: float
    opponent_ranges: List[str]


class MultiOpponentEquity:
    """
    Multi-opponent equity calculator with statistical rigor.

    Features:
    - 2-8 opponent support with predefined ranges
    - Wilson score confidence intervals (95% CI)
    - Variance and standard error tracking
    - Convergence bounds for Monte Carlo stability
    - <100ms for 10K simulations vs 3 opponents

    Statistical Foundations:
    - Wilson score interval: Better than normal approximation for proportions
    - Variance: σ² = p(1-p)/n for binomial trials
    - Standard Error: SE = sqrt(p(1-p)/n)
    - Convergence: |rolling_avg[n] - rolling_avg[n-1000]| < threshold
    """

    # Predefined villain ranges (top X% of hands)
    RANGES = {
        "random": 1.0,       # All hands (169 combos)
        "tight": 0.15,       # Top 15%: 77+, AK, AQ, AJs+, KQs
        "loose": 0.40,       # Top 40%: pairs, broadway, suited connectors
        "very_tight": 0.10,  # Top 10%: TT+, AK, AQs
        "very_loose": 0.60,  # Top 60%: most playable hands
        "fish": 0.80,        # Top 80%: recreational player
    }

    def __init__(self, default_simulations: int = 10000):
        """Initialize multi-opponent equity calculator."""
        self.default_simulations = default_simulations
        self._treys = _get_treys()
        self._evaluator = None

        if self._treys:
            self._evaluator = self._treys.Evaluator()

    @property
    def is_available(self) -> bool:
        """Check if treys library is available."""
        return self._treys is not None and self._evaluator is not None

    def calculate(
        self,
        hero_cards: List[str],
        board: List[str] = None,
        opponent_ranges: List[str] = None,
        opponents: int = 1,
        simulations: int = None
    ) -> MultiOpponentResult:
        """
        Calculate hero equity vs multiple opponents with statistical confidence.

        Args:
            hero_cards: ["Ah", "Kc"] - Hero's hole cards
            board: ["9s", "7h", "3d"] - Community cards (0-5)
            opponent_ranges: ["tight", "loose"] - Range per opponent
            opponents: Number of opponents (1-8)
            simulations: Monte Carlo iterations (default 10,000)

        Returns:
            MultiOpponentResult with equity, confidence intervals, and variance

        Statistical Notes:
            - Wilson CI preferred over normal approximation for p near 0 or 1
            - Variance tracks simulation stability
            - Convergence metric indicates when results stabilize
        """
        if not self.is_available:
            logger.error("treys not available")
            return self._empty_result(opponents)

        simulations = simulations or self.default_simulations
        board = board or []
        opponents = max(1, min(8, opponents))  # Clamp to 1-8

        # Assign ranges to each opponent
        if opponent_ranges is None:
            opponent_ranges = ["random"] * opponents
        elif len(opponent_ranges) < opponents:
            # Extend with "random" if not enough ranges specified
            opponent_ranges = opponent_ranges + ["random"] * (opponents - len(opponent_ranges))
        opponent_ranges = opponent_ranges[:opponents]

        start = time.perf_counter()

        # Parse hero cards
        hero = self._parse_cards(hero_cards)
        if len(hero) != 2:
            logger.error(f"Invalid hero cards: {hero_cards}")
            return self._empty_result(opponents)

        # Parse board
        board_cards = self._parse_cards(board)

        # Run Monte Carlo with convergence tracking
        wins, ties, variance_data = self._run_multi_simulation(
            hero, board_cards, opponent_ranges, simulations
        )

        latency = (time.perf_counter() - start) * 1000

        # Calculate statistics
        n = simulations
        win_prob = wins / n
        tie_prob = ties / n
        equity = win_prob + (tie_prob * 0.5)

        # Wilson score confidence interval (95%, z=1.96)
        ci_low, ci_high = self._wilson_ci(wins + ties // 2, n, z=1.96)

        # Variance and standard error for binomial proportion
        # σ² = p(1-p), SE = sqrt(p(1-p)/n)
        variance = equity * (1 - equity)
        std_error = (variance / n) ** 0.5

        # Convergence metric: how much equity changed in last 10% of sims
        convergence = variance_data.get("convergence", 0.0)

        return MultiOpponentResult(
            equity=equity,
            win_probability=win_prob,
            tie_probability=tie_prob,
            ci_low=ci_low,
            ci_high=ci_high,
            std_error=std_error,
            variance=variance,
            convergence=convergence,
            simulations=n,
            opponents=opponents,
            latency_ms=latency,
            opponent_ranges=opponent_ranges
        )

    def _wilson_ci(self, wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval.

        Better than normal approximation when p is near 0 or 1.

        Formula:
            (p + z²/2n ± z*sqrt(p(1-p)/n + z²/4n²)) / (1 + z²/n)

        Args:
            wins: Number of successes
            n: Total trials
            z: Z-score for confidence level (1.96 for 95%)

        Returns:
            (lower_bound, upper_bound) as floats
        """
        if n == 0:
            return (0.0, 1.0)

        p = wins / n
        z2 = z ** 2

        # Wilson score interval
        denominator = 1 + z2 / n
        center = p + z2 / (2 * n)
        margin = z * ((p * (1 - p) / n + z2 / (4 * n ** 2)) ** 0.5)

        low = (center - margin) / denominator
        high = (center + margin) / denominator

        # Clamp to [0, 1]
        return (max(0.0, low), min(1.0, high))

    def _run_multi_simulation(
        self,
        hero: List[int],
        board_cards: List[int],
        opponent_ranges: List[str],
        simulations: int
    ) -> Tuple[int, int, dict]:
        """
        Run Monte Carlo simulation vs multiple opponents.

        Returns:
            (wins, ties, variance_data)
        """
        wins = 0
        ties = 0
        opponents = len(opponent_ranges)

        # Dead cards: hero + board
        dead_cards = set(hero + board_cards)

        # Build deck
        deck = [c for c in self._treys.Deck().cards if c not in dead_cards]

        # Track convergence (equity after each 10% chunk)
        checkpoints = []
        checkpoint_interval = max(1, simulations // 10)

        for i in range(simulations):
            # Shuffle deck
            sim_deck = deck.copy()
            random.shuffle(sim_deck)

            # Deal all opponents
            all_villains = []
            used_cards = set()

            valid_deal = True
            for range_type in opponent_ranges:
                villain = self._deal_villain_from_range(sim_deck, range_type, used_cards)
                if villain is None:
                    valid_deal = False
                    break
                all_villains.append(villain)
                used_cards.update(villain)

            if not valid_deal:
                continue

            # Remove villain cards from deck for runout
            runout_deck = [c for c in sim_deck if c not in used_cards]

            # Complete the board
            remaining = 5 - len(board_cards)
            runout = board_cards + runout_deck[:remaining]

            # Evaluate hero hand
            hero_score = self._evaluator.evaluate(runout, hero)

            # Evaluate all villain hands, find best
            best_villain_score = float('inf')
            for villain in all_villains:
                score = self._evaluator.evaluate(runout, villain)
                best_villain_score = min(best_villain_score, score)

            # Hero wins if beats ALL opponents (lowest score wins in treys)
            if hero_score < best_villain_score:
                wins += 1
            elif hero_score == best_villain_score:
                ties += 1

            # Track convergence at checkpoints
            if (i + 1) % checkpoint_interval == 0:
                current_equity = (wins + ties * 0.5) / (i + 1)
                checkpoints.append(current_equity)

        # Calculate convergence: max change between consecutive checkpoints
        convergence = 0.0
        if len(checkpoints) >= 2:
            changes = [abs(checkpoints[i] - checkpoints[i-1]) for i in range(1, len(checkpoints))]
            convergence = max(changes) if changes else 0.0

        return wins, ties, {"convergence": convergence, "checkpoints": checkpoints}

    def _deal_villain_from_range(
        self,
        deck: List[int],
        range_type: str,
        used_cards: set
    ) -> Optional[List[int]]:
        """
        Deal villain cards from specified range.

        Args:
            deck: Available deck (shuffled)
            range_type: "random", "tight", "loose", etc.
            used_cards: Cards already dealt to other players

        Returns:
            Two villain cards or None if can't find valid hand
        """
        available = [c for c in deck if c not in used_cards]

        if len(available) < 2:
            return None

        if range_type == "random":
            return available[:2]

        # Try to find a hand in range
        max_attempts = 50
        for _ in range(max_attempts):
            random.shuffle(available)
            cards = available[:2]
            if self._is_in_range(cards, range_type):
                return cards

        # Fallback to random if we can't find a hand in range
        return available[:2]

    def _is_in_range(self, cards: List[int], range_type: str) -> bool:
        """Check if cards fall within specified range."""
        Card = self._treys.Card

        rank1 = Card.get_rank_int(cards[0])
        rank2 = Card.get_rank_int(cards[1])
        suited = Card.get_suit_int(cards[0]) == Card.get_suit_int(cards[1])

        high = max(rank1, rank2)
        low = min(rank1, rank2)
        is_pair = rank1 == rank2
        gap = high - low

        if range_type == "tight":
            # Top 15%: pairs 77+, AK, AQ, AJs+, KQs
            if is_pair and high >= 5:  # 7 = rank 5
                return True
            if high == 12:  # Ace
                if low >= 10:  # Queen+
                    return True
                if suited and low >= 9:  # ATs+
                    return True
            if high == 11 and low == 10 and suited:  # KQs
                return True
            return False

        elif range_type == "loose":
            # Top 40%: any pair, suited connectors, broadway
            if is_pair:
                return True
            if suited and gap <= 1:
                return True
            if low >= 8:  # Broadway
                return True
            if high == 12:  # Any ace
                return True
            return False

        elif range_type == "very_tight":
            # Top 10%: TT+, AK, AQs
            if is_pair and high >= 8:
                return True
            if high == 12 and low == 11:  # AK
                return True
            if high == 12 and low == 10 and suited:  # AQs
                return True
            return False

        elif range_type == "very_loose" or range_type == "fish":
            # Top 60-80%
            if is_pair:
                return True
            if high >= 6:
                return True
            if suited:
                return True
            return False

        return True

    def _parse_cards(self, card_strs: List[str]) -> List[int]:
        """Parse list of card strings to treys format."""
        cards = []
        for c in card_strs:
            try:
                c = c.strip()
                if len(c) == 2:
                    cards.append(self._treys.Card.new(c))
            except Exception:
                pass
        return cards

    def _empty_result(self, opponents: int) -> MultiOpponentResult:
        """Return empty result for error cases."""
        return MultiOpponentResult(
            equity=0.5, win_probability=0.5, tie_probability=0.0,
            ci_low=0.0, ci_high=1.0, std_error=0.5, variance=0.25,
            convergence=1.0, simulations=0, opponents=opponents,
            latency_ms=0, opponent_ranges=[]
        )

    def quick_multi_equity(
        self,
        hero: str,
        board: str = "",
        opponents: int = 2,
        ranges: str = "random"
    ) -> dict:
        """
        Quick multi-opponent equity from string input.

        Args:
            hero: "AhKc" or "Ah Kc"
            board: "9s7h3d" or ""
            opponents: Number of opponents
            ranges: Single range for all opponents or comma-separated

        Returns:
            Dict with equity, CI, and statistics
        """
        hero_cards = self._parse_hand_string(hero)
        board_cards = self._parse_hand_string(board) if board else []

        # Parse ranges
        if "," in ranges:
            opponent_ranges = [r.strip() for r in ranges.split(",")]
        else:
            opponent_ranges = [ranges] * opponents

        result = self.calculate(
            hero_cards=hero_cards,
            board=board_cards,
            opponent_ranges=opponent_ranges,
            opponents=opponents
        )

        return {
            "equity": result.equity,
            "equity_pct": f"{result.equity:.1%}",
            "ci_95": f"[{result.ci_low:.1%}, {result.ci_high:.1%}]",
            "std_error": result.std_error,
            "convergence": result.convergence,
            "opponents": result.opponents,
            "latency_ms": result.latency_ms
        }

    def _parse_hand_string(self, hand_str: str) -> List[str]:
        """Parse hand string like 'AhKc' into list of cards."""
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
        print()

        # === Multi-Opponent Equity Tests ===
        print("=== Multi-Opponent Equity Tests ===\n")
        multi = MultiOpponentEquity()

        # Test 4: AA vs 2 random opponents preflop
        print("Test 4: AsAh vs 2 random opponents preflop")
        result = multi.calculate(
            hero_cards=["As", "Ah"],
            board=[],
            opponents=2,
            simulations=10000
        )
        print(f"  Equity: {result.equity:.1%}")
        print(f"  95% CI: [{result.ci_low:.1%}, {result.ci_high:.1%}]")
        print(f"  Std Error: {result.std_error:.4f}")
        print(f"  Convergence: {result.convergence:.4f}")
        print(f"  Latency: {result.latency_ms:.1f}ms")
        print()

        # Test 5: AKs vs 3 tight opponents
        print("Test 5: AhKh vs 3 tight opponents preflop")
        result = multi.calculate(
            hero_cards=["Ah", "Kh"],
            board=[],
            opponent_ranges=["tight", "tight", "tight"],
            opponents=3,
            simulations=10000
        )
        print(f"  Equity: {result.equity:.1%}")
        print(f"  95% CI: [{result.ci_low:.1%}, {result.ci_high:.1%}]")
        print(f"  Opponents: {result.opponents}")
        print(f"  Latency: {result.latency_ms:.1f}ms")
        print()

        # Test 6: Quick multi-opponent equity
        print("Test 6: Quick multi - QsQh vs 4 loose opponents")
        stats = multi.quick_multi_equity("QsQh", "", opponents=4, ranges="loose")
        print(f"  Equity: {stats['equity_pct']}")
        print(f"  95% CI: {stats['ci_95']}")
        print(f"  Latency: {stats['latency_ms']:.1f}ms")
        print()

        # Test 7: Mixed ranges
        print("Test 7: AcKd vs tight + loose opponents on flop")
        result = multi.calculate(
            hero_cards=["Ac", "Kd"],
            board=["Qs", "Jh", "Td"],  # Flopped Broadway straight
            opponent_ranges=["tight", "loose"],
            opponents=2,
            simulations=10000
        )
        print(f"  Equity: {result.equity:.1%} (should be ~100% - made straight)")
        print(f"  95% CI: [{result.ci_low:.1%}, {result.ci_high:.1%}]")
        print(f"  Latency: {result.latency_ms:.1f}ms")
