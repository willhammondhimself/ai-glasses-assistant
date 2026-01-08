"""Poker equity calculator tool for agent."""
import random
from typing import List, Optional, Tuple
from itertools import combinations
from collections import Counter
from .base import BaseTool, ToolResult


# Card representation
RANKS = '23456789TJQKA'
SUITS = 'cdhs'  # clubs, diamonds, hearts, spades
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}


class PokerTool(BaseTool):
    """Calculate poker pot odds, equity, and expected value."""

    name = "poker"
    description = "Calculate poker pot odds, hand equity via Monte Carlo simulation, and expected value (EV)."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["pot_odds", "equity", "ev"],
                "description": "Calculation type: pot_odds, equity (hand strength), or ev (expected value)"
            },
            "pot": {"type": "number", "description": "Current pot size"},
            "bet": {"type": "number", "description": "Bet amount to call"},
            "hand": {"type": "string", "description": "Your hole cards, e.g., 'AhKd' or 'Ac Kd'"},
            "board": {"type": "string", "description": "Community cards, e.g., 'Qs Jh Tc' (optional)"},
            "opponents": {"type": "integer", "description": "Number of opponents (default: 1)"},
            "simulations": {"type": "integer", "description": "Monte Carlo iterations (default: 10000)"}
        },
        "required": ["action"]
    }

    async def execute(self, action: str, **kwargs) -> ToolResult:
        """Execute poker calculation."""
        try:
            if action == "pot_odds":
                return self._calculate_pot_odds(
                    kwargs.get("pot", 0),
                    kwargs.get("bet", 0)
                )
            elif action == "equity":
                return self._calculate_equity(
                    kwargs.get("hand", ""),
                    kwargs.get("board", ""),
                    kwargs.get("opponents", 1),
                    kwargs.get("simulations", 10000)
                )
            elif action == "ev":
                return self._calculate_ev(
                    kwargs.get("pot", 0),
                    kwargs.get("bet", 0),
                    kwargs.get("hand", ""),
                    kwargs.get("board", ""),
                    kwargs.get("opponents", 1),
                    kwargs.get("simulations", 10000)
                )

            return ToolResult(False, None, f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(False, None, f"Poker calc error: {str(e)}")

    def _calculate_pot_odds(self, pot: float, bet: float) -> ToolResult:
        """Calculate pot odds for a call."""
        # Convert string args from Gemini
        pot = float(pot) if pot else 0
        bet = float(bet) if bet else 0

        if bet <= 0:
            return ToolResult(False, None, "Bet must be greater than 0")

        total_pot = pot + bet
        pot_odds = bet / total_pot
        pot_odds_pct = pot_odds * 100
        implied_odds_ratio = total_pot / bet

        # Required equity to break even
        required_equity = pot_odds_pct

        return ToolResult(
            True,
            {
                "pot": pot,
                "bet": bet,
                "total_pot": total_pot,
                "pot_odds_pct": round(pot_odds_pct, 1),
                "pot_odds_ratio": f"{implied_odds_ratio:.1f}:1",
                "required_equity": round(required_equity, 1),
                "recommendation": f"You need {required_equity:.1f}% equity to call profitably"
            },
            f"Pot odds: {pot_odds_pct:.1f}% ({implied_odds_ratio:.1f}:1). Need {required_equity:.1f}% equity to call."
        )

    def _calculate_equity(self, hand: str, board: str, opponents: int, simulations: int) -> ToolResult:
        """Calculate hand equity via Monte Carlo simulation."""
        # Convert string args from Gemini
        opponents = int(opponents) if opponents else 1
        simulations = int(simulations) if simulations else 10000
        hand = str(hand) if hand else ""
        board = str(board) if board else ""

        if not hand:
            return ToolResult(False, None, "Hand cards required (e.g., 'AhKd')")

        hand_cards = self._parse_cards(hand)
        board_cards = self._parse_cards(board) if board else []

        if len(hand_cards) != 2:
            return ToolResult(False, None, f"Need exactly 2 hole cards, got {len(hand_cards)}")

        if len(board_cards) > 5:
            return ToolResult(False, None, f"Board can have max 5 cards, got {len(board_cards)}")

        # Run Monte Carlo simulation
        wins, ties, losses = self._monte_carlo(hand_cards, board_cards, opponents, simulations)

        equity = (wins + ties / 2) / simulations * 100
        win_pct = wins / simulations * 100
        tie_pct = ties / simulations * 100

        return ToolResult(
            True,
            {
                "hand": hand,
                "board": board or "none",
                "opponents": opponents,
                "simulations": simulations,
                "equity": round(equity, 1),
                "win_pct": round(win_pct, 1),
                "tie_pct": round(tie_pct, 1),
                "losses": losses
            },
            f"Equity: {equity:.1f}% vs {opponents} opponent(s). Win: {win_pct:.1f}%, Tie: {tie_pct:.1f}%"
        )

    def _calculate_ev(self, pot: float, bet: float, hand: str, board: str,
                      opponents: int, simulations: int) -> ToolResult:
        """Calculate expected value of a call."""
        # Convert string args from Gemini
        pot = float(pot) if pot else 0
        bet = float(bet) if bet else 0
        opponents = int(opponents) if opponents else 1
        simulations = int(simulations) if simulations else 10000

        # First get pot odds
        if bet <= 0:
            return ToolResult(False, None, "Bet must be greater than 0")

        # Then get equity
        if not hand:
            return ToolResult(False, None, "Hand cards required for EV calculation")

        equity_result = self._calculate_equity(hand, board, opponents, simulations)
        if not equity_result.success:
            return equity_result

        equity = equity_result.data["equity"] / 100
        pot_odds_result = self._calculate_pot_odds(pot, bet)

        total_pot = pot + bet
        ev_call = (equity * total_pot) - ((1 - equity) * bet)
        ev_fold = 0

        decision = "CALL" if ev_call > ev_fold else "FOLD"

        return ToolResult(
            True,
            {
                "pot": pot,
                "bet": bet,
                "equity_pct": round(equity * 100, 1),
                "ev_call": round(ev_call, 2),
                "ev_fold": 0,
                "decision": decision,
                "pot_odds_pct": pot_odds_result.data["pot_odds_pct"],
                "required_equity": pot_odds_result.data["required_equity"]
            },
            f"EV(call) = ${ev_call:+.2f}. Decision: {decision}. "
            f"Your equity ({equity*100:.1f}%) {'>' if ev_call > 0 else '<'} required ({pot_odds_result.data['required_equity']:.1f}%)"
        )

    def _parse_cards(self, cards_str: str) -> List[Tuple[str, str]]:
        """Parse card string into list of (rank, suit) tuples."""
        if not cards_str:
            return []

        # Remove spaces and normalize
        cards_str = cards_str.replace(" ", "").upper()

        # Handle suit symbols if used
        suit_map = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c',
                    's': 's', 'h': 'h', 'd': 'd', 'c': 'c'}

        cards = []
        i = 0
        while i < len(cards_str):
            if i + 1 < len(cards_str):
                rank = cards_str[i]
                suit = cards_str[i + 1].lower()

                if rank in RANKS and suit in SUITS:
                    cards.append((rank, suit))
                    i += 2
                else:
                    i += 1
            else:
                break

        return cards

    def _make_deck(self, exclude: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Create deck excluding specified cards."""
        exclude_set = set(exclude)
        return [(r, s) for r in RANKS for s in SUITS if (r, s) not in exclude_set]

    def _monte_carlo(self, hand: List[Tuple[str, str]], board: List[Tuple[str, str]],
                     opponents: int, simulations: int) -> Tuple[int, int, int]:
        """Run Monte Carlo simulation for equity calculation."""
        wins = ties = losses = 0
        deck = self._make_deck(hand + board)

        for _ in range(simulations):
            # Shuffle deck for this iteration
            shuffled = deck.copy()
            random.shuffle(shuffled)

            # Deal remaining board cards
            cards_needed = 5 - len(board)
            sim_board = board + shuffled[:cards_needed]
            shuffled = shuffled[cards_needed:]

            # Deal opponent hands
            opp_hands = []
            for i in range(opponents):
                opp_hands.append(shuffled[i*2:(i+1)*2])

            # Evaluate hands
            my_score = self._evaluate_hand(hand + sim_board)
            opp_scores = [self._evaluate_hand(opp + sim_board) for opp in opp_hands]
            best_opp = max(opp_scores)

            if my_score > best_opp:
                wins += 1
            elif my_score == best_opp:
                ties += 1
            else:
                losses += 1

        return wins, ties, losses

    def _evaluate_hand(self, cards: List[Tuple[str, str]]) -> int:
        """Evaluate 7 cards to get best 5-card hand score."""
        best_score = 0
        for combo in combinations(cards, 5):
            score = self._score_hand(list(combo))
            if score > best_score:
                best_score = score
        return best_score

    def _score_hand(self, cards: List[Tuple[str, str]]) -> int:
        """Score a 5-card hand. Higher is better."""
        ranks = sorted([RANK_VALUES[c[0]] for c in cards], reverse=True)
        suits = [c[1] for c in cards]

        is_flush = len(set(suits)) == 1
        is_straight = self._is_straight(ranks)

        rank_counts = Counter(ranks)
        counts = sorted(rank_counts.values(), reverse=True)

        # Hand rankings (higher base = better hand)
        # Royal/Straight flush: 8, Quads: 7, Full house: 6, Flush: 5,
        # Straight: 4, Trips: 3, Two pair: 2, Pair: 1, High card: 0

        if is_straight and is_flush:
            return 8_000000 + max(ranks)
        if counts == [4, 1]:
            return 7_000000 + self._tiebreaker(rank_counts)
        if counts == [3, 2]:
            return 6_000000 + self._tiebreaker(rank_counts)
        if is_flush:
            return 5_000000 + sum(r * (13 ** i) for i, r in enumerate(ranks))
        if is_straight:
            return 4_000000 + max(ranks)
        if counts == [3, 1, 1]:
            return 3_000000 + self._tiebreaker(rank_counts)
        if counts == [2, 2, 1]:
            return 2_000000 + self._tiebreaker(rank_counts)
        if counts == [2, 1, 1, 1]:
            return 1_000000 + self._tiebreaker(rank_counts)

        return sum(r * (13 ** i) for i, r in enumerate(ranks))

    def _is_straight(self, ranks: List[int]) -> bool:
        """Check if sorted ranks form a straight."""
        unique = sorted(set(ranks))
        if len(unique) != 5:
            return False

        # Check normal straight
        if unique[-1] - unique[0] == 4:
            return True

        # Check wheel (A-2-3-4-5)
        if unique == [0, 1, 2, 3, 12]:  # 2,3,4,5,A
            return True

        return False

    def _tiebreaker(self, rank_counts: Counter) -> int:
        """Create tiebreaker score from rank counts."""
        # Sort by count (desc) then by rank (desc)
        sorted_ranks = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        return sum(r * (13 ** (10 + i * 2)) * c for i, (r, c) in enumerate(sorted_ranks))
