"""Live poker game mode for physical table play.

Voice-driven state management without OCR for casino/home games.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class LiveStreet(Enum):
    """Current street in live game."""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class LivePosition(Enum):
    """Positions at live table."""
    BTN = "BTN"
    SB = "SB"
    BB = "BB"
    UTG = "UTG"
    UTG1 = "UTG+1"
    UTG2 = "UTG+2"
    MP = "MP"
    MP1 = "MP+1"
    HJ = "HJ"
    CO = "CO"


@dataclass
class LiveTableState:
    """Current state of a live poker table.

    All values are set via voice commands rather than OCR.
    """
    # Game info
    game_type: str = "NL Hold'em"
    stakes: str = "1/2"  # e.g., "1/2", "2/5", "5/10"
    table_size: int = 9  # 6-max, 9-handed, etc.

    # Current hand state
    hole_cards: Optional[str] = None  # e.g., "Ah Kd"
    community_cards: List[str] = field(default_factory=list)  # ["Qs", "Jh", "Tc"]
    street: LiveStreet = LiveStreet.PREFLOP

    # Pot and betting
    pot: float = 0.0
    current_bet: float = 0.0
    my_stack: float = 0.0
    villain_stacks: Dict[str, float] = field(default_factory=dict)  # position -> stack

    # Position info
    my_position: Optional[LivePosition] = None
    button_seat: int = 1  # Which seat has the button (1-9)
    my_seat: int = 1
    active_players: int = 9
    players_in_hand: int = 0

    # Action tracking
    action_on_me: bool = False
    facing_bet: float = 0.0
    facing_raise: bool = False

    # Session tracking
    session_start: datetime = field(default_factory=datetime.now)
    hands_played: int = 0
    session_profit: float = 0.0

    def get_street_name(self) -> str:
        """Get current street as string."""
        return self.street.value

    def get_position_name(self) -> Optional[str]:
        """Get position as string."""
        return self.my_position.value if self.my_position else None

    def get_spr(self) -> float:
        """Calculate stack-to-pot ratio."""
        if self.pot > 0:
            return round(self.my_stack / self.pot, 1)
        return 0.0

    def get_pot_odds(self) -> Optional[float]:
        """Calculate pot odds if facing a bet."""
        if self.facing_bet > 0 and self.pot > 0:
            return round(self.facing_bet / (self.pot + self.facing_bet) * 100, 1)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "game_type": self.game_type,
            "stakes": self.stakes,
            "table_size": self.table_size,
            "hole_cards": self.hole_cards,
            "community_cards": self.community_cards,
            "street": self.street.value,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "my_stack": self.my_stack,
            "my_position": self.get_position_name(),
            "button_seat": self.button_seat,
            "my_seat": self.my_seat,
            "active_players": self.active_players,
            "players_in_hand": self.players_in_hand,
            "action_on_me": self.action_on_me,
            "facing_bet": self.facing_bet,
            "facing_raise": self.facing_raise,
            "spr": self.get_spr(),
            "pot_odds": self.get_pot_odds(),
            "hands_played": self.hands_played,
            "session_profit": self.session_profit
        }

    def to_engine_format(self) -> Dict[str, Any]:
        """Convert to format expected by PokerEngine."""
        return {
            "hole_cards": self.hole_cards.split() if self.hole_cards else [],
            "community_cards": self.community_cards,
            "pot_size": self.pot,
            "player_stack": self.my_stack,
            "position": self.get_position_name(),
            "street": self.street.value,
            "bet_to_call": self.facing_bet,
            "players_in_hand": self.players_in_hand or self.active_players
        }


class LiveModeManager:
    """Manager for live poker game state.

    Handles voice commands to update table state and provides
    poker advice for physical table play.
    """

    # Card parsing patterns
    CARD_PATTERN = re.compile(
        r'\b([AKQJT2-9])\s*(?:of\s+)?(hearts?|diamonds?|clubs?|spades?|h|d|c|s)\b',
        re.IGNORECASE
    )

    RANK_WORDS = {
        'ace': 'A', 'king': 'K', 'queen': 'Q', 'jack': 'J', 'ten': 'T',
        'nine': '9', 'eight': '8', 'seven': '7', 'six': '6', 'five': '5',
        'four': '4', 'three': '3', 'two': '2', 'deuce': '2'
    }

    SUIT_WORDS = {
        'hearts': 'h', 'heart': 'h', 'diamonds': 'd', 'diamond': 'd',
        'clubs': 'c', 'club': 'c', 'spades': 's', 'spade': 's',
        'h': 'h', 'd': 'd', 'c': 'c', 's': 's'
    }

    POSITION_ALIASES = {
        'button': LivePosition.BTN,
        'btn': LivePosition.BTN,
        'dealer': LivePosition.BTN,
        'small blind': LivePosition.SB,
        'small': LivePosition.SB,
        'sb': LivePosition.SB,
        'big blind': LivePosition.BB,
        'big': LivePosition.BB,
        'bb': LivePosition.BB,
        'under the gun': LivePosition.UTG,
        'utg': LivePosition.UTG,
        'early': LivePosition.UTG,
        'middle': LivePosition.MP,
        'mid': LivePosition.MP,
        'mp': LivePosition.MP,
        'hijack': LivePosition.HJ,
        'hj': LivePosition.HJ,
        'cutoff': LivePosition.CO,
        'cut off': LivePosition.CO,
        'co': LivePosition.CO,
        'late': LivePosition.CO
    }

    def __init__(self):
        """Initialize live mode manager."""
        self.state = LiveTableState()
        self.is_active = False
        self._command_history: List[Tuple[datetime, str]] = []

    def activate(self, stakes: str = "1/2", table_size: int = 9) -> Dict[str, Any]:
        """Activate live mode with table configuration.

        Args:
            stakes: Blind levels (e.g., "1/2", "2/5")
            table_size: Number of seats (6 or 9)

        Returns:
            Activation confirmation
        """
        self.state = LiveTableState(
            stakes=stakes,
            table_size=table_size
        )
        self.is_active = True
        logger.info(f"Live mode activated: {stakes} {table_size}-max")

        return {
            "status": "active",
            "message": f"Live mode active. {stakes} NL {table_size}-max",
            "state": self.state.to_dict()
        }

    def deactivate(self) -> Dict[str, Any]:
        """Deactivate live mode and return session summary."""
        if not self.is_active:
            return {"status": "inactive", "message": "Live mode was not active"}

        summary = {
            "status": "deactivated",
            "session_duration": (datetime.now() - self.state.session_start).seconds // 60,
            "hands_played": self.state.hands_played,
            "session_profit": self.state.session_profit,
            "stakes": self.state.stakes
        }

        self.is_active = False
        logger.info(f"Live mode deactivated: {summary}")

        return summary

    def new_hand(self) -> Dict[str, Any]:
        """Start a new hand, clearing current hand state."""
        self.state.hole_cards = None
        self.state.community_cards = []
        self.state.street = LiveStreet.PREFLOP
        self.state.pot = float(self.state.stakes.split('/')[0]) + float(self.state.stakes.split('/')[1])
        self.state.current_bet = float(self.state.stakes.split('/')[1])  # BB
        self.state.facing_bet = 0.0
        self.state.facing_raise = False
        self.state.action_on_me = False
        self.state.players_in_hand = self.state.active_players
        self.state.hands_played += 1

        # Move button
        self.state.button_seat = (self.state.button_seat % self.state.table_size) + 1
        self._update_my_position()

        logger.info(f"New hand #{self.state.hands_played}")

        return {
            "status": "new_hand",
            "hand_number": self.state.hands_played,
            "position": self.state.get_position_name(),
            "pot": self.state.pot
        }

    def set_hole_cards(self, cards_input: str) -> Dict[str, Any]:
        """Set hero's hole cards from voice input.

        Args:
            cards_input: Voice input like "ace king suited" or "Ah Kh"

        Returns:
            Confirmation with parsed cards
        """
        cards = self._parse_cards(cards_input)

        if len(cards) != 2:
            return {
                "status": "error",
                "message": f"Could not parse two cards from: {cards_input}",
                "parsed": cards
            }

        self.state.hole_cards = " ".join(cards)
        logger.info(f"Hole cards set: {self.state.hole_cards}")

        return {
            "status": "success",
            "hole_cards": self.state.hole_cards,
            "message": f"Holding {self.state.hole_cards}"
        }

    def set_board(self, cards_input: str) -> Dict[str, Any]:
        """Set community cards from voice input.

        Args:
            cards_input: Voice input like "queen jack ten rainbow"

        Returns:
            Confirmation with parsed board
        """
        cards = self._parse_cards(cards_input)

        if len(cards) < 3:
            return {
                "status": "error",
                "message": f"Need at least 3 cards for flop: {cards_input}",
                "parsed": cards
            }

        self.state.community_cards = cards[:5]  # Max 5 cards

        # Update street based on card count
        if len(self.state.community_cards) == 3:
            self.state.street = LiveStreet.FLOP
        elif len(self.state.community_cards) == 4:
            self.state.street = LiveStreet.TURN
        elif len(self.state.community_cards) >= 5:
            self.state.street = LiveStreet.RIVER

        board_str = " ".join(self.state.community_cards)
        logger.info(f"Board set: {board_str} ({self.state.street.value})")

        return {
            "status": "success",
            "board": self.state.community_cards,
            "street": self.state.street.value,
            "message": f"Board is {board_str}"
        }

    def add_card(self, card_input: str) -> Dict[str, Any]:
        """Add a single card to the board (turn or river).

        Args:
            card_input: Voice input like "king of hearts"

        Returns:
            Updated board state
        """
        cards = self._parse_cards(card_input)

        if not cards:
            return {
                "status": "error",
                "message": f"Could not parse card from: {card_input}"
            }

        if len(self.state.community_cards) >= 5:
            return {
                "status": "error",
                "message": "Board already has 5 cards"
            }

        self.state.community_cards.append(cards[0])

        # Update street
        if len(self.state.community_cards) == 4:
            self.state.street = LiveStreet.TURN
        elif len(self.state.community_cards) == 5:
            self.state.street = LiveStreet.RIVER

        return {
            "status": "success",
            "card_added": cards[0],
            "board": self.state.community_cards,
            "street": self.state.street.value
        }

    def set_pot(self, amount: float) -> Dict[str, Any]:
        """Set current pot size.

        Args:
            amount: Pot size in dollars

        Returns:
            Confirmation
        """
        self.state.pot = amount
        logger.info(f"Pot set: ${amount}")

        return {
            "status": "success",
            "pot": amount,
            "spr": self.state.get_spr()
        }

    def set_stack(self, amount: float) -> Dict[str, Any]:
        """Set hero's stack size.

        Args:
            amount: Stack size in dollars

        Returns:
            Confirmation with SPR
        """
        self.state.my_stack = amount
        logger.info(f"Stack set: ${amount}")

        return {
            "status": "success",
            "stack": amount,
            "spr": self.state.get_spr()
        }

    def set_position(self, position_input: str) -> Dict[str, Any]:
        """Set hero's position from voice input.

        Args:
            position_input: Voice input like "button" or "cutoff"

        Returns:
            Confirmation
        """
        position_lower = position_input.lower().strip()

        if position_lower in self.POSITION_ALIASES:
            self.state.my_position = self.POSITION_ALIASES[position_lower]
        else:
            # Try to match position enum directly
            try:
                self.state.my_position = LivePosition(position_input.upper())
            except ValueError:
                return {
                    "status": "error",
                    "message": f"Unknown position: {position_input}",
                    "valid_positions": list(self.POSITION_ALIASES.keys())
                }

        logger.info(f"Position set: {self.state.my_position.value}")

        return {
            "status": "success",
            "position": self.state.my_position.value
        }

    def facing_action(self, action_type: str, amount: float = 0.0) -> Dict[str, Any]:
        """Record action facing hero.

        Args:
            action_type: "bet", "raise", "all-in"
            amount: Bet/raise amount

        Returns:
            Updated state with pot odds
        """
        self.state.facing_bet = amount
        self.state.facing_raise = action_type.lower() in ("raise", "all-in", "allin")
        self.state.action_on_me = True

        pot_odds = self.state.get_pot_odds()

        return {
            "status": "success",
            "facing": action_type,
            "amount": amount,
            "pot_odds": pot_odds,
            "pot": self.state.pot,
            "spr": self.state.get_spr()
        }

    def player_folds(self, count: int = 1) -> Dict[str, Any]:
        """Record player(s) folding.

        Args:
            count: Number of players folding

        Returns:
            Updated player count
        """
        self.state.players_in_hand = max(1, self.state.players_in_hand - count)

        return {
            "status": "success",
            "players_remaining": self.state.players_in_hand
        }

    def record_result(self, won: bool, amount: float = 0.0) -> Dict[str, Any]:
        """Record hand result for session tracking.

        Args:
            won: Whether hero won the pot
            amount: Amount won (if won) or lost (negative if lost)

        Returns:
            Session update
        """
        if won:
            self.state.session_profit += amount
        else:
            self.state.session_profit -= abs(amount)

        logger.info(f"Result recorded: {'Won' if won else 'Lost'} ${abs(amount)}, Session: ${self.state.session_profit}")

        return {
            "status": "success",
            "result": "won" if won else "lost",
            "amount": amount,
            "session_profit": self.state.session_profit,
            "hands_played": self.state.hands_played
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current table state."""
        return {
            "is_active": self.is_active,
            "state": self.state.to_dict() if self.is_active else None
        }

    def get_engine_state(self) -> Optional[Dict[str, Any]]:
        """Get state formatted for PokerEngine analysis."""
        if not self.is_active or not self.state.hole_cards:
            return None
        return self.state.to_engine_format()

    def _parse_cards(self, input_str: str) -> List[str]:
        """Parse card notation from voice input.

        Handles various formats:
        - "Ah Kd" (standard notation)
        - "ace of hearts king of diamonds"
        - "ace king suited" (assumes same suit)
        - "pocket aces"

        Returns:
            List of normalized card strings (e.g., ["Ah", "Kd"])
        """
        cards = []
        input_lower = input_str.lower().strip()

        # Handle "pocket X" for pairs
        pocket_match = re.search(r'pocket\s+(\w+)', input_lower)
        if pocket_match:
            rank = pocket_match.group(1)
            if rank in self.RANK_WORDS:
                r = self.RANK_WORDS[rank]
                # Assume hearts and diamonds for pocket pairs
                return [f"{r}h", f"{r}d"]
            elif len(rank) == 1 and rank.upper() in 'AKQJT98765432':
                r = rank.upper()
                return [f"{r}h", f"{r}d"]

        # Check for "suited" - use same suit for both cards
        is_suited = 'suited' in input_lower
        default_suit = 'h'  # Default to hearts

        # First try standard notation (Ah Kd)
        standard_matches = re.findall(r'\b([AKQJT2-9])([hdcs])\b', input_str, re.IGNORECASE)
        if standard_matches:
            for rank, suit in standard_matches:
                cards.append(f"{rank.upper()}{suit.lower()}")
                if is_suited:
                    default_suit = suit.lower()
            return cards

        # Parse word format (ace of hearts)
        for word in self.RANK_WORDS:
            if word in input_lower:
                rank = self.RANK_WORDS[word]
                # Look for suit after rank word
                pattern = rf'{word}\s+(?:of\s+)?(\w+)'
                suit_match = re.search(pattern, input_lower)
                if suit_match:
                    suit_word = suit_match.group(1)
                    if suit_word in self.SUIT_WORDS:
                        suit = self.SUIT_WORDS[suit_word]
                        cards.append(f"{rank}{suit}")
                        if is_suited:
                            default_suit = suit
                        continue

                # No suit found, use default
                cards.append(f"{rank}{default_suit}")
                if is_suited and len(cards) > 1:
                    # Make second card same suit as first
                    first_suit = cards[0][-1]
                    cards[-1] = cards[-1][:-1] + first_suit

        return cards

    def _update_my_position(self):
        """Update hero's position based on button location."""
        if self.state.my_seat == 0:
            return

        # Calculate position relative to button
        seats_from_button = (self.state.my_seat - self.state.button_seat) % self.state.table_size

        if seats_from_button == 0:
            self.state.my_position = LivePosition.BTN
        elif seats_from_button == 1:
            self.state.my_position = LivePosition.SB
        elif seats_from_button == 2:
            self.state.my_position = LivePosition.BB
        elif self.state.table_size == 6:
            # 6-max positions
            if seats_from_button == 3:
                self.state.my_position = LivePosition.UTG
            elif seats_from_button == 4:
                self.state.my_position = LivePosition.HJ
            else:
                self.state.my_position = LivePosition.CO
        else:
            # 9-handed positions
            if seats_from_button == 3:
                self.state.my_position = LivePosition.UTG
            elif seats_from_button <= 5:
                self.state.my_position = LivePosition.MP
            elif seats_from_button == 6:
                self.state.my_position = LivePosition.HJ
            elif seats_from_button == 7:
                self.state.my_position = LivePosition.CO
            else:
                self.state.my_position = LivePosition.MP

    def process_voice_command(self, command: str) -> Dict[str, Any]:
        """Process a natural language voice command.

        Args:
            command: Voice transcription

        Returns:
            Result of command execution
        """
        cmd_lower = command.lower().strip()
        self._command_history.append((datetime.now(), command))

        # Activation commands
        if 'live mode' in cmd_lower or 'start live' in cmd_lower:
            stakes = "1/2"  # Default
            if '2/5' in cmd_lower or 'two five' in cmd_lower:
                stakes = "2/5"
            elif '5/10' in cmd_lower or 'five ten' in cmd_lower:
                stakes = "5/10"
            elif '1/3' in cmd_lower or 'one three' in cmd_lower:
                stakes = "1/3"

            table_size = 9
            if '6 max' in cmd_lower or 'six max' in cmd_lower:
                table_size = 6

            return self.activate(stakes, table_size)

        if 'stop live' in cmd_lower or 'end live' in cmd_lower or 'exit live' in cmd_lower:
            return self.deactivate()

        if not self.is_active:
            return {"status": "inactive", "message": "Live mode not active. Say 'live mode' to start."}

        # Hand management
        if 'new hand' in cmd_lower or 'next hand' in cmd_lower:
            return self.new_hand()

        # Card setting
        if 'i have' in cmd_lower or 'my cards' in cmd_lower or 'holding' in cmd_lower:
            # Extract cards after the trigger phrase
            for trigger in ['i have', 'my cards are', 'holding']:
                if trigger in cmd_lower:
                    cards_part = cmd_lower.split(trigger, 1)[1]
                    return self.set_hole_cards(cards_part)

        if 'board is' in cmd_lower or 'flop is' in cmd_lower:
            for trigger in ['board is', 'flop is']:
                if trigger in cmd_lower:
                    cards_part = cmd_lower.split(trigger, 1)[1]
                    return self.set_board(cards_part)

        if 'turn is' in cmd_lower or 'turn card' in cmd_lower:
            for trigger in ['turn is', 'turn card']:
                if trigger in cmd_lower:
                    card_part = cmd_lower.split(trigger, 1)[1]
                    return self.add_card(card_part)

        if 'river is' in cmd_lower or 'river card' in cmd_lower:
            for trigger in ['river is', 'river card']:
                if trigger in cmd_lower:
                    card_part = cmd_lower.split(trigger, 1)[1]
                    return self.add_card(card_part)

        # Pot and stack
        if 'pot is' in cmd_lower or 'pot size' in cmd_lower:
            amount = self._extract_amount(cmd_lower)
            if amount:
                return self.set_pot(amount)

        if 'my stack' in cmd_lower or 'stack is' in cmd_lower or 'i have' in cmd_lower and 'stack' in cmd_lower:
            amount = self._extract_amount(cmd_lower)
            if amount:
                return self.set_stack(amount)

        # Position
        if 'position' in cmd_lower or "i'm on" in cmd_lower or "i am" in cmd_lower:
            for pos_alias in self.POSITION_ALIASES:
                if pos_alias in cmd_lower:
                    return self.set_position(pos_alias)

        # Facing action
        if 'facing' in cmd_lower or 'bet to me' in cmd_lower:
            amount = self._extract_amount(cmd_lower)
            if 'raise' in cmd_lower:
                return self.facing_action('raise', amount or 0)
            elif 'all in' in cmd_lower or 'all-in' in cmd_lower:
                return self.facing_action('all-in', amount or 0)
            else:
                return self.facing_action('bet', amount or 0)

        # Folds
        if 'fold' in cmd_lower:
            count = 1
            for word in ['two', '2']:
                if word in cmd_lower:
                    count = 2
            for word in ['three', '3']:
                if word in cmd_lower:
                    count = 3
            return self.player_folds(count)

        # Results
        if 'i won' in cmd_lower or 'won pot' in cmd_lower:
            amount = self._extract_amount(cmd_lower) or self.state.pot
            return self.record_result(True, amount)

        if 'i lost' in cmd_lower or 'lost pot' in cmd_lower:
            amount = self._extract_amount(cmd_lower) or self.state.pot
            return self.record_result(False, amount)

        # State query
        if 'status' in cmd_lower or 'current' in cmd_lower or 'state' in cmd_lower:
            return self.get_state()

        return {
            "status": "unknown",
            "message": f"Could not parse command: {command}",
            "hint": "Try: 'I have ace king suited', 'pot is 50', 'new hand', 'board is queen jack ten'"
        }

    def _extract_amount(self, text: str) -> Optional[float]:
        """Extract dollar amount from text."""
        # Try to find number patterns
        patterns = [
            r'\$(\d+(?:\.\d{2})?)',  # $100 or $100.00
            r'(\d+)\s*(?:dollars?|bucks?)',  # 100 dollars
            r'(\d+(?:\.\d{2})?)\b',  # Just a number
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None


# Global instance for easy access
live_mode = LiveModeManager()
