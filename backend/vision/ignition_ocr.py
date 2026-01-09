"""Ignition Poker specific OCR calibration using Gemini Vision."""

import os
import base64
import re
import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import google.generativeai as genai

logger = logging.getLogger(__name__)


@dataclass
class IgnitionTableState:
    """Parsed state from Ignition Poker table screenshot."""
    hole_cards: Optional[str] = None  # e.g., "Ah Kd"
    community_cards: Optional[str] = None  # e.g., "Qs Jh Tc"
    pot: float = 0.0
    bet_to_call: float = 0.0
    street: str = "preflop"  # preflop, flop, turn, river
    my_stack: float = 0.0
    my_seat: Optional[str] = None  # e.g., "Seat 1"
    villain_stacks: Dict[str, float] = field(default_factory=dict)  # seat -> stack
    active_seats: List[str] = field(default_factory=list)
    dealer_position: Optional[str] = None
    my_position: Optional[str] = None  # e.g., "BTN", "BB", "CO"
    showdown: Optional[str] = None  # e.g., "Seat 3|Two Pair|Ac Kh"
    confidence: float = 0.0
    raw_text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hole_cards": self.hole_cards,
            "community_cards": self.community_cards,
            "pot": self.pot,
            "bet_to_call": self.bet_to_call,
            "street": self.street,
            "my_stack": self.my_stack,
            "my_seat": self.my_seat,
            "villain_stacks": self.villain_stacks,
            "active_seats": self.active_seats,
            "dealer_position": self.dealer_position,
            "my_position": self.my_position,
            "showdown": self.showdown,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


class IgnitionOCR:
    """Ignition Poker table OCR with site-specific calibration.

    Ignition-specific layout characteristics:
    - Anonymous players (Seat 1, Seat 2, etc.)
    - Hole cards at bottom center below player avatar
    - Pot displayed center-top above community cards
    - Community cards centered horizontally
    - Bet amounts displayed near each player seat
    - Dark green felt table background
    - Red card backs for hidden cards
    - White/yellow text for amounts
    """

    # Ignition-specific Gemini prompt for table parsing
    IGNITION_PROMPT = """Analyze this IGNITION POKER table screenshot with extreme precision.

IGNITION CASINO LAYOUT NOTES:
- Table felt: Dark green background
- Players: Anonymous (Seat 1, Seat 2, etc.) around oval table
- Hole cards: Bottom center, face up, below "You" or player avatar
- Community cards: Center of table, horizontal row
- Pot amount: Center-top, above community cards, white text
- Bet amounts: Near each seat, smaller white/yellow text
- Stack sizes: Below player names, shows chip count
- Dealer button: Small "D" disc next to a seat
- Action buttons: Bottom right (Fold, Check, Call, Raise)

EXTRACT EXACTLY (use "none" if not visible):

HOLE_CARDS: [Two cards, e.g., "Ah Kd" or "none"]
COMMUNITY: [Up to 5 cards, e.g., "Qs Jh Tc" or "none"]
POT: [Total pot amount as number, e.g., "125.50"]
BET_TO_CALL: [Current bet to call, e.g., "25.00" or "0"]
STREET: [preflop/flop/turn/river]
MY_STACK: [Hero stack size as number]
MY_SEAT: [Hero's seat number, e.g., "Seat 1" - look for "You" label or hole cards position]
DEALER_SEAT: [Seat number with "D" dealer button chip, e.g., "Seat 3"]
ACTIVE_SEATS: [Comma-separated seats still in hand, e.g., "Seat 1, Seat 4, Seat 6"]
VILLAIN_STACKS: [Format: "Seat 1:150.00, Seat 4:89.50"]
SHOWDOWN: [If hand history panel shows a showdown: "Seat X|Hand Description|Cards" e.g., "Seat 3|Two Pair|Ac Kh" or "none"]
CONFIDENCE: [Your confidence 0.0-1.0 in this reading]

Be precise with card notation:
- Suits: h=hearts, d=diamonds, c=clubs, s=spades
- Ranks: A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
- Example: "As Kh" = Ace of spades, King of hearts"""

    # Ignition color profiles for validation
    IGNITION_COLORS = {
        "table_felt": (0, 80, 40),      # Dark green RGB approximate
        "pot_text": (255, 255, 255),    # White text
        "bet_text": (255, 255, 200),    # Light yellow text
        "card_back": (180, 30, 30),     # Red card backs
        "button_fold": (200, 50, 50),   # Red fold button
        "button_call": (50, 150, 50),   # Green call button
    }

    # Card detection patterns
    CARD_PATTERN = re.compile(r'([AKQJT2-9])([hdcs])', re.IGNORECASE)

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.available = True
        else:
            self.model = None
            self.available = False
            logger.warning("GEMINI_API_KEY not set - Ignition OCR unavailable")

        # Cache for recent readings to smooth out noise
        self._recent_readings: List[IgnitionTableState] = []
        self._max_cache = 5

    async def read_table(self, image_base64: str) -> IgnitionTableState:
        """Read Ignition poker table state from screenshot.

        Args:
            image_base64: Base64 encoded screenshot from glasses/camera

        Returns:
            IgnitionTableState with parsed table information
        """
        if not self.available:
            return self._empty_result("Gemini API not configured")

        try:
            # Clean base64 data
            if ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]

            image_data = base64.b64decode(image_base64)

            # Call Gemini Vision with Ignition-specific prompt
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content([
                    {"mime_type": "image/jpeg", "data": image_data},
                    self.IGNITION_PROMPT
                ])
            )

            # Parse the response
            result = self._parse_ignition_response(response.text)

            # Add to cache for smoothing
            self._add_to_cache(result)

            return result

        except Exception as e:
            logger.error(f"Ignition OCR error: {e}")
            return self._empty_result(f"OCR failed: {str(e)}")

    def _parse_ignition_response(self, response_text: str) -> IgnitionTableState:
        """Parse Gemini response into IgnitionTableState."""
        result = IgnitionTableState(raw_text=response_text)

        # Extract hole cards
        hole_match = re.search(r'HOLE_CARDS:\s*(.+?)(?=\n|COMMUNITY:|$)', response_text, re.IGNORECASE)
        if hole_match:
            cards = hole_match.group(1).strip()
            if cards.lower() != 'none':
                result.hole_cards = self._normalize_cards(cards)

        # Extract community cards
        comm_match = re.search(r'COMMUNITY:\s*(.+?)(?=\n|POT:|$)', response_text, re.IGNORECASE)
        if comm_match:
            cards = comm_match.group(1).strip()
            if cards.lower() != 'none':
                result.community_cards = self._normalize_cards(cards)

        # Extract pot
        pot_match = re.search(r'POT:\s*([\d,.]+)', response_text, re.IGNORECASE)
        if pot_match:
            result.pot = self._parse_amount(pot_match.group(1))

        # Extract bet to call
        bet_match = re.search(r'BET_TO_CALL:\s*([\d,.]+)', response_text, re.IGNORECASE)
        if bet_match:
            result.bet_to_call = self._parse_amount(bet_match.group(1))

        # Extract street
        street_match = re.search(r'STREET:\s*(\w+)', response_text, re.IGNORECASE)
        if street_match:
            street = street_match.group(1).lower()
            if street in ('preflop', 'flop', 'turn', 'river'):
                result.street = street

        # Extract my stack
        stack_match = re.search(r'MY_STACK:\s*([\d,.]+)', response_text, re.IGNORECASE)
        if stack_match:
            result.my_stack = self._parse_amount(stack_match.group(1))

        # Extract hero's seat
        my_seat_match = re.search(r'MY_SEAT:\s*(Seat\s*\d+)', response_text, re.IGNORECASE)
        if my_seat_match:
            result.my_seat = my_seat_match.group(1)

        # Extract dealer position
        dealer_match = re.search(r'DEALER_SEAT:\s*(Seat\s*\d+)', response_text, re.IGNORECASE)
        if dealer_match:
            result.dealer_position = dealer_match.group(1)

        # Extract active seats
        active_match = re.search(r'ACTIVE_SEATS:\s*(.+?)(?=\n|VILLAIN_STACKS:|$)', response_text, re.IGNORECASE)
        if active_match:
            seats_str = active_match.group(1).strip()
            if seats_str.lower() != 'none':
                result.active_seats = [s.strip() for s in seats_str.split(',')]

        # Extract villain stacks
        villain_match = re.search(r'VILLAIN_STACKS:\s*(.+?)(?=\n|CONFIDENCE:|$)', response_text, re.IGNORECASE)
        if villain_match:
            stacks_str = villain_match.group(1).strip()
            if stacks_str.lower() != 'none':
                result.villain_stacks = self._parse_villain_stacks(stacks_str)

        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response_text, re.IGNORECASE)
        if conf_match:
            try:
                result.confidence = float(conf_match.group(1))
            except ValueError:
                result.confidence = 0.5

        # Extract showdown data from hand history panel
        showdown_match = re.search(r'SHOWDOWN:\s*(.+?)(?=\n|CONFIDENCE:|$)', response_text, re.IGNORECASE)
        if showdown_match:
            showdown_str = showdown_match.group(1).strip()
            if showdown_str.lower() != 'none':
                result.showdown = showdown_str

        # Infer street from community cards if not explicitly set
        if result.community_cards and result.street == 'preflop':
            card_count = len(result.community_cards.split())
            if card_count >= 5:
                result.street = 'river'
            elif card_count == 4:
                result.street = 'turn'
            elif card_count >= 3:
                result.street = 'flop'

        # Auto-calculate position if we have the required info
        if result.my_seat and result.dealer_position and result.active_seats:
            result.my_position = self.get_position_from_dealer(
                result.my_seat,
                result.dealer_position,
                result.active_seats
            )

        return result

    def _normalize_cards(self, cards_str: str) -> str:
        """Normalize card notation to standard format."""
        # Find all valid card patterns
        matches = self.CARD_PATTERN.findall(cards_str)
        if matches:
            normalized = []
            for rank, suit in matches:
                # Normalize rank (T for 10)
                rank = rank.upper()
                if rank == '1':
                    rank = 'T'  # 10 sometimes OCR'd as 1
                suit = suit.lower()
                normalized.append(f"{rank}{suit}")
            return ' '.join(normalized)
        return cards_str.strip()

    def _parse_amount(self, amount_str: str) -> float:
        """Parse monetary amount from string."""
        try:
            # Remove commas and extra characters
            cleaned = re.sub(r'[,$]', '', amount_str)
            return float(cleaned)
        except ValueError:
            return 0.0

    def _parse_villain_stacks(self, stacks_str: str) -> Dict[str, float]:
        """Parse villain stack sizes from string."""
        stacks = {}
        # Pattern: "Seat 1:150.00, Seat 4:89.50"
        pattern = re.compile(r'(Seat\s*\d+)\s*:\s*([\d,.]+)', re.IGNORECASE)
        for match in pattern.finditer(stacks_str):
            seat = match.group(1)
            amount = self._parse_amount(match.group(2))
            stacks[seat] = amount
        return stacks

    def _add_to_cache(self, state: IgnitionTableState):
        """Add reading to cache for smoothing."""
        self._recent_readings.append(state)
        if len(self._recent_readings) > self._max_cache:
            self._recent_readings.pop(0)

    def get_smoothed_pot(self) -> float:
        """Get pot size smoothed over recent readings."""
        if not self._recent_readings:
            return 0.0
        pots = [r.pot for r in self._recent_readings if r.pot > 0]
        if pots:
            return sum(pots) / len(pots)
        return 0.0

    def _empty_result(self, message: str) -> IgnitionTableState:
        """Return empty result with error message."""
        return IgnitionTableState(
            raw_text=message,
            confidence=0.0
        )

    def validate_ignition_table(self, image_base64: str) -> Tuple[bool, str]:
        """Validate if image appears to be an Ignition poker table.

        Returns:
            Tuple of (is_valid, reason)
        """
        # This could be enhanced with color profile matching
        # For now, rely on Gemini's detection
        return True, "Validation passed"

    def get_position_from_dealer(self, my_seat: str, dealer_seat: str, active_seats: List[str]) -> str:
        """Calculate position relative to dealer button.

        Returns position like 'BTN', 'SB', 'BB', 'CO', 'HJ', 'UTG', etc.
        """
        if not my_seat or not dealer_seat or not active_seats:
            return "unknown"

        try:
            # Extract seat numbers
            my_num = int(re.search(r'\d+', my_seat).group())
            dealer_num = int(re.search(r'\d+', dealer_seat).group())

            # Sort active seat numbers
            active_nums = sorted([
                int(re.search(r'\d+', s).group())
                for s in active_seats
                if re.search(r'\d+', s)
            ])

            if my_num not in active_nums:
                return "unknown"

            # Calculate position
            dealer_idx = active_nums.index(dealer_num) if dealer_num in active_nums else 0
            my_idx = active_nums.index(my_num)

            # Positions relative to dealer (clockwise)
            total_players = len(active_nums)
            pos_from_dealer = (my_idx - dealer_idx) % total_players

            positions = {
                0: "BTN",
                1: "SB",
                2: "BB",
            }

            if total_players <= 3:
                return positions.get(pos_from_dealer, "BTN")
            elif total_players <= 6:
                # 6-max positions
                return {
                    0: "BTN",
                    1: "SB",
                    2: "BB",
                    3: "UTG",
                    4: "HJ",
                    5: "CO"
                }.get(pos_from_dealer, "MP")
            else:
                # Full ring
                if pos_from_dealer == total_players - 1:
                    return "CO"
                elif pos_from_dealer == total_players - 2:
                    return "HJ"
                elif pos_from_dealer >= 3:
                    return "MP"
                return positions.get(pos_from_dealer, "MP")

        except Exception as e:
            logger.debug(f"Position calculation error: {e}")
            return "unknown"


# Global instance
ignition_ocr = IgnitionOCR()
