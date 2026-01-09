"""Hand history parser for extracting showdown data from OCR.

Parses showdown hands from Ignition's hand history sidebar to update
villain profiles with actual hands shown.
"""

import re
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ShowdownHand:
    """A single showdown hand observed."""
    seat: str  # e.g., "Seat 3"
    hand_description: str  # e.g., "Two Pair, Aces and Kings"
    cards: Optional[str] = None  # e.g., "Ac Kh"
    pot_won: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seat": self.seat,
            "hand_description": self.hand_description,
            "cards": self.cards,
            "pot_won": self.pot_won,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class HandHistoryEntry:
    """A complete hand history entry."""
    hand_id: Optional[str] = None
    showdowns: List[ShowdownHand] = field(default_factory=list)
    winner_seat: Optional[str] = None
    pot_size: float = 0.0
    community_cards: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "showdowns": [s.to_dict() for s in self.showdowns],
            "winner_seat": self.winner_seat,
            "pot_size": self.pot_size,
            "community_cards": self.community_cards,
            "timestamp": self.timestamp.isoformat()
        }


class HandHistoryParser:
    """Parse showdown hands from OCR data to update villain ranges."""

    # Patterns for parsing showdown text
    SEAT_PATTERN = re.compile(r'Seat\s*(\d+)', re.IGNORECASE)
    CARDS_PATTERN = re.compile(r'([AKQJT2-9])([hdcs])\s*([AKQJT2-9])([hdcs])', re.IGNORECASE)
    HAND_RANK_PATTERN = re.compile(
        r'(Royal Flush|Straight Flush|Four of a Kind|Full House|Flush|Straight|'
        r'Three of a Kind|Two Pair|One Pair|Pair|High Card|'
        r'Quads?|Set|Trips?|Boat|Wheel)',
        re.IGNORECASE
    )
    POT_PATTERN = re.compile(r'\$?([\d,]+(?:\.\d{2})?)', re.IGNORECASE)

    # Hand ranking for range estimation
    HAND_STRENGTHS = {
        "royal flush": 10,
        "straight flush": 9,
        "four of a kind": 8,
        "quads": 8,
        "full house": 7,
        "boat": 7,
        "flush": 6,
        "straight": 5,
        "three of a kind": 4,
        "trips": 4,
        "set": 4,
        "two pair": 3,
        "one pair": 2,
        "pair": 2,
        "high card": 1
    }

    def __init__(self):
        self._recent_showdowns: List[ShowdownHand] = []
        self._seen_hands: Dict[str, List[str]] = {}  # seat -> list of cards shown
        self._max_history = 50

    def parse_showdown_text(self, text: str) -> List[ShowdownHand]:
        """Parse showdown information from OCR text.

        Args:
            text: Raw OCR text containing showdown information

        Returns:
            List of ShowdownHand objects
        """
        showdowns = []

        # Split by common delimiters
        lines = text.replace('|', '\n').split('\n')

        current_seat = None
        current_hand = None
        current_cards = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for seat identifier
            seat_match = self.SEAT_PATTERN.search(line)
            if seat_match:
                # Save previous showdown if we have one
                if current_seat and (current_hand or current_cards):
                    showdowns.append(ShowdownHand(
                        seat=current_seat,
                        hand_description=current_hand or "Unknown",
                        cards=current_cards
                    ))

                current_seat = f"Seat {seat_match.group(1)}"
                current_hand = None
                current_cards = None

            # Look for hand ranking
            hand_match = self.HAND_RANK_PATTERN.search(line)
            if hand_match:
                current_hand = hand_match.group(1)

            # Look for actual cards
            cards_match = self.CARDS_PATTERN.search(line)
            if cards_match:
                card1 = f"{cards_match.group(1).upper()}{cards_match.group(2).lower()}"
                card2 = f"{cards_match.group(3).upper()}{cards_match.group(4).lower()}"
                current_cards = f"{card1} {card2}"

        # Don't forget the last one
        if current_seat and (current_hand or current_cards):
            showdowns.append(ShowdownHand(
                seat=current_seat,
                hand_description=current_hand or "Unknown",
                cards=current_cards
            ))

        return showdowns

    def parse_ignition_format(self, showdown_line: str) -> Optional[ShowdownHand]:
        """Parse Ignition-specific showdown format.

        Expected format: "SHOWDOWN: Seat 3|Two Pair|Ac Kh"
        or similar variations.

        Args:
            showdown_line: Single showdown line from OCR

        Returns:
            ShowdownHand or None
        """
        # Try structured format first
        parts = showdown_line.replace('SHOWDOWN:', '').strip().split('|')

        if len(parts) >= 2:
            seat_str = parts[0].strip()
            hand_desc = parts[1].strip() if len(parts) > 1 else "Unknown"
            cards = parts[2].strip() if len(parts) > 2 else None

            # Normalize seat
            seat_match = self.SEAT_PATTERN.search(seat_str)
            if seat_match:
                seat = f"Seat {seat_match.group(1)}"
            else:
                seat = seat_str

            return ShowdownHand(
                seat=seat,
                hand_description=hand_desc,
                cards=cards
            )

        # Fall back to free-form parsing
        showdowns = self.parse_showdown_text(showdown_line)
        return showdowns[0] if showdowns else None

    def record_showdown(self, showdown: ShowdownHand):
        """Record a showdown for villain profiling.

        Args:
            showdown: ShowdownHand to record
        """
        self._recent_showdowns.append(showdown)

        # Keep history bounded
        if len(self._recent_showdowns) > self._max_history:
            self._recent_showdowns = self._recent_showdowns[-self._max_history:]

        # Track cards shown by seat
        if showdown.cards:
            if showdown.seat not in self._seen_hands:
                self._seen_hands[showdown.seat] = []
            self._seen_hands[showdown.seat].append(showdown.cards)

        logger.debug(f"Recorded showdown: {showdown.seat} showed {showdown.cards}")

    def get_villain_showdowns(self, seat: str) -> List[ShowdownHand]:
        """Get all showdowns for a specific seat.

        Args:
            seat: Seat identifier (e.g., "Seat 3")

        Returns:
            List of showdowns for that seat
        """
        return [s for s in self._recent_showdowns if s.seat == seat]

    def get_villain_hands_shown(self, seat: str) -> List[str]:
        """Get actual hands shown by a villain.

        Args:
            seat: Seat identifier

        Returns:
            List of card strings (e.g., ["Ac Kh", "Qd Qc"])
        """
        return self._seen_hands.get(seat, [])

    def estimate_range_from_showdowns(self, seat: str) -> Dict[str, Any]:
        """Estimate villain's range based on observed showdowns.

        Args:
            seat: Seat identifier

        Returns:
            Range estimation data
        """
        showdowns = self.get_villain_showdowns(seat)
        hands_shown = self.get_villain_hands_shown(seat)

        if not showdowns:
            return {
                "seat": seat,
                "hands_observed": 0,
                "range_estimate": "Unknown - no showdowns",
                "strength_profile": "Unknown"
            }

        # Analyze hand strengths shown
        strengths = []
        for s in showdowns:
            for pattern, strength in self.HAND_STRENGTHS.items():
                if pattern in s.hand_description.lower():
                    strengths.append(strength)
                    break

        avg_strength = sum(strengths) / len(strengths) if strengths else 0

        # Estimate range based on average showdown strength
        if avg_strength >= 5:
            range_est = "Very tight - shows strong hands"
            profile = "Nit"
        elif avg_strength >= 3:
            range_est = "Tight - solid holdings"
            profile = "TAG"
        elif avg_strength >= 2:
            range_est = "Standard - mixed range"
            profile = "Reg"
        else:
            range_est = "Loose - wide range"
            profile = "LAG/Fish"

        return {
            "seat": seat,
            "hands_observed": len(showdowns),
            "hands_shown": hands_shown,
            "avg_strength": round(avg_strength, 2),
            "range_estimate": range_est,
            "strength_profile": profile
        }

    def get_recent_showdowns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent showdowns across all seats.

        Args:
            limit: Maximum number to return

        Returns:
            List of showdown dicts
        """
        return [s.to_dict() for s in self._recent_showdowns[-limit:]]

    def clear_history(self, seat: Optional[str] = None):
        """Clear showdown history.

        Args:
            seat: If provided, only clear for this seat. Otherwise clear all.
        """
        if seat:
            self._recent_showdowns = [s for s in self._recent_showdowns if s.seat != seat]
            self._seen_hands.pop(seat, None)
        else:
            self._recent_showdowns = []
            self._seen_hands = {}


# Global instance
hand_history_parser = HandHistoryParser()
