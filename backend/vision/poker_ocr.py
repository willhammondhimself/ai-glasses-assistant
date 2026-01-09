"""Poker table OCR using Gemini Vision for real-time card/pot detection.

Supports multiple poker sites with site-specific calibration:
- Generic (default): Standard poker table detection
- Ignition: Anonymous players, specific layout/colors
"""

import os
import base64
import re
import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import google.generativeai as genai

logger = logging.getLogger(__name__)


class PokerSite(Enum):
    """Supported poker site modes."""
    GENERIC = "generic"
    IGNITION = "ignition"
    # Future: POKERSTARS = "pokerstars"
    # Future: GG_POKER = "gg_poker"


@dataclass
class PokerTableState:
    """Detected poker table state from webcam frame."""
    # Cards
    hole_cards: List[str]  # e.g., ["Ah", "Kd"]
    community_cards: List[str]  # e.g., ["Qs", "Jh", "Tc"]

    # Money
    pot_size: float
    current_bet: float
    player_stack: Optional[float]

    # Game state
    street: str  # preflop, flop, turn, river
    num_players: int
    position: Optional[str]

    # Raw OCR
    raw_text: str
    confidence: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hole_cards": self.hole_cards,
            "community_cards": self.community_cards,
            "pot_size": self.pot_size,
            "current_bet": self.current_bet,
            "player_stack": self.player_stack,
            "street": self.street,
            "num_players": self.num_players,
            "position": self.position,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


class PokerOCR:
    """Real-time poker table OCR using Gemini Vision."""

    # Card pattern: rank + suit
    CARD_PATTERN = re.compile(r'\b([2-9TJQKA])([cdhs])\b', re.IGNORECASE)
    MONEY_PATTERN = re.compile(r'\$?([\d,]+(?:\.\d{2})?)')

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.available = True
        else:
            self.model = None
            self.available = False
            logger.warning("GEMINI_API_KEY not set - Poker OCR unavailable")

        # Session stats tracking
        self._session_start = datetime.utcnow()
        self._hand_history: List[Dict] = []
        self._vpip_hands = 0
        self._pfr_hands = 0  # Preflop raise hands
        self._total_hands = 0

        # State tracking for auto-detection
        self._last_hole_cards: Optional[List[str]] = None
        self._last_street: str = "unknown"
        self._current_hand_tracked = False
        self._pending_action: Optional[str] = None

    async def analyze_frame(self, image_base64: str) -> PokerTableState:
        """Analyze a single webcam frame for poker table state.

        Args:
            image_base64: Base64 encoded image from webcam

        Returns:
            PokerTableState with detected cards, pot, bets
        """
        if not self.available:
            return self._empty_state("Gemini API not configured")

        try:
            # Clean base64
            if ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]

            image_data = base64.b64decode(image_base64)

            # Gemini Vision prompt optimized for poker tables
            prompt = """Analyze this poker table image. Extract:

1. HOLE_CARDS: Player's two hole cards (e.g., "Ah Kd" for Ace of hearts, King of diamonds)
2. COMMUNITY: Board cards (e.g., "Qs Jh Tc 2d" for flop/turn/river)
3. POT: Current pot size (number only, e.g., "150")
4. BET: Current bet to call (number only, e.g., "25")
5. STACK: Player's remaining chips (number only)
6. PLAYERS: Number of players still in hand
7. STREET: preflop/flop/turn/river
8. DEALER_SEAT: Seat number with "D" dealer button (e.g., "2" or "Seat 2")
9. HERO_SEAT: Hero's seat number (where hole cards are shown)

Card format: Rank (2-9, T, J, Q, K, A) + Suit (c=clubs, d=diamonds, h=hearts, s=spades)

Respond EXACTLY in this format:
HOLE_CARDS: [cards or "unknown"]
COMMUNITY: [cards or "none"]
POT: [number or "0"]
BET: [number or "0"]
STACK: [number or "unknown"]
PLAYERS: [number]
STREET: [preflop/flop/turn/river]
DEALER_SEAT: [seat number or "unknown"]
HERO_SEAT: [seat number or "unknown"]
CONFIDENCE: [0.0-1.0]"""

            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt, {"mime_type": "image/jpeg", "data": image_data}]
            )

            return self._parse_response(response.text)

        except Exception as e:
            logger.error(f"Poker OCR error: {e}")
            return self._empty_state(str(e))

    def _parse_response(self, text: str) -> PokerTableState:
        """Parse Gemini response into PokerTableState."""
        lines = text.strip().split('\n')
        data = {}

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip().upper()] = value.strip()

        # Parse hole cards
        hole_cards = []
        if 'HOLE_CARDS' in data and data['HOLE_CARDS'].lower() != 'unknown':
            hole_cards = self._parse_cards(data['HOLE_CARDS'])

        # Parse community cards
        community_cards = []
        if 'COMMUNITY' in data and data['COMMUNITY'].lower() != 'none':
            community_cards = self._parse_cards(data['COMMUNITY'])

        # Determine street from community cards
        if len(community_cards) == 0:
            street = "preflop"
        elif len(community_cards) == 3:
            street = "flop"
        elif len(community_cards) == 4:
            street = "turn"
        else:
            street = "river"

        # Parse numbers
        pot_size = self._parse_number(data.get('POT', '0'))
        current_bet = self._parse_number(data.get('BET', '0'))
        player_stack = self._parse_number(data.get('STACK', '0'))
        num_players = int(self._parse_number(data.get('PLAYERS', '2')) or 2)
        confidence = float(data.get('CONFIDENCE', '0.5'))

        # Extract dealer and hero seats for position calculation
        dealer_seat = None
        hero_seat = None
        position = None

        dealer_match = re.search(r'DEALER_SEAT:\s*(?:Seat\s*)?(\d+)', text, re.IGNORECASE)
        if dealer_match:
            dealer_seat = int(dealer_match.group(1))

        hero_match = re.search(r'HERO_SEAT:\s*(?:Seat\s*)?(\d+)', text, re.IGNORECASE)
        if hero_match:
            hero_seat = int(hero_match.group(1))

        # Calculate position if we have both seats
        if dealer_seat is not None and hero_seat is not None and num_players > 0:
            position = self._calculate_position(hero_seat, dealer_seat, num_players)

        return PokerTableState(
            hole_cards=hole_cards,
            community_cards=community_cards,
            pot_size=pot_size,
            current_bet=current_bet,
            player_stack=player_stack if player_stack else None,
            street=street,
            num_players=num_players,
            position=position,
            raw_text=text,
            confidence=confidence,
            timestamp=datetime.utcnow()
        )

    def _parse_cards(self, text: str) -> List[str]:
        """Parse card string into list of standardized cards."""
        cards = []
        # Match patterns like "Ah", "Kd", "Tc"
        matches = self.CARD_PATTERN.findall(text)
        for rank, suit in matches:
            cards.append(f"{rank.upper()}{suit.lower()}")
        return cards

    def _parse_number(self, text: str) -> float:
        """Parse money/number string."""
        if not text or text.lower() in ('unknown', 'none', ''):
            return 0.0

        # Remove currency symbols and commas
        cleaned = re.sub(r'[,$]', '', text)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _empty_state(self, error: str) -> PokerTableState:
        """Return empty state with error."""
        return PokerTableState(
            hole_cards=[],
            community_cards=[],
            pot_size=0,
            current_bet=0,
            player_stack=None,
            street="unknown",
            num_players=0,
            position=None,
            raw_text=error,
            confidence=0,
            timestamp=datetime.utcnow()
        )

    def _calculate_position(self, hero_seat: int, dealer_seat: int, num_players: int) -> str:
        """Calculate hero position relative to dealer button.

        Args:
            hero_seat: Hero's seat number (1-9)
            dealer_seat: Dealer button seat number (1-9)
            num_players: Number of players at table

        Returns:
            Position string: BTN, SB, BB, UTG, UTG+1, MP, HJ, CO
        """
        if num_players <= 0:
            return "unknown"

        # Calculate seats from dealer (0 = BTN, 1 = SB, 2 = BB, etc.)
        # Seats are numbered 1-9 typically, positions go clockwise from dealer
        offset = (hero_seat - dealer_seat) % num_players

        if num_players <= 3:
            # 3 or fewer players: BTN, SB, BB only
            positions = {0: "BTN", 1: "SB", 2: "BB"}
            return positions.get(offset, "BTN")

        elif num_players <= 6:
            # 6-max positions
            positions_6max = {
                0: "BTN",
                1: "SB",
                2: "BB",
                3: "UTG",
                4: "HJ",  # Hijack
                5: "CO"   # Cutoff
            }
            return positions_6max.get(offset, "MP")

        else:
            # Full ring (9-max)
            if offset == 0:
                return "BTN"
            elif offset == 1:
                return "SB"
            elif offset == 2:
                return "BB"
            elif offset == 3:
                return "UTG"
            elif offset == 4:
                return "UTG+1"
            elif offset == num_players - 1:
                return "CO"
            elif offset == num_players - 2:
                return "HJ"
            elif offset == num_players - 3:
                return "LJ"  # Lojack
            else:
                return "MP"

    def update_vpip(self, hole_cards: List[str], action: str, is_raise: bool = False):
        """Track VPIP (Voluntarily Put $ In Pot) for leak detection.

        Args:
            hole_cards: Player's hole cards
            action: 'fold', 'call', 'raise'
            is_raise: Whether this was a raise/bet (for PFR tracking)
        """
        self._total_hands += 1
        if action in ('call', 'raise'):
            self._vpip_hands += 1

        if is_raise or action == 'raise':
            self._pfr_hands += 1

        self._hand_history.append({
            'cards': hole_cards,
            'action': action,
            'is_raise': is_raise,
            'timestamp': datetime.utcnow().isoformat()
        })

        # Keep last 100 hands
        if len(self._hand_history) > 100:
            self._hand_history = self._hand_history[-100:]

        # Mark hand as tracked
        self._current_hand_tracked = True

    def detect_new_hand(self, state: 'PokerTableState') -> bool:
        """Detect if a new hand has started from OCR state.

        Args:
            state: Current table state

        Returns:
            True if new hand detected
        """
        # New hand detection: hole cards changed or went from postflop to preflop
        is_new_hand = False

        if state.hole_cards and len(state.hole_cards) == 2:
            if self._last_hole_cards is None:
                # First cards we've seen
                is_new_hand = True
            elif state.hole_cards != self._last_hole_cards:
                # Different hole cards = new hand
                is_new_hand = True
            elif state.street == "preflop" and self._last_street in ("turn", "river"):
                # Back to preflop = new hand
                is_new_hand = True

        if is_new_hand:
            self._last_hole_cards = state.hole_cards.copy()
            self._last_street = state.street
            self._current_hand_tracked = False
            logger.debug(f"New hand detected: {state.hole_cards}")

        return is_new_hand

    def get_vpip_stats(self) -> Dict[str, Any]:
        """Get VPIP statistics for leak analysis."""
        if self._total_hands == 0:
            return {
                "vpip": 0,
                "pfr": 0,
                "hands": 0,
                "leak": None,
                "session_minutes": self._get_session_minutes()
            }

        vpip = (self._vpip_hands / self._total_hands) * 100
        pfr = (self._pfr_hands / self._total_hands) * 100

        # VPIP analysis - optimal is 18-25% for 6-max
        leak = None
        if vpip > 30:
            leak = "Too loose - playing too many hands"
        elif vpip < 15:
            leak = "Too tight - missing value spots"
        elif vpip - pfr > 10:
            leak = "Too passive - call too much, raise more"

        return {
            "vpip": round(vpip, 1),
            "pfr": round(pfr, 1),
            "vpip_pfr_gap": round(vpip - pfr, 1),
            "hands": self._total_hands,
            "hands_played": self._vpip_hands,
            "hands_raised": self._pfr_hands,
            "leak": leak,
            "recommendation": self._get_vpip_recommendation(vpip, pfr),
            "session_minutes": self._get_session_minutes(),
            "hands_per_hour": self._get_hands_per_hour()
        }

    def _get_session_minutes(self) -> int:
        """Get session duration in minutes."""
        return int((datetime.utcnow() - self._session_start).total_seconds() / 60)

    def _get_hands_per_hour(self) -> float:
        """Calculate hands per hour rate."""
        minutes = self._get_session_minutes()
        if minutes == 0:
            return 0
        return round((self._total_hands / minutes) * 60, 1)

    def _get_vpip_recommendation(self, vpip: float, pfr: float = 0) -> str:
        """Get VPIP/PFR improvement recommendation."""
        gap = vpip - pfr

        if vpip > 35:
            return "Tighten up significantly. Focus on premium hands."
        elif vpip > 30:
            return "Fold more marginal hands like suited connectors from early position."
        elif vpip > 25:
            return "Slightly too loose. Cut weaker Ax and low pairs from EP."
        elif vpip < 15:
            return "Too tight. Add suited connectors and small pairs in position."
        elif vpip < 18:
            return "Slightly tight. Widen range in late position."
        elif gap > 12:
            return "Too passive. Raise more preflop instead of calling."
        elif gap < 3:
            return "Very aggressive - ensure you have hands to call with."
        else:
            return "VPIP/PFR in optimal range."

    def reset_session(self) -> Dict[str, Any]:
        """Reset session statistics.

        Returns:
            Final session stats before reset
        """
        final_stats = self.get_vpip_stats()
        final_stats["reset_at"] = datetime.utcnow().isoformat()

        # Reset all counters
        self._session_start = datetime.utcnow()
        self._hand_history = []
        self._vpip_hands = 0
        self._pfr_hands = 0
        self._total_hands = 0
        self._last_hole_cards = None
        self._last_street = "unknown"
        self._current_hand_tracked = False

        logger.info("Session stats reset")
        return final_stats

    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        stats = self.get_vpip_stats()

        # Add recent hand history summary
        recent_hands = self._hand_history[-10:] if self._hand_history else []

        return {
            **stats,
            "session_start": self._session_start.isoformat(),
            "recent_hands": recent_hands,
            "current_hole_cards": self._last_hole_cards
        }


class SPRCalculator:
    """Stack-to-Pot Ratio calculator for commitment decisions."""

    # SPR commitment zones
    COMMITMENT_ZONES = {
        "committed": (0, 4),      # SPR < 4: All-in territory
        "medium": (4, 13),        # SPR 4-13: Standard play
        "deep": (13, float('inf'))  # SPR > 13: Deep stack poker
    }

    @staticmethod
    def calculate_spr(stack: float, pot: float) -> Dict[str, Any]:
        """Calculate SPR and commitment zone.

        Args:
            stack: Effective stack size
            pot: Current pot size

        Returns:
            Dict with SPR value, zone, and recommendation
        """
        if pot <= 0:
            return {
                "spr": float('inf'),
                "zone": "deep",
                "color": "green",
                "recommendation": "Deep SPR - speculative hands valuable"
            }

        spr = stack / pot

        # Determine zone
        if spr < 4:
            zone = "committed"
            color = "red"
            recommendation = "Low SPR - commit with top pair+, big draws. All-in territory."
        elif spr < 13:
            zone = "medium"
            color = "yellow"
            recommendation = "Medium SPR - standard hand selection applies."
        else:
            zone = "deep"
            color = "green"
            recommendation = "Deep SPR - set mining, implied odds valuable. Speculative hands gain value."

        return {
            "spr": round(spr, 1),
            "zone": zone,
            "color": color,
            "recommendation": recommendation,
            "stack": stack,
            "pot": pot
        }

    @staticmethod
    def get_commitment_threshold(spr: float) -> str:
        """Get hand strength needed for commitment at given SPR.

        Args:
            spr: Stack-to-pot ratio

        Returns:
            Minimum hand strength recommendation
        """
        if spr < 3:
            return "Any pair, any draw with 8+ outs"
        elif spr < 6:
            return "Top pair good kicker+, strong draws"
        elif spr < 10:
            return "Overpair+, nut draws"
        elif spr < 15:
            return "Two pair+, combo draws"
        else:
            return "Sets+, nut straights/flushes"


class EVCalculator:
    """Expected Value calculator with bet sizing options and meta awareness."""

    BET_SIZES = {
        "1/4": 0.25,
        "1/3": 0.33,
        "1/2": 0.50,
        "2/3": 0.67,
        "3/4": 0.75,
        "pot": 1.0,
        "1.5x": 1.5,
        "2x": 2.0
    }

    def __init__(self):
        self._meta_advisor = None

    def _get_meta_advisor(self):
        """Lazy load meta advisor."""
        if self._meta_advisor is None:
            try:
                from backend.poker.meta_advisor import meta_advisor
                self._meta_advisor = meta_advisor
            except ImportError:
                logger.warning("Meta advisor not available")
        return self._meta_advisor

    def calculate_ev_table(
        self,
        pot: float,
        equity: float,
        villain_fold_pct: float = 30,
        street: str = "flop",
        use_meta: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Calculate EV for different bet sizes with optional meta adjustment.

        Args:
            pot: Current pot size
            equity: Our hand equity (0-100)
            villain_fold_pct: Estimated fold percentage to bets
            street: Current street (preflop, flop, turn, river)
            use_meta: Whether to apply meta-based adjustments

        Returns:
            Dict mapping bet size to EV and profit calculations
        """
        # Get meta adjustments
        meta_advisor = self._get_meta_advisor() if use_meta else None
        adjusted_fold_pct = villain_fold_pct

        if meta_advisor:
            adjusted_fold_pct = meta_advisor.adjust_fold_equity(villain_fold_pct, street)

        equity_decimal = equity / 100
        fold_decimal = adjusted_fold_pct / 100

        results = {}

        for size_name, size_pct in self.BET_SIZES.items():
            bet_amount = pot * size_pct

            # EV = (fold% * pot) + (call% * (equity * new_pot - (1-equity) * bet))
            ev_when_fold = fold_decimal * pot

            new_pot = pot + bet_amount
            ev_when_call = (1 - fold_decimal) * (
                equity_decimal * new_pot - (1 - equity_decimal) * bet_amount
            )

            base_ev = ev_when_fold + ev_when_call

            # Apply meta EV modifier
            total_ev = base_ev
            meta_modifier = 1.0
            if meta_advisor:
                meta_data = meta_advisor.calculate_adjusted_ev(
                    base_ev, pot, bet_amount, street, "bet"
                )
                total_ev = meta_data["adjusted_ev"]
                meta_modifier = meta_data["ev_modifier"]

            results[size_name] = {
                "bet_amount": round(bet_amount, 2),
                "ev": round(total_ev, 2),
                "base_ev": round(base_ev, 2),
                "meta_modifier": meta_modifier,
                "profitable": total_ev > 0,
                "ev_bb": round(total_ev / max(pot * 0.01, 1), 1)
            }

        # Find optimal size
        best_size = max(results.items(), key=lambda x: x[1]['ev'])
        results["optimal"] = {
            "size": best_size[0],
            "ev": best_size[1]['ev']
        }

        # Add meta context
        if meta_advisor:
            meta_snapshot = meta_advisor.get_current_meta()
            results["meta"] = {
                "fold_equity_adjust": round(adjusted_fold_pct - villain_fold_pct, 1),
                "adjusted_fold_pct": round(adjusted_fold_pct, 1),
                "trend_count": meta_snapshot.to_dict()["trend_count"],
                "summary": meta_advisor._get_meta_summary()
            }

        return results

    @staticmethod
    def pot_odds_required(bet: float, pot: float) -> float:
        """Calculate equity needed to call profitably."""
        return (bet / (pot + bet)) * 100

    @staticmethod
    def implied_odds(
        pot: float,
        bet: float,
        expected_future_bets: float
    ) -> float:
        """Calculate implied odds equity requirement."""
        total_pot = pot + bet + expected_future_bets
        return (bet / total_pot) * 100

    async def refresh_meta(self, force: bool = False) -> Dict[str, Any]:
        """Refresh meta trends from Perplexity.

        Args:
            force: Force refresh even if cache valid

        Returns:
            Current meta snapshot data
        """
        meta_advisor = self._get_meta_advisor()
        if meta_advisor:
            snapshot = await meta_advisor.refresh_meta(force)
            return snapshot.to_dict()
        return {"error": "Meta advisor not available"}


class PokerSiteRouter:
    """Routes OCR requests to site-specific implementations.

    Supports automatic site detection or manual mode switching via voice commands.
    """

    def __init__(self):
        self._current_site = PokerSite.GENERIC
        self._generic_ocr = PokerOCR()
        self._ignition_ocr = None  # Lazy load

        # Site detection keywords/patterns
        self._site_indicators = {
            PokerSite.IGNITION: [
                "ignition", "ignition casino", "bovada",
                "seat 1", "seat 2", "seat 3",  # Anonymous player names
            ],
        }

    @property
    def current_site(self) -> PokerSite:
        """Get current site mode."""
        return self._current_site

    def set_site(self, site: Union[PokerSite, str]) -> str:
        """Set the poker site mode.

        Args:
            site: PokerSite enum or string name

        Returns:
            Confirmation message
        """
        if isinstance(site, str):
            site_lower = site.lower()
            if "ignition" in site_lower or "bovada" in site_lower:
                site = PokerSite.IGNITION
            else:
                site = PokerSite.GENERIC

        self._current_site = site
        logger.info(f"Poker OCR mode set to: {site.value}")

        # Lazy load site-specific OCR
        if site == PokerSite.IGNITION and self._ignition_ocr is None:
            try:
                from backend.vision.ignition_ocr import ignition_ocr
                self._ignition_ocr = ignition_ocr
            except ImportError:
                logger.error("Failed to load Ignition OCR module")
                self._current_site = PokerSite.GENERIC
                return "Ignition OCR not available, using generic mode"

        return f"Switched to {site.value} mode"

    async def analyze_frame(self, image_base64: str) -> Union[PokerTableState, Any]:
        """Route frame analysis to appropriate site-specific OCR.

        Args:
            image_base64: Base64 encoded image

        Returns:
            PokerTableState or site-specific result
        """
        if self._current_site == PokerSite.IGNITION and self._ignition_ocr:
            # Use Ignition-specific OCR
            return await self._ignition_ocr.read_table(image_base64)
        else:
            # Use generic OCR
            return await self._generic_ocr.analyze_frame(image_base64)

    async def auto_detect_site(self, image_base64: str) -> PokerSite:
        """Attempt to auto-detect poker site from screenshot.

        This uses a quick Gemini call to identify the site.
        """
        # For now, return current site
        # Future: implement visual site detection
        return self._current_site

    def get_ocr_instance(self):
        """Get the current OCR instance for direct access."""
        if self._current_site == PokerSite.IGNITION and self._ignition_ocr:
            return self._ignition_ocr
        return self._generic_ocr


# Global instances
poker_ocr = PokerOCR()
ev_calculator = EVCalculator()
spr_calculator = SPRCalculator()
poker_router = PokerSiteRouter()
