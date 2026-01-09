"""Voice tool for poker HUD control and EV calculations."""
import logging
import re
from typing import Optional

from .base import VoiceTool, VoiceToolResult
from backend.vision.poker_ocr import poker_ocr, ev_calculator, poker_router, PokerSite
from backend.agent.tools.poker import PokerTool
from backend.poker.live_mode import live_mode

logger = logging.getLogger(__name__)


class PokerVoiceTool(VoiceTool):
    """Voice tool for poker HUD and odds calculations.

    Handles voice commands like:
    - "Show HUD odds" - Activates poker HUD stream
    - "What are my odds with pocket aces?" - Hand strength query
    - "Calculate EV for ace king" - EV calculations
    - "Show my VPIP stats" - Leak analysis

    HUD Ultimate features:
    - "Enable 4 tables" - Multi-table mode with 2-4 cameras
    - "Physics mode on" - OCR debug overlay with bounding boxes
    - "OCR debug" - Same as physics mode
    - "Pocket aces vs 3 opponents" - Multi-opponent equity
    - "ICM with stacks 5000, 3000, 2000" - ICM calculations
    - "Tournament equity" - ICM guidance

    Offline mode:
    - "Go offline" / "Offline mode" - Enable local LLM and voice
    - "Go online" / "Online mode" - Return to cloud services
    """

    name = "poker"
    description = "Poker HUD control and odds calculations"

    # Keywords for activation
    keywords = [
        r'\bpoker\b',
        r'\bhud\b',
        r'\bodds\b',
        r'\bequity\b',
        r'\bev\b',
        r'\bexpected value\b',
        r'\bpot odds\b',
        r'\bvpip\b',
        r'\bhand\s+strength\b',
        r'\bshow\s+(?:me\s+)?(?:the\s+)?(?:hud|odds|equity)\b',
        r'\bpocket\s+\w+',  # pocket aces, pocket kings, etc.
        r'\bace\s+king\b',
        r'\bking\s+queen\b',
        r'\b(?:AK|AQ|AA|KK|QQ|JJ|TT)\b',  # Hand shorthand
        # HUD control commands
        r'\bhud\s+(?:zoom|bigger|larger|smaller)\b',
        r'\b(?:hide|show)\s+(?:vpip|chart|stats)\b',
        r'\bmulti[- ]?table\b',
        r'\bcalibrate\s+(?:pot|ocr)\b',
        r'\b(?:export|download)\s+(?:hands?|log|csv)\b',
        r'\b(?:export|generate|create)\s+(?:vpip|session)\s+(?:report|pdf|stats)\b',
        # HUD Ultimate features
        r'\bphysics\s+mode\b',
        r'\bocr\s+debug\b',
        r'\bdebug\s+(?:mode|ocr)\b',
        r'\bshow\s+(?:bounding\s+)?boxes\b',
        r'\b(?:two|three|four|2|3|4)\s+(?:tables?|opponents?)\b',
        r'\bvs\s+(?:\d+|two|three|four|multiple)\b',
        r'\bicm\b',
        r'\btournament\s+(?:equity|value|decision)\b',
        r'\bbubble\s+(?:factor|icm)\b',
        r'\bchip\s+(?:ev|value)\b',
        r'\bstack\s+(?:size|value)s?\b',
        # Offline mode commands
        r'\boffline\s+mode\b',
        r'\bgo\s+offline\b',
        r'\bwork\s+offline\b',
        r'\bgo\s+online\b',
        r'\bonline\s+mode\b',
        # Site-specific mode commands
        r'\bignition\s+mode\b',
        r'\bswitch\s+to\s+ignition\b',
        r'\bignition\s+poker\b',
        r'\bbovada\s+mode\b',
        r'\bgeneric\s+mode\b',
        r'\bdefault\s+mode\b',
        # Live game mode commands
        r'\blive\s+(?:mode|game|poker)\b',
        r'\bstart\s+live\b',
        r'\bend\s+live\b',
        r'\bexit\s+live\b',
        r'\bcasino\s+(?:mode|game)\b',
        r'\bhome\s+game\b',
        r'\bnew\s+hand\b',
        r'\bi\s+have\b',
        r'\bholding\b',
        r'\bpot\s+is\b',
        r'\bmy\s+stack\b',
        r'\bboard\s+is\b',
        r'\bflop\s+is\b',
        r'\bturn\s+(?:is|card)\b',
        r'\briver\s+(?:is|card)\b',
        r'\bfacing\b',
        r'\bi\s+won\b',
        r'\bi\s+lost\b',
    ]

    priority = 7  # Moderate priority

    # Card mappings for voice input
    CARD_NAMES = {
        'ace': 'A', 'aces': 'A',
        'king': 'K', 'kings': 'K',
        'queen': 'Q', 'queens': 'Q',
        'jack': 'J', 'jacks': 'J',
        'ten': 'T', 'tens': 'T',
        'nine': '9', 'nines': '9',
        'eight': '8', 'eights': '8',
        'seven': '7', 'sevens': '7',
        'six': '6', 'sixes': '6',
        'five': '5', 'fives': '5',
        'four': '4', 'fours': '4',
        'three': '3', 'threes': '3',
        'two': '2', 'twos': '2', 'deuce': '2', 'deuces': '2',
    }

    SUIT_NAMES = {
        'hearts': 'h', 'heart': 'h',
        'diamonds': 'd', 'diamond': 'd',
        'clubs': 'c', 'club': 'c',
        'spades': 's', 'spade': 's',
        'suited': 's',  # Generic "suited"
        'offsuit': 'o',
    }

    def __init__(self):
        self.poker_tool = PokerTool()

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute poker-related voice command."""
        query_lower = query.lower()

        try:
            # Check for live mode commands first (casino/home game support)
            if self._is_live_mode_command(query_lower):
                return await self._handle_live_mode(query_lower)

            # Check for HUD control commands (zoom, hide, show, etc.)
            if self._is_hud_control_command(query_lower):
                return await self._handle_hud_control(query_lower)

            # Check for HUD activation commands
            if self._is_hud_command(query_lower):
                return await self._handle_hud_command(query_lower)

            # Check for VPIP stats request
            if 'vpip' in query_lower or 'leak' in query_lower:
                return await self._handle_vpip_stats()

            # Check for ICM calculation request
            if self._is_icm_query(query_lower):
                return await self._handle_icm_query(query_lower)

            # Check for multi-opponent equity request
            if self._is_multi_opponent_query(query_lower):
                hand = self._extract_hand(query_lower)
                if hand:
                    return await self._handle_multi_opponent_query(hand, query_lower)

            # Check for hand-specific queries
            hand = self._extract_hand(query_lower)
            if hand:
                return await self._handle_hand_query(hand, query_lower)

            # Default: general poker info
            return VoiceToolResult(
                success=True,
                message="Open the poker HUD at hud poker dot html. Say 'show HUD odds' to start the webcam stream, or ask about specific hands like 'odds with pocket aces'.",
                data={"action": "info"}
            )

        except Exception as e:
            logger.error(f"Poker tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble with that poker query.",
                data={"error": str(e)}
            )

    def _is_hud_command(self, query: str) -> bool:
        """Check if this is a HUD activation command."""
        hud_patterns = [
            'show hud', 'start hud', 'open hud',
            'show odds', 'show my odds',
            'start poker', 'open poker',
            'hud on', 'activate hud',
            'poker stream', 'start stream',
        ]
        return any(pattern in query for pattern in hud_patterns)

    def _is_hud_control_command(self, query: str) -> bool:
        """Check if this is a HUD control command (zoom, hide, show, etc.)."""
        control_patterns = [
            r'hud\s+(zoom|bigger|larger|smaller)',
            r'(hide|show)\s+(vpip|chart|stats)',
            r'multi[- ]?table',
            r'calibrate\s+(pot|ocr)',
            r'(export|download)\s+(hands?|log|csv)',
            # HUD Ultimate patterns
            r'physics\s+mode',
            r'ocr\s+debug',
            r'debug\s+(mode|ocr)',
            r'show\s+(bounding\s+)?boxes',
            r'(two|three|four|2|3|4)\s+tables?',
            # Offline mode patterns
            r'offline\s+mode',
            r'go\s+(offline|online)',
            r'work\s+offline',
            r'online\s+mode',
            # VPIP PDF report patterns
            r'(export|generate|create)\s+(vpip|session)\s+(report|pdf|stats)',
            # Site mode patterns
            r'ignition\s+mode',
            r'switch\s+to\s+ignition',
            r'ignition\s+poker',
            r'bovada\s+mode',
            r'generic\s+mode',
            r'default\s+mode',
        ]
        return any(re.search(pattern, query) for pattern in control_patterns)

    async def _handle_hud_control(self, query: str) -> VoiceToolResult:
        """Handle HUD control commands."""
        import re

        # Zoom commands
        if re.search(r'hud\s+(zoom|bigger|larger)', query):
            return VoiceToolResult(
                success=True,
                message="Zooming in on the HUD. The display is now larger.",
                data={
                    "action": "hud_control",
                    "command": "zoom_in",
                    "hud_message": {"type": "control", "action": "zoom", "direction": "in"}
                }
            )

        if re.search(r'hud\s+smaller', query):
            return VoiceToolResult(
                success=True,
                message="Zooming out on the HUD. The display is now smaller.",
                data={
                    "action": "hud_control",
                    "command": "zoom_out",
                    "hud_message": {"type": "control", "action": "zoom", "direction": "out"}
                }
            )

        # Hide/show VPIP
        if re.search(r'hide\s+vpip', query):
            return VoiceToolResult(
                success=True,
                message="Hiding VPIP stats panel.",
                data={
                    "action": "hud_control",
                    "command": "hide_vpip",
                    "hud_message": {"type": "control", "action": "toggle", "element": "vpip", "visible": False}
                }
            )

        if re.search(r'show\s+vpip', query):
            return VoiceToolResult(
                success=True,
                message="Showing VPIP stats panel.",
                data={
                    "action": "hud_control",
                    "command": "show_vpip",
                    "hud_message": {"type": "control", "action": "toggle", "element": "vpip", "visible": True}
                }
            )

        # Hide/show chart
        if re.search(r'hide\s+chart', query):
            return VoiceToolResult(
                success=True,
                message="Hiding the VPIP trend chart.",
                data={
                    "action": "hud_control",
                    "command": "hide_chart",
                    "hud_message": {"type": "control", "action": "toggle", "element": "chart", "visible": False}
                }
            )

        if re.search(r'show\s+chart', query):
            return VoiceToolResult(
                success=True,
                message="Showing the VPIP trend chart.",
                data={
                    "action": "hud_control",
                    "command": "show_chart",
                    "hud_message": {"type": "control", "action": "toggle", "element": "chart", "visible": True}
                }
            )

        # Multi-table mode with specific count
        table_count_match = re.search(r'(two|three|four|2|3|4)\s+tables?', query)
        if table_count_match:
            count_map = {'two': 2, 'three': 3, 'four': 4, '2': 2, '3': 3, '4': 4}
            count_str = table_count_match.group(1).lower()
            num_tables = count_map.get(count_str, 2)
            return VoiceToolResult(
                success=True,
                message=f"Enabling {num_tables}-table mode. Connect {num_tables} webcams for multi-table grinding.",
                data={
                    "action": "hud_control",
                    "command": "multi_table",
                    "hud_message": {"type": "control", "action": "multi_table", "enabled": True, "count": num_tables}
                }
            )

        # Generic multi-table mode
        if re.search(r'multi[- ]?table', query):
            return VoiceToolResult(
                success=True,
                message="Toggling multi-table mode. Connect a second webcam for the second table.",
                data={
                    "action": "hud_control",
                    "command": "multi_table",
                    "hud_message": {"type": "control", "action": "multi_table", "enabled": True}
                }
            )

        # Physics Mode / OCR Debug
        if re.search(r'physics\s+mode|ocr\s+debug|debug\s+(mode|ocr)|show\s+(bounding\s+)?boxes', query):
            # Check if disabling
            if 'off' in query or 'disable' in query or 'hide' in query:
                return VoiceToolResult(
                    success=True,
                    message="Disabling physics mode. OCR debug overlay hidden.",
                    data={
                        "action": "hud_control",
                        "command": "physics_mode",
                        "hud_message": {"type": "control", "action": "physics_mode", "enabled": False}
                    }
                )
            return VoiceToolResult(
                success=True,
                message="Enabling physics mode. Showing OCR bounding boxes and debug stats. FPS, latency, and confidence will be displayed.",
                data={
                    "action": "hud_control",
                    "command": "physics_mode",
                    "hud_message": {"type": "control", "action": "physics_mode", "enabled": True}
                }
            )

        # Offline mode toggle
        if re.search(r'offline\s+mode|go\s+offline|work\s+offline', query):
            # Enable offline mode
            return await self._handle_offline_toggle(enabled=True)

        if re.search(r'online\s+mode|go\s+online', query):
            # Disable offline mode (go back online)
            return await self._handle_offline_toggle(enabled=False)

        # Calibrate pot
        if re.search(r'calibrate\s+(pot|ocr)', query):
            return VoiceToolResult(
                success=True,
                message="Calibration mode active. Tap the pot area on screen to set the OCR focus region.",
                data={
                    "action": "hud_control",
                    "command": "calibrate",
                    "hud_message": {"type": "control", "action": "calibrate", "target": "pot"}
                }
            )

        # Export hands
        if re.search(r'(export|download)\s+(hands?|log|csv)', query):
            return VoiceToolResult(
                success=True,
                message="Exporting hand log to CSV. Check your downloads folder.",
                data={
                    "action": "hud_control",
                    "command": "export_hands",
                    "hud_message": {"type": "control", "action": "export", "format": "csv"}
                }
            )

        # VPIP PDF Report
        if re.search(r'(export|generate|create)\s+(vpip|session)\s+(report|pdf|stats)', query):
            return await self._handle_vpip_report()

        # Site mode switching (Ignition, Bovada, Generic)
        if re.search(r'ignition\s+mode|switch\s+to\s+ignition|ignition\s+poker|bovada\s+mode', query):
            return await self._handle_site_mode(PokerSite.IGNITION)

        if re.search(r'generic\s+mode|default\s+mode', query):
            return await self._handle_site_mode(PokerSite.GENERIC)

        # Default control response
        return VoiceToolResult(
            success=True,
            message="HUD control command received.",
            data={"action": "hud_control", "command": "unknown"}
        )

    async def _handle_hud_command(self, query: str) -> VoiceToolResult:
        """Handle HUD activation command."""
        if 'stop' in query or 'off' in query or 'close' in query:
            return VoiceToolResult(
                success=True,
                message="Poker HUD stopped. You can close the browser tab or say 'show HUD odds' to restart.",
                data={
                    "action": "stop_hud",
                    "hud_url": "/hud/poker.html"
                }
            )

        return VoiceToolResult(
            success=True,
            message="Starting poker HUD. Open the HUD page in your browser at hud slash poker dot html. Point your webcam at the poker table for live odds.",
            data={
                "action": "start_hud",
                "hud_url": "/hud/poker.html",
                "stream_endpoint": "/ws/poker-stream",
                "instructions": [
                    "Navigate to /hud/poker.html in your browser",
                    "Allow webcam access when prompted",
                    "Click 'Start Stream' to begin OCR",
                    "Point camera at poker table for live analysis"
                ]
            }
        )

    async def _handle_vpip_stats(self) -> VoiceToolResult:
        """Handle VPIP stats request."""
        stats = poker_ocr.get_vpip_stats()

        if stats['total_hands'] == 0:
            return VoiceToolResult(
                success=True,
                message="No poker hands tracked yet. Start playing to build your stats.",
                data={"vpip_stats": stats}
            )

        vpip_pct = stats['vpip_percentage']
        pfr_pct = stats['pfr_percentage']
        hands = stats['total_hands']

        # Provide coaching based on stats
        if vpip_pct > 35:
            coaching = "You're playing too loose. Try tightening up your range."
        elif vpip_pct < 15:
            coaching = "You're playing very tight. Consider adding more hands in position."
        else:
            coaching = "Your VPIP is in a solid range."

        message = f"After {hands} hands, your VPIP is {vpip_pct:.1f}% and PFR is {pfr_pct:.1f}%. {coaching}"

        return VoiceToolResult(
            success=True,
            message=message,
            data={"vpip_stats": stats}
        )

    def _extract_hand(self, query: str) -> Optional[str]:
        """Extract poker hand from voice query."""
        import re

        # Direct shorthand like "AK" or "AA"
        match = re.search(r'\b([AKQJT98765432]{2}[so]?)\b', query.upper())
        if match:
            return match.group(1)

        # Pocket pairs: "pocket aces", "pocket kings"
        pocket_match = re.search(r'pocket\s+(\w+)', query)
        if pocket_match:
            card_name = pocket_match.group(1).lower()
            if card_name in self.CARD_NAMES:
                rank = self.CARD_NAMES[card_name]
                return f"{rank}{rank}"  # Pocket pair

        # Two card names: "ace king", "king queen"
        two_cards = re.search(r'(\w+)\s+(\w+)', query)
        if two_cards:
            card1 = two_cards.group(1).lower()
            card2 = two_cards.group(2).lower()
            if card1 in self.CARD_NAMES and card2 in self.CARD_NAMES:
                r1 = self.CARD_NAMES[card1]
                r2 = self.CARD_NAMES[card2]
                # Check for suited/offsuit
                if 'suited' in query:
                    return f"{r1}{r2}s"
                elif 'offsuit' in query:
                    return f"{r1}{r2}o"
                return f"{r1}{r2}"

        return None

    async def _handle_hand_query(self, hand: str, query: str) -> VoiceToolResult:
        """Handle query about a specific poker hand."""
        # Normalize hand
        if len(hand) >= 2:
            r1, r2 = hand[0].upper(), hand[1].upper()
            suited = 's' in hand.lower()

            # Calculate preflop equity vs random hand
            if r1 == r2:
                # Pocket pair
                hand_str = f"{r1}h{r2}d"  # Different suits
                hand_display = f"pocket {self._rank_name(r1)}s"
            else:
                if suited:
                    hand_str = f"{r1}h{r2}h"  # Same suit
                    hand_display = f"{self._rank_name(r1)}-{self._rank_name(r2)} suited"
                else:
                    hand_str = f"{r1}h{r2}d"  # Different suits
                    hand_display = f"{self._rank_name(r1)}-{self._rank_name(r2)} offsuit"

            # Get equity using Monte Carlo
            result = await self.poker_tool.execute(
                action="equity",
                hand=hand_str,
                board="",
                opponents=1,
                simulations=5000
            )

            if result.success:
                equity = result.data.get('equity', 50.0)

                # Classify hand strength
                if equity > 80:
                    strength = "premium hand"
                elif equity > 65:
                    strength = "strong hand"
                elif equity > 55:
                    strength = "playable hand"
                elif equity > 45:
                    strength = "marginal hand"
                else:
                    strength = "weak hand"

                message = f"{hand_display} has about {equity:.0f}% equity heads up. That's a {strength}."

                return VoiceToolResult(
                    success=True,
                    message=message,
                    data={
                        "hand": hand_str,
                        "display": hand_display,
                        "equity": equity,
                        "strength": strength
                    }
                )

            return VoiceToolResult(
                success=False,
                message=f"Couldn't calculate odds for {hand_display}.",
                data={"hand": hand_str}
            )

        return VoiceToolResult(
            success=False,
            message="I didn't catch the hand. Try saying something like 'odds with pocket aces' or 'ace king suited'.",
            data={}
        )

    def _rank_name(self, rank: str) -> str:
        """Convert rank code to spoken name."""
        names = {
            'A': 'ace', 'K': 'king', 'Q': 'queen', 'J': 'jack', 'T': 'ten',
            '9': 'nine', '8': 'eight', '7': 'seven', '6': 'six', '5': 'five',
            '4': 'four', '3': 'three', '2': 'two'
        }
        return names.get(rank.upper(), rank)

    def _is_icm_query(self, query: str) -> bool:
        """Check if query is about ICM calculations."""
        import re
        icm_patterns = [
            r'\bicm\b',
            r'\btournament\s+(equity|value|decision)',
            r'\bbubble\s+(factor|icm)',
            r'\bchip\s+(ev|value)',
            r'\bstack\s+(size|value)s?',
            r'\bprize\s+pool',
            r'\bpayout\s+structure',
        ]
        return any(re.search(pattern, query) for pattern in icm_patterns)

    def _is_multi_opponent_query(self, query: str) -> bool:
        """Check if query is about multi-opponent equity."""
        import re
        multi_patterns = [
            r'\bvs\s+(\d+|two|three|four|multiple)',
            r'\b(two|three|four|2|3|4)\s+opponents?',
            r'\bmulti[- ]?way',
            r'\bthree[- ]?way',
            r'\bfour[- ]?way',
            r'\b(3|4)[- ]?way',
        ]
        return any(re.search(pattern, query) for pattern in multi_patterns)

    def _extract_opponent_count(self, query: str) -> int:
        """Extract number of opponents from query."""
        import re
        count_map = {'two': 2, 'three': 3, 'four': 4, 'multiple': 3}

        # Check for numeric
        num_match = re.search(r'(\d+)\s*opponents?', query)
        if num_match:
            return min(int(num_match.group(1)), 8)

        # Check for word-based
        word_match = re.search(r'(two|three|four|multiple)\s*opponents?', query)
        if word_match:
            return count_map.get(word_match.group(1).lower(), 2)

        # Check for "vs N"
        vs_match = re.search(r'vs\s+(\d+|two|three|four|multiple)', query)
        if vs_match:
            val = vs_match.group(1).lower()
            if val.isdigit():
                return min(int(val), 8)
            return count_map.get(val, 2)

        # Check for N-way
        way_match = re.search(r'(three|four|3|4)[- ]?way', query)
        if way_match:
            val = way_match.group(1).lower()
            if val.isdigit():
                return int(val) - 1  # N-way means N-1 opponents
            return count_map.get(val, 3) - 1

        return 2  # Default to 2 opponents

    async def _handle_icm_query(self, query: str) -> VoiceToolResult:
        """Handle ICM calculation queries."""
        import re

        # Try to extract stack sizes from query
        stack_match = re.search(r'(\d+)[,\s]+(\d+)[,\s]+(\d+)', query)
        if stack_match:
            stacks = [int(stack_match.group(i)) for i in range(1, 4)]
        else:
            # Provide general ICM guidance
            return VoiceToolResult(
                success=True,
                message="ICM calculations require stack sizes. Say something like 'ICM with stacks 5000, 3000, 2000' or provide your stack and remaining players.",
                data={
                    "action": "icm_info",
                    "examples": [
                        "ICM with stacks 5000, 3000, 2000",
                        "What's my ICM equity with 10000 chips, 3 players left",
                        "Bubble factor with 4 players remaining"
                    ]
                }
            )

        # Calculate ICM equity
        try:
            from phone_client.engines.icm import ICMCalculator
            icm = ICMCalculator()

            # Default prize pool (can be customized)
            total_chips = sum(stacks)
            # Standard 50/30/20 payout
            payouts = [0.5, 0.3, 0.2][:len(stacks)]

            equities = icm.calculate_icm(stacks, payouts)

            # Format response
            equity_strs = [f"Stack {i+1} ({stacks[i]}): {eq*100:.1f}%" for i, eq in enumerate(equities)]
            message = f"ICM equities for stacks {stacks}: " + ", ".join(equity_strs)

            return VoiceToolResult(
                success=True,
                message=message,
                data={
                    "action": "icm_calculation",
                    "stacks": stacks,
                    "equities": equities,
                    "payouts": payouts
                }
            )
        except Exception as e:
            logger.error(f"ICM calculation error: {e}")
            return VoiceToolResult(
                success=False,
                message="ICM calculation failed. Make sure to provide valid stack sizes.",
                data={"error": str(e)}
            )

    async def _handle_multi_opponent_query(self, hand: str, query: str) -> VoiceToolResult:
        """Handle multi-opponent equity queries."""
        num_opponents = self._extract_opponent_count(query)

        # Normalize hand
        if len(hand) >= 2:
            r1, r2 = hand[0].upper(), hand[1].upper()
            suited = 's' in hand.lower()

            if r1 == r2:
                hand_str = f"{r1}h{r2}d"
                hand_display = f"pocket {self._rank_name(r1)}s"
            else:
                if suited:
                    hand_str = f"{r1}h{r2}h"
                    hand_display = f"{self._rank_name(r1)}-{self._rank_name(r2)} suited"
                else:
                    hand_str = f"{r1}h{r2}d"
                    hand_display = f"{self._rank_name(r1)}-{self._rank_name(r2)} offsuit"

            # Calculate multi-way equity
            result = await self.poker_tool.execute(
                action="equity",
                hand=hand_str,
                board="",
                opponents=num_opponents,
                simulations=5000
            )

            if result.success:
                equity = result.data.get('equity', 50.0)

                # Classify hand strength for multi-way
                if equity > 50:
                    strength = "strong in multi-way"
                elif equity > 35:
                    strength = "playable multi-way"
                elif equity > 25:
                    strength = "marginal multi-way"
                else:
                    strength = "weak multi-way, consider folding"

                message = f"{hand_display} has {equity:.0f}% equity against {num_opponents} opponents. That's {strength}."

                return VoiceToolResult(
                    success=True,
                    message=message,
                    data={
                        "hand": hand_str,
                        "display": hand_display,
                        "equity": equity,
                        "opponents": num_opponents,
                        "strength": strength
                    }
                )

            return VoiceToolResult(
                success=False,
                message=f"Couldn't calculate multi-way odds for {hand_display}.",
                data={"hand": hand_str, "opponents": num_opponents}
            )

        return VoiceToolResult(
            success=False,
            message="I need a hand to calculate multi-way equity. Try 'pocket aces vs 3 opponents'.",
            data={}
        )

    async def _handle_offline_toggle(self, enabled: bool) -> VoiceToolResult:
        """Handle offline mode toggle via voice command."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8000/offline/toggle",
                    json={"enabled": enabled}
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        if enabled:
                            # Get status to show what's available
                            llm_status = result.get("llm", {})
                            voice_status = result.get("voice", {})

                            llm_ready = llm_status.get("available", False)
                            voice_stt = voice_status.get("stt", {}).get("available", False)
                            voice_tts = voice_status.get("tts", {}).get("available", False)

                            features = []
                            if llm_ready:
                                features.append("local LLM")
                            if voice_stt:
                                features.append("offline speech recognition")
                            if voice_tts:
                                features.append("offline text to speech")

                            if features:
                                feature_str = ", ".join(features)
                                message = f"Offline mode enabled. Available: {feature_str}. Poker equity calculations work offline. OCR still uses cloud for best accuracy."
                            else:
                                message = "Offline mode enabled but no local models found. Download Mistral 7B GGUF for LLM and Piper models for voice. Poker equity works offline."

                            return VoiceToolResult(
                                success=True,
                                message=message,
                                data={
                                    "action": "offline_mode",
                                    "enabled": True,
                                    "status": result
                                }
                            )
                        else:
                            return VoiceToolResult(
                                success=True,
                                message="Online mode restored. Using cloud services for LLM and voice processing.",
                                data={
                                    "action": "offline_mode",
                                    "enabled": False,
                                    "status": result
                                }
                            )
                    else:
                        return VoiceToolResult(
                            success=False,
                            message="Failed to toggle offline mode. Server returned an error.",
                            data={"error": f"HTTP {response.status}"}
                        )

        except aiohttp.ClientError as e:
            logger.error(f"Offline toggle connection error: {e}")
            return VoiceToolResult(
                success=False,
                message="Could not connect to server to toggle offline mode.",
                data={"error": str(e)}
            )
        except Exception as e:
            logger.error(f"Offline toggle error: {e}")
            return VoiceToolResult(
                success=False,
                message="Error toggling offline mode.",
                data={"error": str(e)}
            )

    async def _handle_vpip_report(self) -> VoiceToolResult:
        """Generate and export VPIP session report as PDF."""
        from backend.voice.tools.report_pdf import generate_session_pdf

        # Get VPIP stats from poker OCR
        vpip_stats = poker_ocr.get_vpip_stats()

        # Convert to format expected by report_pdf
        vpip_data = {
            "vpip_percentage": vpip_stats.get("vpip_percentage", 0),
            "hands_played": vpip_stats.get("total_hands", 0),
            "vpip_hands": vpip_stats.get("vpip_hands", 0),
        }

        # Add leak analysis based on VPIP
        vpip_pct = vpip_data["vpip_percentage"]
        if vpip_pct > 35:
            vpip_data["leak_analysis"] = (
                f"Your VPIP of {vpip_pct:.1f}% is too high for most games. "
                "You're playing too many hands and likely losing money on marginal holdings."
            )
            vpip_data["recommendation"] = (
                "Tighten your preflop range. Focus on playing premium hands "
                "and position-aware poker. Fold more from early position."
            )
        elif vpip_pct > 28:
            vpip_data["leak_analysis"] = (
                f"Your VPIP of {vpip_pct:.1f}% is slightly loose. "
                "This can be profitable in soft games but may leak EV in tougher lineups."
            )
            vpip_data["recommendation"] = (
                "Consider tightening from early and middle position. "
                "Save looser play for the button and blinds."
            )
        elif vpip_pct < 15:
            vpip_data["leak_analysis"] = (
                f"Your VPIP of {vpip_pct:.1f}% is very tight. "
                "You may be missing profitable spots and becoming predictable."
            )
            vpip_data["recommendation"] = (
                "Add more suited connectors and broadway hands in position. "
                "Attack weak players more frequently."
            )
        else:
            vpip_data["leak_analysis"] = (
                f"Your VPIP of {vpip_pct:.1f}% is in a solid range for most games. "
                "Continue playing position-aware poker."
            )
            vpip_data["recommendation"] = (
                "Maintain your current approach. Focus on post-flop decisions "
                "and exploiting opponent tendencies."
            )

        if vpip_data["hands_played"] == 0:
            return VoiceToolResult(
                success=False,
                message="No hands recorded this session. Play some hands first to generate a report!",
                data={"error": "no_data"}
            )

        try:
            pdf_path = generate_session_pdf(vpip_data)
            hands = vpip_data["hands_played"]
            vpip = vpip_data["vpip_percentage"]

            return VoiceToolResult(
                success=True,
                message=f"VPIP report generated! Your VPIP is {vpip:.1f}% over {hands} hands. PDF saved to {pdf_path}",
                data={
                    "action": "vpip_report",
                    "pdf_path": pdf_path,
                    "stats": vpip_data
                }
            )
        except Exception as e:
            logger.error(f"VPIP report generation error: {e}")
            return VoiceToolResult(
                success=False,
                message=f"Failed to generate VPIP report: {str(e)}",
                data={"error": str(e)}
            )

    async def _handle_site_mode(self, site: PokerSite) -> VoiceToolResult:
        """Handle poker site mode switching.

        Args:
            site: The PokerSite to switch to

        Returns:
            VoiceToolResult with status and instructions
        """
        try:
            # Switch the site mode in the router
            result_msg = poker_router.set_site(site)

            if site == PokerSite.IGNITION:
                return VoiceToolResult(
                    success=True,
                    message=f"Ignition mode activated. {result_msg}. OCR is now calibrated for Ignition Poker tables with anonymous players. Point your camera at the table for optimized detection.",
                    data={
                        "action": "site_mode",
                        "site": "ignition",
                        "hud_message": {
                            "type": "control",
                            "action": "site_mode",
                            "site": "ignition",
                            "calibration": {
                                "table_type": "ignition",
                                "anonymous_players": True,
                                "pot_position": "center_top",
                                "hole_cards_position": "bottom_center"
                            }
                        }
                    }
                )
            else:
                return VoiceToolResult(
                    success=True,
                    message=f"Generic mode activated. {result_msg}. OCR will use standard poker table detection.",
                    data={
                        "action": "site_mode",
                        "site": "generic",
                        "hud_message": {
                            "type": "control",
                            "action": "site_mode",
                            "site": "generic"
                        }
                    }
                )

        except Exception as e:
            logger.error(f"Site mode switch error: {e}")
            return VoiceToolResult(
                success=False,
                message=f"Failed to switch site mode: {str(e)}",
                data={"error": str(e)}
            )

    def _is_live_mode_command(self, query: str) -> bool:
        """Check if query is a live game mode command."""
        live_patterns = [
            r'live\s+(mode|game|poker)',
            r'start\s+live',
            r'end\s+live',
            r'exit\s+live',
            r'casino\s+(mode|game)',
            r'home\s+game',
            r'new\s+hand',
            r'next\s+hand',
            r'i\s+have\s+\w',
            r'holding\s+\w',
            r'my\s+cards?\s+(are|is)',
            r'pot\s+is\s+\d',
            r'my\s+stack\s+is',
            r'stack\s+is\s+\d',
            r'board\s+is\s+\w',
            r'flop\s+is\s+\w',
            r'turn\s+(is|card)\s+\w',
            r'river\s+(is|card)\s+\w',
            r'facing\s+(bet|raise|all)',
            r'i\s+won',
            r'won\s+pot',
            r'i\s+lost',
            r'lost\s+pot',
        ]

        # Also match if live mode is already active and query looks like a state update
        if live_mode.is_active:
            active_patterns = [
                r'(button|cutoff|hijack|big blind|small blind|utg|middle|late)',
                r'\d+\s+(players?|opponents?)',
                r'fold|check|call|raise|bet|all.?in',
            ]
            live_patterns.extend(active_patterns)

        return any(re.search(pattern, query) for pattern in live_patterns)

    async def _handle_live_mode(self, query: str) -> VoiceToolResult:
        """Handle live game mode commands for casino/home games.

        Supports voice-driven state management without OCR.
        """
        try:
            # Process the command through live mode manager
            result = live_mode.process_voice_command(query)

            status = result.get("status", "unknown")
            message = result.get("message", "Command processed")

            # Enhanced messages for specific actions
            if status == "active":
                return VoiceToolResult(
                    success=True,
                    message=f"Live poker mode activated. {result.get('state', {}).get('stakes', '1/2')} game ready. Say 'new hand' to start, then tell me your cards with 'I have ace king suited'.",
                    data={"action": "live_mode", "result": result}
                )

            elif status == "deactivated":
                duration = result.get("session_duration", 0)
                hands = result.get("hands_played", 0)
                profit = result.get("session_profit", 0)
                return VoiceToolResult(
                    success=True,
                    message=f"Live mode ended. Session: {duration} minutes, {hands} hands, ${profit:+.0f} profit.",
                    data={"action": "live_mode_end", "result": result}
                )

            elif status == "new_hand":
                position = result.get("position", "unknown")
                hand_num = result.get("hand_number", 0)
                return VoiceToolResult(
                    success=True,
                    message=f"Hand {hand_num}. You're on the {position}. Tell me your cards.",
                    data={"action": "new_hand", "result": result}
                )

            elif status == "success":
                # Get advice if we have enough state for analysis
                if live_mode.state.hole_cards and live_mode.state.pot > 0:
                    engine_state = live_mode.get_engine_state()
                    if engine_state:
                        # Get EV advice using the poker engine
                        ev_result = await self._get_live_advice(engine_state)
                        if ev_result:
                            message = f"{message}. {ev_result}"

                return VoiceToolResult(
                    success=True,
                    message=message,
                    data={"action": "live_update", "result": result}
                )

            elif status == "inactive":
                return VoiceToolResult(
                    success=True,
                    message="Live mode is not active. Say 'live mode' or 'casino game' to start tracking a live poker session.",
                    data={"action": "live_inactive"}
                )

            elif status == "error":
                return VoiceToolResult(
                    success=False,
                    message=message,
                    data={"action": "live_error", "result": result}
                )

            else:
                hint = result.get("hint", "")
                return VoiceToolResult(
                    success=True,
                    message=f"{message}. {hint}" if hint else message,
                    data={"action": "live_unknown", "result": result}
                )

        except Exception as e:
            logger.error(f"Live mode error: {e}")
            return VoiceToolResult(
                success=False,
                message="Error processing live mode command.",
                data={"error": str(e)}
            )

    async def _get_live_advice(self, engine_state: dict) -> Optional[str]:
        """Get poker advice for live game state.

        Args:
            engine_state: State dict from live mode

        Returns:
            Advice string or None
        """
        try:
            hole_cards = engine_state.get("hole_cards", [])
            community = engine_state.get("community_cards", [])
            pot = engine_state.get("pot_size", 0)
            stack = engine_state.get("player_stack", 0)
            position = engine_state.get("position", "")

            if not hole_cards or len(hole_cards) < 2:
                return None

            # Calculate equity
            hand_str = "".join(hole_cards)
            board_str = " ".join(community) if community else ""

            result = await self.poker_tool.execute(
                action="equity",
                hand=hand_str,
                board=board_str,
                opponents=live_mode.state.players_in_hand - 1 if live_mode.state.players_in_hand > 1 else 1,
                simulations=3000
            )

            if result.success:
                equity = result.data.get('equity', 50.0)

                # Calculate SPR
                spr = stack / pot if pot > 0 else 0

                # Generate advice based on equity and SPR
                if equity > 70:
                    advice = f"Strong hand with {equity:.0f}% equity. Value bet or raise for value."
                elif equity > 55:
                    advice = f"Good hand at {equity:.0f}% equity. Bet for value, consider pot control."
                elif equity > 45:
                    advice = f"Marginal at {equity:.0f}% equity. Check or small bet, avoid big pots."
                else:
                    advice = f"Weak at {equity:.0f}% equity. Check-fold or bluff if position is good."

                # Add SPR context
                if spr < 4:
                    advice += " Low SPR - commit or fold decision."
                elif spr > 13:
                    advice += " Deep stacked - play cautiously post-flop."

                return advice

            return None

        except Exception as e:
            logger.debug(f"Live advice error: {e}")
            return None
