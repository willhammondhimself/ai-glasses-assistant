"""
Poker HUD Display - Live poker coaching display for Halo glasses.
Renders DeepSeek recommendations to the 640x400 OLED display.

Now powered by OLED-optimized renderer with:
- Power-aware color adjustment
- Burn-in prevention
- Smooth animations
- Enhanced confidence indicators
"""
import time
from typing import Optional, List, Dict, Any

from .colors import Colors
from .renderer import HUDRenderer
from ..halo.oled_renderer import OLEDRenderer, OLEDColors, create_oled_renderer
from ..halo.animations import AnimationEngine, create_animation_engine
from ..core.power_manager import PowerManager


class PokerHUD:
    """
    Live poker HUD with DeepSeek recommendations.

    Layout (640x400):
    ┌─────────────────────────────────┐
    │ Ah Kc          [THINKING...] 3s │ ← Hero cards + timer
    │ Board: 9s 7h 3d                 │
    │                                 │
    │ POT: 12bb | BET: 8bb            │
    │ ODDS: 35% | EQUITY: 68%         │
    │                                 │
    │ ▶ RAISE 3/4 POT (9bb)           │ ← Big action
    │                                 │
    │ Villain: Calling Station        │
    │ They call too much. Value bet.  │ ← Reasoning
    │                                 │
    │ Cost: $0.009 | Session: $0.47   │ ← API cost
    └─────────────────────────────────┘
    """

    # Card display colors (RGB tuples for OLED)
    CARD_COLORS = {
        'h': (255, 68, 68),    # Hearts - red
        'd': (68, 136, 255),   # Diamonds - blue
        'c': (68, 204, 68),    # Clubs - green
        's': (255, 255, 255),  # Spades - white
    }

    # Action colors (RGB tuples)
    ACTION_COLORS = {
        'FOLD': (136, 136, 136),    # Gray
        'CALL': (68, 204, 68),      # Green
        'RAISE': (255, 136, 0),     # Orange
        'ALL-IN': (255, 68, 68),    # Red
    }

    # Legacy hex colors for backwards compatibility
    ACTION_COLORS_HEX = {
        'FOLD': '#888888',
        'CALL': '#44CC44',
        'RAISE': '#FF8800',
        'ALL-IN': '#FF4444',
    }

    def __init__(
        self,
        width: int = 640,
        height: int = 400,
        power_manager: PowerManager = None,
        config: dict = None
    ):
        """
        Initialize poker HUD.

        Args:
            width: Display width in pixels
            height: Display height in pixels
            power_manager: Optional power manager for battery-aware rendering
            config: Optional configuration dict
        """
        self.width = width
        self.height = height
        self.config = config or {}
        self.power_manager = power_manager

        # Use new OLED renderer if available
        self.oled_renderer = create_oled_renderer(config, power_manager)
        self.animations = create_animation_engine(config, self.oled_renderer)

        # Keep legacy renderer for backwards compatibility
        self.renderer = HUDRenderer(width, height)

        # Track rendering mode
        self.use_oled = True  # Prefer OLED renderer

    def render_thinking(
        self,
        hero_cards: List[str],
        board: List[str] = None,
        pot_bb: float = 0,
        bet_bb: float = 0,
        elapsed_s: float = 0
    ) -> str:
        """
        Render thinking screen while DeepSeek processes.

        Args:
            hero_cards: ["Ah", "Kc"]
            board: Community cards or None
            pot_bb: Pot size
            bet_bb: Bet facing
            elapsed_s: Seconds elapsed

        Returns:
            Lua code for display
        """
        # Format cards with colors
        hero_str = self._format_cards(hero_cards)
        board_str = self._format_cards(board) if board else "Preflop"

        # Thinking animation dots
        dots = "." * (int(elapsed_s) % 4)
        thinking_str = f"THINKING{dots}"

        lua = f'''
-- Poker HUD: Thinking
frame.display.text("{hero_str}", 20, 30, {{color = "{Colors.CYAN}", spacing = 2}})
frame.display.text("{thinking_str}", 450, 30, {{color = "{Colors.YELLOW}"}})
frame.display.text("{elapsed_s:.1f}s", 580, 30, {{color = "{Colors.WHITE}"}})

frame.display.text("Board: {board_str}", 20, 80, {{color = "{Colors.WHITE}"}})

frame.display.text("POT: {pot_bb:.0f}bb | BET: {bet_bb:.0f}bb", 20, 140, {{color = "{Colors.WHITE}"}})

-- Pulsing thinking indicator
frame.display.text("Analyzing...", 220, 220, {{color = "{Colors.CYAN}", spacing = 1}})

frame.display.show()
'''
        return lua

    def render_recommendation(
        self,
        hero_cards: List[str],
        board: List[str],
        action: str,
        sizing: str = "",
        equity: float = 0,
        pot_odds: float = 0,
        pot_bb: float = 0,
        bet_bb: float = 0,
        villain_type: str = "",
        reasoning: str = "",
        cost: float = 0,
        session_cost: float = 0,
        latency_ms: float = 0,
        cached: bool = False,
        confidence: float = None
    ) -> str:
        """
        Render full recommendation after analysis.

        Args:
            hero_cards: Hero's hole cards
            board: Community cards
            action: FOLD, CALL, RAISE
            sizing: Bet sizing if raising
            equity: Estimated equity (0-1)
            pot_odds: Required pot odds (0-1)
            pot_bb: Pot size in bb
            bet_bb: Bet facing in bb
            villain_type: Villain archetype
            reasoning: AI's reasoning
            cost: Cost for this analysis
            session_cost: Total session cost
            latency_ms: Analysis latency
            cached: Whether result was cached
            confidence: AI confidence level (0-1), optional

        Returns:
            Lua code for display
        """
        # Format cards
        hero_str = self._format_cards(hero_cards)
        board_str = self._format_cards(board) if board else "Preflop"

        # Action with color
        action_color = self.ACTION_COLORS.get(action.upper(), Colors.WHITE)
        action_str = action.upper()
        if sizing:
            action_str += f" {sizing}"

        # Equity vs pot odds comparison
        equity_color = Colors.GREEN if equity > pot_odds else Colors.RED

        # Truncate reasoning if too long
        reasoning = reasoning[:60] + "..." if len(reasoning) > 60 else reasoning

        # Cache indicator
        cache_str = " (cached)" if cached else ""

        # Confidence bar generation
        confidence_lua = ""
        if confidence is not None:
            bar_length = 10
            filled = int(confidence * bar_length)
            empty = bar_length - filled
            bar = "\u2588" * filled + "\u2591" * empty  # █ and ░

            # Color-code confidence
            if confidence >= 0.8:
                conf_color = Colors.GREEN
            elif confidence >= 0.6:
                conf_color = Colors.YELLOW
            else:
                conf_color = Colors.RED

            confidence_lua = f'''
-- Confidence indicator
frame.display.text("Conf: {bar} {confidence * 100:.0f}%", 350, 220, {{color = "{conf_color}"}})
'''

        lua = f'''
-- Poker HUD: Recommendation
frame.display.text("{hero_str}", 20, 25, {{color = "{Colors.CYAN}", spacing = 2}})
frame.display.text("{latency_ms:.0f}ms{cache_str}", 480, 25, {{color = "{Colors.GRAY}"}})

frame.display.text("Board: {board_str}", 20, 70, {{color = "{Colors.WHITE}"}})

-- Pot info line
frame.display.text("POT: {pot_bb:.0f}bb", 20, 120, {{color = "{Colors.WHITE}"}})
frame.display.text("|", 150, 120, {{color = "{Colors.GRAY}"}})
frame.display.text("BET: {bet_bb:.0f}bb", 180, 120, {{color = "{Colors.WHITE}"}})

-- Odds vs Equity
frame.display.text("ODDS: {pot_odds:.0%}", 20, 160, {{color = "{Colors.WHITE}"}})
frame.display.text("|", 150, 160, {{color = "{Colors.GRAY}"}})
frame.display.text("EQUITY: {equity:.0%}", 180, 160, {{color = "{equity_color}"}})

-- Big action display
frame.display.text(">> {action_str}", 20, 220, {{color = "{action_color}", spacing = 2}})
{confidence_lua}
-- Villain info
frame.display.text("Villain: {villain_type.title()}", 20, 280, {{color = "{Colors.YELLOW}"}})
frame.display.text("{reasoning}", 20, 320, {{color = "{Colors.WHITE}"}})

-- Cost tracking
frame.display.text("Cost: ${cost:.3f} | Session: ${session_cost:.2f}", 20, 380, {{color = "{Colors.GRAY}"}})

frame.display.show()
'''
        return lua

    def render_villain_stats(
        self,
        villain_type: str,
        vpip: float,
        aggression: float,
        hands_observed: int,
        exploits: List[str]
    ) -> str:
        """
        Render villain stats overlay.

        Args:
            villain_type: Archetype
            vpip: VPIP percentage
            aggression: Aggression factor
            hands_observed: Sample size
            exploits: List of exploitation strategies

        Returns:
            Lua code for display
        """
        # Color code villain type
        type_colors = {
            "calling_station": Colors.GREEN,   # Easy to exploit
            "tight_passive": Colors.CYAN,      # Fold equity
            "maniac": Colors.RED,              # Danger
            "tag": Colors.WHITE,               # Standard
            "lag": Colors.YELLOW,              # Tricky
        }
        type_color = type_colors.get(villain_type, Colors.WHITE)

        # Format exploits (max 4)
        exploit_lines = exploits[:4]

        lua = f'''
-- Villain Stats HUD
frame.display.text("VILLAIN PROFILE", 180, 30, {{color = "{Colors.CYAN}", spacing = 1}})

frame.display.text("{villain_type.upper()}", 220, 80, {{color = "{type_color}", spacing = 2}})
frame.display.text("({hands_observed} hands)", 230, 120, {{color = "{Colors.GRAY}"}})

-- Stats
frame.display.text("VPIP: {vpip:.0%}", 150, 170, {{color = "{Colors.WHITE}"}})
frame.display.text("AGG: {aggression:.0%}", 350, 170, {{color = "{Colors.WHITE}"}})

-- Exploits header
frame.display.text("EXPLOITS:", 20, 230, {{color = "{Colors.YELLOW}"}})
'''

        # Add exploit lines
        y = 270
        for exploit in exploit_lines:
            exploit = exploit[:50]  # Truncate
            lua += f'frame.display.text("* {exploit}", 40, {y}, {{color = "{Colors.WHITE}"}})\n'
            y += 35

        lua += '\nframe.display.show()\n'
        return lua

    def render_session_summary(
        self,
        hands_played: int,
        profit_bb: float,
        profit_usd: float,
        win_rate: float,
        api_cost: float,
        net_profit: float,
        biggest_win: float,
        biggest_loss: float
    ) -> str:
        """
        Render session summary screen.

        Returns:
            Lua code for display
        """
        # Color profit
        profit_color = Colors.GREEN if profit_bb >= 0 else Colors.RED
        net_color = Colors.GREEN if net_profit >= 0 else Colors.RED

        lua = f'''
-- Session Summary
frame.display.text("SESSION COMPLETE", 180, 30, {{color = "{Colors.CYAN}", spacing = 1}})

frame.display.text("Hands: {hands_played}", 20, 90, {{color = "{Colors.WHITE}"}})

frame.display.text("Profit: {profit_bb:+.1f}bb (${profit_usd:+.2f})", 20, 140, {{color = "{profit_color}"}})
frame.display.text("Win Rate: {win_rate:+.1f}bb/100", 20, 180, {{color = "{profit_color}"}})

frame.display.text("Best: {biggest_win:+.1f}bb | Worst: {biggest_loss:+.1f}bb", 20, 240, {{color = "{Colors.WHITE}"}})

frame.display.text("AI Cost: ${api_cost:.2f}", 20, 300, {{color = "{Colors.GRAY}"}})
frame.display.text("Net Profit: ${net_profit:+.2f}", 20, 340, {{color = "{net_color}", spacing = 1}})

frame.display.show()
'''
        return lua

    def render_preflop_chart(
        self,
        hero_cards: List[str],
        position: str,
        action: str,
        range_strength: str = "premium"
    ) -> str:
        """
        Render preflop decision with range context.

        Args:
            hero_cards: Hero's cards
            position: BTN, CO, etc.
            action: Recommended action
            range_strength: premium, strong, marginal, weak

        Returns:
            Lua code for display
        """
        hero_str = self._format_cards(hero_cards)

        # Color code range strength
        strength_colors = {
            "premium": Colors.GREEN,
            "strong": Colors.CYAN,
            "marginal": Colors.YELLOW,
            "weak": Colors.RED,
        }
        strength_color = strength_colors.get(range_strength, Colors.WHITE)
        action_color = self.ACTION_COLORS.get(action.upper(), Colors.WHITE)

        lua = f'''
-- Preflop Decision
frame.display.text("{hero_str}", 200, 80, {{color = "{Colors.CYAN}", spacing = 3}})

frame.display.text("Position: {position}", 230, 150, {{color = "{Colors.WHITE}"}})
frame.display.text("Range: {range_strength.upper()}", 220, 200, {{color = "{strength_color}"}})

frame.display.text(">> {action.upper()}", 220, 280, {{color = "{action_color}", spacing = 2}})

frame.display.show()
'''
        return lua

    def _format_cards(self, cards: List[str]) -> str:
        """
        Format cards for display.
        Converts ['Ah', 'Kc'] to 'Ah Kc' with proper spacing.
        """
        if not cards:
            return ""
        return " ".join(cards)

    def _get_card_color(self, card: str) -> str:
        """Get color for a card based on suit."""
        if len(card) < 2:
            return Colors.WHITE
        suit = card[-1].lower()
        return self.CARD_COLORS.get(suit, Colors.WHITE)

    def _get_card_color_rgb(self, card: str) -> tuple:
        """Get RGB color for a card based on suit."""
        if len(card) < 2:
            return OLEDColors.TEXT_PRIMARY
        suit = card[-1].lower()
        return self.CARD_COLORS.get(suit, OLEDColors.TEXT_PRIMARY)

    # ============================================================
    # OLED-Optimized Render Methods
    # ============================================================

    def render_thinking_oled(
        self,
        hero_cards: List[str],
        board: List[str] = None,
        pot_bb: float = 0,
        bet_bb: float = 0,
        elapsed_s: float = 0
    ) -> str:
        """
        Render thinking screen using OLED renderer with animations.

        Args:
            hero_cards: ["Ah", "Kc"]
            board: Community cards or None
            pot_bb: Pot size
            bet_bb: Bet facing
            elapsed_s: Seconds elapsed

        Returns:
            Lua code for display
        """
        r = self.oled_renderer
        r.clear()

        # Hero cards with suit colors
        x = 20
        for card in hero_cards:
            color = self._get_card_color_rgb(card)
            r.text(card, x, 30, color, 36, spacing=2)
            x += 60

        # Thinking indicator with pulsing dots
        dots = "." * (int(elapsed_s * 2) % 4)
        r.text(f"THINKING{dots}", 450, 30, OLEDColors.WARNING, 24)
        r.text(f"{elapsed_s:.1f}s", 580, 30, OLEDColors.TEXT_DIM, 20)

        # Board
        board_str = self._format_cards(board) if board else "Preflop"
        r.text(f"Board: {board_str}", 20, 80, OLEDColors.TEXT_PRIMARY, 24)

        # Pot info
        r.text(f"POT: {pot_bb:.0f}bb  |  BET: {bet_bb:.0f}bb", 20, 140, OLEDColors.TEXT_SECONDARY, 22)

        # Centered analyzing text with subtle pulse
        r.text("Analyzing...", self.width // 2, 220, OLEDColors.PRIMARY, 28, "center")

        # Progress indicator
        progress = min(1.0, elapsed_s / 6.0)  # Assume ~6s typical analysis
        r.progress_bar(150, 280, 340, 8, progress, color=OLEDColors.PRIMARY)

        r.show()
        return r.get_lua()

    def render_recommendation_oled(
        self,
        hero_cards: List[str],
        board: List[str],
        action: str,
        sizing: str = "",
        equity: float = 0,
        pot_odds: float = 0,
        pot_bb: float = 0,
        bet_bb: float = 0,
        villain_type: str = "",
        reasoning: str = "",
        cost: float = 0,
        session_cost: float = 0,
        latency_ms: float = 0,
        cached: bool = False,
        confidence: float = None
    ) -> str:
        """
        Render recommendation using OLED renderer with enhanced visuals.

        Returns:
            Lua code for display
        """
        r = self.oled_renderer
        r.clear()

        # Hero cards with suit colors
        x = 20
        for card in hero_cards:
            color = self._get_card_color_rgb(card)
            r.text(card, x, 25, color, 32, spacing=2)
            x += 55

        # Latency/cache indicator
        cache_str = " (cached)" if cached else ""
        r.text(f"{latency_ms:.0f}ms{cache_str}", 480, 25, OLEDColors.TEXT_DIM, 18)

        # Board with suit colors
        if board:
            x = 20
            r.text("Board:", x, 70, OLEDColors.TEXT_SECONDARY, 20)
            x = 100
            for card in board:
                color = self._get_card_color_rgb(card)
                r.text(card, x, 70, color, 24)
                x += 45

        # Pot and bet info
        r.text(f"POT: {pot_bb:.0f}bb", 20, 115, OLEDColors.TEXT_PRIMARY, 22)
        r.text("|", 150, 115, OLEDColors.TEXT_DIM, 22)
        r.text(f"BET: {bet_bb:.0f}bb", 175, 115, OLEDColors.TEXT_PRIMARY, 22)

        # Equity vs odds with color coding
        equity_color = OLEDColors.SUCCESS if equity > pot_odds else OLEDColors.ERROR
        r.text(f"ODDS: {pot_odds:.0%}", 20, 155, OLEDColors.TEXT_SECONDARY, 20)
        r.text("|", 150, 155, OLEDColors.TEXT_DIM, 20)
        r.text(f"EQUITY: {equity:.0%}", 175, 155, equity_color, 20)

        # Big action display
        action_upper = action.upper()
        action_color = self.ACTION_COLORS.get(action_upper, OLEDColors.TEXT_PRIMARY)
        action_str = f">> {action_upper}"
        if sizing:
            action_str += f" {sizing}"

        r.text(action_str, 20, 210, action_color, 32, spacing=2)

        # Confidence bar (enhanced)
        if confidence is not None:
            r.confidence_bar(350, 215, confidence, show_label=True, width=120)

        # Villain info
        r.text(f"Villain: {villain_type.title()}", 20, 270, OLEDColors.WARNING, 22)

        # Reasoning (truncated)
        if len(reasoning) > 55:
            reasoning = reasoning[:52] + "..."
        r.text(reasoning, 20, 310, OLEDColors.TEXT_SECONDARY, 18)

        # Cost tracking
        r.text(f"Cost: ${cost:.3f} | Session: ${session_cost:.2f}", 20, 375, OLEDColors.TEXT_DIM, 16)

        r.show()
        return r.get_lua()

    def render_session_summary_oled(
        self,
        hands_played: int,
        profit_bb: float,
        profit_usd: float,
        win_rate: float,
        api_cost: float,
        net_profit: float,
        biggest_win: float,
        biggest_loss: float
    ) -> str:
        """
        Render session summary using OLED renderer.

        Returns:
            Lua code for display
        """
        r = self.oled_renderer
        r.clear()

        # Title
        r.text("SESSION COMPLETE", self.width // 2, 30, OLEDColors.PRIMARY, 28, "center")

        # Divider
        r.divider(60)

        # Stats
        r.text(f"Hands: {hands_played}", 20, 90, OLEDColors.TEXT_PRIMARY, 22)

        # Profit with color
        profit_color = OLEDColors.SUCCESS if profit_bb >= 0 else OLEDColors.ERROR
        r.text(f"Profit: {profit_bb:+.1f}bb (${profit_usd:+.2f})", 20, 130, profit_color, 24)
        r.text(f"Win Rate: {win_rate:+.1f}bb/100", 20, 170, profit_color, 22)

        # Best/worst
        r.text(f"Best: {biggest_win:+.1f}bb  |  Worst: {biggest_loss:+.1f}bb", 20, 220, OLEDColors.TEXT_SECONDARY, 20)

        # API cost
        r.text(f"AI Cost: ${api_cost:.2f}", 20, 280, OLEDColors.TEXT_DIM, 20)

        # Net profit
        net_color = OLEDColors.SUCCESS if net_profit >= 0 else OLEDColors.ERROR
        r.text(f"Net Profit: ${net_profit:+.2f}", 20, 320, net_color, 26)

        r.show()
        return r.get_lua()

    def get_animated_recommendation(
        self,
        hero_cards: List[str],
        board: List[str],
        action: str,
        **kwargs
    ) -> List[tuple]:
        """
        Get animated recommendation with action pulse.

        Returns:
            List of (lua_code, delay_ms) tuples for animation
        """
        # First render the static recommendation
        static = self.render_recommendation_oled(
            hero_cards=hero_cards,
            board=board,
            action=action,
            **kwargs
        )

        # For high-impact actions, add pulse animation
        if action.upper() in ("RAISE", "ALL-IN"):
            action_color = self.ACTION_COLORS.get(action.upper(), OLEDColors.WARNING)
            pulse_frames = self.animations.pulse_element(
                x=150, y=210,
                text=f">> {action.upper()}",
                color=action_color,
                cycles=2,
                duration_ms=600
            )
            return [(static, 0)] + pulse_frames

        return [(static, 0)]


# Test
def test_poker_hud():
    """Test poker HUD rendering."""
    print("=== Poker HUD Test ===\n")

    hud = PokerHUD()

    # Test thinking screen
    print("Thinking screen:")
    print("-" * 40)
    lua = hud.render_thinking(
        hero_cards=["Ah", "Kc"],
        board=["9s", "7h", "3d"],
        pot_bb=12,
        bet_bb=8,
        elapsed_s=2.5
    )
    print(lua[:500] + "...")
    print()

    # Test recommendation screen
    print("Recommendation screen:")
    print("-" * 40)
    lua = hud.render_recommendation(
        hero_cards=["Ah", "Kc"],
        board=["9s", "7h", "3d"],
        action="RAISE",
        sizing="3/4 pot",
        equity=0.68,
        pot_odds=0.40,
        pot_bb=12,
        bet_bb=8,
        villain_type="calling_station",
        reasoning="Value bet thin - villain calls too much",
        cost=0.009,
        session_cost=0.47,
        latency_ms=4200
    )
    print(lua[:500] + "...")


if __name__ == "__main__":
    test_poker_hud()
