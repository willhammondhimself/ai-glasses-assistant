"""
Poker HUD Display - Live poker coaching display for Halo glasses.
Renders DeepSeek recommendations to the 640x400 OLED display.
"""
import time
from typing import Optional, List, Dict, Any
from .colors import Colors
from .renderer import HUDRenderer


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

    # Card display colors
    CARD_COLORS = {
        'h': '#FF4444',  # Hearts - red
        'd': '#4488FF',  # Diamonds - blue
        'c': '#44CC44',  # Clubs - green
        's': '#FFFFFF',  # Spades - white
    }

    # Action colors
    ACTION_COLORS = {
        'FOLD': '#888888',     # Gray
        'CALL': '#44CC44',     # Green
        'RAISE': '#FF8800',    # Orange
        'ALL-IN': '#FF4444',   # Red
    }

    def __init__(self, width: int = 640, height: int = 400):
        """
        Initialize poker HUD.

        Args:
            width: Display width in pixels
            height: Display height in pixels
        """
        self.width = width
        self.height = height
        self.renderer = HUDRenderer(width, height)

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
        cached: bool = False
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
