"""
Live Poker Coach - Real-time poker coaching with DeepSeek V3.1.
Decision time: 4-6 seconds per hand.
Cost: ~$0.009/hand ($0.90/100 hands)
"""
import os
import asyncio
import logging
import re
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass

from ..api_clients import GeminiClient, DeepSeekClient, ClaudeClient
from ..poker import OpponentTracker, build_poker_prompt, PokerCache
from ..core import CostTracker
from .poker_session import PokerSession

logger = logging.getLogger(__name__)


@dataclass
class PokerRecommendation:
    """AI recommendation for a poker hand."""
    action: str         # FOLD, CALL, RAISE
    sizing: str         # "3/4 pot", "pot", etc.
    equity: float       # Estimated equity (0.0-1.0)
    reasoning: str      # Brief explanation
    villain_type: str   # Detected villain archetype
    latency_ms: float   # API response time
    cost: float         # API cost for this analysis
    cached: bool        # Whether result was from cache
    confidence: float = None  # AI confidence level (0.0-1.0)


class LivePokerCoach:
    """
    Real-time poker coach using DeepSeek V3.1.

    Flow:
    1. Camera captures table (1s)
    2. Gemini OCR extracts cards (1s)
    3. DeepSeek V3.1 analyzes and recommends (4s)
    4. Display on HUD

    Total: ~6 seconds per decision
    Cost: ~$0.009 per hand
    """

    def __init__(
        self,
        stakes: str = "$0.25/$0.50",
        bb_value: float = 0.50,
        persist_dir: Optional[str] = None
    ):
        """
        Initialize poker coach.

        Args:
            stakes: Stakes description
            bb_value: Big blind value in USD
            persist_dir: Directory for saving session data
        """
        self.stakes = stakes
        self.bb_value = bb_value

        # API clients
        self.gemini = GeminiClient()
        self.deepseek = DeepSeekClient()
        self.claude = ClaudeClient()  # Fallback

        # Components
        self.opponent_tracker = OpponentTracker()
        self.cache = PokerCache()
        self.cost_tracker = CostTracker()
        self.session = PokerSession(
            stakes=stakes,
            bb_value=bb_value,
            persist_dir=persist_dir
        )

        # State
        self.active = False
        self.position = "BTN"
        self.stack_bb = 100.0
        self._pending_analysis: Optional[asyncio.Task] = None

        logger.info(f"Poker coach initialized for {stakes}")

    async def start(self):
        """Start poker coaching session."""
        self.active = True
        self.cost_tracker.reset()
        logger.info("Poker coach started")

    async def stop(self) -> Dict:
        """
        Stop coaching and return session summary.

        Returns:
            Session summary dictionary
        """
        self.active = False

        if self._pending_analysis:
            self._pending_analysis.cancel()

        # Get session summary
        summary = self.session.get_summary()

        # Add cost tracker data
        cost_summary = self.cost_tracker.get_summary()
        summary["cost_breakdown"] = cost_summary.model_breakdown
        summary["cache_stats"] = self.cache.get_stats()

        # Save session
        self.session.save()
        self.opponent_tracker.save()

        logger.info(f"Session ended: {summary['profit_bb']:+.1f}bb, cost ${summary['total_api_cost']:.2f}")
        return summary

    def set_position(self, position: str):
        """Set hero's current position."""
        self.position = position.upper()

    def set_stack(self, stack_bb: float):
        """Set hero's stack size in big blinds."""
        self.stack_bb = stack_bb

    def set_villain(self, villain_id: str, name: str = ""):
        """Set current villain being played against."""
        self.opponent_tracker.set_villain(villain_id, name)

    def update_villain(
        self,
        action_type: str,
        voluntarily_invested: bool = False
    ):
        """Update villain stats based on observed action."""
        self.opponent_tracker.update(
            action_type=action_type,
            voluntarily_invested=voluntarily_invested
        )

    async def analyze_hand(
        self,
        camera_image: bytes,
        on_thinking: Optional[Callable[[], Awaitable[None]]] = None
    ) -> PokerRecommendation:
        """
        Full hand analysis from camera image.

        Args:
            camera_image: Raw image bytes from glasses camera
            on_thinking: Optional callback while AI is thinking

        Returns:
            PokerRecommendation with action, sizing, reasoning
        """
        import time
        start = time.perf_counter()

        # Notify thinking started
        if on_thinking:
            await on_thinking()

        # Step 1: OCR cards from camera (1s)
        cards = await self.gemini.extract_cards(camera_image)
        self.cost_tracker.add("gemini_flash", context="ocr")

        hero_cards = cards.hole_cards
        board = cards.board
        pot_size = cards.pot_size or 1.5
        bet_facing = cards.bet_facing or 0

        # Step 2: Get villain stats (local, instant)
        villain_stats = self.opponent_tracker.get_stats()

        # Step 3: Check cache
        cached_result = self.cache.get(
            hero_cards=hero_cards,
            board=board,
            villain_type=villain_stats["type"]
        )

        if cached_result:
            latency = (time.perf_counter() - start) * 1000
            return PokerRecommendation(
                action=cached_result["action"],
                sizing=cached_result.get("sizing", ""),
                equity=cached_result.get("equity", 0),
                reasoning=cached_result.get("reasoning", "Cached result"),
                villain_type=villain_stats["type"],
                latency_ms=latency,
                cost=0.002,  # Just OCR cost
                cached=True,
                confidence=cached_result.get("confidence")
            )

        # Step 4: Build prompt
        prompt = build_poker_prompt(
            hero_cards=hero_cards,
            board=board,
            pot_size=pot_size,
            bet_size=bet_facing,
            villain_stats=villain_stats,
            position=self.position,
            stack_size=self.stack_bb,
            stakes=self.stakes
        )

        # Step 5: DeepSeek analysis (4s)
        response = await self.deepseek.analyze_hand(prompt, thinking=False)
        self.cost_tracker.add("deepseek_v3.1", context="analysis")

        # Step 6: Parse response
        parsed = self._parse_response(response.content)

        # Step 7: Cache result
        self.cache.set(
            hero_cards=hero_cards,
            board=board,
            villain_type=villain_stats["type"],
            result=parsed
        )

        latency = (time.perf_counter() - start) * 1000

        recommendation = PokerRecommendation(
            action=parsed.get("action", "FOLD"),
            sizing=parsed.get("sizing", ""),
            equity=parsed.get("equity", 0),
            reasoning=parsed.get("reasoning", ""),
            villain_type=villain_stats["type"],
            latency_ms=latency,
            cost=0.009,  # OCR + DeepSeek
            cached=False,
            confidence=parsed.get("confidence")
        )

        # Record in session
        self.session.start_hand(hero_cards, self.position, self.stack_bb)
        self.session.update_board(board)
        self.session.update_pot(pot_size)
        self.session.set_villain(villain_stats["type"], villain_stats)
        self.session.record_ai_recommendation(
            action=recommendation.action,
            reasoning=recommendation.reasoning,
            equity=recommendation.equity,
            api_cost=recommendation.cost
        )

        logger.info(
            f"Analysis: {recommendation.action} ({recommendation.equity:.0%}) "
            f"in {recommendation.latency_ms:.0f}ms"
        )

        return recommendation

    async def analyze_text(
        self,
        hero_cards: str,
        board: str = "",
        pot_bb: float = 0,
        bet_bb: float = 0
    ) -> PokerRecommendation:
        """
        Analyze hand from text input (no camera).

        Args:
            hero_cards: "Ah Kc"
            board: "9s 7h 3d" or ""
            pot_bb: Pot size in bb
            bet_bb: Bet facing in bb

        Returns:
            PokerRecommendation
        """
        import time
        start = time.perf_counter()

        # Parse cards
        hero_list = hero_cards.split()
        board_list = board.split() if board else []

        # Get villain stats
        villain_stats = self.opponent_tracker.get_stats()

        # Check cache
        cached = self.cache.get(hero_list, board_list, villain_stats["type"])
        if cached:
            return PokerRecommendation(
                action=cached["action"],
                sizing=cached.get("sizing", ""),
                equity=cached.get("equity", 0),
                reasoning="Cached",
                villain_type=villain_stats["type"],
                latency_ms=(time.perf_counter() - start) * 1000,
                cost=0,
                cached=True,
                confidence=cached.get("confidence")
            )

        # Build prompt
        prompt = build_poker_prompt(
            hero_cards=hero_list,
            board=board_list,
            pot_size=pot_bb or 1.5,
            bet_size=bet_bb,
            villain_stats=villain_stats,
            position=self.position,
            stack_size=self.stack_bb,
            stakes=self.stakes
        )

        # DeepSeek analysis
        response = await self.deepseek.analyze_hand(prompt, thinking=False)
        self.cost_tracker.add("deepseek_v3.1", context="text_analysis")

        parsed = self._parse_response(response.content)

        # Cache
        self.cache.set(hero_list, board_list, villain_stats["type"], parsed)

        latency = (time.perf_counter() - start) * 1000

        return PokerRecommendation(
            action=parsed.get("action", "FOLD"),
            sizing=parsed.get("sizing", ""),
            equity=parsed.get("equity", 0),
            reasoning=parsed.get("reasoning", ""),
            villain_type=villain_stats["type"],
            latency_ms=latency,
            cost=0.007,
            cached=False,
            confidence=parsed.get("confidence")
        )

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse DeepSeek response into structured format."""
        result = {
            "action": "FOLD",
            "sizing": "",
            "equity": 0.0,
            "reasoning": "",
            "confidence": None
        }

        # Extract ACTION
        action_match = re.search(r'ACTION:\s*(FOLD|CALL|RAISE)', response, re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).upper()

        # Extract SIZING
        sizing_match = re.search(r'SIZING:\s*([^\n]+)', response, re.IGNORECASE)
        if sizing_match:
            result["sizing"] = sizing_match.group(1).strip()

        # Extract EQUITY
        equity_match = re.search(r'EQUITY:\s*(\d+)%?', response, re.IGNORECASE)
        if equity_match:
            result["equity"] = float(equity_match.group(1)) / 100

        # Extract REASONING
        reasoning_match = re.search(r'REASONING:\s*([^\n]+)', response, re.IGNORECASE)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        else:
            # Fallback: use last line
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            if lines:
                result["reasoning"] = lines[-1][:100]

        # Extract CONFIDENCE (1-10 scale, convert to 0.0-1.0)
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response, re.IGNORECASE)
        if conf_match:
            conf_value = float(conf_match.group(1))
            # Clamp to 1-10 range and convert to 0.0-1.0
            conf_value = max(1, min(10, conf_value))
            result["confidence"] = conf_value / 10.0

        return result

    def record_result(
        self,
        action_taken: str,
        result_bb: float,
        showdown: bool = False,
        notes: str = ""
    ):
        """
        Record the result of a hand.

        Args:
            action_taken: What hero actually did
            result_bb: Won/lost in big blinds
            showdown: Whether hand went to showdown
            notes: Optional notes
        """
        self.session.record_action(action_taken)
        self.session.end_hand(
            result_bb=result_bb,
            showdown=showdown,
            notes=notes
        )

        # Update poker profit in cost tracker
        self.cost_tracker.record_poker_result(result_bb, self.bb_value)

    async def post_session_review(self) -> str:
        """
        Get deep analysis of session using DeepSeek V3.2.

        Returns:
            Detailed session review
        """
        from ..poker.prompt_builder import build_review_prompt

        hands = self.session.get_hands_for_review(limit=20)

        if not hands:
            return "No hands to review"

        prompt = build_review_prompt(
            hands=hands,
            session_context=f"{self.stakes}, {self.session.stats.hands_played} hands"
        )

        # Use thinking mode for deep analysis (35s)
        response = await self.deepseek.analyze_hand(prompt, thinking=True)
        self.cost_tracker.add("deepseek_v3.2", context="session_review")

        return response.content

    def get_villain_exploits(self) -> str:
        """Get exploitative adjustments for current villain."""
        exploits = self.opponent_tracker.get_exploits()
        return "\n".join(f"• {e}" for e in exploits)

    def get_session_status(self) -> str:
        """Get current session status for HUD."""
        summary = self.session.get_summary()
        cost_summary = self.cost_tracker.get_summary()

        return (
            f"Session: {summary['hands_played']} hands | "
            f"{summary['profit_bb']:+.1f}bb (${summary['profit_usd']:+.2f})\n"
            f"Cost: ${cost_summary.total_cost:.2f} | "
            f"Cache: {self.cache.get_stats()['hit_rate']:.0%}"
        )


# Test
async def test_poker_coach():
    """Test poker coach."""
    print("=== Poker Coach Test ===\n")

    coach = LivePokerCoach(stakes="$0.25/$0.50")
    await coach.start()

    # Set villain
    coach.set_villain("seat_3", "FishyPlayer")
    for _ in range(10):
        coach.update_villain("call", voluntarily_invested=True)

    # Test text analysis (no camera)
    if coach.deepseek.is_available:
        print("Analyzing hand...")
        rec = await coach.analyze_text(
            hero_cards="Ah Kc",
            board="9s 7h 3d",
            pot_bb=12,
            bet_bb=8
        )
        print(f"Recommendation: {rec.action} {rec.sizing}")
        print(f"Equity: {rec.equity:.0%}")
        print(f"Reasoning: {rec.reasoning}")
        print(f"Latency: {rec.latency_ms:.0f}ms")
        print(f"Cost: ${rec.cost:.3f}")
        print(f"Cached: {rec.cached}")
    else:
        print("DeepSeek not available (no API key)")

    print()
    print("Villain exploits:")
    print(coach.get_villain_exploits())

    print()
    summary = await coach.stop()
    print(coach.session.format_summary())


if __name__ == "__main__":
    asyncio.run(test_poker_coach())
