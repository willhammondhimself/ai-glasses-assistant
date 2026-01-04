"""
5-Tier Intelligence Router.
Routes queries to optimal model based on task type, latency needs, and cost.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Union

from .cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class RoutingTier(Enum):
    """Intelligence routing tiers."""
    LOCAL = 1      # Mental math, poker equity (instant, free)
    GEMINI = 2     # OCR/vision tasks (1s, $0.002)
    PERPLEXITY = 3 # Facts/search (2s, $0.001-0.005)
    DEEPSEEK_FAST = 4  # Reasoning V3.1 (4s, $0.007)
    DEEPSEEK_DEEP = 5  # Deep analysis V3.2 (35s, $0.014)


@dataclass
class RouteDecision:
    """Routing decision with reasoning."""
    tier: RoutingTier
    model: str
    estimated_latency_ms: float
    estimated_cost: float
    reasoning: str


@dataclass
class RouterResponse:
    """Response from routed query."""
    content: str
    tier_used: RoutingTier
    latency_ms: float
    cost: float
    fallback_used: bool = False
    error: Optional[str] = None


class IntelligenceRouter:
    """
    Routes queries to optimal AI model.

    Routing Logic:
    - Tier 1 (LOCAL): Mental math, poker equity, syntax checking
    - Tier 2 (GEMINI): OCR, card detection, equation extraction
    - Tier 3 (PERPLEXITY): Facts, current events, research
    - Tier 4 (DEEPSEEK_FAST): Strategy, analysis, debugging
    - Tier 5 (DEEPSEEK_DEEP): Complex reasoning, post-session review
    """

    def __init__(self):
        # API clients (lazy loaded)
        self._gemini = None
        self._perplexity = None
        self._deepseek = None

        # Local engines (lazy loaded)
        self._math_solver = None
        self._poker_engine = None
        self._syntax_checker = None

        # Cost tracking
        self.cost_tracker = CostTracker()
        self.tier_usage: Dict[RoutingTier, int] = {tier: 0 for tier in RoutingTier}

    def _lazy_load_clients(self):
        """Lazy load API clients when first needed."""
        if self._gemini is None:
            try:
                from ..api_clients import GeminiClient
                self._gemini = GeminiClient()
            except Exception as e:
                logger.warning(f"Could not load GeminiClient: {e}")

        if self._perplexity is None:
            try:
                from ..api_clients import PerplexityClient
                self._perplexity = PerplexityClient()
            except Exception as e:
                logger.warning(f"Could not load PerplexityClient: {e}")

        if self._deepseek is None:
            try:
                from ..api_clients import DeepSeekClient
                self._deepseek = DeepSeekClient()
            except Exception as e:
                logger.warning(f"Could not load DeepSeekClient: {e}")

    def _lazy_load_engines(self):
        """Lazy load local engines when first needed."""
        if self._math_solver is None:
            try:
                from ..local_engines import LocalMathSolver
                self._math_solver = LocalMathSolver()
            except Exception as e:
                logger.debug(f"Could not load LocalMathSolver: {e}")

        if self._poker_engine is None:
            try:
                from ..engines import PokerEquityEngine
                self._poker_engine = PokerEquityEngine()
            except Exception as e:
                logger.debug(f"Could not load PokerEquityEngine: {e}")

        if self._syntax_checker is None:
            try:
                from ..local_engines import LocalSyntaxChecker
                self._syntax_checker = LocalSyntaxChecker()
            except Exception as e:
                logger.debug(f"Could not load LocalSyntaxChecker: {e}")

    def decide_tier(
        self,
        query: str,
        context: Optional[Dict] = None,
        mode: str = "general"
    ) -> RouteDecision:
        """
        Determine optimal routing tier.

        Args:
            query: The user query or task description
            context: Additional context (has_image, urgency, etc.)
            mode: Task mode (poker_strategy, homework, debug, etc.)

        Returns:
            RouteDecision with tier and reasoning
        """
        context = context or {}

        # Tier 1: Local engines
        if mode == "mental_math" or mode == "arithmetic":
            return RouteDecision(
                tier=RoutingTier.LOCAL,
                model="mental_math_engine",
                estimated_latency_ms=5,
                estimated_cost=0,
                reasoning="Mental math solved locally"
            )

        if mode == "poker_equity":
            return RouteDecision(
                tier=RoutingTier.LOCAL,
                model="poker_equity_engine",
                estimated_latency_ms=50,
                estimated_cost=0,
                reasoning="Equity calculated via Monte Carlo"
            )

        if mode == "syntax_check":
            return RouteDecision(
                tier=RoutingTier.LOCAL,
                model="syntax_checker",
                estimated_latency_ms=10,
                estimated_cost=0,
                reasoning="Syntax check done locally"
            )

        # Tier 2: Vision/OCR
        if context.get("has_image") or mode in ["ocr", "card_detection", "math_ocr", "code_ocr"]:
            return RouteDecision(
                tier=RoutingTier.GEMINI,
                model="gemini-2.0-flash",
                estimated_latency_ms=1000,
                estimated_cost=0.002,
                reasoning="Image processing requires Gemini"
            )

        # Tier 3: Facts/Search
        if self._is_factual_query(query):
            return RouteDecision(
                tier=RoutingTier.PERPLEXITY,
                model="sonar",
                estimated_latency_ms=2000,
                estimated_cost=0.001,
                reasoning="Factual query routed to Perplexity"
            )

        # Tier 4: Reasoning (default for most tasks)
        if mode in ["poker_strategy", "debug", "homework", "analysis", "explain"]:
            return RouteDecision(
                tier=RoutingTier.DEEPSEEK_FAST,
                model="deepseek-chat",
                estimated_latency_ms=4000,
                estimated_cost=0.007,
                reasoning="Reasoning task routed to DeepSeek V3.1"
            )

        # Tier 5: Deep analysis
        if mode in ["post_session", "deep_review", "complex_analysis", "proof"]:
            return RouteDecision(
                tier=RoutingTier.DEEPSEEK_DEEP,
                model="deepseek-reasoner",
                estimated_latency_ms=35000,
                estimated_cost=0.014,
                reasoning="Complex analysis routed to DeepSeek V3.2"
            )

        # Default to fast DeepSeek
        return RouteDecision(
            tier=RoutingTier.DEEPSEEK_FAST,
            model="deepseek-chat",
            estimated_latency_ms=4000,
            estimated_cost=0.007,
            reasoning="Default reasoning route"
        )

    def _is_factual_query(self, query: str) -> bool:
        """Check if query is factual (best for Perplexity)."""
        factual_indicators = [
            "what is", "who is", "when did", "where is",
            "how many", "latest", "current", "news",
            "price of", "weather", "score", "definition",
            "what does", "explain what"
        ]
        query_lower = query.lower()
        return any(ind in query_lower for ind in factual_indicators)

    async def route_query(
        self,
        query: str,
        context: Optional[Dict] = None,
        mode: str = "general",
        image_data: bytes = None
    ) -> RouterResponse:
        """
        Route query to optimal model.

        Args:
            query: The user query
            context: Additional context (has_image, urgency, etc.)
            mode: Task mode (poker_strategy, homework, debug, etc.)
            image_data: Optional image bytes for vision tasks

        Returns:
            RouterResponse with result
        """
        import time

        # Add image flag to context
        context = context or {}
        if image_data:
            context["has_image"] = True

        decision = self.decide_tier(query, context, mode)
        self.tier_usage[decision.tier] += 1

        try:
            start = time.perf_counter()
            response = await self._execute_tier(decision, query, context, image_data)
            latency = (time.perf_counter() - start) * 1000

            # Update response with actual latency
            response.latency_ms = latency

            # Track cost
            if decision.tier != RoutingTier.LOCAL:
                self.cost_tracker.add(
                    decision.model.replace("-", "_").replace(".", "_"),
                    decision.estimated_cost,
                    context=mode
                )
            else:
                self.cost_tracker.add_local(context=mode)

            return response

        except Exception as e:
            logger.error(f"Tier {decision.tier.name} failed: {e}")
            # Attempt fallback
            return await self._fallback(decision, query, context, str(e))

    async def _execute_tier(
        self,
        decision: RouteDecision,
        query: str,
        context: Dict,
        image_data: bytes = None
    ) -> RouterResponse:
        """Execute query on decided tier."""

        if decision.tier == RoutingTier.LOCAL:
            return await self._execute_local(decision, query, context)

        # Lazy load clients
        self._lazy_load_clients()

        if decision.tier == RoutingTier.GEMINI:
            if not self._gemini or not self._gemini.is_available:
                raise Exception("Gemini not available")

            # Determine extraction type from mode
            mode = context.get("mode", "general")
            if mode == "math_ocr":
                result = await self._gemini.extract_math(image_data)
                content = f"Equation: {result.equation}\nType: {result.problem_type}"
            elif mode == "code_ocr":
                result = await self._gemini.extract_code(image_data)
                content = f"Language: {result.language}\nCode:\n{result.code}"
            elif mode == "card_detection":
                result = await self._gemini.extract_cards(image_data)
                content = f"Cards: {result.hole_cards}, Board: {result.board}"
            else:
                content = await self._gemini.analyze_image(image_data, query)

            return RouterResponse(
                content=content,
                tier_used=decision.tier,
                latency_ms=0,  # Will be updated
                cost=decision.estimated_cost
            )

        elif decision.tier == RoutingTier.PERPLEXITY:
            if not self._perplexity or not self._perplexity.is_available:
                raise Exception("Perplexity not available")

            response = await self._perplexity.query(query)
            return RouterResponse(
                content=response.content,
                tier_used=decision.tier,
                latency_ms=response.latency_ms,
                cost=response.cost
            )

        elif decision.tier == RoutingTier.DEEPSEEK_FAST:
            if not self._deepseek or not self._deepseek.is_available:
                raise Exception("DeepSeek not available")

            content = await self._deepseek.live_analysis(query)
            return RouterResponse(
                content=content,
                tier_used=decision.tier,
                latency_ms=0,
                cost=decision.estimated_cost
            )

        elif decision.tier == RoutingTier.DEEPSEEK_DEEP:
            if not self._deepseek or not self._deepseek.is_available:
                raise Exception("DeepSeek not available")

            response = await self._deepseek.analyze_hand(query, thinking=True)
            return RouterResponse(
                content=response.content,
                tier_used=decision.tier,
                latency_ms=response.latency_ms,
                cost=decision.estimated_cost
            )

        raise Exception(f"Unknown tier: {decision.tier}")

    async def _execute_local(
        self,
        decision: RouteDecision,
        query: str,
        context: Dict
    ) -> RouterResponse:
        """Execute local computation."""
        self._lazy_load_engines()

        if "math" in decision.model and self._math_solver:
            result = await self._math_solver.solve(query)
            if result:
                content = f"Answer: {result.answer}\nSteps: {', '.join(result.steps)}"
            else:
                content = "Could not solve locally"

            return RouterResponse(
                content=content,
                tier_used=RoutingTier.LOCAL,
                latency_ms=5,
                cost=0
            )

        elif "poker" in decision.model and self._poker_engine:
            hero = context.get("hero_cards", [])
            board = context.get("board", [])
            if hero:
                result = self._poker_engine.calculate_equity(hero, board)
                content = f"Equity: {result.equity:.1%} (Win: {result.win_probability:.1%})"
            else:
                content = "No cards provided"

            return RouterResponse(
                content=content,
                tier_used=RoutingTier.LOCAL,
                latency_ms=50,
                cost=0
            )

        elif "syntax" in decision.model and self._syntax_checker:
            language = context.get("language", "python")
            if language == "python":
                result = self._syntax_checker.check_python(query)
            elif language == "javascript":
                result = self._syntax_checker.check_javascript(query)
            else:
                result = None

            if result:
                if result.valid:
                    content = "Syntax OK"
                else:
                    errors = [f"Line {e.line}: {e.message}" for e in result.errors]
                    content = "Errors:\n" + "\n".join(errors)
            else:
                content = "Could not check syntax"

            return RouterResponse(
                content=content,
                tier_used=RoutingTier.LOCAL,
                latency_ms=10,
                cost=0
            )

        return RouterResponse(
            content="Local engine not available",
            tier_used=RoutingTier.LOCAL,
            latency_ms=0,
            cost=0,
            error="Engine not loaded"
        )

    async def _fallback(
        self,
        original_decision: RouteDecision,
        query: str,
        context: Dict,
        error: str
    ) -> RouterResponse:
        """Handle tier failures with fallback."""
        logger.info(f"Falling back from {original_decision.tier.name}")

        # Determine next tier
        if original_decision.tier == RoutingTier.GEMINI:
            # Can't fallback from vision without image processing
            return RouterResponse(
                content="Vision processing failed and no fallback available",
                tier_used=original_decision.tier,
                latency_ms=0,
                cost=0,
                fallback_used=True,
                error=error
            )

        elif original_decision.tier == RoutingTier.PERPLEXITY:
            # Fallback to DeepSeek for factual queries
            fallback = RouteDecision(
                tier=RoutingTier.DEEPSEEK_FAST,
                model="deepseek-chat",
                estimated_latency_ms=4000,
                estimated_cost=0.007,
                reasoning="Fallback from Perplexity"
            )

        elif original_decision.tier == RoutingTier.DEEPSEEK_FAST:
            # Try DeepSeek deep mode
            fallback = RouteDecision(
                tier=RoutingTier.DEEPSEEK_DEEP,
                model="deepseek-reasoner",
                estimated_latency_ms=35000,
                estimated_cost=0.014,
                reasoning="Fallback from DeepSeek Fast"
            )

        else:
            # No more fallbacks
            return RouterResponse(
                content=f"All tiers failed: {error}",
                tier_used=original_decision.tier,
                latency_ms=0,
                cost=0,
                fallback_used=True,
                error=error
            )

        try:
            response = await self._execute_tier(fallback, query, context)
            response.fallback_used = True
            return response
        except Exception as e:
            return RouterResponse(
                content=f"Fallback also failed: {e}",
                tier_used=fallback.tier,
                latency_ms=0,
                cost=0,
                fallback_used=True,
                error=str(e)
            )

    def get_session_stats(self) -> Dict:
        """Get routing statistics for session."""
        cost_summary = self.cost_tracker.get_summary()
        return {
            "total_cost": cost_summary["total"],
            "tier_usage": {t.name: c for t, c in self.tier_usage.items()},
            "total_queries": sum(self.tier_usage.values()),
            "avg_cost_per_query": (
                cost_summary["total"] / sum(self.tier_usage.values())
                if sum(self.tier_usage.values()) > 0 else 0
            ),
            "cost_breakdown": cost_summary["breakdown"],
            "warnings": cost_summary["warnings"]
        }

    def reset_stats(self):
        """Reset session statistics."""
        self.tier_usage = {tier: 0 for tier in RoutingTier}
        self.cost_tracker.reset()
