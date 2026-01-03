"""
Intelligent Router

Core routing orchestrator that coordinates:
- Problem classification
- Cache lookups (with semantic similarity)
- Engine selection and fallback
- Cost tracking and budget enforcement
"""

import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .problem_classifier import ProblemClassifier, Classification
from .cache_manager import SemanticCacheManager, get_semantic_cache
from .cost_tracker import CostTracker, BudgetZone, get_cost_tracker
from .adapters.base import IEngineAdapter, EngineResult

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Record of a routing decision for analysis."""
    problem_type: str
    complexity: str
    engine_tried: List[str]
    engine_used: str
    cache_hit: bool
    cache_type: Optional[str]
    latency_ms: float
    cost: float
    success: bool
    budget_zone: str


class IntelligentRouter:
    """
    Intelligent problem routing with cost optimization.

    Routing Logic:
    1. Check cache first (always)
    2. Classify problem type and complexity
    3. Get engine preference order based on:
       - Problem complexity
       - Current budget zone
       - Engine capabilities
    4. Try engines in order with fallback
    5. Cache successful results
    6. Track metrics
    """

    # Default engine preferences by problem type
    # Order: prefer local/free engines first, Claude as fallback
    ENGINE_PREFERENCES = {
        "math": ["sympy", "numpy", "claude"],
        "cs": ["local_analysis", "claude"],
        "chemistry": ["rdkit", "claude"],
        "biology": ["claude"],  # Usually needs Claude
        "statistics": ["scipy", "numpy", "claude"],
        "poker": ["poker_calc", "claude"],
        "quant": ["numpy", "scipy", "claude"],
        "vision": ["claude"],  # Always requires Claude
    }

    # Complexity thresholds for local vs Claude
    LOCAL_COMPLEXITY_MAX = {
        "math": "moderate",
        "cs": "simple",
        "chemistry": "simple",
        "statistics": "moderate",
        "poker": "moderate",
        "quant": "simple",
    }

    def __init__(
        self,
        cache: Optional[SemanticCacheManager] = None,
        cost_tracker: Optional[CostTracker] = None,
        classifier: Optional[ProblemClassifier] = None,
        adapters: Optional[Dict[str, IEngineAdapter]] = None
    ):
        """
        Initialize the router.

        Args:
            cache: Semantic cache manager
            cost_tracker: Cost tracking and budget enforcement
            classifier: Problem classifier
            adapters: Dict mapping engine names to adapters
        """
        self.cache = cache or get_semantic_cache()
        self.cost_tracker = cost_tracker or get_cost_tracker()
        self.classifier = classifier or ProblemClassifier()
        self.adapters: Dict[str, IEngineAdapter] = adapters or {}

        # Routing history for analysis
        self._routing_history: List[RoutingDecision] = []
        self._max_history = 1000

    def register_adapter(self, name: str, adapter: IEngineAdapter):
        """
        Register an engine adapter.

        Args:
            name: Unique name for the adapter
            adapter: The adapter instance
        """
        self.adapters[name] = adapter
        logger.info(f"Registered adapter: {name} with capabilities: {adapter.capabilities}")

    async def route(
        self,
        problem: str,
        problem_type: Optional[str] = None,
        **kwargs
    ) -> EngineResult:
        """
        Route a problem to the best available engine.

        Args:
            problem: The problem to solve
            problem_type: Optional type hint (auto-detected if not provided)
            **kwargs: Additional parameters for engines

        Returns:
            EngineResult with solution or error
        """
        start_time = time.time()
        engines_tried = []

        # Step 1: Classify the problem
        classification = self.classifier.classify(problem)
        if problem_type:
            classification.problem_type = problem_type

        # Step 2: Check cache first (always)
        cache_result = self.cache.get(problem, classification.problem_type)
        if cache_result.hit:
            latency = (time.time() - start_time) * 1000
            self._record_routing(
                classification, ["cache"], "cache",
                cache_hit=True, cache_type=cache_result.cache_type,
                latency_ms=latency, cost=0.0, success=True
            )
            return EngineResult.cached(
                data=cache_result.data,
                cache_type=cache_result.cache_type,
                latency_ms=latency
            )

        # Step 3: Check budget zone
        budget_zone = self.cost_tracker.get_current_zone()
        if self.cost_tracker.should_block_paid_calls():
            # Only try local engines in critical zone
            logger.warning("Budget critical - blocking paid API calls")

        # Step 4: Get engine order based on classification and budget
        engine_order = self._get_engine_order(classification, budget_zone)

        # Step 5: Try engines in order
        last_error = None
        for engine_name in engine_order:
            adapter = self.adapters.get(engine_name)
            if not adapter:
                continue

            # Check if engine can handle this problem
            confidence = adapter.can_handle(problem, **kwargs)
            if confidence < 0.3:
                continue

            # Skip expensive engines in critical zone
            if budget_zone == BudgetZone.CRITICAL and not adapter.is_local:
                continue

            engines_tried.append(engine_name)

            try:
                result = await adapter.solve(problem, **kwargs)

                if result.success:
                    # Cache successful result
                    self.cache.set(
                        problem,
                        classification.problem_type,
                        result.data,
                        metadata={
                            "engine": engine_name,
                            "complexity": classification.complexity,
                            "confidence": result.confidence
                        }
                    )

                    # Record cost
                    self.cost_tracker.record_call(
                        engine=engine_name,
                        model=result.method if not adapter.is_local else None,
                        input_tokens=result.tokens_used,
                        output_tokens=0,  # Would need to track separately
                        latency_ms=result.latency_ms,
                        success=True,
                        cached=False,
                        problem_type=classification.problem_type,
                        complexity=classification.complexity
                    )

                    # Record routing decision
                    total_latency = (time.time() - start_time) * 1000
                    self._record_routing(
                        classification, engines_tried, engine_name,
                        cache_hit=False, cache_type=None,
                        latency_ms=total_latency, cost=result.cost, success=True
                    )

                    return result

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Engine {engine_name} failed: {e}")
                continue

        # Step 6: All engines failed
        total_latency = (time.time() - start_time) * 1000
        self._record_routing(
            classification, engines_tried, "none",
            cache_hit=False, cache_type=None,
            latency_ms=total_latency, cost=0.0, success=False
        )

        return EngineResult.failed(
            error=last_error or "No engine could handle the problem",
            latency_ms=total_latency
        )

    def _get_engine_order(
        self,
        classification: Classification,
        budget_zone: BudgetZone
    ) -> List[str]:
        """
        Determine engine order based on classification and budget.

        Args:
            classification: Problem classification
            budget_zone: Current budget zone

        Returns:
            Ordered list of engine names to try
        """
        base_order = self.ENGINE_PREFERENCES.get(
            classification.problem_type,
            ["claude"]  # Default to Claude for unknown types
        )

        # For trivial/simple problems, strongly prefer local
        complexity_order = ["trivial", "simple", "moderate", "complex", "expert"]
        problem_complexity_idx = complexity_order.index(classification.complexity) \
            if classification.complexity in complexity_order else 2

        max_local = self.LOCAL_COMPLEXITY_MAX.get(classification.problem_type, "simple")
        max_local_idx = complexity_order.index(max_local) if max_local in complexity_order else 1

        # If problem is simple enough for local engines
        if problem_complexity_idx <= max_local_idx:
            # Local engines first, Claude as fallback
            return [e for e in base_order if e != "claude"] + ["claude"]

        # For expert problems, Claude might need to go first
        if classification.complexity == "expert":
            return ["claude"] + [e for e in base_order if e != "claude"]

        # Budget-based adjustments
        if budget_zone in (BudgetZone.RED, BudgetZone.CRITICAL):
            # Force local engines first when budget is tight
            return [e for e in base_order if e != "claude"] + ["claude"]

        return base_order

    def _record_routing(
        self,
        classification: Classification,
        engines_tried: List[str],
        engine_used: str,
        cache_hit: bool,
        cache_type: Optional[str],
        latency_ms: float,
        cost: float,
        success: bool
    ):
        """Record a routing decision for analysis."""
        decision = RoutingDecision(
            problem_type=classification.problem_type,
            complexity=classification.complexity,
            engine_tried=engines_tried,
            engine_used=engine_used,
            cache_hit=cache_hit,
            cache_type=cache_type,
            latency_ms=latency_ms,
            cost=cost,
            success=success,
            budget_zone=self.cost_tracker.get_current_zone().value
        )

        self._routing_history.append(decision)

        # Trim history if needed
        if len(self._routing_history) > self._max_history:
            self._routing_history = self._routing_history[-self._max_history:]

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get routing statistics.

        Returns:
            Dict with routing metrics
        """
        if not self._routing_history:
            return {"total_routes": 0}

        total = len(self._routing_history)
        cache_hits = sum(1 for r in self._routing_history if r.cache_hit)
        successes = sum(1 for r in self._routing_history if r.success)
        total_cost = sum(r.cost for r in self._routing_history)
        avg_latency = sum(r.latency_ms for r in self._routing_history) / total

        # Engine usage
        engine_usage = {}
        for r in self._routing_history:
            engine_usage[r.engine_used] = engine_usage.get(r.engine_used, 0) + 1

        # Problem type distribution
        type_dist = {}
        for r in self._routing_history:
            type_dist[r.problem_type] = type_dist.get(r.problem_type, 0) + 1

        return {
            "total_routes": total,
            "cache_hit_rate": cache_hits / total,
            "success_rate": successes / total,
            "total_cost": round(total_cost, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "engine_usage": engine_usage,
            "problem_type_distribution": type_dist,
            "current_budget_zone": self.cost_tracker.get_current_zone().value
        }


# Global router instance
_router: Optional[IntelligentRouter] = None


def get_router() -> IntelligentRouter:
    """Get the global router instance."""
    global _router
    if _router is None:
        _router = IntelligentRouter()
    return _router


def init_router(
    cache: Optional[SemanticCacheManager] = None,
    cost_tracker: Optional[CostTracker] = None,
    classifier: Optional[ProblemClassifier] = None,
    adapters: Optional[Dict[str, IEngineAdapter]] = None
) -> IntelligentRouter:
    """
    Initialize the global router with custom components.

    Args:
        cache: Custom cache manager
        cost_tracker: Custom cost tracker
        classifier: Custom classifier
        adapters: Dict of engine adapters

    Returns:
        The initialized router
    """
    global _router
    _router = IntelligentRouter(
        cache=cache,
        cost_tracker=cost_tracker,
        classifier=classifier,
        adapters=adapters
    )
    return _router
