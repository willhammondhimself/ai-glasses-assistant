"""
Intelligent Problem Routing System

Smart routing layer between API endpoints and solution engines,
optimizing for cost, speed, and accuracy.
"""

from .intelligent_router import IntelligentRouter, get_router, init_router
from .problem_classifier import ProblemClassifier, Classification
from .cache_manager import CacheManager, SemanticCacheManager, get_semantic_cache
from .cost_tracker import CostTracker, CostMetrics, BudgetZone, get_cost_tracker
from .decorators import routed, RoutedEndpoint

# Core engine adapters (always available)
from .adapters import IEngineAdapter, EngineResult, ClaudeAdapter


def __getattr__(name):
    """Lazy load engine-specific adapters."""
    if name in ('MathEngineAdapter', 'SymPyOnlyAdapter', 'ClaudeMathAdapter'):
        from . import adapters
        return getattr(adapters, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core router
    'IntelligentRouter',
    'get_router',
    'init_router',
    # Classification
    'ProblemClassifier',
    'Classification',
    # Caching
    'CacheManager',
    'SemanticCacheManager',
    'get_semantic_cache',
    # Cost tracking
    'CostTracker',
    'CostMetrics',
    'BudgetZone',
    'get_cost_tracker',
    # Decorators
    'routed',
    'RoutedEndpoint',
    # Adapters
    'IEngineAdapter',
    'EngineResult',
    'ClaudeAdapter',
    'MathEngineAdapter',
    'SymPyOnlyAdapter',
    'ClaudeMathAdapter',
]
