"""
Router Decorators

Decorators for gradual router integration with FastAPI endpoints.
Provides opt-in routing with fallback to original implementation.
"""

import logging
from functools import wraps
from typing import Optional, Callable, Any
import asyncio
import inspect

from .intelligent_router import get_router

logger = logging.getLogger(__name__)


def routed(
    problem_type: Optional[str] = None,
    problem_field: str = "expression",
    cache_ttl: int = 86400,
    fallback_on_error: bool = True,
    enabled: bool = True
):
    """
    Decorator for routing API requests through the intelligent router.

    Usage:
        @app.post("/math/solve")
        @routed(problem_type="math", problem_field="expression")
        async def math_solve(request: MathRequest):
            # Original implementation (used as fallback)
            ...

    Args:
        problem_type: Type hint for problem classification (auto-detected if None)
        problem_field: Field name in request containing the problem text
        cache_ttl: Cache TTL in seconds for successful results
        fallback_on_error: If True, fall back to original function on router error
        enabled: If False, skip router entirely (for easy disable)

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Skip routing if disabled
            if not enabled:
                return await _call_original(func, *args, **kwargs)

            # Extract problem from request
            problem = _extract_problem(args, kwargs, problem_field)
            if not problem:
                logger.debug(f"No problem found in field '{problem_field}', using original")
                return await _call_original(func, *args, **kwargs)

            try:
                # Try router first
                router = get_router()
                result = await router.route(
                    problem=problem,
                    problem_type=problem_type
                )

                if result.success:
                    logger.debug(f"Router success via {result.method}")
                    return result.data

                # Router didn't succeed, fall back
                if fallback_on_error:
                    logger.debug("Router failed, using fallback")
                    return await _call_original(func, *args, **kwargs)
                else:
                    return {"error": "Router failed", "details": result.data}

            except Exception as e:
                logger.warning(f"Router error: {e}")
                if fallback_on_error:
                    return await _call_original(func, *args, **kwargs)
                raise

        return wrapper
    return decorator


def _extract_problem(args: tuple, kwargs: dict, field: str) -> Optional[str]:
    """
    Extract problem text from function arguments.

    Handles:
    - Direct kwargs (problem_field in kwargs)
    - Pydantic request objects (request.field)
    - First positional arg with the field
    """
    # Try kwargs directly
    if field in kwargs:
        value = kwargs[field]
        return str(value) if value else None

    # Try request object in kwargs
    for key in ["request", "req", "body"]:
        if key in kwargs:
            obj = kwargs[key]
            if hasattr(obj, field):
                value = getattr(obj, field)
                return str(value) if value else None

    # Try positional args (usually Pydantic request is first after self)
    for arg in args:
        if hasattr(arg, field):
            value = getattr(arg, field)
            return str(value) if value else None

    return None


async def _call_original(func: Callable, *args, **kwargs) -> Any:
    """Call the original function, handling both sync and async."""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)


class RoutedEndpoint:
    """
    Class-based alternative to the decorator for more control.

    Usage:
        router_endpoint = RoutedEndpoint(
            problem_type="math",
            problem_field="expression"
        )

        @app.post("/math/solve")
        async def math_solve(request: MathRequest):
            # Check router first
            result = await router_endpoint.try_route(request)
            if result:
                return result

            # Original implementation
            ...
    """

    def __init__(
        self,
        problem_type: Optional[str] = None,
        problem_field: str = "expression",
        cache_ttl: int = 86400
    ):
        self.problem_type = problem_type
        self.problem_field = problem_field
        self.cache_ttl = cache_ttl

    async def try_route(self, request: Any) -> Optional[dict]:
        """
        Try to route the request.

        Args:
            request: The request object

        Returns:
            Result dict if successful, None if should use fallback
        """
        problem = getattr(request, self.problem_field, None)
        if not problem:
            return None

        try:
            router = get_router()
            result = await router.route(
                problem=str(problem),
                problem_type=self.problem_type
            )

            if result.success:
                return result.data

        except Exception as e:
            logger.warning(f"RoutedEndpoint error: {e}")

        return None


def with_routing(problem_type: str):
    """
    Simple decorator that adds routing info to a function.
    Does not actually route - just marks the endpoint for documentation.

    Usage:
        @app.post("/math/solve")
        @with_routing("math")
        async def math_solve(request: MathRequest):
            ...
    """
    def decorator(func: Callable) -> Callable:
        func._routing_info = {
            "problem_type": problem_type,
            "routed": True
        }
        return func
    return decorator
