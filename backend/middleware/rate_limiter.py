"""Rate limiting middleware for API protection.

Implements a sliding window rate limiter with per-IP tracking and
configurable limits for different endpoint categories.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from functools import wraps

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit rule."""
    requests: int  # Number of requests allowed
    window_seconds: int  # Time window in seconds
    burst_multiplier: float = 1.5  # Allow burst up to this multiplier

    @property
    def burst_limit(self) -> int:
        """Maximum burst limit."""
        return int(self.requests * self.burst_multiplier)


@dataclass
class RateLimitBucket:
    """Sliding window rate limit bucket."""
    requests: list = field(default_factory=list)  # Timestamps of requests
    blocked_until: Optional[float] = None  # Timestamp when block expires

    def clean_old_requests(self, window_seconds: int) -> None:
        """Remove requests older than the window."""
        cutoff = time.time() - window_seconds
        self.requests = [ts for ts in self.requests if ts > cutoff]

    def add_request(self) -> None:
        """Add a request timestamp."""
        self.requests.append(time.time())

    def is_blocked(self) -> bool:
        """Check if currently blocked."""
        if self.blocked_until is None:
            return False
        if time.time() > self.blocked_until:
            self.blocked_until = None
            return False
        return True

    def block_for(self, seconds: int) -> None:
        """Block for a number of seconds."""
        self.blocked_until = time.time() + seconds


class RateLimiter:
    """In-memory rate limiter with sliding window algorithm."""

    # Default rate limits by endpoint category
    DEFAULT_LIMITS = {
        "default": RateLimitConfig(requests=100, window_seconds=60),  # 100 req/min
        "expensive": RateLimitConfig(requests=20, window_seconds=60),  # 20 req/min for LLM calls
        "voice": RateLimitConfig(requests=10, window_seconds=60),  # 10 req/min for voice tokens
        "search": RateLimitConfig(requests=30, window_seconds=60),  # 30 req/min for searches
        "health": RateLimitConfig(requests=200, window_seconds=60),  # 200 req/min for health checks
        "websocket": RateLimitConfig(requests=50, window_seconds=60),  # 50 connections/min
    }

    # Endpoint patterns to category mapping
    ENDPOINT_PATTERNS = {
        "/health": "health",
        "/voice/status": "health",
        "/voice/agent-status": "health",
        "/voice/token": "voice",
        "/ws/": "websocket",
        "/solve": "expensive",
        "/math/": "expensive",
        "/vision/": "expensive",
        "/cs/": "expensive",
        "/poker/": "expensive",
        "/chemistry/": "expensive",
        "/biology/": "expensive",
        "/statistics/": "expensive",
        "/quant/": "expensive",
        "/search": "search",
    }

    def __init__(self, enabled: bool = True):
        """Initialize rate limiter.

        Args:
            enabled: Whether rate limiting is enabled
        """
        self.enabled = enabled
        self._buckets: Dict[str, Dict[str, RateLimitBucket]] = defaultdict(
            lambda: defaultdict(RateLimitBucket)
        )
        self._limits = self.DEFAULT_LIMITS.copy()
        self._blocked_ips: Dict[str, float] = {}  # IP -> blocked_until timestamp
        self._global_stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "unique_clients": set(),
        }

    def set_limit(self, category: str, config: RateLimitConfig) -> None:
        """Set or update a rate limit for a category.

        Args:
            category: The category name
            config: Rate limit configuration
        """
        self._limits[category] = config
        logger.info(f"Rate limit set for {category}: {config.requests} req/{config.window_seconds}s")

    def get_category(self, path: str) -> str:
        """Determine the rate limit category for a path.

        Args:
            path: The request path

        Returns:
            Category name
        """
        for pattern, category in self.ENDPOINT_PATTERNS.items():
            if path.startswith(pattern):
                return category
        return "default"

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request.

        Args:
            request: FastAPI request

        Returns:
            Client identifier (IP address)
        """
        # Check for X-Forwarded-For header (behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check for X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to client host
        if request.client:
            return request.client.host

        return "unknown"

    def check_rate_limit(self, client_id: str, category: str) -> tuple[bool, dict]:
        """Check if a request should be allowed.

        Args:
            client_id: Client identifier
            category: Rate limit category

        Returns:
            Tuple of (allowed: bool, info: dict)
        """
        if not self.enabled:
            return True, {"rate_limiting": "disabled"}

        # Update global stats
        self._global_stats["total_requests"] += 1
        self._global_stats["unique_clients"].add(client_id)

        # Get limit config
        config = self._limits.get(category, self._limits["default"])

        # Get or create bucket
        bucket = self._buckets[category][client_id]

        # Check if currently blocked
        if bucket.is_blocked():
            self._global_stats["blocked_requests"] += 1
            return False, {
                "error": "rate_limited",
                "category": category,
                "retry_after": int(bucket.blocked_until - time.time()) + 1,
            }

        # Clean old requests
        bucket.clean_old_requests(config.window_seconds)

        # Check rate limit
        current_count = len(bucket.requests)
        if current_count >= config.burst_limit:
            # Exceeded burst limit - block temporarily
            block_time = config.window_seconds
            bucket.block_for(block_time)
            self._global_stats["blocked_requests"] += 1
            logger.warning(
                f"Rate limit exceeded for {client_id} on {category}: "
                f"{current_count} requests in {config.window_seconds}s"
            )
            return False, {
                "error": "rate_limited",
                "category": category,
                "retry_after": block_time,
                "limit": config.requests,
                "window": config.window_seconds,
            }

        # Allow request
        bucket.add_request()
        remaining = config.requests - len(bucket.requests)
        return True, {
            "limit": config.requests,
            "remaining": max(0, remaining),
            "reset": int(time.time() + config.window_seconds),
            "category": category,
        }

    def get_stats(self) -> dict:
        """Get rate limiter statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "enabled": self.enabled,
            "total_requests": self._global_stats["total_requests"],
            "blocked_requests": self._global_stats["blocked_requests"],
            "unique_clients": len(self._global_stats["unique_clients"]),
            "block_rate": (
                self._global_stats["blocked_requests"] / max(1, self._global_stats["total_requests"])
            ),
            "limits": {
                category: {
                    "requests": config.requests,
                    "window_seconds": config.window_seconds,
                    "burst_limit": config.burst_limit,
                }
                for category, config in self._limits.items()
            },
        }


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, limiter: RateLimiter = None):
        super().__init__(app)
        self.limiter = limiter or rate_limiter

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and apply rate limiting.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response
        """
        # Skip rate limiting for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Get client ID and category
        client_id = self.limiter._get_client_id(request)
        category = self.limiter.get_category(request.url.path)

        # Check rate limit
        allowed, info = self.limiter.check_rate_limit(client_id, category)

        if not allowed:
            # Return 429 Too Many Requests
            return Response(
                content=f'{{"error": "Too many requests", "retry_after": {info.get("retry_after", 60)}}}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Content-Type": "application/json",
                    "Retry-After": str(info.get("retry_after", 60)),
                    "X-RateLimit-Category": category,
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(info.get("limit", 100))
        response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(info.get("reset", 0))
        response.headers["X-RateLimit-Category"] = category

        return response


def rate_limit(category: str = "default"):
    """Decorator for rate limiting specific endpoints.

    Args:
        category: Rate limit category

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, request: Request = None, **kwargs) -> Any:
            if request is None:
                # Try to find request in args
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request:
                client_id = rate_limiter._get_client_id(request)
                allowed, info = rate_limiter.check_rate_limit(client_id, category)

                if not allowed:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "error": "Too many requests",
                            "retry_after": info.get("retry_after", 60),
                            "category": category,
                        },
                        headers={"Retry-After": str(info.get("retry_after", 60))},
                    )

            return await func(*args, **kwargs)

        return wrapper
    return decorator
