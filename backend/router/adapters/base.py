"""
Base Engine Adapter Interface

Provides a unified interface for all solution engines,
enabling consistent routing and fallback behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import time


@dataclass
class EngineResult:
    """Result from an engine solving a problem."""
    success: bool
    data: Any
    method: str  # "local" | "claude" | "cached" | "error"
    tokens_used: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    cache_type: Optional[str] = None  # "exact" | "semantic" | None
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def cached(cls, data: Any, cache_type: str, latency_ms: float) -> 'EngineResult':
        """Create a cached result."""
        return cls(
            success=True,
            data=data,
            method="cached",
            tokens_used=0,
            latency_ms=latency_ms,
            cost=0.0,
            cache_type=cache_type
        )

    @classmethod
    def failed(cls, error: str, latency_ms: float) -> 'EngineResult':
        """Create a failed result."""
        return cls(
            success=False,
            data={"error": error},
            method="error",
            tokens_used=0,
            latency_ms=latency_ms,
            cost=0.0
        )


class IEngineAdapter(ABC):
    """
    Abstract base class for engine adapters.

    Each adapter wraps an existing engine and provides:
    - Unified solve interface
    - Confidence scoring for problem matching
    - Capability declaration
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this adapter."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> list[str]:
        """List of problem types/operations this engine can handle."""
        pass

    @property
    def is_local(self) -> bool:
        """Whether this engine runs locally (free) vs external API (paid)."""
        return True

    @property
    def priority(self) -> int:
        """Priority for routing (lower = try first). Default 50."""
        return 50

    @abstractmethod
    async def solve(self, problem: str, **kwargs) -> EngineResult:
        """
        Attempt to solve the problem.

        Args:
            problem: The problem text to solve
            **kwargs: Additional parameters (operation type, variables, etc.)

        Returns:
            EngineResult with success/failure and data
        """
        pass

    @abstractmethod
    def can_handle(self, problem: str, **kwargs) -> float:
        """
        Confidence score for handling this problem.

        Args:
            problem: The problem text
            **kwargs: Additional context

        Returns:
            Float 0.0-1.0 indicating confidence (0 = cannot handle, 1 = perfect match)
        """
        pass

    def _measure_time(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.time() - start_time) * 1000


class ClaudeAdapter(IEngineAdapter):
    """
    Base adapter for Claude-powered engines.
    Handles token extraction and cost calculation.
    """

    # Anthropic pricing (per 1M tokens)
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-haiku-3-5-20241022": {"input": 0.25, "output": 1.25},
    }
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    @property
    def is_local(self) -> bool:
        return False

    @property
    def priority(self) -> int:
        return 100  # Claude is expensive, try local first

    def extract_tokens(self, response: dict) -> tuple[int, int]:
        """Extract token counts from Anthropic response."""
        usage = response.get("usage", {})
        return usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost from token usage."""
        pricing = self.PRICING.get(model, self.PRICING[self.DEFAULT_MODEL])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
