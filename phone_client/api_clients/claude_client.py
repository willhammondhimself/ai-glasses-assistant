"""
Claude Client for complex reasoning fallback.
Use when DeepSeek fails or for particularly complex spots.
Latency: ~16s (Sonnet non-thinking)
Cost: $0.028/request
"""
import os
import asyncio
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic not installed. Run: pip install anthropic")


@dataclass
class ClaudeResponse:
    """Response from Claude API."""
    content: str
    model: str
    latency_ms: float
    tokens_used: int


class ClaudeClient:
    """
    Claude client for complex reasoning.

    Use as fallback when:
    - DeepSeek is unavailable
    - Particularly complex decision needed
    - Need highest quality analysis

    Model: claude-sonnet-4-20250514
    Latency: ~16s
    Cost: ~$0.028/request
    """

    MODEL = "claude-sonnet-4-20250514"
    COST_PER_REQUEST = 0.028

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        self._initialized = False

        if not ANTHROPIC_AVAILABLE:
            logger.error("anthropic package not available")
            return

        if not self.api_key:
            logger.warning("No Anthropic API key provided. Set ANTHROPIC_API_KEY env var.")
            return

        try:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self._initialized = True
            logger.info("Claude client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Claude: {e}")

    @property
    def is_available(self) -> bool:
        """Check if client is ready to use."""
        return self._initialized and self.client is not None

    async def analyze(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3
    ) -> ClaudeResponse:
        """
        Analyze with Claude.

        Args:
            prompt: The analysis prompt
            system: Optional system prompt
            temperature: Response randomness

        Returns:
            ClaudeResponse with analysis
        """
        if not self.is_available:
            return ClaudeResponse(
                content="Claude client not available",
                model="none",
                latency_ms=0,
                tokens_used=0
            )

        try:
            import time
            start = time.perf_counter()

            # Run in thread to avoid blocking
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.MODEL,
                max_tokens=1024,
                system=system or "You are a poker coach. Be concise and actionable.",
                messages=[{"role": "user", "content": prompt}]
            )

            latency_ms = (time.perf_counter() - start) * 1000

            content = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens

            return ClaudeResponse(
                content=content,
                model=self.MODEL,
                latency_ms=latency_ms,
                tokens_used=tokens
            )

        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            return ClaudeResponse(
                content=f"Error: {e}",
                model=self.MODEL,
                latency_ms=0,
                tokens_used=0
            )

    async def poker_analysis(self, prompt: str) -> str:
        """
        Poker-specific analysis with Claude.

        Args:
            prompt: Poker situation prompt

        Returns:
            Analysis text
        """
        system = """You are an expert poker coach specializing in exploitative play.

Your style:
- Be concise and direct
- Focus on exploiting opponent tendencies
- Recommend specific actions with sizing
- Explain the "why" briefly

Format responses as:
ACTION: [FOLD/CALL/RAISE]
SIZING: [if raising]
EQUITY: [estimated %]
REASONING: [1-2 sentences]"""

        response = await self.analyze(prompt, system=system)
        return response.content

    def get_cost(self) -> float:
        """Get cost estimate for a request."""
        return self.COST_PER_REQUEST


# Test
async def test_claude():
    """Test Claude client."""
    print("=== Claude Client Test ===\n")

    client = ClaudeClient()

    if not client.is_available:
        print("Claude not available. Set ANTHROPIC_API_KEY env var.")
        return

    prompt = "What is 2+2? Answer with just the number."

    print("Testing Claude...")
    response = await client.analyze(prompt)
    print(f"Model: {response.model}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"Response: {response.content}")


if __name__ == "__main__":
    asyncio.run(test_claude())
