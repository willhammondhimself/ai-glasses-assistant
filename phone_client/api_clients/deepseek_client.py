"""
DeepSeek Client for poker analysis and reasoning.
- V3.1 (deepseek-chat): Live play, 4s latency, $0.007/request
- V3.2 (deepseek-reasoner): Post-session, 35s latency, $0.014/request
"""
import os
import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# DeepSeek uses OpenAI-compatible API
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed. Run: pip install openai")


@dataclass
class DeepSeekResponse:
    """Response from DeepSeek API."""
    content: str
    model: str
    thinking: bool
    latency_ms: float
    tokens_used: int


class DeepSeekClient:
    """
    DeepSeek client for poker analysis.

    Two modes:
    - V3.1 (deepseek-chat): Fast mode for live play (~4s)
    - V3.2 (deepseek-reasoner): Deep thinking for post-session (~35s)

    Uses OpenAI-compatible API.
    """

    BASE_URL = "https://api.deepseek.com"

    # Model selection
    MODEL_FAST = "deepseek-chat"       # V3.1 - 4s latency
    MODEL_THINKING = "deepseek-reasoner"  # V3.2 - 35s latency

    # Cost per request (approximate)
    COST_FAST = 0.007
    COST_THINKING = 0.014

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.client = None
        self._initialized = False

        if not OPENAI_AVAILABLE:
            logger.error("openai package not available")
            return

        if not self.api_key:
            logger.warning("No DeepSeek API key provided. Set DEEPSEEK_API_KEY env var.")
            return

        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.BASE_URL
            )
            self._initialized = True
            logger.info("DeepSeek client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek: {e}")

    @property
    def is_available(self) -> bool:
        """Check if client is ready to use."""
        return self._initialized and self.client is not None

    async def analyze_hand(
        self,
        prompt: str,
        thinking: bool = False,
        temperature: float = 0.3
    ) -> DeepSeekResponse:
        """
        Analyze a poker hand.

        Args:
            prompt: The poker analysis prompt
            thinking: Use V3.2 thinking mode (slower, deeper analysis)
            temperature: Response randomness (0.0-1.0)

        Returns:
            DeepSeekResponse with analysis
        """
        if not self.is_available:
            return DeepSeekResponse(
                content="DeepSeek client not available",
                model="none",
                thinking=thinking,
                latency_ms=0,
                tokens_used=0
            )

        model = self.MODEL_THINKING if thinking else self.MODEL_FAST

        try:
            import time
            start = time.perf_counter()

            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024
            )

            latency_ms = (time.perf_counter() - start) * 1000

            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0

            return DeepSeekResponse(
                content=content,
                model=model,
                thinking=thinking,
                latency_ms=latency_ms,
                tokens_used=tokens
            )

        except Exception as e:
            logger.error(f"DeepSeek analysis failed: {e}")
            return DeepSeekResponse(
                content=f"Error: {e}",
                model=model,
                thinking=thinking,
                latency_ms=0,
                tokens_used=0
            )

    async def live_analysis(self, prompt: str) -> str:
        """
        Fast analysis for live poker play (~4s).

        Args:
            prompt: Poker situation prompt

        Returns:
            Analysis text
        """
        response = await self.analyze_hand(prompt, thinking=False)
        return response.content

    async def deep_review(self, prompt: str) -> str:
        """
        Deep analysis for post-session review (~35s).

        Args:
            prompt: Hand history or session review prompt

        Returns:
            Detailed analysis with reasoning
        """
        response = await self.analyze_hand(prompt, thinking=True)
        return response.content

    async def batch_review(
        self,
        hands: list,
        session_context: str = ""
    ) -> str:
        """
        Review multiple hands from a session.

        Args:
            hands: List of hand dictionaries
            session_context: Stakes, villain types, etc.

        Returns:
            Comprehensive session review
        """
        if not hands:
            return "No hands to review"

        prompt = f"""Review this poker session and identify leaks/mistakes.

SESSION CONTEXT:
{session_context}

HANDS PLAYED:
"""
        for i, hand in enumerate(hands, 1):
            prompt += f"\n--- Hand {i} ---\n"
            prompt += f"Hero: {hand.get('hero_cards', '?')}\n"
            prompt += f"Board: {hand.get('board', '?')}\n"
            prompt += f"Action: {hand.get('action', '?')}\n"
            prompt += f"Result: {hand.get('result', '?')}\n"

        prompt += """

ANALYZE:
1. Identify the 3 biggest mistakes
2. Spot any patterns or leaks
3. Suggest specific improvements
4. Rate overall session play (1-10)

Be specific and actionable."""

        return await self.deep_review(prompt)

    def get_cost(self, thinking: bool = False) -> float:
        """Get cost estimate for a request."""
        return self.COST_THINKING if thinking else self.COST_FAST

    async def solve_math(self, problem: str) -> str:
        """
        Solve math problem with step-by-step reasoning.

        Args:
            problem: Math problem description

        Returns:
            Solution with steps
        """
        prompt = f"""Solve this math problem step by step:

{problem}

Show your work clearly:
1. Identify what we're solving for
2. Show each step of the solution
3. Provide the final answer

Format your answer clearly with the final answer on its own line."""

        return await self.live_analysis(prompt)

    async def explain_code(self, code: str, error: str = None) -> str:
        """
        Explain code or debug an error.

        Args:
            code: The code to explain
            error: Optional error message

        Returns:
            Explanation string
        """
        if error:
            prompt = f"""Explain this error and how to fix it:

Code:
```
{code}
```

Error:
{error}

Provide:
1. What the error means
2. Why it's happening
3. The corrected code
4. How to prevent this in the future"""
        else:
            prompt = f"""Explain what this code does:

```
{code}
```

Provide a clear, concise explanation of:
1. What the code does
2. How it works
3. Any potential issues"""

        return await self.live_analysis(prompt)


# Test
async def test_deepseek():
    """Test DeepSeek client."""
    print("=== DeepSeek Client Test ===\n")

    client = DeepSeekClient()

    if not client.is_available:
        print("DeepSeek not available. Set DEEPSEEK_API_KEY env var.")
        return

    # Test fast analysis
    prompt = """You are a poker coach. Analyze this hand briefly.

Hero: Ah Kc | Position: BTN | Stack: 100bb
Board: 9s 7h 3d
Pot: 12bb | Facing: 8bb

Villain is a calling station (VPIP 45%, Aggression 25%).

Recommend action (FOLD/CALL/RAISE) with brief reasoning."""

    print("Testing fast analysis (V3.1)...")
    response = await client.analyze_hand(prompt, thinking=False)
    print(f"Model: {response.model}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"Response:\n{response.content}\n")


if __name__ == "__main__":
    asyncio.run(test_deepseek())
