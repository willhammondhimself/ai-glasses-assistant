"""
Perplexity AI client for fact-checking and web search.
Best for: current events, factual queries, research questions.
"""
import os
import time
import logging
import aiohttp
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class PerplexityResponse:
    """Response from Perplexity API."""
    content: str
    citations: List[str]
    model: str
    latency_ms: float
    cost: float


class PerplexityClient:
    """
    Perplexity API for facts and search.

    Tier 3 in Intelligence Router:
    - Latency: ~2s
    - Cost: $0.001-0.005 per query
    - Best for: factual queries, current events, research
    """

    BASE_URL = "https://api.perplexity.ai/chat/completions"

    MODELS = {
        "sonar-reasoning": {"cost": 0.005, "best_for": "complex research"},
        "sonar": {"cost": 0.001, "best_for": "quick facts"},
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def query(
        self,
        prompt: str,
        model: str = "sonar",
        search_domain_filter: List[str] = None,
        search_recency_filter: str = None
    ) -> PerplexityResponse:
        """
        Query Perplexity for facts/research.

        Args:
            prompt: The question to research
            model: "sonar" (fast) or "sonar-reasoning" (deep)
            search_domain_filter: Limit search to specific domains
            search_recency_filter: "day", "week", "month", "year"

        Returns:
            PerplexityResponse with content and citations
        """
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not set")

        session = await self._get_session()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        if search_domain_filter:
            payload["search_domain_filter"] = search_domain_filter

        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter

        start = time.perf_counter()
        try:
            async with session.post(
                self.BASE_URL,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Perplexity API error: {response.status} - {error_text}")
                    raise Exception(f"Perplexity API error: {response.status}")

                data = await response.json()
                latency = (time.perf_counter() - start) * 1000

            return PerplexityResponse(
                content=data["choices"][0]["message"]["content"],
                citations=data.get("citations", []),
                model=model,
                latency_ms=latency,
                cost=self.MODELS[model]["cost"]
            )

        except aiohttp.ClientError as e:
            logger.error(f"Perplexity request failed: {e}")
            raise

    async def quick_fact(self, question: str) -> str:
        """
        Quick factual query using sonar model.

        Args:
            question: Simple factual question

        Returns:
            Answer string
        """
        response = await self.query(question, model="sonar")
        return response.content

    async def research(self, topic: str, recency: str = None) -> PerplexityResponse:
        """
        In-depth research using sonar-reasoning model.

        Args:
            topic: Research topic or complex question
            recency: Optional time filter

        Returns:
            Full response with citations
        """
        return await self.query(
            topic,
            model="sonar-reasoning",
            search_recency_filter=recency
        )

    @property
    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
