"""
Perplexity AI client for fact-checking and web search.
Best for: current events, factual queries, research questions.
"""
import os
import time
import logging
import aiohttp
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerplexityResponse:
    """Response from Perplexity API."""
    content: str
    citations: List[str]
    model: str
    latency_ms: float
    cost: float


@dataclass
class ResearchQuery:
    """Track research query with context."""
    query: str
    timestamp: datetime
    model: str
    cost: float
    citations: List[Dict[str, str]]
    content: str
    context: Dict[str, Any]  # mode, skill_level, etc.


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
        "sonar-pro": {"cost": 0.005, "best_for": "complex research"},
        "sonar": {"cost": 0.001, "best_for": "quick facts"},
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[PerplexityResponse, float]] = {}
        self._cache_ttl = 3600  # 1 hour

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
            model: "sonar" (fast) or "sonar-pro" (deep)
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
            model="sonar-pro",
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

    async def research_with_context(
        self,
        query: str,
        context: Dict[str, Any],
        model: str = "sonar-pro"
    ) -> PerplexityResponse:
        """
        Context-aware research that adapts query based on mode and skill level.

        Args:
            query: Research question
            context: {'mode': 'poker'|'homework'|'general', 'skill_level': 'beginner'|'intermediate'|'advanced'}
            model: 'sonar' (fast) or 'sonar-pro' (deep)

        Returns:
            PerplexityResponse with enhanced context
        """
        # Enhance query with context prefix
        enhanced_query = self._add_context_prefix(query, context)

        # Get mode-specific domain filter
        domain_filter = self._get_domain_filter(context.get('mode'))

        # Execute query
        response = await self.query(
            enhanced_query,
            model=model,
            search_domain_filter=domain_filter
        )

        # Save to research history
        await self._save_research_query(query, response, context)

        return response

    def _add_context_prefix(self, query: str, context: Dict) -> str:
        """Add appropriate context prefix to query."""
        prefixes = {
            'poker': lambda s: f"For {s} poker player: ",
            'homework': lambda s: f"For {s} level student: ",
            'debug': lambda s: f"For {s} developer: "
        }

        mode = context.get('mode', 'general')
        skill = context.get('skill_level', 'intermediate')

        if mode in prefixes:
            return prefixes[mode](skill) + query
        return query

    def _get_domain_filter(self, mode: Optional[str]) -> List[str]:
        """Get authoritative domains for mode."""
        filters = {
            'poker': ['pokerstrategy.com', '2plus2.com', 'upswingpoker.com'],
            'homework': ['.edu', 'khanacademy.org', 'wolframalpha.com'],
            'debug': ['stackoverflow.com', 'github.com'],
            'general': ['.edu', '.gov', '.org']
        }
        return filters.get(mode, filters['general'])

    async def _save_research_query(
        self,
        query: str,
        response: PerplexityResponse,
        context: Dict
    ):
        """Save research query to history."""
        # This will be called by session_summary after we implement that
        # For now, just log it
        logger.info(f"Research query saved: {query[:50]}... (cost: ${response.cost:.3f})")

    async def verify_claim(
        self,
        claim: str,
        search_domains: List[str] = None
    ) -> Dict[str, Any]:
        """
        Fact-check a claim with confidence scoring.

        Returns:
        {
            'claim': str,
            'verdict': 'TRUE' | 'FALSE' | 'DISPUTED' | 'UNVERIFIABLE',
            'confidence': 0.0-1.0,
            'explanation': str,
            'sources': List[Dict]
        }
        """
        prompt = f"""Fact-check this claim with authoritative sources:

CLAIM: {claim}

Provide:
1. VERDICT: TRUE, FALSE, DISPUTED, or UNVERIFIABLE
2. CONFIDENCE: 0.0 to 1.0
3. EXPLANATION: Brief reasoning (2-3 sentences)

Format as:
VERDICT: [verdict]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [explanation]
"""

        response = await self.query(
            prompt,
            model='sonar-pro',
            search_domain_filter=search_domains or ['.edu', '.gov', '.org']
        )

        # Parse structured response
        verdict = self._extract_field(response.content, 'VERDICT')
        confidence_str = self._extract_field(response.content, 'CONFIDENCE')
        explanation = self._extract_field(response.content, 'EXPLANATION')

        return {
            'claim': claim,
            'verdict': verdict,
            'confidence': float(confidence_str) if confidence_str else 0.7,
            'explanation': explanation,
            'sources': response.citations,
            'cost': response.cost
        }

    def _extract_field(self, text: str, field: str) -> str:
        """Extract field from formatted response."""
        match = re.search(f"{field}:?\\s*(.+?)(?:\\n|$)", text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    async def cached_query(self, query: str, **kwargs) -> PerplexityResponse:
        """Query with 1-hour cache."""
        cache_key = f"{query}:{kwargs.get('model', 'sonar')}"

        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return result

        result = await self.query(query, **kwargs)
        self._cache[cache_key] = (result, time.time())
        return result
