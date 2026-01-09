"""Dedicated news voice tool using Perplexity for poker and current news."""
import os
import httpx
import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


class PerplexityNewsTool(VoiceTool):
    """Get poker and current news via Perplexity AI.

    Specialized news tool with higher priority than generic search.
    Handles voice commands like:
    - "poker news" / "news poker meta"
    - "latest news"
    - "what's happening in poker"
    - "GTO trends"
    - "poker strategy news"
    - "WSOP updates"
    """

    name = "news"
    description = "Get poker and current news via Perplexity"

    # Higher priority keywords for news queries
    keywords = [
        # Poker news specific
        r"\bpoker\s+news\b",
        r"\bnews\s+poker\b",
        r"\bpoker\s+meta\b",
        r"\bnews\s+meta\b",  # "news meta" trigger
        r"\bmeta\s+(?:news|trends|analysis)\b",
        r"\bgto\s+(?:news|trends|updates|meta)\b",
        r"\bwsop\s+(?:news|updates|results)\b",
        r"\bworld\s+series\s+of\s+poker\b",
        r"\bpoker\s+(?:strategy|tips)\s+news\b",
        r"\bpoker\s+(?:results|tournaments)\b",
        r"\bonline\s+poker\s+(?:news|updates)\b",
        r"\bsolver\s+(?:news|updates|trends)\b",
        # General news
        r"\b(?:latest|current|today'?s?)\s+news\b",
        r"\bheadlines?\b",
        r"\bwhat'?s?\s+(?:happening|going\s+on)\b",
        r"\bnews\s+(?:update|brief|summary)\b",
        r"\btop\s+(?:stories|news)\b",
        r"\bbreaking\s+news\b",
    ]

    priority = 8  # Higher than generic Perplexity (5)

    # Enhanced poker news categories with search queries
    POKER_CATEGORIES = {
        "meta": "poker GTO meta strategy trends solvers current plays exploits",
        "tournaments": "poker tournament results WSOP WPT EPT final table",
        "cash": "cash game strategy high stakes nosebleeds live poker",
        "online": "online poker site news PokerStars GGPoker promotions",
        "strategy": "poker strategy training coaching GTO solver updates",
        "players": "poker player news interviews profiles rankings"
    }

    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            logger.warning("PERPLEXITY_API_KEY not set - News tool disabled")

        # Cache for recent news queries (simple in-memory cache)
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = 300  # 5 minutes

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute a news query using Perplexity API.

        Args:
            query: The user's voice query

        Returns:
            VoiceToolResult with news summary
        """
        if not self.api_key:
            return VoiceToolResult(
                success=False,
                message="News service is not configured. Please set the Perplexity API key.",
                data={"error": "missing_api_key"}
            )

        query_lower = query.lower()

        # Determine news type - check for meta first (highest priority)
        is_meta_query = self._is_meta_query(query_lower)
        is_poker_news = self._is_poker_news(query_lower)

        try:
            if is_meta_query:
                result = await self._get_meta_analysis(query_lower)
            elif is_poker_news:
                result = await self._get_poker_news(query_lower)
            else:
                result = await self._get_general_news(query_lower)

            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"Perplexity API error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, the news service is temporarily unavailable.",
                data={"error": str(e)}
            )
        except Exception as e:
            logger.error(f"News query failed: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I couldn't fetch the news right now.",
                data={"error": str(e)}
            )

    def _is_meta_query(self, query: str) -> bool:
        """Check if query is asking for poker meta/GTO analysis."""
        meta_indicators = [
            "meta", "gto trends", "solver", "exploits", "current meta",
            "strategy trends", "what's working", "popular plays"
        ]
        return any(indicator in query for indicator in meta_indicators)

    def _is_poker_news(self, query: str) -> bool:
        """Check if query is asking for poker-related news."""
        poker_indicators = [
            "poker", "wsop", "gto", "tournament", "cash game",
            "hand history", "solver", "strategy", "bovada", "ignition",
            "pokerstars", "gg poker", "acr", "americas cardroom"
        ]
        return any(indicator in query for indicator in poker_indicators)

    async def _get_meta_analysis(self, query: str) -> VoiceToolResult:
        """Get poker meta/GTO trends analysis with sources."""
        search_query = self.POKER_CATEGORIES["meta"]

        system_prompt = """You are a poker meta analyst. Provide current GTO and meta strategy insights.
Focus on:
1. Current solver-approved plays and adjustments
2. Popular exploits and counter-strategies
3. Preflop range trends (3-bet frequencies, cold-call ranges)
4. Postflop tendencies (c-bet frequencies, check-raise spots)
5. What's working at different stake levels

Be specific with percentages and frequencies when possible.
Keep response to 3-4 sentences, voice-friendly.
Example: "Current meta favors aggressive 3-betting in the small blind, around 12-15% vs button opens. Players are overfolding to river raises, making thin value bets profitable."
"""

        content, sources = await self._call_perplexity_with_sources(search_query, system_prompt)

        return VoiceToolResult(
            success=True,
            message=content,
            data={
                "query": search_query,
                "news_type": "meta",
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "hud_message": {
                    "type": "news",
                    "category": "meta",
                    "content": content,
                    "sources": sources,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

    async def _get_poker_news(self, query: str) -> VoiceToolResult:
        """Get poker-specific news."""
        # Build a targeted poker news query
        search_query = self._build_poker_query(query)

        system_prompt = """You are a poker news assistant. Provide concise poker news updates.
Focus on:
1. Recent tournament results and notable hands
2. Strategy and GTO developments
3. Online poker site updates
4. Player interviews and profiles
5. Industry news and regulations

Keep responses concise (2-4 sentences) and voice-friendly.
Mention specific events, players, or developments when relevant."""

        content, sources = await self._call_perplexity_with_sources(search_query, system_prompt)

        return VoiceToolResult(
            success=True,
            message=content,
            data={
                "query": search_query,
                "news_type": "poker",
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "hud_message": {
                    "type": "news",
                    "category": "poker",
                    "content": content,
                    "sources": sources,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

    async def _get_general_news(self, query: str) -> VoiceToolResult:
        """Get general news headlines."""
        # Extract topic if specified
        topic = self._extract_topic(query)

        if topic:
            search_query = f"latest news about {topic}"
        else:
            search_query = "top news headlines today"

        system_prompt = """You are a news assistant. Provide concise news updates.
Focus on the most important and relevant current events.
Keep responses to 2-3 sentences suitable for voice output.
Mention specific events, people, or developments when relevant."""

        content, sources = await self._call_perplexity_with_sources(search_query, system_prompt)

        return VoiceToolResult(
            success=True,
            message=content,
            data={
                "query": search_query,
                "news_type": "general",
                "topic": topic,
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "hud_message": {
                    "type": "news",
                    "category": "general",
                    "content": content,
                    "sources": sources,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

    def _build_poker_query(self, query: str) -> str:
        """Build a targeted poker news search query."""
        # Check for specific poker topics
        if "wsop" in query or "world series" in query:
            return "World Series of Poker WSOP latest news and results"
        elif "gto" in query:
            return "poker GTO strategy latest developments and solver updates"
        elif "tournament" in query or "results" in query:
            return "poker tournament results and notable hands today"
        elif "strategy" in query or "tips" in query:
            return "poker strategy news and training site updates"
        elif "online" in query:
            return "online poker news site updates and promotions"
        elif "meta" in query:
            return "current poker meta strategy trends and popular plays"
        else:
            return "poker news today latest developments and tournament results"

    def _extract_topic(self, query: str) -> Optional[str]:
        """Extract topic from news query."""
        # Remove common news-related prefixes
        prefixes = [
            "latest news", "current news", "today's news", "news about",
            "news on", "what's happening", "what is happening",
            "headlines", "top news", "breaking news", "news update"
        ]

        cleaned = query.lower()
        for prefix in prefixes:
            if prefix in cleaned:
                # Get the part after the prefix
                idx = cleaned.find(prefix) + len(prefix)
                remainder = cleaned[idx:].strip()
                if remainder and len(remainder) > 2:
                    # Clean up common words
                    remainder = re.sub(r'^(in|with|about|on|for)\s+', '', remainder)
                    return remainder if remainder else None

        return None

    async def _call_perplexity(self, query: str, system_prompt: str) -> str:
        """Call the Perplexity API for news query (voice-only, no sources).

        Args:
            query: Search query
            system_prompt: System context for the model

        Returns:
            Summary response from Perplexity
        """
        content, _ = await self._call_perplexity_with_sources(query, system_prompt)
        return content

    async def _call_perplexity_with_sources(
        self, query: str, system_prompt: str
    ) -> tuple[str, List[Dict[str, str]]]:
        """Call the Perplexity API and return content with sources.

        Args:
            query: Search query
            system_prompt: System context for the model

        Returns:
            Tuple of (content, sources) where sources is list of {url, title}
        """
        # Use online model for real-time search
        model = "llama-3.1-sonar-small-128k-online"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.2,
            "max_tokens": 400,  # Slightly longer for news
            "return_citations": True,  # Get sources
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                PERPLEXITY_API_URL,
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Extract sources/citations if available
            sources = []
            if "citations" in data:
                # Perplexity returns citations as list of URLs
                for i, url in enumerate(data.get("citations", [])[:5]):  # Max 5 sources
                    sources.append({
                        "url": url,
                        "title": f"Source {i + 1}"  # Perplexity doesn't always give titles
                    })

            # Clean up content for voice
            clean_content = self._clean_for_voice(content)

            return clean_content, sources

    def _clean_for_voice(self, text: str) -> str:
        """Clean up text for voice output."""
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        text = re.sub(r'#{1,6}\s+', '', text)  # Headers

        # Remove bullet points and numbered lists markers
        text = re.sub(r'^[\s]*[-•]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s*', '', text, flags=re.MULTILINE)

        # Clean up citations like [1], [2]
        text = re.sub(r'\[\d+\]', '', text)

        # Clean up extra whitespace
        text = re.sub(r'\n{2,}', '. ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)

        return text.strip()

    async def get_ticker_headlines(self, count: int = 5) -> List[Dict[str, str]]:
        """Get news headlines for HUD ticker.

        Args:
            count: Number of headlines to return

        Returns:
            List of headline dicts with title, category, and priority
        """
        if not self.api_key:
            return []

        try:
            # Get poker news headlines
            poker_headlines = await self._get_headlines_for_category("poker tournament results WSOP", "poker")

            # Get general headlines
            general_headlines = await self._get_headlines_for_category("top news headlines", "general")

            # Combine and return
            all_headlines = poker_headlines + general_headlines
            return all_headlines[:count]

        except Exception as e:
            logger.error(f"Failed to get ticker headlines: {e}")
            return []

    async def _get_headlines_for_category(self, query: str, category: str) -> List[Dict[str, str]]:
        """Get headlines for a specific category."""
        system_prompt = """Extract 3 brief news headlines from current events.
Format each headline as a single sentence (10-15 words max).
Return only the headlines, one per line, no numbering or bullets."""

        try:
            content = await self._call_perplexity(query, system_prompt)
            lines = [line.strip() for line in content.split('.') if line.strip() and len(line.strip()) > 10]

            return [
                {
                    "title": line[:100],  # Limit length
                    "category": category,
                    "priority": "high" if category == "poker" else "normal"
                }
                for line in lines[:3]
            ]
        except Exception:
            return []
