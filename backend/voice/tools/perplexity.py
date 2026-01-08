"""Perplexity AI tool for web search and news."""
import os
import httpx
import logging
from typing import Optional
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


class PerplexityTool(VoiceTool):
    """Search the web and get news using Perplexity AI."""

    name = "perplexity"
    description = "Search the web and get current news using Perplexity AI"

    keywords = [
        r"\bsearch\s+(for|about)\b",
        r"\blook\s+up\b",
        r"\bwhat\s+is\b",
        r"\bwho\s+is\b",
        r"\bnews\s+(about|on)\b",
        r"\blatest\s+(on|news)\b",
        r"\btell\s+me\s+about\b",
        r"\bresearch\b",
        r"\bfind\s+(out|information)\b",
        r"\bcurrent\s+events\b",
        r"\bwhat('s| is)\s+happening\b",
    ]

    priority = 5  # Lower priority - catch-all for general queries

    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            logger.warning("PERPLEXITY_API_KEY not set - Perplexity tool will be disabled")

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute a search using Perplexity API.

        Args:
            query: The user's voice query

        Returns:
            VoiceToolResult with search results
        """
        if not self.api_key:
            return VoiceToolResult(
                success=False,
                message="Perplexity search is not configured. Please set the API key."
            )

        # Determine if this is a news query
        is_news = any(word in query.lower() for word in ["news", "latest", "current events", "happening"])

        # Clean up the query
        search_query = self._extract_search_query(query)

        try:
            result = await self._call_perplexity(search_query, is_news)
            return VoiceToolResult(
                success=True,
                message=result,
                data={"query": search_query, "is_news": is_news}
            )
        except Exception as e:
            logger.error(f"Perplexity search failed: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I couldn't complete the search right now."
            )

    def _extract_search_query(self, query: str) -> str:
        """Extract the actual search terms from the voice query."""
        # Remove common prefixes
        prefixes = [
            "search for", "search about", "look up", "tell me about",
            "what is", "who is", "news about", "news on", "latest on",
            "latest news", "find out", "find information about", "research"
        ]

        cleaned = query.lower()
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        return cleaned.strip() if cleaned else query

    async def _call_perplexity(self, query: str, is_news: bool) -> str:
        """Call the Perplexity API.

        Args:
            query: Search query
            is_news: Whether to focus on recent news

        Returns:
            Summary response from Perplexity
        """
        # Use online model for real-time search
        model = "llama-3.1-sonar-small-128k-online"

        system_prompt = (
            "You are a helpful voice assistant. Provide a concise answer suitable for voice output. "
            "Keep responses to 2-3 sentences unless more detail is specifically needed. "
            "Focus on the most important and relevant information."
        )

        if is_news:
            system_prompt += " Focus on recent news and current events."

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.2,
            "max_tokens": 300,  # Keep responses concise for voice
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

            # Clean up for voice
            return self._clean_for_voice(content)

    def _clean_for_voice(self, text: str) -> str:
        """Clean up text for voice output."""
        import re

        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links

        # Remove bullet points and numbered lists markers
        text = re.sub(r'^[\s]*[-•]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s*', '', text, flags=re.MULTILINE)

        # Clean up extra whitespace
        text = re.sub(r'\n{2,}', '. ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)

        return text.strip()
