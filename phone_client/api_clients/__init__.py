"""
API Clients for WHAM multi-model routing.
- GeminiClient: OCR and vision (1s latency, $0.002)
- DeepSeekClient: V3.1 live play, V3.2 post-session
- PerplexityClient: Facts and web search (2s latency, $0.001-0.005)
- ClaudeClient: Fallback for complex reasoning
"""
from .gemini_client import GeminiClient, CardExtraction, MathExtraction, CodeExtraction
from .deepseek_client import DeepSeekClient, DeepSeekResponse
from .perplexity_client import PerplexityClient, PerplexityResponse
from .claude_client import ClaudeClient

__all__ = [
    "GeminiClient",
    "CardExtraction",
    "MathExtraction",
    "CodeExtraction",
    "DeepSeekClient",
    "DeepSeekResponse",
    "PerplexityClient",
    "PerplexityResponse",
    "ClaudeClient",
]
