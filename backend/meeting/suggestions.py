"""
Suggestion Engine - Gemini 2.5 Pro integration with WHAM personality.

Handles:
- Quick suggestions (double-tap, 2-3s latency target)
- Detailed suggestions (voice command, 3-4s latency target)
- WHAM personality traits: concise, tactical, confident but humble
"""

import asyncio
import logging
import os
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import google.generativeai as genai

from .models import (
    SuggestionRequest,
    SuggestionResponse,
    SuggestionType,
    TriggerType,
    TranscriptSegment,
    SpeakerRole,
)
from .context import ContextWindow

logger = logging.getLogger(__name__)


# WHAM System Prompt - Core personality for meeting assistance
WHAM_SYSTEM_PROMPT = """You are WHAM (Will Hammond's Augmented Mind), an AI assistant embedded in AR glasses helping Will during live meetings.

CORE TRAITS:
- You AUGMENT Will's intelligence, you don't replace it
- Be CONCISE - Will needs quick answers under pressure (1-3 sentences max for quick mode)
- Be TACTICAL - understand meeting dynamics and goals
- Be CONFIDENT but HUMBLE - suggest, don't dictate
- Be PROFESSIONAL but CONVERSATIONAL - sound human, not robotic

YOUR CAPABILITIES:
- Provide suggested responses or talking points
- Flag tactical opportunities or concerns (leverage, being talked over, commitments made)
- Offer 2-3 alternatives when appropriate
- Fill in context Will might be missing
- Help navigate difficult conversations

FORMATTING RULES:
- For QUICK MODE (double-tap): Ultra-concise, 1-2 sentences max. Action-oriented.
- For FULL MODE (voice command): Can be more detailed but still focused. Max 3-4 sentences.
- Start with the most important point
- Use natural language, not bullet points for speech
- If suggesting what to say, phrase it naturally (don't use "Say:" prefix)

WHAT NOT TO DO:
- Don't be preachy or lecture Will
- Don't over-explain
- Don't provide generic advice - be specific to the conversation
- Don't assume Will doesn't understand things - he's smart
- Don't use corporate jargon or filler words

TACTICAL AWARENESS:
- Notice if someone is dominating the conversation
- Catch commitments or promises being made
- Identify when Will is being put on the spot
- Recognize negotiation dynamics
- Flag if something said contradicts earlier statements
"""


@dataclass
class SuggestionConfig:
    """Configuration for suggestion engine."""
    model: str = "gemini-2.0-flash"  # Fast model for low latency
    model_full: str = "gemini-2.5-pro-preview-06-05"  # Full model for detailed requests
    temperature_quick: float = 0.3  # More focused for quick mode
    temperature_full: float = 0.5  # Slightly more creative for full mode
    max_tokens_quick: int = 100
    max_tokens_full: int = 500
    timeout_quick_s: float = 3.0
    timeout_full_s: float = 5.0


class SuggestionEngine:
    """
    Gemini-powered suggestion engine with WHAM personality.

    Uses:
    - gemini-2.0-flash for quick double-tap responses (~1.5s)
    - gemini-2.5-pro for detailed voice command responses (~3s)

    Cost: Free on Google AI Studio student tier
    """

    def __init__(self, config: SuggestionConfig = None):
        self.config = config or SuggestionConfig()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set - suggestions will fail")
            self._available = False
        else:
            genai.configure(api_key=api_key)
            self._available = True

        self._model_quick = None
        self._model_full = None

    def _get_model(self, is_quick: bool):
        """Get the appropriate Gemini model."""
        if is_quick:
            if self._model_quick is None:
                self._model_quick = genai.GenerativeModel(
                    self.config.model,
                    system_instruction=WHAM_SYSTEM_PROMPT,
                )
            return self._model_quick
        else:
            if self._model_full is None:
                self._model_full = genai.GenerativeModel(
                    self.config.model_full,
                    system_instruction=WHAM_SYSTEM_PROMPT,
                )
            return self._model_full

    async def get_suggestion(
        self,
        context: ContextWindow,
        trigger: TriggerType,
        user_query: str = None,
    ) -> SuggestionResponse:
        """
        Generate a meeting suggestion.

        Args:
            context: Context window from ContextManager
            trigger: How the suggestion was triggered
            user_query: Optional user question (for voice commands)

        Returns:
            SuggestionResponse with WHAM's suggestion
        """
        if not self._available:
            return SuggestionResponse(
                suggestion="WHAM unavailable - check GEMINI_API_KEY",
                suggestion_type=SuggestionType.QUICK_RESPONSE,
                confidence=0.0,
            )

        start_time = time.perf_counter()
        is_quick = trigger == TriggerType.DOUBLE_TAP

        try:
            # Build the prompt
            prompt = self._build_prompt(context, trigger, user_query)

            # Get model and configure
            model = self._get_model(is_quick)

            generation_config = genai.types.GenerationConfig(
                temperature=self.config.temperature_quick if is_quick else self.config.temperature_full,
                max_output_tokens=self.config.max_tokens_quick if is_quick else self.config.max_tokens_full,
            )

            # Set timeout
            timeout = self.config.timeout_quick_s if is_quick else self.config.timeout_full_s

            # Generate response
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=generation_config,
                ),
                timeout=timeout,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse response
            return self._parse_response(response, latency_ms, is_quick)

        except asyncio.TimeoutError:
            logger.warning(f"Suggestion timeout after {timeout}s")
            return SuggestionResponse(
                suggestion="Taking too long - try again",
                suggestion_type=SuggestionType.QUICK_RESPONSE,
                confidence=0.5,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Suggestion error: {e}")
            return SuggestionResponse(
                suggestion="Error generating suggestion",
                suggestion_type=SuggestionType.QUICK_RESPONSE,
                confidence=0.0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

    def _build_prompt(
        self,
        context: ContextWindow,
        trigger: TriggerType,
        user_query: str = None,
    ) -> str:
        """Build the prompt for Gemini."""
        parts = []

        # Mode indicator
        if trigger == TriggerType.DOUBLE_TAP:
            parts.append("MODE: QUICK (1-2 sentences max, action-oriented)")
        else:
            parts.append("MODE: FULL (can be more detailed, up to 3-4 sentences)")

        # Meeting context if available
        if context.meeting_context:
            parts.append(f"\nMEETING CONTEXT:\n{context.meeting_context}")

        # Transcript
        parts.append("\nRECENT CONVERSATION:")
        if context.segments:
            for seg in context.segments:
                speaker = "Will" if seg.speaker == SpeakerRole.USER else "Other"
                parts.append(f"[{speaker}]: {seg.text}")
        else:
            parts.append("[No conversation yet]")

        # User query for voice commands
        if user_query:
            parts.append(f"\nWILL'S QUESTION: {user_query}")
            parts.append("\nProvide a helpful response to Will's question based on the conversation context.")
        else:
            parts.append("\nWhat should Will say or consider right now? Be specific and actionable.")

        return "\n".join(parts)

    def _parse_response(
        self,
        response,
        latency_ms: float,
        is_quick: bool,
    ) -> SuggestionResponse:
        """Parse Gemini response into SuggestionResponse."""
        try:
            text = response.text.strip()

            # Determine suggestion type based on content
            suggestion_type = self._classify_suggestion(text)

            # Extract alternatives if present (for longer responses)
            alternatives = []
            if not is_quick and "\n" in text:
                lines = text.split("\n")
                main_suggestion = lines[0]

                for line in lines[1:]:
                    line = line.strip()
                    if line and not line.startswith("-"):
                        alternatives.append(line)
            else:
                main_suggestion = text

            # Extract tactical notes if present
            tactical_notes = None
            if "[tactical:" in text.lower():
                start = text.lower().find("[tactical:")
                end = text.find("]", start)
                if end > start:
                    tactical_notes = text[start + 10:end].strip()
                    main_suggestion = text[:start].strip()

            return SuggestionResponse(
                suggestion=main_suggestion,
                suggestion_type=suggestion_type,
                confidence=0.9 if latency_ms < 2000 else 0.8,
                alternatives=alternatives[:3],
                tactical_notes=tactical_notes,
                latency_ms=latency_ms,
                tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
                cost=0.0,  # Free tier
            )

        except Exception as e:
            logger.error(f"Response parse error: {e}")
            return SuggestionResponse(
                suggestion="Could not parse response",
                suggestion_type=SuggestionType.QUICK_RESPONSE,
                confidence=0.5,
                latency_ms=latency_ms,
            )

    def _classify_suggestion(self, text: str) -> SuggestionType:
        """Classify the type of suggestion based on content."""
        text_lower = text.lower()

        if any(w in text_lower for w in ["negotiate", "leverage", "position", "counter"]):
            return SuggestionType.NEGOTIATION

        if any(w in text_lower for w in ["clarify", "ask", "what do you mean", "could you"]):
            return SuggestionType.CLARIFICATION

        if any(w in text_lower for w in ["actually", "correct", "not quite", "verify"]):
            return SuggestionType.FACT_CHECK

        if any(w in text_lower for w in ["consider", "keep in mind", "note that", "strategically"]):
            return SuggestionType.TACTICAL_ADVICE

        if any(w in text_lower for w in ["background", "context", "for reference", "historically"]):
            return SuggestionType.CONTEXT_FILL

        return SuggestionType.QUICK_RESPONSE

    async def analyze_dynamics(
        self,
        context: ContextWindow
    ) -> Dict[str, Any]:
        """
        Analyze meeting dynamics (who's dominating, tension, etc.).

        Useful for proactive suggestions.
        """
        if not self._available or not context.segments:
            return {}

        prompt = f"""Analyze the meeting dynamics from this transcript.
Return a brief JSON with:
- dominant_speaker: "user" or "other" or "balanced"
- tension_level: 1-5 (1=relaxed, 5=tense)
- user_participation: 1-5 (1=quiet, 5=very active)
- key_topics: list of 2-3 main topics
- concerns: any issues or red flags

TRANSCRIPT:
{self._build_prompt(context, TriggerType.VOICE_COMMAND)}

Respond with only the JSON, no other text."""

        try:
            model = self._get_model(is_quick=True)
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=200,
                ),
            )

            import json
            return json.loads(response.text)

        except Exception as e:
            logger.error(f"Dynamics analysis error: {e}")
            return {}


# Singleton instance
_suggestion_engine: Optional[SuggestionEngine] = None


def get_suggestion_engine() -> SuggestionEngine:
    """Get the global suggestion engine instance."""
    global _suggestion_engine
    if _suggestion_engine is None:
        _suggestion_engine = SuggestionEngine()
    return _suggestion_engine
