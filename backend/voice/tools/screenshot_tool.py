"""Screenshot analysis voice tool - help with errors, forms, and UI."""
import logging
from typing import Optional
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class ScreenshotVoiceTool(VoiceTool):
    """Voice-controlled screenshot analysis for error parsing and UI help."""

    name = "screenshot"
    description = "Analyze screenshots - parse errors, help with forms, explain UI"

    keywords = [
        r"\bscreenshot\b",
        r"\bscreen\b",
        r"\berror\s*(message)?\b",
        r"\bwhat.*say\b",
        r"\bhelp.*form\b",
        r"\bwhat\s+went\s+wrong\b",
        r"\bfix\s+this\b",
        r"\bwhat\s+should\s+i\s+(do|click)\b",
        r"\banalyze\s+(this\s+)?error\b",
        r"\bexplain\s+this\s+(error|screen)\b",
    ]

    priority = 8

    def __init__(self):
        self._vision_engine = None
        self._last_screenshot: Optional[str] = None

    def _get_vision_engine(self):
        """Get vision engine (lazy load)."""
        if self._vision_engine is None:
            from backend.vision.engine import VisionEngine
            self._vision_engine = VisionEngine()
        return self._vision_engine

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute screenshot analysis.

        Args:
            query: The user's voice query
            **kwargs: Additional context including 'image' (base64)

        Returns:
            VoiceToolResult with analysis
        """
        query_lower = query.lower()
        image_base64 = kwargs.get("image") or self._last_screenshot

        if not image_base64:
            return VoiceToolResult(
                success=False,
                message="I need a screenshot to analyze. Please share your screen or take a screenshot.",
                data={"needs_image": True}
            )

        # Store for follow-up
        self._last_screenshot = image_base64
        engine = self._get_vision_engine()

        try:
            # Analyze screenshot with context
            context = self._extract_context(query_lower)
            result = engine.analyze_screenshot(image_base64, context)

            if result.get("error"):
                return VoiceToolResult(
                    success=False,
                    message="I had trouble analyzing this screenshot.",
                    data=result
                )

            return self._format_result(result, query_lower)

        except Exception as e:
            logger.error(f"Screenshot tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I couldn't analyze this screenshot.",
                data={"error": str(e)}
            )

    def _extract_context(self, query: str) -> str:
        """Extract what the user needs help with."""
        if "error" in query:
            return "User needs help understanding and fixing an error"
        if "form" in query:
            return "User needs help filling out a form"
        if "click" in query or "do" in query:
            return "User needs guidance on what action to take"
        if "fix" in query:
            return "User wants to fix something that's wrong"
        return ""

    def _format_result(self, result: dict, query: str) -> VoiceToolResult:
        """Format analysis result for voice response."""
        screen_type = result.get("screen_type", "screen")
        summary = result.get("content_summary", "")
        errors = result.get("errors", [])
        suggestions = result.get("suggestions", [])

        # Build response based on what was found
        parts = []

        # Errors are highest priority
        if errors:
            if len(errors) == 1:
                parts.append(f"I see an error: {errors[0]}")
            else:
                parts.append(f"I see {len(errors)} errors. Main issue: {errors[0]}")

            # Add fix suggestion
            if suggestions:
                parts.append(f"To fix this: {suggestions[0]}")
        elif summary:
            # No errors, describe what's on screen
            parts.append(f"This is {screen_type}. {summary}")

            if suggestions:
                parts.append(suggestions[0])
        else:
            parts.append("I can see the screen but I'm not sure what you need help with. Could you be more specific?")

        message = " ".join(parts)

        return VoiceToolResult(
            success=True,
            message=message,
            data=result
        )
