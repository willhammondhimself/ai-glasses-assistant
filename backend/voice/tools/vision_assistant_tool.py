"""Vision Assistant voice tool - "What am I looking at?" functionality."""
import logging
import re
from typing import Optional
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class VisionAssistantTool(VoiceTool):
    """Voice-controlled vision assistant for scene analysis, OCR, and object identification."""

    name = "vision"
    description = "Analyze what the camera sees - describe scenes, read text, identify objects"

    keywords = [
        r"\bwhat.*(this|that|looking\s+at|see|is\s+this)\b",
        r"\bidentify\b",
        r"\bdescribe\b",
        r"\bread\s+this\b",
        r"\btranslate\s+this\b",
        r"\bwhat\s+kind\s+of\b",
        r"\bwhat\s+type\s+of\b",
        r"\bwhat\s+is\s+that\b",
        r"\btell\s+me\s+about\s+this\b",
        r"\banalyze\s+this\b",
        r"\bwhat\s+does\s+this\s+say\b",
        r"\bhelp\s+me\s+(read|understand)\b",
    ]

    priority = 10  # High priority for vision queries

    def __init__(self):
        self._vision_engine = None
        self._last_image: Optional[str] = None

    def _get_vision_engine(self):
        """Get vision engine (lazy load)."""
        if self._vision_engine is None:
            from backend.vision.engine import VisionEngine
            self._vision_engine = VisionEngine()
        return self._vision_engine

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute vision analysis.

        Args:
            query: The user's voice query
            **kwargs: Additional context including 'image' (base64)

        Returns:
            VoiceToolResult with analysis
        """
        query_lower = query.lower()
        image_base64 = kwargs.get("image") or self._last_image

        if not image_base64:
            return VoiceToolResult(
                success=False,
                message="I need an image to analyze. Please take a photo or share your screen.",
                data={"needs_image": True}
            )

        # Store for follow-up queries
        self._last_image = image_base64
        engine = self._get_vision_engine()

        try:
            # Route to appropriate analysis type
            if self._is_translation_request(query_lower):
                return await self._handle_translation(engine, image_base64, query_lower)

            if self._is_read_request(query_lower):
                return await self._handle_read(engine, image_base64)

            if self._is_identify_request(query_lower):
                object_type = self._extract_object_type(query_lower)
                return await self._handle_identify(engine, image_base64, object_type)

            # Default: general scene analysis
            return await self._handle_scene_analysis(engine, image_base64, query)

        except Exception as e:
            logger.error(f"Vision tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble analyzing the image.",
                data={"error": str(e)}
            )

    def _is_translation_request(self, query: str) -> bool:
        patterns = ["translate", "in english", "what does this say", "foreign"]
        return any(p in query for p in patterns)

    def _is_read_request(self, query: str) -> bool:
        patterns = ["read this", "read that", "what does it say", "ocr", "text in this"]
        return any(p in query for p in patterns)

    def _is_identify_request(self, query: str) -> bool:
        patterns = ["identify", "what kind of", "what type of", "what species", "what is that"]
        return any(p in query for p in patterns)

    def _extract_object_type(self, query: str) -> Optional[str]:
        """Extract object type hint from query."""
        patterns = [
            r"what\s+(?:kind|type|species)\s+of\s+(\w+)",
            r"identify\s+(?:this|that|the)\s+(\w+)",
            r"what\s+(\w+)\s+is\s+(?:this|that)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)
        return None

    async def _handle_scene_analysis(
        self,
        engine,
        image_base64: str,
        query: str
    ) -> VoiceToolResult:
        """Handle general scene analysis."""
        # Check if it's a specific question or general "what is this"
        is_specific_question = any(word in query.lower() for word in [
            "how", "why", "can", "should", "where", "which"
        ])

        result = engine.analyze_scene(
            image_base64,
            query=query if is_specific_question else None
        )

        if result.get("error"):
            return VoiceToolResult(
                success=False,
                message=f"Vision analysis failed: {result['error']}",
                data=result
            )

        # Build voice-friendly response
        if is_specific_question:
            message = result["description"]
        else:
            message = result["description"] or "I see something, but I'm not sure what it is."

            # Add suggestions if available
            if result.get("suggestions"):
                message += f" {result['suggestions'][0]}"

        return VoiceToolResult(
            success=True,
            message=message,
            data=result
        )

    async def _handle_read(self, engine, image_base64: str) -> VoiceToolResult:
        """Handle OCR/text reading request."""
        result = engine.extract_text(image_base64)

        if result.get("error"):
            return VoiceToolResult(
                success=False,
                message="I couldn't read any text from this image.",
                data=result
            )

        text = result.get("text", "")
        if not text:
            return VoiceToolResult(
                success=True,
                message="I don't see any readable text in this image.",
                data=result
            )

        # Truncate for voice if too long
        if len(text) > 500:
            text = text[:500] + "... There's more text. Would you like me to continue?"

        return VoiceToolResult(
            success=True,
            message=f"I see: {text}",
            data=result
        )

    async def _handle_translation(
        self,
        engine,
        image_base64: str,
        query: str
    ) -> VoiceToolResult:
        """Handle translation request."""
        # Detect target language from query
        target = "English"
        lang_patterns = {
            "spanish": "Spanish",
            "french": "French",
            "german": "German",
            "japanese": "Japanese",
            "chinese": "Chinese",
            "korean": "Korean",
            "italian": "Italian",
            "portuguese": "Portuguese",
        }
        for pattern, lang in lang_patterns.items():
            if pattern in query:
                target = lang
                break

        result = engine.translate_text(image_base64, target)

        if result.get("error"):
            return VoiceToolResult(
                success=False,
                message="I couldn't translate this. The text might not be visible.",
                data=result
            )

        if not result.get("translated_text"):
            return VoiceToolResult(
                success=True,
                message="I don't see any text to translate.",
                data=result
            )

        source_lang = result.get("source_language", "unknown language")
        translated = result["translated_text"]

        return VoiceToolResult(
            success=True,
            message=f"From {source_lang}: {translated}",
            data=result
        )

    async def _handle_identify(
        self,
        engine,
        image_base64: str,
        object_type: Optional[str]
    ) -> VoiceToolResult:
        """Handle object identification request."""
        result = engine.identify_object(image_base64, object_type)

        if result.get("error"):
            return VoiceToolResult(
                success=False,
                message="I couldn't identify this clearly.",
                data=result
            )

        name = result.get("name", "Unknown")
        category = result.get("category", "")
        details = result.get("details", "")

        # Build natural response
        if category:
            message = f"This appears to be a {name} ({category})."
        else:
            message = f"This looks like {name}."

        if details:
            message += f" {details}"

        return VoiceToolResult(
            success=True,
            message=message,
            data=result
        )
