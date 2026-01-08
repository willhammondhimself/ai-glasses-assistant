"""
Gemini Flash Client for OCR and Vision.
Primary use: Extract poker cards from camera images.
Latency: ~1 second
Cost: $0.002 per request

Uses the new google-genai SDK (unified SDK).
"""
import os
import json
import asyncio
import logging
import re
import base64
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# New unified SDK import
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai not installed. Run: pip install google-genai")


@dataclass
class CardExtraction:
    """Result of card extraction from image."""
    hole_cards: list  # ["Ah", "Kc"]
    board: list       # ["9s", "7h", "3d"]
    pot_size: float   # In big blinds
    bet_facing: float # In big blinds
    confidence: float # 0.0-1.0
    raw_response: str = ""


@dataclass
class MathExtraction:
    """Result of math equation extraction from image."""
    equation: str
    problem_type: str  # "algebra", "calculus", "word_problem"
    variables: list
    confidence: float
    raw_response: str = ""


@dataclass
class CodeExtraction:
    """Result of code extraction from image."""
    code: str
    language: str
    error_visible: bool
    confidence: float
    raw_response: str = ""


class GeminiClient:
    """
    Gemini Flash for OCR and vision tasks.

    Primary use cases:
    - Extract poker cards from camera images
    - Read text from glasses camera
    - Quick factual lookups

    Model: gemini-2.0-flash
    Latency: ~1 second
    Cost: ~$0.002 per request
    """

    CARD_EXTRACTION_PROMPT = """Analyze this poker table image and extract the visible cards.

EXTRACT:
1. Hero's hole cards (the 2 cards closest to camera/bottom of image)
2. Community board cards (0-5 cards in the center)
3. Pot size (if visible, estimate in big blinds)
4. Current bet to call (if visible, in big blinds)

CARD FORMAT: Use standard notation like "Ah" (Ace of hearts), "Ks" (King of spades), "9d" (9 of diamonds), "2c" (2 of clubs).

Suits: h=hearts, d=diamonds, c=clubs, s=spades

RESPOND IN JSON:
{
    "hole_cards": ["Ah", "Kc"],
    "board": ["9s", "7h", "3d"],
    "pot_size": 12,
    "bet_facing": 8,
    "confidence": 0.95
}

If you cannot see certain cards clearly, use "?" for unknown cards.
If pot/bet sizes aren't visible, estimate or use 0."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self._initialized = False
        self._model = "gemini-2.0-flash"

        if not GENAI_AVAILABLE:
            logger.error("google-genai not available")
            return

        if not self.api_key:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY env var.")
            return

        try:
            self.client = genai.Client(api_key=self.api_key)
            self._initialized = True
            logger.info("Gemini client initialized with new SDK")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")

    @property
    def is_available(self) -> bool:
        """Check if client is ready to use."""
        return self._initialized and self.client is not None

    async def extract_cards(self, image_data: bytes) -> CardExtraction:
        """
        Extract poker cards from camera image.

        Args:
            image_data: Raw image bytes from camera

        Returns:
            CardExtraction with hole cards, board, pot size, bet facing
        """
        if not self.is_available:
            return CardExtraction(
                hole_cards=["?", "?"],
                board=[],
                pot_size=0,
                bet_facing=0,
                confidence=0.0,
                raw_response="Gemini client not available"
            )

        try:
            # Create image part for new SDK
            image_part = types.Part.from_bytes(
                data=image_data,
                mime_type="image/jpeg"
            )

            # Generate content with image and prompt
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self._model,
                contents=[self.CARD_EXTRACTION_PROMPT, image_part]
            )

            raw_text = response.text
            return self._parse_card_response(raw_text)

        except Exception as e:
            logger.error(f"Card extraction failed: {e}")
            return CardExtraction(
                hole_cards=["?", "?"],
                board=[],
                pot_size=0,
                bet_facing=0,
                confidence=0.0,
                raw_response=str(e)
            )

    def _parse_card_response(self, response: str) -> CardExtraction:
        """Parse JSON response from Gemini."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            return CardExtraction(
                hole_cards=data.get("hole_cards", ["?", "?"]),
                board=data.get("board", []),
                pot_size=float(data.get("pot_size", 0)),
                bet_facing=float(data.get("bet_facing", 0)),
                confidence=float(data.get("confidence", 0.5)),
                raw_response=response
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse card response: {e}")
            return CardExtraction(
                hole_cards=["?", "?"],
                board=[],
                pot_size=0,
                bet_facing=0,
                confidence=0.0,
                raw_response=response
            )

    async def analyze_image(self, image_data: bytes, prompt: str) -> str:
        """
        General image analysis with custom prompt.

        Args:
            image_data: Raw image bytes
            prompt: Analysis prompt

        Returns:
            Gemini's response text
        """
        if not self.is_available:
            return "Gemini client not available"

        try:
            image_part = types.Part.from_bytes(
                data=image_data,
                mime_type="image/jpeg"
            )

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self._model,
                contents=[prompt, image_part]
            )

            return response.text

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return f"Error: {e}"

    async def quick_query(self, prompt: str) -> str:
        """
        Quick text query without image.

        Args:
            prompt: Question or request

        Returns:
            Gemini's response text
        """
        if not self.is_available:
            return "Gemini client not available"

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self._model,
                contents=prompt
            )
            return response.text

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Error: {e}"

    async def extract_math(self, image_data: bytes) -> MathExtraction:
        """
        Extract mathematical equation from image.

        Args:
            image_data: Raw image bytes

        Returns:
            MathExtraction with equation and type
        """
        if not self.is_available:
            return MathExtraction(
                equation="",
                problem_type="unknown",
                variables=[],
                confidence=0.0,
                raw_response="Gemini client not available"
            )

        prompt = """Extract the mathematical content from this image.

Identify:
1. The equation or expression exactly as shown
2. Problem type: arithmetic, algebra, polynomial, derivative, integral, trigonometry, word_problem, proof
3. Variables used (e.g., x, y, z)

RESPOND IN JSON ONLY:
{"equation": "2x + 5 = 15", "problem_type": "algebra",
 "variables": ["x"], "confidence": 0.95}"""

        try:
            image_part = types.Part.from_bytes(
                data=image_data,
                mime_type="image/jpeg"
            )

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self._model,
                contents=[prompt, image_part]
            )

            raw_text = response.text
            return self._parse_math_response(raw_text)

        except Exception as e:
            logger.error(f"Math extraction failed: {e}")
            return MathExtraction(
                equation="",
                problem_type="unknown",
                variables=[],
                confidence=0.0,
                raw_response=str(e)
            )

    def _parse_math_response(self, response: str) -> MathExtraction:
        """Parse JSON response for math extraction."""
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            return MathExtraction(
                equation=data.get("equation", ""),
                problem_type=data.get("problem_type", "unknown"),
                variables=data.get("variables", []),
                confidence=float(data.get("confidence", 0.5)),
                raw_response=response
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse math response: {e}")
            return MathExtraction(
                equation="",
                problem_type="unknown",
                variables=[],
                confidence=0.0,
                raw_response=response
            )

    async def extract_code(self, image_data: bytes) -> CodeExtraction:
        """
        Extract code snippet from image.

        Args:
            image_data: Raw image bytes

        Returns:
            CodeExtraction with code and language
        """
        if not self.is_available:
            return CodeExtraction(
                code="",
                language="unknown",
                error_visible=False,
                confidence=0.0,
                raw_response="Gemini client not available"
            )

        prompt = """Extract the code from this image.

Identify:
1. The code exactly as shown (preserve indentation)
2. Programming language (python, javascript, java, c++, etc.)
3. Whether an error message is visible in the image

RESPOND IN JSON ONLY:
{"code": "def hello():\\n    print('world')",
 "language": "python", "error_visible": false, "confidence": 0.95}"""

        try:
            image_part = types.Part.from_bytes(
                data=image_data,
                mime_type="image/jpeg"
            )

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self._model,
                contents=[prompt, image_part]
            )

            raw_text = response.text
            return self._parse_code_response(raw_text)

        except Exception as e:
            logger.error(f"Code extraction failed: {e}")
            return CodeExtraction(
                code="",
                language="unknown",
                error_visible=False,
                confidence=0.0,
                raw_response=str(e)
            )

    def _parse_code_response(self, response: str) -> CodeExtraction:
        """Parse JSON response for code extraction."""
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            return CodeExtraction(
                code=data.get("code", ""),
                language=data.get("language", "unknown"),
                error_visible=bool(data.get("error_visible", False)),
                confidence=float(data.get("confidence", 0.5)),
                raw_response=response
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse code response: {e}")
            return CodeExtraction(
                code="",
                language="unknown",
                error_visible=False,
                confidence=0.0,
                raw_response=response
            )


# Test
async def test_gemini():
    """Test Gemini client."""
    print("=== Gemini Client Test (New SDK) ===\n")

    client = GeminiClient()

    if not client.is_available:
        print("Gemini not available. Set GEMINI_API_KEY env var.")
        return

    # Test quick query
    response = await client.quick_query("What is 2+2? Answer with just the number.")
    print(f"Quick query test: {response}")


if __name__ == "__main__":
    asyncio.run(test_gemini())
