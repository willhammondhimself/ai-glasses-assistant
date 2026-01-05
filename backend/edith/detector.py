"""
EDITH Content Detectors

Specialized detectors for different content types.
Each detector handles classification and extraction for its domain.
"""

import logging
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from .scanner import ScanResult, DetectionType

# Import GeminiClient from phone_client
PHONE_CLIENT_PATH = Path(__file__).parent.parent.parent / "phone_client"
sys.path.insert(0, str(PHONE_CLIENT_PATH))

from api_clients.gemini_client import GeminiClient, MathExtraction, CodeExtraction, CardExtraction

# Import vision models
from backend.dashboard.models import (
    MathDetectionResult,
    CodeDetectionResult,
    PokerDetectionResult,
    TextDetectionResult,
)

logger = logging.getLogger(__name__)


class ContentDetector(ABC):
    """Base class for content detectors."""

    @property
    @abstractmethod
    def detection_type(self) -> DetectionType:
        """The type of content this detector handles."""
        pass

    @abstractmethod
    async def detect(
        self,
        image_data: bytes,
        quick_mode: bool = False,
    ) -> Optional[ScanResult]:
        """
        Detect content in image.

        Args:
            image_data: Raw image bytes
            quick_mode: If True, use fast heuristics only

        Returns:
            ScanResult if content found, None otherwise
        """
        pass

    @abstractmethod
    def extract_content(self, raw_result: Dict[str, Any]) -> str:
        """Extract structured content from raw detection result."""
        pass


class EquationDetector(ContentDetector):
    """
    Detects mathematical equations in images.

    Recognizes:
    - Handwritten equations
    - Printed math notation
    - LaTeX-style expressions
    - Calculator displays
    """

    # Common math symbols and patterns
    MATH_PATTERNS = [
        r'[0-9]+\s*[\+\-\×\÷\*\/]\s*[0-9]+',  # Basic arithmetic
        r'[a-z]\s*=\s*[0-9]+',                  # Variable assignment
        r'∫|∑|∏|√|∞',                           # Calculus symbols
        r'd[xy]/d[xy]',                         # Derivatives
        r'\([^)]+\)\s*[\+\-\*\/]',              # Parenthesized expressions
    ]

    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    @property
    def is_available(self) -> bool:
        """Check if Gemini vision is available."""
        return self.client.is_available

    @property
    def detection_type(self) -> DetectionType:
        return DetectionType.EQUATION

    async def detect(
        self,
        image_data: bytes,
        quick_mode: bool = False,
    ) -> Optional[ScanResult]:
        """Detect math equation in image using Gemini."""
        # Call through to Gemini
        extraction = await self.client.extract_math(image_data)

        # Convert to MathDetectionResult
        result = MathDetectionResult(
            equation=extraction.equation.strip(),
            problem_type=extraction.problem_type,
            variables=list(set(extraction.variables)),  # Unique variables
            confidence=max(0.0, min(1.0, extraction.confidence)),  # Clamp
            raw_response=extraction.raw_response
        )

        # Return None if no equation found
        if not result.equation:
            return None

        # Wrap in ScanResult for compatibility
        return ScanResult(
            detected=True,
            confidence=result.confidence,
            bounding_box=None,
            data=result.model_dump()
        )

    def extract_content(self, raw_result: Dict[str, Any]) -> str:
        """Extract equation as string."""
        return raw_result.get("equation", "")

    def classify_equation_type(self, equation: str) -> str:
        """Classify the type of math problem."""
        equation_lower = equation.lower()

        if any(s in equation for s in ['∫', 'integral']):
            return "calculus_integral"
        elif any(s in equation for s in ['d/d', "'"]):
            return "calculus_derivative"
        elif any(s in equation for s in ['∑', 'sigma', 'sum']):
            return "series"
        elif 'lim' in equation_lower:
            return "limit"
        elif any(s in equation for s in ['sin', 'cos', 'tan']):
            return "trigonometry"
        elif any(s in equation for s in ['log', 'ln', 'exp']):
            return "logarithm"
        elif '^' in equation or '²' in equation or '³' in equation:
            return "polynomial"
        elif any(s in equation for s in ['+', '-', '×', '÷', '*', '/']):
            return "arithmetic"
        else:
            return "algebra"


class CodeDetector(ContentDetector):
    """
    Detects code snippets in images.

    Recognizes:
    - Syntax-highlighted code
    - Terminal/console output
    - Error messages
    - IDE screenshots
    """

    # Language detection patterns
    LANGUAGE_PATTERNS = {
        "python": [
            r'def\s+\w+\s*\(',
            r'import\s+\w+',
            r'class\s+\w+:',
            r'if\s+__name__\s*==',
            r'print\s*\(',
        ],
        "javascript": [
            r'const\s+\w+\s*=',
            r'let\s+\w+\s*=',
            r'function\s+\w+\s*\(',
            r'=>',
            r'console\.log',
        ],
        "java": [
            r'public\s+class',
            r'private\s+\w+\s+\w+',
            r'System\.out\.println',
        ],
        "c++": [
            r'#include\s*<',
            r'std::',
            r'int\s+main\s*\(',
        ],
    }

    # Error pattern recognition
    ERROR_PATTERNS = [
        r'Error:',
        r'Exception',
        r'Traceback',
        r'undefined',
        r'null',
        r'TypeError',
        r'SyntaxError',
        r'ReferenceError',
    ]

    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    @property
    def is_available(self) -> bool:
        """Check if Gemini vision is available."""
        return self.client.is_available

    @property
    def detection_type(self) -> DetectionType:
        return DetectionType.CODE

    async def detect(
        self,
        image_data: bytes,
        quick_mode: bool = False,
    ) -> Optional[ScanResult]:
        """Detect code in image using Gemini."""
        extraction = await self.client.extract_code(image_data)

        result = CodeDetectionResult(
            code=extraction.code,  # Preserve indentation
            language=extraction.language.lower(),
            error_visible=extraction.error_visible,
            confidence=max(0.0, min(1.0, extraction.confidence)),
            raw_response=extraction.raw_response
        )

        # Return None if no code or confidence 0
        if not result.code or result.confidence == 0.0:
            return None

        return ScanResult(
            detected=True,
            confidence=result.confidence,
            bounding_box=None,
            data=result.model_dump()
        )

    def extract_content(self, raw_result: Dict[str, Any]) -> str:
        """Extract code as string."""
        return raw_result.get("code", "")

    def detect_language(self, code: str) -> str:
        """Detect programming language from code content."""
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code):
                    return lang
        return "unknown"

    def detect_error(self, code: str) -> Optional[Dict[str, str]]:
        """Detect if code contains an error message."""
        for pattern in self.ERROR_PATTERNS:
            match = re.search(pattern, code, re.IGNORECASE)
            if match:
                return {
                    "type": match.group(),
                    "context": code[max(0, match.start()-50):match.end()+100],
                }
        return None


class PokerDetector(ContentDetector):
    """
    Detects poker cards and game situations.

    Recognizes:
    - Playing cards (rank and suit)
    - Community board cards
    - Chip stacks (approximate)
    - Player positions
    """

    # Card patterns
    RANKS = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
    SUITS = ['♠', '♥', '♦', '♣', 's', 'h', 'd', 'c']

    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    @property
    def is_available(self) -> bool:
        """Check if Gemini vision is available."""
        return self.client.is_available

    @property
    def detection_type(self) -> DetectionType:
        return DetectionType.POKER_CARDS

    async def detect(
        self,
        image_data: bytes,
        quick_mode: bool = False,
    ) -> Optional[ScanResult]:
        """Detect poker cards and game state using Gemini."""
        extraction = await self.client.extract_cards(image_data)

        # Ensure exactly 2 hole cards
        hole_cards = extraction.hole_cards[:2]  # Truncate if > 2
        while len(hole_cards) < 2:
            hole_cards.append("?")  # Pad if < 2

        # Ensure board max 5 cards
        board = extraction.board[:5]

        result = PokerDetectionResult(
            hole_cards=hole_cards,
            board=board,
            pot_size_bb=max(0.0, extraction.pot_size),  # No negatives
            bet_facing_bb=max(0.0, extraction.bet_facing),
            confidence=max(0.0, min(1.0, extraction.confidence)),
            raw_response=extraction.raw_response
        )

        return ScanResult(
            detected=True,
            confidence=result.confidence,
            bounding_box=None,
            data=result.model_dump()
        )

    def extract_content(self, raw_result: Dict[str, Any]) -> str:
        """Extract cards as string representation."""
        cards = raw_result.get("hole_cards", []) + raw_result.get("board", [])
        return " ".join(cards)

    def parse_hand(self, hand_str: str) -> List[Tuple[str, str]]:
        """Parse hand string into (rank, suit) tuples."""
        cards = []
        # Pattern: rank + suit (e.g., "As", "Kh", "10d")
        pattern = r'([AKQJ]|10|[2-9])([shdc♠♥♦♣])'
        matches = re.findall(pattern, hand_str, re.IGNORECASE)
        for rank, suit in matches:
            cards.append((rank.upper(), suit.lower()))
        return cards

    def classify_hand_strength(
        self,
        hole_cards: List[Tuple[str, str]],
        board: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """Quick classification of hand strength."""
        if len(hole_cards) != 2:
            return "unknown"

        r1, s1 = hole_cards[0]
        r2, s2 = hole_cards[1]

        # Premium hands
        if r1 == r2 and r1 in ['A', 'K', 'Q', 'J']:
            return "premium_pair"
        if {r1, r2} == {'A', 'K'}:
            return "premium_broadway"
        if r1 == r2:
            return "pair"
        if s1 == s2:
            return "suited"
        return "offsuit"


class TextDetector(ContentDetector):
    """
    Detects general text content.

    Recognizes:
    - Printed text
    - Handwritten text
    - Signs and labels
    - Document pages
    """

    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    @property
    def is_available(self) -> bool:
        """Check if Gemini vision is available."""
        return self.client.is_available

    @property
    def detection_type(self) -> DetectionType:
        return DetectionType.TEXT

    async def detect(
        self,
        image_data: bytes,
        quick_mode: bool = False,
    ) -> Optional[ScanResult]:
        """Extract text from image using Gemini OCR."""
        # Use analyze_image with OCR-specific prompt
        prompt = "Extract all visible text from this image. Return only the text, preserving formatting and structure. Do not summarize."

        text = await self.client.analyze_image(image_data, prompt)

        # Estimate confidence (Gemini doesn't provide OCR confidence directly)
        confidence = 0.9 if text and len(text) > 10 else 0.5 if text else 0.0

        result = TextDetectionResult(
            text=text,
            confidence=confidence,
            raw_response=text
        )

        if not result.text:
            return None

        return ScanResult(
            detected=True,
            confidence=result.confidence,
            bounding_box=None,
            data=result.model_dump()
        )

    def extract_content(self, raw_result: Dict[str, Any]) -> str:
        """Extract text content."""
        return raw_result.get("text", "")

    def classify_text_type(self, text: str) -> str:
        """Classify the type of text content."""
        text_lower = text.lower()

        # Check for question patterns
        if '?' in text or text_lower.startswith(('what', 'how', 'why', 'when', 'where')):
            return "question"

        # Check for definition patterns
        if ' is ' in text_lower or ' means ' in text_lower:
            return "definition"

        # Check for list patterns
        if re.search(r'^\s*[\d•\-]\s*', text, re.MULTILINE):
            return "list"

        # Check word count for passage
        word_count = len(text.split())
        if word_count > 100:
            return "passage"
        elif word_count > 20:
            return "paragraph"

        return "short_text"


# Detector registry
DETECTORS: Dict[DetectionType, type] = {
    DetectionType.EQUATION: EquationDetector,
    DetectionType.CODE: CodeDetector,
    DetectionType.POKER_CARDS: PokerDetector,
    DetectionType.TEXT: TextDetector,
}


def get_detector(detection_type: DetectionType) -> ContentDetector:
    """Get detector instance for a detection type."""
    detector_class = DETECTORS.get(detection_type)
    if detector_class:
        return detector_class()
    raise ValueError(f"No detector for type: {detection_type}")
