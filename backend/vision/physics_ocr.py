"""Physics equation OCR using Gemini Vision for real-time equation detection."""

import os
import base64
import re
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

import google.generativeai as genai

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box for detected equation region."""
    x: int
    y: int
    width: int
    height: int
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "text": self.text
        }


@dataclass
class EquationDetectionResult:
    """Result of equation detection from camera frame."""
    equation: Optional[str]  # Detected equation in symbolic form
    equation_latex: Optional[str]  # LaTeX representation
    equation_type: str  # integral, derivative, equation, expression
    confidence: float  # 0.0 to 1.0
    boxes: List[Dict] = field(default_factory=list)  # Bounding boxes
    raw_text: str = ""  # Raw OCR text
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equation": self.equation,
            "equation_latex": self.equation_latex,
            "equation_type": self.equation_type,
            "confidence": self.confidence,
            "boxes": self.boxes,
            "raw_text": self.raw_text,
            "timestamp": self.timestamp.isoformat()
        }


class PhysicsOCR:
    """Real-time physics equation OCR using Gemini Vision."""

    # Common equation patterns
    INTEGRAL_PATTERN = re.compile(r'∫|\\int|integral', re.IGNORECASE)
    DERIVATIVE_PATTERN = re.compile(r'd/d[xyt]|\\frac\{d\}|derivative|d\'', re.IGNORECASE)
    EQUATION_PATTERN = re.compile(r'=')

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.available = True
        else:
            self.model = None
            self.available = False
            logger.warning("GEMINI_API_KEY not set - Physics OCR unavailable")

    async def detect_equation(self, image_base64: str) -> EquationDetectionResult:
        """Detect and extract equations from a camera frame.

        Args:
            image_base64: Base64 encoded image from webcam

        Returns:
            EquationDetectionResult with detected equation and metadata
        """
        if not self.available:
            return self._empty_result("Gemini API not configured")

        try:
            # Clean base64
            if ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]

            image_data = base64.b64decode(image_base64)

            # Gemini Vision prompt optimized for math equations
            prompt = """Analyze this image for mathematical equations or physics formulas.

Extract ALL equations you can find. For each equation:

1. EQUATION: The equation in plain text, computer-readable format
   - Use ^ for exponents (x^2, not x²)
   - Use sqrt() for square roots
   - Use standard operators: +, -, *, /, =
   - For integrals: write as "integral of [expression] dx"
   - For derivatives: write as "derivative of [expression] with respect to x"

2. LATEX: The equation in LaTeX format

3. TYPE: One of: integral, derivative, equation, expression, physics_formula

4. CONFIDENCE: Your confidence level (high, medium, low)

Format your response exactly like this:
EQUATION: [plain text equation]
LATEX: [latex version]
TYPE: [type]
CONFIDENCE: [level]

If multiple equations are found, list them separately.
If no equations are visible, respond with:
EQUATION: none
TYPE: none
CONFIDENCE: low"""

            # Call Gemini Vision
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content([
                    {"mime_type": "image/jpeg", "data": image_data},
                    prompt
                ])
            )

            # Parse response
            result = self._parse_response(response.text)
            return result

        except Exception as e:
            logger.error(f"Physics OCR error: {e}")
            return self._empty_result(f"OCR failed: {str(e)}")

    def _parse_response(self, response_text: str) -> EquationDetectionResult:
        """Parse Gemini response into EquationDetectionResult."""
        equation = None
        latex = None
        eq_type = "unknown"
        confidence = 0.0

        # Extract equation
        eq_match = re.search(r'EQUATION:\s*(.+?)(?=\n|LATEX:|$)', response_text, re.IGNORECASE)
        if eq_match:
            equation = eq_match.group(1).strip()
            if equation.lower() == 'none':
                equation = None

        # Extract LaTeX
        latex_match = re.search(r'LATEX:\s*(.+?)(?=\n|TYPE:|$)', response_text, re.IGNORECASE)
        if latex_match:
            latex = latex_match.group(1).strip()

        # Extract type
        type_match = re.search(r'TYPE:\s*(\w+)', response_text, re.IGNORECASE)
        if type_match:
            eq_type = type_match.group(1).lower()

        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*(\w+)', response_text, re.IGNORECASE)
        if conf_match:
            conf_level = conf_match.group(1).lower()
            confidence = {'high': 0.9, 'medium': 0.7, 'low': 0.4}.get(conf_level, 0.5)

        # Normalize equation for parsing
        if equation:
            equation = self._normalize_equation(equation)

        return EquationDetectionResult(
            equation=equation,
            equation_latex=latex,
            equation_type=eq_type,
            confidence=confidence,
            raw_text=response_text,
            boxes=[]  # Bounding boxes would need additional processing
        )

    def _normalize_equation(self, equation: str) -> str:
        """Normalize equation text for the physics engine."""
        result = equation

        # Common normalizations
        replacements = [
            # Unicode to ASCII
            ('²', '^2'),
            ('³', '^3'),
            ('⁴', '^4'),
            ('√', 'sqrt'),
            ('×', '*'),
            ('÷', '/'),
            ('π', 'pi'),
            ('∞', 'infinity'),
            # LaTeX remnants
            (r'\cdot', '*'),
            (r'\times', '*'),
            (r'\div', '/'),
            (r'\pi', 'pi'),
            (r'\sqrt', 'sqrt'),
            # Common OCR errors
            ('1ntegral', 'integral'),
            ('d1/dx', 'd/dx'),
        ]

        for old, new in replacements:
            result = result.replace(old, new)

        # Clean whitespace
        result = ' '.join(result.split())

        return result

    def _empty_result(self, message: str) -> EquationDetectionResult:
        """Return empty result with error message."""
        return EquationDetectionResult(
            equation=None,
            equation_latex=None,
            equation_type="none",
            confidence=0.0,
            raw_text=message,
            boxes=[]
        )

    def detect_equation_type(self, text: str) -> str:
        """Detect the type of equation from text."""
        text_lower = text.lower()

        if self.INTEGRAL_PATTERN.search(text_lower):
            return "integral"
        elif self.DERIVATIVE_PATTERN.search(text_lower):
            return "derivative"
        elif '=' in text:
            return "equation"
        else:
            return "expression"


# Global instance
physics_ocr = PhysicsOCR()
