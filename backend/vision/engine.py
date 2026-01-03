"""
VisionEngine: Extracts mathematical equations from images using GPT-4o Vision via OpenRouter.

Uses OpenRouter API for cost-effective access to GPT-4o Vision model.
"""

import os
import base64
import re
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class VisionEngine:
    """Vision processing engine for equation extraction from images."""

    def __init__(self):
        self.client: Optional[OpenAI] = None
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )

    def extract_equation(self, image_base64: str) -> dict:
        """
        Extract mathematical equations from an image.

        Args:
            image_base64: Base64 encoded image string (PNG, JPG, etc.)

        Returns:
            dict with keys: latex, raw_text, confidence, error
        """
        if not self.client:
            return {
                "latex": None,
                "raw_text": None,
                "confidence": 0.0,
                "error": "OPENROUTER_API_KEY not configured"
            }

        try:
            # Detect image type from base64 header or assume PNG
            image_type = self._detect_image_type(image_base64)

            # Clean the base64 string (remove data URL prefix if present)
            clean_base64 = self._clean_base64(image_base64)

            # Construct the image URL
            image_url = f"data:image/{image_type};base64,{clean_base64}"

            response = self.client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this image and extract any mathematical equations, formulas, or expressions.

Please provide:
1. LATEX: The equation(s) in LaTeX format
2. RAW_TEXT: The equation(s) in plain text format
3. CONFIDENCE: Your confidence level (0.0 to 1.0) in the extraction accuracy

If there are multiple equations, separate them with newlines.
If no mathematical content is found, respond with "NO_MATH_CONTENT".

Format your response exactly as:
LATEX: [latex equations]
RAW_TEXT: [plain text equations]
CONFIDENCE: [0.0-1.0]"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024
            )

            response_text = response.choices[0].message.content

            # Parse the response
            return self._parse_extraction_response(response_text)

        except Exception as e:
            return {
                "latex": None,
                "raw_text": None,
                "confidence": 0.0,
                "error": f"Vision API error: {str(e)}"
            }

    def extract_text(self, image_base64: str) -> dict:
        """
        General OCR - extract all text from an image.

        Args:
            image_base64: Base64 encoded image string

        Returns:
            dict with keys: text, confidence, error
        """
        if not self.client:
            return {
                "text": None,
                "confidence": 0.0,
                "error": "OPENROUTER_API_KEY not configured"
            }

        try:
            image_type = self._detect_image_type(image_base64)
            clean_base64 = self._clean_base64(image_base64)
            image_url = f"data:image/{image_type};base64,{clean_base64}"

            response = self.client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all visible text from this image. Return only the extracted text, preserving layout where possible."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2048
            )

            text = response.choices[0].message.content

            return {
                "text": text,
                "confidence": 0.9,  # GPT-4o is generally reliable
                "error": None
            }

        except Exception as e:
            return {
                "text": None,
                "confidence": 0.0,
                "error": f"Vision API error: {str(e)}"
            }

    def analyze_diagram(self, image_base64: str, context: str = "") -> dict:
        """
        Analyze a diagram or figure (useful for CS/math diagrams).

        Args:
            image_base64: Base64 encoded image string
            context: Optional context about what the diagram represents

        Returns:
            dict with keys: description, elements, relationships, error
        """
        if not self.client:
            return {
                "description": None,
                "elements": [],
                "relationships": [],
                "error": "OPENROUTER_API_KEY not configured"
            }

        try:
            image_type = self._detect_image_type(image_base64)
            clean_base64 = self._clean_base64(image_base64)
            image_url = f"data:image/{image_type};base64,{clean_base64}"

            context_prompt = f"\nContext: {context}" if context else ""

            response = self.client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Analyze this diagram or figure.{context_prompt}

Provide:
1. DESCRIPTION: A brief description of what the diagram shows
2. ELEMENTS: List the key elements/components (one per line, prefixed with "- ")
3. RELATIONSHIPS: Describe how elements relate to each other (one per line, prefixed with "- ")

Format your response exactly as:
DESCRIPTION: [description]
ELEMENTS:
- [element 1]
- [element 2]
RELATIONSHIPS:
- [relationship 1]
- [relationship 2]"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024
            )

            response_text = response.choices[0].message.content

            return self._parse_diagram_response(response_text)

        except Exception as e:
            return {
                "description": None,
                "elements": [],
                "relationships": [],
                "error": f"Vision API error: {str(e)}"
            }

    def _detect_image_type(self, base64_string: str) -> str:
        """Detect image type from base64 string."""
        if base64_string.startswith("data:image/"):
            # Extract type from data URL
            match = re.match(r'data:image/(\w+);', base64_string)
            if match:
                return match.group(1)

        # Try to detect from base64 magic bytes
        try:
            clean = self._clean_base64(base64_string)
            decoded = base64.b64decode(clean[:20])

            if decoded.startswith(b'\x89PNG'):
                return 'png'
            elif decoded.startswith(b'\xff\xd8\xff'):
                return 'jpeg'
            elif decoded.startswith(b'GIF'):
                return 'gif'
            elif decoded.startswith(b'RIFF') and b'WEBP' in decoded:
                return 'webp'
        except Exception:
            pass

        # Default to PNG
        return 'png'

    def _clean_base64(self, base64_string: str) -> str:
        """Remove data URL prefix if present."""
        if ',' in base64_string and base64_string.startswith('data:'):
            return base64_string.split(',', 1)[1]
        return base64_string

    def _parse_extraction_response(self, response_text: str) -> dict:
        """Parse the equation extraction response."""
        if "NO_MATH_CONTENT" in response_text:
            return {
                "latex": None,
                "raw_text": None,
                "confidence": 1.0,
                "error": "No mathematical content found in image"
            }

        latex = None
        raw_text = None
        confidence = 0.8  # Default confidence

        # Parse LATEX
        latex_match = re.search(r'LATEX:\s*(.+?)(?=RAW_TEXT:|$)', response_text, re.DOTALL)
        if latex_match:
            latex = latex_match.group(1).strip()

        # Parse RAW_TEXT
        raw_match = re.search(r'RAW_TEXT:\s*(.+?)(?=CONFIDENCE:|$)', response_text, re.DOTALL)
        if raw_match:
            raw_text = raw_match.group(1).strip()

        # Parse CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response_text)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass

        return {
            "latex": latex,
            "raw_text": raw_text,
            "confidence": confidence,
            "error": None
        }

    def _parse_diagram_response(self, response_text: str) -> dict:
        """Parse the diagram analysis response."""
        description = None
        elements = []
        relationships = []

        # Parse DESCRIPTION
        desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=ELEMENTS:|$)', response_text, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()

        # Parse ELEMENTS
        elem_match = re.search(r'ELEMENTS:\s*(.+?)(?=RELATIONSHIPS:|$)', response_text, re.DOTALL)
        if elem_match:
            elem_text = elem_match.group(1).strip()
            elements = [line.strip().lstrip('- ') for line in elem_text.split('\n') if line.strip().startswith('-')]

        # Parse RELATIONSHIPS
        rel_match = re.search(r'RELATIONSHIPS:\s*(.+?)$', response_text, re.DOTALL)
        if rel_match:
            rel_text = rel_match.group(1).strip()
            relationships = [line.strip().lstrip('- ') for line in rel_text.split('\n') if line.strip().startswith('-')]

        return {
            "description": description,
            "elements": elements,
            "relationships": relationships,
            "error": None
        }
