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

    def analyze_scene(self, image_base64: str, query: Optional[str] = None) -> dict:
        """
        General scene analysis - describe what's in the image.

        Args:
            image_base64: Base64 encoded image string
            query: Optional specific question about the image

        Returns:
            dict with keys: description, objects, text_found, suggestions, error
        """
        if not self.client:
            return {
                "description": None,
                "objects": [],
                "text_found": None,
                "suggestions": [],
                "error": "OPENROUTER_API_KEY not configured"
            }

        try:
            image_type = self._detect_image_type(image_base64)
            clean_base64 = self._clean_base64(image_base64)
            image_url = f"data:image/{image_type};base64,{clean_base64}"

            # Customize prompt based on query
            if query:
                prompt = f"""Analyze this image to answer: {query}

Provide a helpful, voice-friendly response. Be concise but informative.
If the image contains text, include relevant parts.
If you can identify objects, mention them.
Suggest follow-up actions if appropriate."""
            else:
                prompt = """Describe what you see in this image. Be concise and voice-friendly.

Include:
1. SCENE: What is this? (1-2 sentences)
2. OBJECTS: Key objects or people visible (comma-separated list)
3. TEXT: Any visible text (summarize if long)
4. SUGGESTIONS: What might the user want to do? (1-2 suggestions)

Format as:
SCENE: [description]
OBJECTS: [list]
TEXT: [any text found, or "None"]
SUGGESTIONS: [helpful suggestions]"""

            response = self.client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=1024
            )

            response_text = response.choices[0].message.content

            # If custom query, return simple response
            if query:
                return {
                    "description": response_text,
                    "objects": [],
                    "text_found": None,
                    "suggestions": [],
                    "error": None
                }

            # Parse structured response
            return self._parse_scene_response(response_text)

        except Exception as e:
            return {
                "description": None,
                "objects": [],
                "text_found": None,
                "suggestions": [],
                "error": f"Vision API error: {str(e)}"
            }

    def translate_text(self, image_base64: str, target_language: str = "English") -> dict:
        """
        OCR and translate text from an image.

        Args:
            image_base64: Base64 encoded image string
            target_language: Language to translate to

        Returns:
            dict with keys: original_text, translated_text, source_language, error
        """
        if not self.client:
            return {
                "original_text": None,
                "translated_text": None,
                "source_language": None,
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
                                "text": f"""Extract and translate the text in this image to {target_language}.

Format your response as:
ORIGINAL: [original text]
LANGUAGE: [detected source language]
TRANSLATION: [translated text in {target_language}]

If no text is found, respond with "NO_TEXT_FOUND"."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=2048
            )

            response_text = response.choices[0].message.content

            if "NO_TEXT_FOUND" in response_text:
                return {
                    "original_text": None,
                    "translated_text": None,
                    "source_language": None,
                    "error": "No text found in image"
                }

            return self._parse_translation_response(response_text)

        except Exception as e:
            return {
                "original_text": None,
                "translated_text": None,
                "source_language": None,
                "error": f"Vision API error: {str(e)}"
            }

    def identify_object(self, image_base64: str, object_type: Optional[str] = None) -> dict:
        """
        Identify objects in the image (plants, animals, products, etc.).

        Args:
            image_base64: Base64 encoded image string
            object_type: Optional hint about what to identify (plant, bird, product, etc.)

        Returns:
            dict with keys: name, category, confidence, details, error
        """
        if not self.client:
            return {
                "name": None,
                "category": None,
                "confidence": 0.0,
                "details": None,
                "error": "OPENROUTER_API_KEY not configured"
            }

        try:
            image_type = self._detect_image_type(image_base64)
            clean_base64 = self._clean_base64(image_base64)
            image_url = f"data:image/{image_type};base64,{clean_base64}"

            type_hint = f" Focus on identifying the {object_type}." if object_type else ""

            response = self.client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Identify the main subject in this image.{type_hint}

Provide:
NAME: [specific name or species if possible]
CATEGORY: [general category - plant, animal, product, food, vehicle, etc.]
CONFIDENCE: [0.0-1.0 how confident you are]
DETAILS: [brief interesting facts or useful information]

Be voice-friendly and concise."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=512
            )

            response_text = response.choices[0].message.content
            return self._parse_identification_response(response_text)

        except Exception as e:
            return {
                "name": None,
                "category": None,
                "confidence": 0.0,
                "details": None,
                "error": f"Vision API error: {str(e)}"
            }

    def analyze_screenshot(self, image_base64: str, context: str = "") -> dict:
        """
        Analyze a screenshot - detect errors, UI elements, help with forms.

        Args:
            image_base64: Base64 encoded image string
            context: What the user needs help with

        Returns:
            dict with keys: screen_type, content_summary, errors, suggestions, error
        """
        if not self.client:
            return {
                "screen_type": None,
                "content_summary": None,
                "errors": [],
                "suggestions": [],
                "error": "OPENROUTER_API_KEY not configured"
            }

        try:
            image_type = self._detect_image_type(image_base64)
            clean_base64 = self._clean_base64(image_base64)
            image_url = f"data:image/{image_type};base64,{clean_base64}"

            context_prompt = f"\nUser context: {context}" if context else ""

            response = self.client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Analyze this screenshot.{context_prompt}

Provide:
SCREEN_TYPE: [app/website/error/form/settings/terminal/etc.]
SUMMARY: [brief description of what's on screen]
ERRORS: [any error messages or issues visible, one per line with "- "]
SUGGESTIONS: [how to fix errors or what actions to take, one per line with "- "]

Be concise and helpful. If there are errors, explain them simply."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=1024
            )

            response_text = response.choices[0].message.content
            return self._parse_screenshot_response(response_text)

        except Exception as e:
            return {
                "screen_type": None,
                "content_summary": None,
                "errors": [],
                "suggestions": [],
                "error": f"Vision API error: {str(e)}"
            }

    def _parse_scene_response(self, response_text: str) -> dict:
        """Parse scene analysis response."""
        description = None
        objects = []
        text_found = None
        suggestions = []

        # Parse SCENE
        scene_match = re.search(r'SCENE:\s*(.+?)(?=OBJECTS:|$)', response_text, re.DOTALL)
        if scene_match:
            description = scene_match.group(1).strip()

        # Parse OBJECTS
        obj_match = re.search(r'OBJECTS:\s*(.+?)(?=TEXT:|$)', response_text, re.DOTALL)
        if obj_match:
            objects = [o.strip() for o in obj_match.group(1).split(',') if o.strip()]

        # Parse TEXT
        text_match = re.search(r'TEXT:\s*(.+?)(?=SUGGESTIONS:|$)', response_text, re.DOTALL)
        if text_match:
            text = text_match.group(1).strip()
            if text.lower() != "none":
                text_found = text

        # Parse SUGGESTIONS
        sug_match = re.search(r'SUGGESTIONS:\s*(.+?)$', response_text, re.DOTALL)
        if sug_match:
            suggestions = [s.strip() for s in sug_match.group(1).split('\n') if s.strip()]

        return {
            "description": description,
            "objects": objects,
            "text_found": text_found,
            "suggestions": suggestions,
            "error": None
        }

    def _parse_translation_response(self, response_text: str) -> dict:
        """Parse translation response."""
        original = None
        translated = None
        language = None

        orig_match = re.search(r'ORIGINAL:\s*(.+?)(?=LANGUAGE:|$)', response_text, re.DOTALL)
        if orig_match:
            original = orig_match.group(1).strip()

        lang_match = re.search(r'LANGUAGE:\s*(.+?)(?=TRANSLATION:|$)', response_text, re.DOTALL)
        if lang_match:
            language = lang_match.group(1).strip()

        trans_match = re.search(r'TRANSLATION:\s*(.+?)$', response_text, re.DOTALL)
        if trans_match:
            translated = trans_match.group(1).strip()

        return {
            "original_text": original,
            "translated_text": translated,
            "source_language": language,
            "error": None
        }

    def _parse_identification_response(self, response_text: str) -> dict:
        """Parse object identification response."""
        name = None
        category = None
        confidence = 0.8
        details = None

        name_match = re.search(r'NAME:\s*(.+?)(?=CATEGORY:|$)', response_text, re.DOTALL)
        if name_match:
            name = name_match.group(1).strip()

        cat_match = re.search(r'CATEGORY:\s*(.+?)(?=CONFIDENCE:|$)', response_text, re.DOTALL)
        if cat_match:
            category = cat_match.group(1).strip()

        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response_text)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                pass

        det_match = re.search(r'DETAILS:\s*(.+?)$', response_text, re.DOTALL)
        if det_match:
            details = det_match.group(1).strip()

        return {
            "name": name,
            "category": category,
            "confidence": confidence,
            "details": details,
            "error": None
        }

    def _parse_screenshot_response(self, response_text: str) -> dict:
        """Parse screenshot analysis response."""
        screen_type = None
        summary = None
        errors = []
        suggestions = []

        type_match = re.search(r'SCREEN_TYPE:\s*(.+?)(?=SUMMARY:|$)', response_text, re.DOTALL)
        if type_match:
            screen_type = type_match.group(1).strip()

        sum_match = re.search(r'SUMMARY:\s*(.+?)(?=ERRORS:|$)', response_text, re.DOTALL)
        if sum_match:
            summary = sum_match.group(1).strip()

        err_match = re.search(r'ERRORS:\s*(.+?)(?=SUGGESTIONS:|$)', response_text, re.DOTALL)
        if err_match:
            errors = [e.strip().lstrip('- ') for e in err_match.group(1).split('\n') if e.strip().startswith('-')]

        sug_match = re.search(r'SUGGESTIONS:\s*(.+?)$', response_text, re.DOTALL)
        if sug_match:
            suggestions = [s.strip().lstrip('- ') for s in sug_match.group(1).split('\n') if s.strip().startswith('-')]

        return {
            "screen_type": screen_type,
            "content_summary": summary,
            "errors": errors,
            "suggestions": suggestions,
            "error": None
        }
