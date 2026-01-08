"""Vision analysis for voice agent - REST API hybrid for image context injection."""
import base64
import logging
import os
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VisionRequest(BaseModel):
    """Request for vision analysis."""
    image: str  # Base64-encoded image
    context: Optional[str] = None  # Optional voice context
    analyze_type: Optional[str] = "general"  # general, text, objects, scene


class VisionResponse(BaseModel):
    """Response from vision analysis."""
    success: bool
    description: str  # Voice-friendly description
    details: Optional[dict] = None  # Detailed structured data


async def analyze_for_voice(
    image_base64: str,
    context: str = "",
    analyze_type: str = "general"
) -> VisionResponse:
    """Analyze an image and return a voice-friendly description.

    This is designed to inject visual context into the voice conversation.

    Args:
        image_base64: Base64-encoded image data
        context: Optional voice context (what the user asked about)
        analyze_type: Type of analysis (general, text, objects, scene)

    Returns:
        VisionResponse with voice-friendly description
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return VisionResponse(
            success=False,
            description="Vision analysis is not configured."
        )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        # Build prompt based on analysis type
        prompts = {
            "general": "Describe what you see in this image briefly. Focus on the most important elements. Keep your response to 2-3 sentences suitable for voice output.",
            "text": "Read any text visible in this image. If there's no text, describe what you see briefly.",
            "objects": "List the main objects or items visible in this image. Keep it brief and suitable for voice output.",
            "scene": "Describe the scene or environment shown in this image. Keep your response brief and conversational.",
        }

        base_prompt = prompts.get(analyze_type, prompts["general"])

        if context:
            prompt = f"{base_prompt}\n\nThe user asked: '{context}'. Focus your description on what's relevant to their question."
        else:
            prompt = base_prompt

        # Decode base64 image
        # Handle data URL format if present
        if image_base64.startswith("data:"):
            # Extract base64 part from data URL
            image_base64 = image_base64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_base64)

        # Create image part for Gemini
        image_part = {
            "inline_data": {
                "mime_type": "image/jpeg",  # Assume JPEG, could detect from header
                "data": image_base64
            }
        }

        # Call Gemini Vision
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                {"parts": [{"text": prompt}, image_part]}
            ]
        )

        description = response.text.strip()

        # Clean up for voice
        description = _clean_for_voice(description)

        return VisionResponse(
            success=True,
            description=description,
            details={"analyze_type": analyze_type, "has_context": bool(context)}
        )

    except Exception as e:
        logger.error(f"Vision analysis failed: {e}")
        return VisionResponse(
            success=False,
            description="Sorry, I couldn't analyze that image."
        )


async def analyze_for_auto_detection(image_base64: str) -> Optional[VisionResponse]:
    """Analyze image for auto-detection mode (glasses always-on recording).

    Only returns a result if something noteworthy is detected.

    Args:
        image_base64: Base64-encoded image data

    Returns:
        VisionResponse if something noteworthy, None otherwise
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        prompt = """Analyze this image. If you see something the user might want to know about, describe it briefly.

Things worth mentioning:
- Text that might be important (signs, documents, labels)
- People the user might be interacting with
- Notable objects or situations
- QR codes or barcodes

If this is just a normal, unremarkable scene (like a wall, floor, or empty space), respond with exactly: "NOTHING_NOTABLE"

Keep any description to 1-2 sentences, suitable for voice output."""

        # Handle data URL format
        if image_base64.startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]

        image_part = {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image_base64
            }
        }

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                {"parts": [{"text": prompt}, image_part]}
            ]
        )

        result = response.text.strip()

        if result == "NOTHING_NOTABLE":
            return None

        return VisionResponse(
            success=True,
            description=_clean_for_voice(result),
            details={"auto_detected": True}
        )

    except Exception as e:
        logger.error(f"Auto-detection vision analysis failed: {e}")
        return None


def _clean_for_voice(text: str) -> str:
    """Clean up text for voice output."""
    import re

    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove bullet points
    text = re.sub(r'^[\s]*[-•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.\s*', '', text, flags=re.MULTILINE)

    # Clean up whitespace
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()
