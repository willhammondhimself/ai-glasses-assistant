"""Wake Word Service - Local wake word detection using faster-whisper.

Energy-first approach:
1. Check audio energy to filter out silence (FREE)
2. If speech detected, transcribe with faster-whisper (LOCAL, FREE)
3. Check if transcription starts with wake word
4. Return remaining query if detected
"""
import io
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import faster-whisper to avoid startup cost
_whisper_model = None


def _get_whisper_model():
    """Lazy load the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            logger.info("Loading faster-whisper 'tiny' model...")
            _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            logger.info("Whisper model loaded successfully")
        except ImportError:
            logger.error("faster-whisper not installed. Run: pip install faster-whisper")
            raise
    return _whisper_model


class WakeWordService:
    """Local wake word detection using energy detection + faster-whisper.

    Flow:
    1. Audio comes in from microphone
    2. Check energy level - if below threshold, return immediately (no cost)
    3. If energy detected, transcribe with local faster-whisper
    4. Check if transcription starts with wake word
    5. Return (detected, remaining_query)
    """

    def __init__(
        self,
        wake_word: str = "wham",
        energy_threshold: float = 0.01,
        sample_rate: int = 16000
    ):
        """Initialize wake word service.

        Args:
            wake_word: The wake word to listen for (default: "wham")
            energy_threshold: Minimum RMS energy to consider as speech (default: 0.01)
            sample_rate: Expected audio sample rate (default: 16000)
        """
        self.wake_word = wake_word.lower()
        self.energy_threshold = energy_threshold
        self.sample_rate = sample_rate

        # Alternative wake word variations
        self.wake_word_variants = [
            self.wake_word,
            f"hey {self.wake_word}",
            f"ok {self.wake_word}",
            f"okay {self.wake_word}",
        ]

        logger.info(f"WakeWordService initialized with wake word: '{self.wake_word}'")

    def has_speech(self, audio_bytes: bytes, dtype: str = "int16") -> bool:
        """Check if audio chunk contains speech based on energy level.

        This is a fast, local check that filters out silence without
        any API calls or model inference.

        Args:
            audio_bytes: Raw audio data
            dtype: Audio data type (default: int16)

        Returns:
            True if energy exceeds threshold (likely speech)
        """
        try:
            # Convert bytes to numpy array
            if dtype == "int16":
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif dtype == "float32":
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
            else:
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Calculate RMS energy
            energy = np.sqrt(np.mean(audio ** 2))

            return energy > self.energy_threshold

        except Exception as e:
            logger.warning(f"Error checking speech energy: {e}")
            return False

    def detect(self, audio_bytes: bytes, dtype: str = "int16") -> Tuple[bool, str]:
        """Detect wake word in audio and extract remaining query.

        Args:
            audio_bytes: Raw audio data
            dtype: Audio data type (default: int16)

        Returns:
            Tuple of (wake_word_detected, remaining_query)
            - If wake word detected: (True, "what's the weather")
            - If no wake word: (False, "")
        """
        # Step 1: Energy check (fast, free)
        if not self.has_speech(audio_bytes, dtype):
            return False, ""

        # Step 2: Transcribe with local faster-whisper
        try:
            model = _get_whisper_model()

            # Convert audio to format expected by faster-whisper
            if dtype == "int16":
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif dtype == "float32":
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
            else:
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe
            segments, info = model.transcribe(
                audio,
                language="en",
                beam_size=1,  # Faster
                vad_filter=True,
            )

            # Combine segments
            text = " ".join(segment.text for segment in segments).lower().strip()

            if not text:
                return False, ""

            logger.debug(f"Transcribed: '{text}'")

            # Step 3: Check for wake word variants
            for variant in self.wake_word_variants:
                if text.startswith(variant):
                    # Extract remaining query after wake word
                    query = text[len(variant):].strip(" ,:.!?")
                    logger.info(f"Wake word detected! Query: '{query}'")
                    return True, query

            # Wake word not found
            return False, ""

        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return False, ""

    def detect_in_text(self, text: str) -> Tuple[bool, str]:
        """Check if text starts with wake word (for pre-transcribed audio).

        Useful when audio is already transcribed by another service.

        Args:
            text: Transcribed text to check

        Returns:
            Tuple of (wake_word_detected, remaining_query)
        """
        text_lower = text.lower().strip()

        for variant in self.wake_word_variants:
            if text_lower.startswith(variant):
                query = text_lower[len(variant):].strip(" ,:.!?")
                return True, query

        return False, ""


# Global service instance
_wake_word_service: Optional[WakeWordService] = None


def get_wake_word_service() -> WakeWordService:
    """Get or create global WakeWordService instance."""
    global _wake_word_service
    if _wake_word_service is None:
        _wake_word_service = WakeWordService()
    return _wake_word_service
