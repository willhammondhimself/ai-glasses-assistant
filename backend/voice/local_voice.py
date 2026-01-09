"""Local voice processing via faster-whisper and Piper TTS.

Provides offline speech-to-text and text-to-speech capabilities.
Designed to run on phone/MacBook with 8GB+ RAM.

Usage:
    from backend.voice.local_voice import get_local_voice

    voice = get_local_voice()
    if voice.enabled:
        # Transcribe audio
        text = voice.transcribe(audio_bytes)

        # Synthesize speech
        audio = voice.synthesize("Hello world")
"""
import os
import io
import wave
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Default model paths
DEFAULT_WHISPER_MODEL = "base.en"  # ~150MB, good accuracy/speed balance
DEFAULT_PIPER_MODEL = "en_US-lessac-medium"
DEFAULT_PIPER_PATH = "models/en_US-lessac-medium.onnx"


class LocalVoice:
    """Offline voice processing with faster-whisper STT and Piper TTS."""

    def __init__(
        self,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        piper_model_path: Optional[str] = None,
    ):
        """Initialize LocalVoice.

        Args:
            whisper_model: Whisper model size (tiny.en, base.en, small.en, medium.en)
            piper_model_path: Path to Piper ONNX model file
        """
        self.whisper_model = whisper_model
        self.piper_model_path = piper_model_path or self._find_piper_model()

        self._whisper = None
        self._piper = None
        self._whisper_loaded = False
        self._piper_loaded = False
        self.enabled = True  # Can be toggled via settings

    def _find_piper_model(self) -> Optional[str]:
        """Find Piper model from various sources."""
        # Check environment variable
        env_path = os.getenv("PIPER_MODEL_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # Check default locations
        paths_to_check = [
            DEFAULT_PIPER_PATH,
            os.path.expanduser(f"~/models/{DEFAULT_PIPER_MODEL}.onnx"),
            f"/opt/models/{DEFAULT_PIPER_MODEL}.onnx",
        ]

        for path in paths_to_check:
            if os.path.exists(path):
                return path

        return None

    @property
    def stt_available(self) -> bool:
        """Check if STT is available."""
        try:
            import faster_whisper
            return True
        except ImportError:
            return False

    @property
    def tts_available(self) -> bool:
        """Check if TTS model file exists."""
        return self.piper_model_path is not None and os.path.exists(self.piper_model_path)

    @property
    def is_ready(self) -> bool:
        """Check if voice processing is ready."""
        return self.enabled and (self.stt_available or self.tts_available)

    def load_stt(self) -> bool:
        """Load faster-whisper model.

        Returns:
            True if loaded successfully
        """
        if self._whisper_loaded:
            return True

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading faster-whisper model: {self.whisper_model}")
            self._whisper = WhisperModel(
                self.whisper_model,
                device="cpu",
                compute_type="int8",  # Optimized for CPU
            )
            self._whisper_loaded = True
            logger.info("faster-whisper loaded successfully")
            return True

        except ImportError:
            logger.error("faster-whisper not installed. Run: pip install faster-whisper")
            return False
        except Exception as e:
            logger.error(f"Failed to load faster-whisper: {e}")
            return False

    def load_tts(self) -> bool:
        """Load Piper TTS model.

        Returns:
            True if loaded successfully
        """
        if self._piper_loaded:
            return True

        if not self.tts_available:
            logger.warning(f"Piper model not found. Download from piper releases.")
            return False

        try:
            from piper import PiperVoice

            logger.info(f"Loading Piper TTS from: {self.piper_model_path}")
            self._piper = PiperVoice.load(self.piper_model_path)
            self._piper_loaded = True
            logger.info("Piper TTS loaded successfully")
            return True

        except ImportError:
            logger.error("piper-tts not installed. Run: pip install piper-tts")
            return False
        except Exception as e:
            logger.error(f"Failed to load Piper TTS: {e}")
            return False

    def unload_stt(self):
        """Unload STT model from memory."""
        if self._whisper:
            del self._whisper
            self._whisper = None
            self._whisper_loaded = False
            logger.info("faster-whisper unloaded")

    def unload_tts(self):
        """Unload TTS model from memory."""
        if self._piper:
            del self._piper
            self._piper = None
            self._piper_loaded = False
            logger.info("Piper TTS unloaded")

    def unload_all(self):
        """Unload all models from memory."""
        self.unload_stt()
        self.unload_tts()

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "en",
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio_bytes: Raw audio bytes (WAV format preferred)
            language: Language code (default: English)

        Returns:
            Transcribed text
        """
        if not self.enabled:
            return ""

        if not self.load_stt():
            return ""

        try:
            # Create file-like object from bytes
            audio_file = io.BytesIO(audio_bytes)

            segments, info = self._whisper.transcribe(
                audio_file,
                language=language,
                beam_size=5,
                vad_filter=True,  # Filter out silence
            )

            # Concatenate all segments
            text = " ".join(segment.text for segment in segments)
            return text.strip()

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def transcribe_file(self, audio_path: str, language: str = "en") -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code

        Returns:
            Transcribed text
        """
        if not self.enabled:
            return ""

        if not self.load_stt():
            return ""

        try:
            segments, info = self._whisper.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,
            )

            text = " ".join(segment.text for segment in segments)
            return text.strip()

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def synthesize(
        self,
        text: str,
        sample_rate: int = 22050,
    ) -> bytes:
        """Convert text to speech audio.

        Args:
            text: Text to synthesize
            sample_rate: Output sample rate (default: 22050)

        Returns:
            WAV audio as bytes
        """
        if not self.enabled:
            return b""

        if not self.load_tts():
            return b""

        try:
            # Create WAV buffer
            audio_buffer = io.BytesIO()

            with wave.open(audio_buffer, "wb") as wav:
                wav.setframerate(sample_rate)
                wav.setsampwidth(2)  # 16-bit audio
                wav.setnchannels(1)  # Mono

                # Synthesize and write audio
                self._piper.synthesize(text, wav)

            return audio_buffer.getvalue()

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return b""

    def synthesize_stream(self, text: str):
        """Stream audio synthesis chunk by chunk.

        Yields:
            Audio chunks as bytes
        """
        if not self.enabled:
            return

        if not self.load_tts():
            return

        try:
            for audio_chunk in self._piper.synthesize_stream_raw(text):
                yield audio_chunk

        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")

    def toggle(self, enabled: bool):
        """Enable or disable local voice processing.

        Args:
            enabled: True to enable, False to disable
        """
        self.enabled = enabled
        logger.info(f"Local voice {'enabled' if enabled else 'disabled'}")

        if not enabled:
            # Optionally unload models when disabled to free memory
            # self.unload_all()
            pass

    def get_status(self) -> dict:
        """Get current status of local voice processing."""
        return {
            "enabled": self.enabled,
            "stt": {
                "available": self.stt_available,
                "loaded": self._whisper_loaded,
                "model": self.whisper_model,
            },
            "tts": {
                "available": self.tts_available,
                "loaded": self._piper_loaded,
                "model_path": self.piper_model_path,
            },
        }


# Singleton instance
_local_voice: Optional[LocalVoice] = None


def get_local_voice() -> LocalVoice:
    """Get singleton LocalVoice instance."""
    global _local_voice
    if _local_voice is None:
        _local_voice = LocalVoice()
    return _local_voice
