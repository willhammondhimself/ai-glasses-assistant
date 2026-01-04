"""
Transcription Service - OpenAI Whisper API integration.

Handles:
- Continuous audio transcription from 1-second chunks
- Speaker diarization (basic, via voice characteristics)
- Confidence scoring
- Error recovery with backoff
"""

import asyncio
import base64
import logging
import os
import time
import io
from typing import Optional, List, Callable, Awaitable
from datetime import datetime
from dataclasses import dataclass

import httpx

from .models import TranscriptSegment, AudioChunk, SpeakerRole

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionConfig:
    """Configuration for transcription service."""
    model: str = "whisper-1"
    language: str = "en"
    temperature: float = 0.0  # More deterministic
    response_format: str = "verbose_json"  # Get word-level timestamps
    chunk_duration_ms: int = 1000
    max_retries: int = 3
    retry_delay_ms: int = 500


class TranscriptionService:
    """
    Whisper API transcription service for continuous audio.

    Cost: ~$0.006/minute of audio (very cheap)

    Usage:
        service = TranscriptionService()
        segment = await service.transcribe(audio_chunk)
    """

    API_URL = "https://api.openai.com/v1/audio/transcriptions"

    def __init__(self, config: TranscriptionConfig = None):
        self.config = config or TranscriptionConfig()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set - transcription will fail")

        self._client: Optional[httpx.AsyncClient] = None
        self._buffer: List[AudioChunk] = []
        self._last_speaker: SpeakerRole = SpeakerRole.UNKNOWN
        self._user_voice_signature: Optional[bytes] = None  # For speaker detection

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._client

    async def transcribe(self, chunk: AudioChunk) -> Optional[TranscriptSegment]:
        """
        Transcribe a single audio chunk.

        Args:
            chunk: AudioChunk with raw audio bytes

        Returns:
            TranscriptSegment or None if transcription failed/empty
        """
        if not self.api_key:
            return None

        start_time = time.perf_counter()

        for attempt in range(self.config.max_retries):
            try:
                segment = await self._call_whisper(chunk)
                if segment:
                    segment.duration_ms = chunk.duration_ms
                    latency = (time.perf_counter() - start_time) * 1000
                    logger.debug(f"Transcribed in {latency:.0f}ms: {segment.text[:50]}...")
                return segment

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    await asyncio.sleep(self.config.retry_delay_ms / 1000 * (attempt + 1))
                    continue
                logger.error(f"Whisper API error: {e}")
                return None

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_ms / 1000)
                    continue
                return None

        return None

    async def _call_whisper(self, chunk: AudioChunk) -> Optional[TranscriptSegment]:
        """Make the actual Whisper API call."""
        client = await self._get_client()

        # Prepare audio file for upload
        audio_file = self._prepare_audio_file(chunk)

        files = {
            "file": ("audio.wav", audio_file, "audio/wav"),
            "model": (None, self.config.model),
            "language": (None, self.config.language),
            "response_format": (None, self.config.response_format),
            "temperature": (None, str(self.config.temperature)),
        }

        response = await client.post(self.API_URL, files=files)
        response.raise_for_status()

        data = response.json()

        text = data.get("text", "").strip()
        if not text:
            return None

        # Detect speaker (basic heuristic)
        speaker = self._detect_speaker(chunk, text)

        return TranscriptSegment(
            text=text,
            speaker=speaker,
            timestamp=chunk.timestamp,
            confidence=self._estimate_confidence(data),
            is_final=True,
        )

    def _prepare_audio_file(self, chunk: AudioChunk) -> io.BytesIO:
        """Convert raw audio bytes to WAV format for Whisper."""
        import wave

        buffer = io.BytesIO()

        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(chunk.channels)
            wav.setsampwidth(2)  # 16-bit audio
            wav.setframerate(chunk.sample_rate)
            wav.writeframes(chunk.data)

        buffer.seek(0)
        return buffer

    def _detect_speaker(self, chunk: AudioChunk, text: str) -> SpeakerRole:
        """
        Basic speaker detection.

        For now, uses simple heuristics. Future: voice embeddings.
        """
        # If we have a user voice signature, compare
        if self._user_voice_signature:
            # TODO: Implement voice embedding comparison
            pass

        # Heuristic: Questions directed at "you" are likely from counterpart
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in ["what do you think", "your opinion", "you mentioned"]):
            return SpeakerRole.COUNTERPART

        # Heuristic: "I" statements with context clues
        if text_lower.startswith("i ") or " i " in text_lower:
            # Could be either speaker - use last known
            return self._last_speaker

        return SpeakerRole.UNKNOWN

    def _estimate_confidence(self, data: dict) -> float:
        """Estimate transcription confidence from Whisper response."""
        # Whisper verbose_json includes word-level info
        if "segments" in data and data["segments"]:
            # Average the no_speech_prob (lower is better)
            no_speech_probs = [
                seg.get("no_speech_prob", 0.5)
                for seg in data["segments"]
            ]
            avg_no_speech = sum(no_speech_probs) / len(no_speech_probs)
            return 1.0 - avg_no_speech

        return 0.8  # Default confidence

    def set_user_voice(self, audio_sample: bytes):
        """
        Set user's voice signature for speaker detection.

        Args:
            audio_sample: A sample of the user's voice (5+ seconds recommended)
        """
        self._user_voice_signature = audio_sample
        logger.info("User voice signature set for speaker detection")

    async def transcribe_batch(
        self,
        chunks: List[AudioChunk],
        on_segment: Callable[[TranscriptSegment], Awaitable[None]] = None
    ) -> List[TranscriptSegment]:
        """
        Transcribe multiple chunks, optionally streaming results.

        Args:
            chunks: List of audio chunks
            on_segment: Optional callback for each transcribed segment

        Returns:
            List of all transcribed segments
        """
        segments = []

        for chunk in chunks:
            segment = await self.transcribe(chunk)
            if segment:
                segments.append(segment)
                if on_segment:
                    await on_segment(segment)

        return segments

    async def close(self):
        """Clean up HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class AudioBuffer:
    """
    Buffer for accumulating audio before transcription.

    Useful for:
    - Combining very short audio clips
    - Voice command recording with silence detection
    """

    def __init__(
        self,
        min_duration_ms: int = 500,
        max_duration_ms: int = 5000,
        silence_threshold_ms: int = 1000,
    ):
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.silence_threshold_ms = silence_threshold_ms

        self._chunks: List[AudioChunk] = []
        self._total_duration_ms: int = 0
        self._silence_start: Optional[datetime] = None

    def add(self, chunk: AudioChunk) -> bool:
        """
        Add chunk to buffer.

        Returns:
            True if buffer is ready to flush (max duration or silence detected)
        """
        self._chunks.append(chunk)
        self._total_duration_ms += chunk.duration_ms

        # Check for silence (simplified - real impl would analyze audio)
        is_silence = self._is_silence(chunk)

        if is_silence:
            if self._silence_start is None:
                self._silence_start = chunk.timestamp
            else:
                silence_duration = (chunk.timestamp - self._silence_start).total_seconds() * 1000
                if silence_duration >= self.silence_threshold_ms:
                    return True
        else:
            self._silence_start = None

        return self._total_duration_ms >= self.max_duration_ms

    def flush(self) -> Optional[AudioChunk]:
        """
        Combine buffered chunks into a single chunk.

        Returns:
            Combined AudioChunk or None if buffer is empty/too short
        """
        if not self._chunks or self._total_duration_ms < self.min_duration_ms:
            return None

        # Combine audio data
        combined_data = b"".join(c.data for c in self._chunks)
        first_chunk = self._chunks[0]

        combined = AudioChunk(
            data=combined_data,
            timestamp=first_chunk.timestamp,
            duration_ms=self._total_duration_ms,
            sample_rate=first_chunk.sample_rate,
            channels=first_chunk.channels,
        )

        self.clear()
        return combined

    def clear(self):
        """Clear the buffer."""
        self._chunks = []
        self._total_duration_ms = 0
        self._silence_start = None

    def _is_silence(self, chunk: AudioChunk) -> bool:
        """
        Detect if chunk is silence.

        Uses RMS energy threshold.
        """
        if len(chunk.data) < 2:
            return True

        # Calculate RMS energy
        import struct
        samples = struct.unpack(f"<{len(chunk.data)//2}h", chunk.data)
        rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5

        # Threshold (tune based on mic sensitivity)
        return rms < 500

    @property
    def duration_ms(self) -> int:
        return self._total_duration_ms

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0
