"""
Meeting Mode

Real-time meeting assistant with WHAM (Will Hammond's Augmented Mind).

Features:
- Continuous audio transcription to backend
- Double-tap for quick suggestions (2-3s latency)
- Voice command "WHAM, [question]" for detailed help (3-4s latency)
- Real-time transcript display
- Suggestion HUD overlay
"""

import asyncio
import base64
import logging
import struct
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from glasses_client.core.state import AppMode
from glasses_client.core.display import HALO_COLORS
from glasses_client.input.tap import TapGesture
from .base import BaseMode

logger = logging.getLogger(__name__)


@dataclass
class MeetingConfig:
    """Configuration for meeting mode."""
    meeting_type: str = "general"  # general, negotiation, interview, sales
    participants: List[str] = None
    context: str = ""  # Pre-meeting context/goals
    proactive_suggestions: bool = False

    def __post_init__(self):
        if self.participants is None:
            self.participants = []


class MeetingMode(BaseMode):
    """
    Meeting assistant mode with WHAM personality.

    Input methods:
    - Double-tap: Quick suggestion (last 5 segments, ~2s response)
    - "WHAM, [question]": Detailed suggestion (full context, ~3s response)

    Audio is continuously streamed to backend for transcription.
    """

    mode_type = AppMode.MEETING

    # Audio recording settings
    SAMPLE_RATE = 16000
    CHUNK_DURATION_MS = 1000  # 1 second chunks
    VOICE_COMMAND_MAX_MS = 5000  # Max 5 seconds for voice command
    VOICE_COMMAND_SILENCE_MS = 1000  # 1 second silence = stop

    # Wake word for voice commands
    WAKE_WORD = "wham"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_id = str(uuid.uuid4())[:8]
        self._config: Optional[MeetingConfig] = None
        self._is_listening = False
        self._last_transcript: List[Dict] = []
        self._suggestions_received = 0
        self._audio_task: Optional[asyncio.Task] = None
        self._processing = False

    async def run(
        self,
        meeting_type: str = "general",
        context: str = "",
        participants: List[str] = None,
    ):
        """
        Run meeting mode.

        Args:
            meeting_type: Type of meeting (general, negotiation, interview, sales)
            context: Pre-meeting context or goals
            participants: List of participant names
        """
        self._config = MeetingConfig(
            meeting_type=meeting_type,
            context=context,
            participants=participants or [],
        )

        # Connect to WebSocket
        ws_path = f"/ws/meeting/{self._session_id}"
        ws = await self.connect_websocket(ws_path)

        # Set up message handlers
        self._setup_handlers(ws)

        # Show welcome
        await self._show_welcome()

        # Wait for start command
        started = await self._wait_for_start()
        if not started:
            return

        # Send meeting start
        await ws.send({
            "type": "meeting_start",
            "config": {
                "meeting_type": self._config.meeting_type,
                "participants": self._config.participants,
                "context": self._config.context,
            }
        })

        # Start audio streaming
        self._is_listening = True
        self._audio_task = asyncio.create_task(self._audio_stream_loop(ws))

        # Show listening status
        await self._show_listening()

        # Main input loop
        try:
            await self._input_loop(ws)
        finally:
            # Clean up
            self._is_listening = False
            if self._audio_task:
                self._audio_task.cancel()
                try:
                    await self._audio_task
                except asyncio.CancelledError:
                    pass

            # End meeting
            await ws.send({"type": "meeting_end"})

            # Show summary
            await self._show_summary()

    def _setup_handlers(self, ws):
        """Set up WebSocket message handlers."""

        @ws.on_message("status")
        async def handle_status(data):
            status = data.get("status", "")
            message = data.get("message", "")

            if status == "meeting_started":
                logger.info(f"Meeting started: {data.get('session_id')}")
            elif status == "processing":
                self._processing = True
                await self._show_processing(message)
            elif status == "meeting_ended":
                logger.info("Meeting ended")

        @ws.on_message("transcript_update")
        async def handle_transcript(data):
            segment = data.get("segment", {})
            if segment:
                self._last_transcript.append(segment)
                # Keep last 10 for display
                self._last_transcript = self._last_transcript[-10:]

                # Update display with latest transcript
                await self._show_transcript_update(segment)

        @ws.on_message("suggestion")
        async def handle_suggestion(data):
            self._processing = False
            self._suggestions_received += 1

            suggestion = data.get("suggestion", {})
            trigger = data.get("trigger", "double_tap")
            latency = data.get("total_latency_ms", 0)

            await self._show_suggestion(suggestion, trigger, latency)

        @ws.on_message("error")
        async def handle_error(data):
            self._processing = False
            message = data.get("message", "Unknown error")
            logger.error(f"Meeting error: {message}")
            await self.display.show_error(message)

    async def _input_loop(self, ws):
        """
        Main input loop - handles taps and voice commands.

        - Double-tap: Quick suggestion
        - Voice starting with "WHAM": Detailed suggestion
        - Triple-tap: End meeting
        """
        while self._running and self._is_listening:
            # Check for tap gestures
            tap = await self.tap.wait_for_tap(timeout_ms=100)

            if tap:
                if tap == TapGesture.DOUBLE_TAP:
                    # Quick suggestion request
                    await self._handle_double_tap(ws)
                elif tap == TapGesture.TRIPLE_TAP:
                    # End meeting
                    break

            # Check for voice command
            # This is non-blocking - just checks if wake word detected
            voice_result = await self._check_for_wake_word()

            if voice_result:
                await self._handle_voice_command(ws, voice_result)

            # Small sleep to prevent tight loop
            await asyncio.sleep(0.05)

    async def _handle_double_tap(self, ws):
        """Handle double-tap for quick suggestion."""
        if self._processing:
            return

        logger.info("Double-tap detected - requesting quick suggestion")

        await ws.send({"type": "double_tap"})

    async def _handle_voice_command(self, ws, initial_audio: bytes = None):
        """
        Handle voice command starting with "WHAM, [question]".

        Records up to 5 seconds or until 1 second of silence.
        """
        if self._processing:
            return

        logger.info("Voice command detected - recording question")

        # Show recording indicator
        await self._show_recording()

        # Record the voice command
        audio_data, query_text = await self._record_voice_command()

        if audio_data or query_text:
            # Send voice command
            message = {"type": "voice_command"}

            if query_text:
                # If we got transcribed text, send that
                message["query"] = query_text
            elif audio_data:
                # Otherwise send raw audio for backend transcription
                message["audio"] = base64.b64encode(audio_data).decode()
                message["duration_ms"] = len(audio_data) // (self.SAMPLE_RATE * 2) * 1000

            await ws.send(message)
        else:
            # No command detected
            await self._show_listening()

    async def _audio_stream_loop(self, ws):
        """
        Continuously stream audio chunks to backend for transcription.

        Sends 1-second chunks of audio.
        """
        try:
            while self._is_listening:
                # Record 1 second of audio
                audio_chunk = await self._record_audio_chunk()

                if audio_chunk and len(audio_chunk) > 0:
                    # Send to backend
                    await ws.send({
                        "type": "audio_chunk",
                        "audio": base64.b64encode(audio_chunk).decode(),
                        "duration_ms": self.CHUNK_DURATION_MS,
                        "sample_rate": self.SAMPLE_RATE,
                    })

                # Small sleep between chunks
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Audio stream error: {e}")

    # ==========================================================================
    # HALO SDK INTEGRATION POINTS
    # ==========================================================================
    #
    # This section contains placeholder methods that need to be replaced with
    # actual Halo Frame SDK calls when the hardware is available.
    #
    # Integration Checklist:
    # ----------------------
    # [ ] Microphone streaming  - frame.microphone.record()
    # [ ] IMU tap detection     - frame.imu.wait_for_tap()
    # [ ] Wake word detection   - frame.voice.check_wake_word()
    # [ ] OLED display          - frame.display.show_text()
    # [ ] Bone conduction audio - frame.audio.speak()
    #
    # Documentation: https://docs.brilliant.xyz/frame/
    # ==========================================================================

    async def _record_audio_chunk(self) -> Optional[bytes]:
        """
        Record a single audio chunk from the Halo Frame microphone.

        TODO: HALO SDK INTEGRATION
        ==========================
        Replace this placeholder with:

            audio = await self.frame.microphone.record(
                sample_rate=self.SAMPLE_RATE,
                duration_ms=self.CHUNK_DURATION_MS,
                channels=1,  # Mono
                bit_depth=16  # 16-bit PCM
            )
            return audio.raw_bytes

        The Halo Frame has a built-in MEMS microphone optimized for voice.

        Audio Format Requirements:
        - Sample rate: 16000 Hz (self.SAMPLE_RATE)
        - Channels: 1 (mono)
        - Bit depth: 16-bit signed PCM
        - Duration: 1000ms chunks (self.CHUNK_DURATION_MS)

        See: https://docs.brilliant.xyz/frame/audio#microphone

        Returns:
            Raw audio bytes in WAV format, or None if recording failed
        """
        # Placeholder for development - simulates recording delay
        await asyncio.sleep(self.CHUNK_DURATION_MS / 1000)
        return None

    async def _check_for_wake_word(self) -> Optional[str]:
        """
        Check if the wake word "WHAM" was detected.

        TODO: HALO SDK INTEGRATION
        ==========================
        Replace this placeholder with dual-trigger detection:

        1. IMU Tap Detection (double-tap on temple):
            tap_result = await self.frame.imu.wait_for_tap(
                tap_type=TapType.DOUBLE,
                timeout_ms=100,
                sensitivity=0.7  # Adjust for user preference
            )
            if tap_result.detected:
                # Double-tap triggers quick suggestion (no voice needed)
                return "TAP_TRIGGER"

        2. Voice Wake Word Detection:
            result = await self.frame.voice.check_wake_word(
                wake_word=self.WAKE_WORD,  # "wham"
                timeout_ms=100,
                threshold=0.8  # Confidence threshold
            )
            if result.detected:
                # Return any audio captured after the wake word
                return result.trailing_audio

        IMU Tap Types:
        - TapType.SINGLE: Single tap (not used)
        - TapType.DOUBLE: Double tap (quick suggestion)
        - TapType.TRIPLE: Triple tap (end meeting)

        See: https://docs.brilliant.xyz/frame/imu#tap-detection
        See: https://docs.brilliant.xyz/frame/voice#wake-word

        Returns:
            Audio bytes after wake word, "TAP_TRIGGER" for tap, or None
        """
        # Placeholder - no detection in development mode
        return None

    async def _record_voice_command(self) -> tuple[Optional[bytes], Optional[str]]:
        """
        Record voice command until silence or max duration.

        TODO: HALO SDK INTEGRATION
        ==========================
        Replace this placeholder with:

            # Record until silence detected
            recording = await self.frame.microphone.record_until_silence(
                max_duration_ms=self.VOICE_COMMAND_MAX_MS,  # 5000ms
                silence_threshold_ms=self.VOICE_COMMAND_SILENCE_MS,  # 1000ms
                sample_rate=self.SAMPLE_RATE,
                silence_amplitude=0.02  # RMS threshold for silence
            )

            # If Halo has on-device ASR, use it for faster response
            if self.frame.has_local_asr:
                text = await self.frame.asr.transcribe(
                    recording.raw_bytes,
                    language="en"
                )
                # Strip wake word from beginning if present
                if text.lower().startswith(self.WAKE_WORD):
                    text = text[len(self.WAKE_WORD):].strip(" ,:")
                return recording.raw_bytes, text

            # Otherwise, return raw audio for backend transcription
            return recording.raw_bytes, None

        Recording Parameters:
        - Max duration: 5 seconds (prevents runaway recording)
        - Silence threshold: 1 second of silence = stop recording
        - Silence amplitude: ~2% RMS = considered silence

        On-device ASR (if available):
        - Faster response (no network round-trip)
        - May have lower accuracy than Whisper
        - Falls back to backend transcription if unavailable

        See: https://docs.brilliant.xyz/frame/audio#recording
        See: https://docs.brilliant.xyz/frame/asr

        Returns:
            Tuple of (raw_audio_bytes, transcribed_text)
            - Both present: on-device transcription succeeded
            - Only audio: needs backend transcription
            - Both None: recording failed
        """
        # Placeholder - simulates recording delay
        await asyncio.sleep(0.5)
        return None, None

    # ==========================================================================
    # HALO SDK DISPLAY INTEGRATION NOTES
    # ==========================================================================
    #
    # The display methods below use DisplayController which abstracts the
    # Halo Frame's micro OLED display. When integrating with actual hardware:
    #
    # Display Specs:
    # - Resolution: 640x400 pixels
    # - FOV: 20 degrees
    # - Max lines: 4 (with current font sizes)
    # - Max chars per line: 45
    #
    # Direct SDK Usage (if bypassing DisplayController):
    #
    #     # Text display
    #     await self.frame.display.show_text(
    #         lines=["Line 1", "Line 2"],
    #         x=20, y=20,
    #         color=(78, 205, 196),  # RGB
    #         font_size=32,
    #         align="left"  # or "center"
    #     )
    #     await self.frame.display.show()  # Flush to display
    #
    #     # Clear display
    #     await self.frame.display.clear()
    #
    # Bone Conduction Audio Feedback:
    #
    #     # Speak suggestion via bone conduction
    #     await self.frame.audio.speak(
    #         text=suggestion[:100],  # First 100 chars
    #         speed=1.2,  # Slightly faster than normal
    #         voice="en-US-Neural2-J"  # Male voice
    #     )
    #
    #     # Audio cues
    #     await self.frame.audio.beep(
    #         frequency=880,  # A5 note
    #         duration_ms=100
    #     )
    #
    # See: https://docs.brilliant.xyz/frame/display
    # See: https://docs.brilliant.xyz/frame/audio#bone-conduction
    # ==========================================================================

    # === Display Methods ===

    async def _show_welcome(self):
        """Show welcome screen."""
        from glasses_client.core.display import DisplaySlide

        lines = [
            "[WHAM MEETING MODE]",
            f"Type: {self._config.meeting_type.upper()}",
            "",
            "Say 'start' or tap to begin",
        ]

        if self._config.context:
            lines[2] = f"Goal: {self._config.context[:30]}..."

        slide = DisplaySlide(
            lines=lines,
            color=HALO_COLORS.get("meeting", HALO_COLORS["info"]),
            centered=True
        )
        await self.display.render_slide(slide)

    async def _wait_for_start(self) -> bool:
        """Wait for start command."""
        from glasses_client.input.voice import VoiceCommand

        while self._running:
            # Check voice
            result = await self.voice.listen(timeout_ms=5000)
            if result:
                if result.command in [VoiceCommand.START, VoiceCommand.NEXT]:
                    return True
                elif result.command == VoiceCommand.EXIT:
                    return False

            # Check tap
            tap = await self.tap.wait_for_tap(timeout_ms=100)
            if tap:
                return True

        return False

    async def _show_listening(self):
        """Show listening status."""
        from glasses_client.core.display import DisplaySlide

        slide = DisplaySlide(
            lines=[
                "[LISTENING]",
                "",
                "Double-tap: Quick help",
                "Say 'WHAM, ...': Ask question",
                "Triple-tap: End meeting",
            ],
            color=HALO_COLORS.get("meeting", HALO_COLORS["info"]),
            centered=True
        )
        await self.display.render_slide(slide)

    async def _show_recording(self):
        """Show recording indicator."""
        from glasses_client.core.display import DisplaySlide

        slide = DisplaySlide(
            lines=[
                "[RECORDING]",
                "",
                "Speak your question...",
            ],
            color=HALO_COLORS.get("recording", "#FF6B6B"),
            centered=True
        )
        await self.display.render_slide(slide)

    async def _show_processing(self, message: str = ""):
        """Show processing indicator."""
        from glasses_client.core.display import DisplaySlide

        slide = DisplaySlide(
            lines=[
                "[THINKING]",
                "",
                message or "Processing...",
            ],
            color=HALO_COLORS.get("processing", "#FFE66D"),
            centered=True
        )
        await self.display.render_slide(slide)

    async def _show_transcript_update(self, segment: Dict):
        """Show transcript update in subtle overlay."""
        # Don't interrupt suggestion display
        if self._processing:
            return

        text = segment.get("text", "")
        speaker = segment.get("speaker", "unknown")

        # Just log for now - could show subtle indicator
        logger.debug(f"[{speaker}]: {text[:50]}...")

    async def _show_suggestion(
        self,
        suggestion: Dict,
        trigger: str,
        latency_ms: float
    ):
        """Display WHAM's suggestion."""
        from glasses_client.core.display import DisplaySlide

        text = suggestion.get("suggestion", "No suggestion")
        suggestion_type = suggestion.get("type", "quick_response")
        alternatives = suggestion.get("alternatives", [])
        tactical_notes = suggestion.get("tactical_notes")

        # Build display lines
        lines = []

        # Header with type
        if trigger == "double_tap":
            lines.append("[QUICK HELP]")
        else:
            lines.append("[WHAM SUGGESTS]")

        lines.append("")

        # Main suggestion (wrap to multiple lines if needed)
        wrapped = self._wrap_text(text, max_chars=35)
        lines.extend(wrapped[:4])  # Max 4 lines for main suggestion

        # Tactical note if present
        if tactical_notes:
            lines.append("")
            lines.append(f"TIP: {tactical_notes[:30]}...")

        # Determine color based on type
        color = HALO_COLORS.get("suggestion", HALO_COLORS["info"])
        if suggestion_type == "negotiation":
            color = HALO_COLORS.get("warning", "#FFE66D")
        elif suggestion_type == "fact_check":
            color = HALO_COLORS.get("alert", "#FF6B6B")

        slide = DisplaySlide(
            lines=lines[:8],  # Max 8 lines
            color=color,
            centered=False  # Left-align for readability
        )
        await self.display.render_slide(slide)

        # Log latency
        logger.info(f"Suggestion displayed ({trigger}) in {latency_ms:.0f}ms")

        # Auto-dismiss after delay
        await asyncio.sleep(5.0)

        # Return to listening view if still running
        if self._running and self._is_listening:
            await self._show_listening()

    async def _show_summary(self):
        """Show meeting summary."""
        from glasses_client.core.display import DisplaySlide

        slide = DisplaySlide(
            lines=[
                "[MEETING ENDED]",
                "",
                f"Suggestions given: {self._suggestions_received}",
                "",
                "Good work, Will.",
            ],
            color=HALO_COLORS.get("result_green", "#4ECDC4"),
            centered=True
        )
        await self.display.render_slide(slide)

        await asyncio.sleep(3)

    # === Utility Methods ===

    def _wrap_text(self, text: str, max_chars: int = 35) -> List[str]:
        """Wrap text to fit display width."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines
