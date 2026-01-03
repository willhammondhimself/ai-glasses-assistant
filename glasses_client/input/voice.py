"""
Voice Command Handler

Processes voice input for hands-free control of the glasses.
Supports command recognition and number parsing.
"""

import asyncio
import logging
import re
from typing import Optional, Callable, Awaitable, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VoiceCommand(Enum):
    """Recognized voice commands."""
    # Navigation
    NEXT = "next"
    BACK = "back"
    EXIT = "exit"

    # Mode activation
    MENTAL_MATH = "mental_math"
    CAMERA = "camera"
    DEBUG = "debug"

    # Actions
    START = "start"
    SKIP = "skip"
    HINT = "hint"
    EXPLAIN = "explain"
    SAVE = "save"

    # Confirmations
    YES = "yes"
    NO = "no"

    # Numbers (handled separately)
    NUMBER = "number"

    # Unknown
    UNKNOWN = "unknown"


# Command aliases for recognition
COMMAND_ALIASES = {
    # Navigation
    "next": VoiceCommand.NEXT,
    "continue": VoiceCommand.NEXT,
    "forward": VoiceCommand.NEXT,
    "back": VoiceCommand.BACK,
    "previous": VoiceCommand.BACK,
    "go back": VoiceCommand.BACK,
    "exit": VoiceCommand.EXIT,
    "quit": VoiceCommand.EXIT,
    "done": VoiceCommand.EXIT,
    "stop": VoiceCommand.EXIT,

    # Mode activation
    "mental math": VoiceCommand.MENTAL_MATH,
    "speed run": VoiceCommand.MENTAL_MATH,
    "jane street": VoiceCommand.MENTAL_MATH,
    "camera": VoiceCommand.CAMERA,
    "scan": VoiceCommand.CAMERA,
    "capture": VoiceCommand.CAMERA,
    "help with this": VoiceCommand.CAMERA,
    "debug": VoiceCommand.DEBUG,
    "fix code": VoiceCommand.DEBUG,
    "what's wrong": VoiceCommand.DEBUG,

    # Actions
    "start": VoiceCommand.START,
    "begin": VoiceCommand.START,
    "skip": VoiceCommand.SKIP,
    "pass": VoiceCommand.SKIP,
    "hint": VoiceCommand.HINT,
    "help": VoiceCommand.HINT,
    "explain": VoiceCommand.EXPLAIN,
    "why": VoiceCommand.EXPLAIN,
    "save": VoiceCommand.SAVE,
    "flashcard": VoiceCommand.SAVE,

    # Confirmations
    "yes": VoiceCommand.YES,
    "yeah": VoiceCommand.YES,
    "yep": VoiceCommand.YES,
    "correct": VoiceCommand.YES,
    "no": VoiceCommand.NO,
    "nope": VoiceCommand.NO,
    "cancel": VoiceCommand.NO,
}

# Number words for parsing
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000,
}


@dataclass
class VoiceResult:
    """Result of voice recognition."""
    command: VoiceCommand
    raw_text: str
    confidence: float = 1.0
    number_value: Optional[float] = None


class VoiceHandler:
    """
    Handles voice input from the glasses microphone.

    Features:
    - Command recognition with aliases
    - Number parsing (spoken and digits)
    - Continuous listening mode
    - Wake word detection (optional)
    """

    def __init__(self, frame=None):
        """
        Initialize voice handler.

        Args:
            frame: Frame SDK instance. If None, uses mock for testing.
        """
        self.frame = frame
        self._mock_mode = frame is None
        self._listening = False
        self._wake_word = "hey coach"
        self._use_wake_word = False

    async def listen(self, timeout_ms: int = 5000) -> Optional[VoiceResult]:
        """
        Listen for a voice command.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            VoiceResult if command recognized, None if timeout
        """
        self._listening = True

        try:
            if self._mock_mode:
                # Mock mode: simulate with input
                text = await self._mock_listen(timeout_ms)
            else:
                text = await self._real_listen(timeout_ms)

            if not text:
                return None

            return self._parse_input(text)

        finally:
            self._listening = False

    async def listen_for_number(self, timeout_ms: int = 5000) -> Optional[float]:
        """
        Listen specifically for a numeric answer.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            Numeric value if recognized, None otherwise
        """
        result = await self.listen(timeout_ms)

        if result and result.command == VoiceCommand.NUMBER:
            return result.number_value

        # Also try to parse the raw text as a number
        if result:
            number = self._parse_number(result.raw_text)
            if number is not None:
                return number

        return None

    async def confirm(self, prompt: str = "continue") -> bool:
        """
        Wait for yes/no confirmation.

        Args:
            prompt: What we're confirming (for context)

        Returns:
            True for yes, False for no/timeout
        """
        result = await self.listen(timeout_ms=5000)

        if result:
            return result.command == VoiceCommand.YES

        return False

    async def wait_for_command(
        self,
        allowed: Optional[List[VoiceCommand]] = None,
        timeout_ms: int = 10000
    ) -> Optional[VoiceCommand]:
        """
        Wait for a specific command from a set of allowed commands.

        Args:
            allowed: List of allowed commands (None = all)
            timeout_ms: Timeout in milliseconds

        Returns:
            Recognized command or None
        """
        result = await self.listen(timeout_ms)

        if not result:
            return None

        if allowed and result.command not in allowed:
            return None

        return result.command

    def _parse_input(self, text: str) -> VoiceResult:
        """Parse recognized text into a VoiceResult."""
        text_lower = text.lower().strip()

        # Check for number first
        number = self._parse_number(text_lower)
        if number is not None:
            return VoiceResult(
                command=VoiceCommand.NUMBER,
                raw_text=text,
                number_value=number
            )

        # Check command aliases
        for alias, command in COMMAND_ALIASES.items():
            if alias in text_lower:
                return VoiceResult(
                    command=command,
                    raw_text=text
                )

        return VoiceResult(
            command=VoiceCommand.UNKNOWN,
            raw_text=text
        )

    def _parse_number(self, text: str) -> Optional[float]:
        """
        Parse a number from text.

        Handles:
        - Digit strings: "42", "3.14"
        - Spoken numbers: "forty-two", "three point one four"
        - Mixed: "42 thousand"
        """
        text = text.lower().strip()

        # Try direct numeric parsing first
        try:
            # Handle negative
            if text.startswith("negative ") or text.startswith("minus "):
                rest = text.split(" ", 1)[1]
                return -self._parse_number(rest)

            # Handle decimals
            if "point" in text:
                parts = text.split("point")
                whole = self._parse_whole_number(parts[0].strip())
                decimal = self._parse_whole_number(parts[1].strip())
                if whole is not None and decimal is not None:
                    decimal_str = str(int(decimal))
                    return whole + float(f"0.{decimal_str}")

            # Try as numeric string
            cleaned = text.replace(",", "").replace(" ", "")
            return float(cleaned)

        except ValueError:
            pass

        # Try parsing spoken number
        return self._parse_whole_number(text)

    def _parse_whole_number(self, text: str) -> Optional[float]:
        """Parse a whole number from spoken text."""
        if not text:
            return None

        # Check for simple number word
        if text in NUMBER_WORDS:
            return float(NUMBER_WORDS[text])

        # Parse compound numbers like "forty-seven"
        text = text.replace("-", " ")
        words = text.split()

        total = 0
        current = 0

        for word in words:
            if word in NUMBER_WORDS:
                value = NUMBER_WORDS[word]

                if value == 100:
                    current = (current or 1) * 100
                elif value == 1000:
                    current = (current or 1) * 1000
                    total += current
                    current = 0
                elif value == 1000000:
                    current = (current or 1) * 1000000
                    total += current
                    current = 0
                else:
                    current += value
            else:
                # Try as digit
                try:
                    current += int(word)
                except ValueError:
                    return None

        total += current
        return float(total) if total > 0 or current == 0 else None

    async def _mock_listen(self, timeout_ms: int) -> Optional[str]:
        """Mock listening for testing."""
        import sys

        # For testing, we can use console input or simulate
        logger.debug(f"[MOCK VOICE] Listening for {timeout_ms}ms...")

        try:
            # Simulate brief delay
            await asyncio.sleep(0.2)

            # In real testing, you might want to use asyncio stdin reading
            # For now, return None to simulate timeout
            return None

        except Exception as e:
            logger.warning(f"Mock listen error: {e}")
            return None

    async def _real_listen(self, timeout_ms: int) -> Optional[str]:
        """Real voice recognition using Frame SDK."""
        try:
            # Record audio from Frame microphone
            audio_data = await asyncio.wait_for(
                self.frame.microphone.record(duration_ms=timeout_ms),
                timeout=timeout_ms / 1000 + 1
            )

            # Use speech recognition
            # This is a simplified version - real implementation would use
            # a proper speech recognition service
            import speech_recognition as sr

            recognizer = sr.Recognizer()

            # Convert audio data to AudioData
            audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)

            # Recognize using Google (or other service)
            text = recognizer.recognize_google(audio)
            logger.info(f"Recognized: {text}")
            return text

        except asyncio.TimeoutError:
            logger.debug("Voice recognition timeout")
            return None

        except Exception as e:
            logger.warning(f"Voice recognition error: {e}")
            return None

    def set_wake_word(self, wake_word: str, enabled: bool = True):
        """Set the wake word for activation."""
        self._wake_word = wake_word.lower()
        self._use_wake_word = enabled
