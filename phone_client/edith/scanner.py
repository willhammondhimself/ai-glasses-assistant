"""
EDITH - Environmental Detection and Image Tracking Helper.
Background scanning for proactive assistance.
"""
import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Awaitable, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

# Optional OpenCV for vision
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class DetectionType(Enum):
    """Types of content EDITH can detect."""
    EQUATION = "equation"
    CODE = "code"
    TEXT = "text"
    FACE = "face"
    QR_CODE = "qr_code"
    POKER_CARDS = "poker_cards"

    # Screen-based detection (using Gemini vision)
    POKER_TABLE = "poker_table"      # Full poker table UI
    CODE_EDITOR = "code_editor"      # IDE/editor interface
    MATH_PROBLEM = "math_problem"    # Math equations/problems


@dataclass
class Detection:
    """A detected item in the frame."""
    type: DetectionType
    confidence: float
    bounding_box: tuple  # (x, y, width, height)
    content: str = ""    # Extracted text/code if applicable
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScanConfig:
    """EDITH scanning configuration."""
    enabled: bool = True
    scan_interval_seconds: float = 5.0
    battery_saver_interval: float = 15.0
    stare_threshold_ms: float = 1500
    suggestion_cooldown_ms: float = 10000
    detection_types: List[str] = field(default_factory=lambda: ["equation", "code", "text"])


class EdithScanner:
    """
    EDITH Background Scanner.

    Periodically captures images and looks for:
    - Equations (offer to solve)
    - Code (offer to debug/explain)
    - Text (offer to summarize)
    - Faces (identify if known)
    - QR codes (decode)
    - Poker cards (offer GTO analysis)

    Uses lightweight on-device detection to minimize battery drain.
    """

    def __init__(
        self,
        config: ScanConfig,
        capture_image: Callable[[], Awaitable[Optional[bytes]]],
        on_detection: Callable[[Detection], Awaitable[None]],
    ):
        self.config = config
        self.capture_image = capture_image
        self.on_detection = on_detection

        self._running = False
        self._scan_task: Optional[asyncio.Task] = None
        self._last_detections: Dict[DetectionType, float] = {}
        self._stare_tracker: Dict[str, float] = {}  # content_hash -> first_seen_time

        # Detection state
        self._battery_saver = False
        self._paused = False
        self._current_fps = 1.0 / config.scan_interval_seconds

    async def start(self):
        """Start background scanning."""
        if not self.config.enabled:
            logger.info("EDITH disabled in config")
            return

        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        logger.info("EDITH scanner started")

    async def stop(self):
        """Stop background scanning."""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        logger.info("EDITH scanner stopped")

    def set_battery_saver(self, enabled: bool):
        """Enable/disable battery saver mode."""
        self._battery_saver = enabled
        if enabled:
            self._current_fps = 1.0 / self.config.battery_saver_interval
        else:
            self._current_fps = 1.0 / self.config.scan_interval_seconds
        logger.info(f"Battery saver: {enabled}, FPS: {self._current_fps}")

    async def pause(self):
        """Pause scanning without stopping."""
        self._paused = True
        logger.info("EDITH paused")

    async def resume(self):
        """Resume scanning."""
        self._paused = False
        logger.info("EDITH resumed")

    @property
    def is_paused(self) -> bool:
        """Check if EDITH is paused."""
        return self._paused

    async def _scan_loop(self):
        """Main scanning loop."""
        while self._running:
            try:
                # Check if paused
                if self._paused:
                    await asyncio.sleep(0.5)
                    continue

                interval = self.config.battery_saver_interval if self._battery_saver else self.config.scan_interval_seconds
                await asyncio.sleep(interval)

                if not self._running:
                    break

                # Capture image
                image_data = await self.capture_image()
                if image_data is None:
                    continue

                # Analyze image
                detections = await self._analyze_image(image_data)

                # Process detections
                for detection in detections:
                    if self._should_notify(detection):
                        await self.on_detection(detection)
                        self._last_detections[detection.type] = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scan error: {e}")

    async def _analyze_image(self, image_data: bytes) -> List[Detection]:
        """
        Analyze image for detections.

        Uses Gemini for screen-based classification (poker tables, code editors, math problems),
        falls back to OpenCV detectors for traditional vision tasks.
        """
        detections = []

        if not image_data:
            return detections

        # Try Gemini classification for screen-based detection first
        try:
            from ..api_clients.gemini_client import GeminiClient
            gemini = GeminiClient()

            if gemini.is_available:
                classification = await gemini.analyze_image(
                    image_data,
                    prompt="Classify this image. Return ONLY one of: POKER_TABLE, CODE_EDITOR, MATH_PROBLEM, POKER_CARDS, TEXT, OTHER"
                )

                classification = classification.strip().upper()

                # Map classification to DetectionType
                if "POKER_TABLE" in classification:
                    detections.append(Detection(
                        type=DetectionType.POKER_TABLE,
                        confidence=0.9,
                        bounding_box=(0, 0, 1280, 720),
                        content="Poker table detected via Gemini"
                    ))

                elif "CODE_EDITOR" in classification or "CODE" in classification:
                    detections.append(Detection(
                        type=DetectionType.CODE_EDITOR,
                        confidence=0.9,
                        bounding_box=(0, 0, 1280, 720),
                        content="Code editor detected via Gemini"
                    ))

                elif "MATH" in classification:
                    detections.append(Detection(
                        type=DetectionType.MATH_PROBLEM,
                        confidence=0.9,
                        bounding_box=(0, 0, 1280, 720),
                        content="Math problem detected via Gemini"
                    ))

                elif "POKER_CARDS" in classification:
                    detections.append(Detection(
                        type=DetectionType.POKER_CARDS,
                        confidence=0.9,
                        bounding_box=(0, 0, 1280, 720),
                        content="Poker cards detected via Gemini"
                    ))

                # If Gemini found something, return early
                if detections:
                    return detections

        except Exception as e:
            logger.debug(f"Gemini classification failed, falling back to OpenCV: {e}")

        # Fallback to OpenCV detectors
        if not CV2_AVAILABLE:
            return detections

        try:
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return detections

            # Run enabled detectors
            if "equation" in self.config.detection_types:
                eq_detections = self._detect_equations(frame)
                detections.extend(eq_detections)

            if "code" in self.config.detection_types:
                code_detections = self._detect_code(frame)
                detections.extend(code_detections)

            if "text" in self.config.detection_types:
                text_detections = self._detect_text(frame)
                detections.extend(text_detections)

            if "face" in self.config.detection_types:
                face_detections = self._detect_faces(frame)
                detections.extend(face_detections)

            if "qr_code" in self.config.detection_types:
                qr_detections = self._detect_qr(frame)
                detections.extend(qr_detections)

        except Exception as e:
            logger.error(f"OpenCV analysis error: {e}")

        return detections

    def _detect_equations(self, frame) -> List[Detection]:
        """
        Detect mathematical equations in frame.

        Uses OpenCV-based pattern detection from detectors module.
        """
        try:
            from .detectors import EquationDetector
            detector = EquationDetector()
            result = detector.detect(frame)

            if result and result.detected:
                return [Detection(
                    type=DetectionType.EQUATION,
                    confidence=result.confidence,
                    bounding_box=result.bounding_box or (0, 0, frame.shape[1], frame.shape[0]),
                    content=""
                )]
        except Exception as e:
            logger.debug(f"Equation detector error: {e}")

        return []

    def _detect_code(self, frame) -> List[Detection]:
        """
        Detect code snippets in frame.

        Uses OpenCV-based dark background and structure detection.
        """
        try:
            from .detectors import CodeDetector
            detector = CodeDetector()
            result = detector.detect(frame)

            if result and result.detected:
                return [Detection(
                    type=DetectionType.CODE,
                    confidence=result.confidence,
                    bounding_box=result.bounding_box or (0, 0, frame.shape[1], frame.shape[0]),
                    content=""
                )]
        except Exception as e:
            logger.debug(f"Code detector error: {e}")

        return []

    def _detect_text(self, frame) -> List[Detection]:
        """
        Detect general text in frame.

        Uses OpenCV-based paragraph structure detection.
        """
        try:
            from .detectors import TextDetector
            detector = TextDetector()
            result = detector.detect(frame)

            if result and result.detected:
                return [Detection(
                    type=DetectionType.TEXT,
                    confidence=result.confidence,
                    bounding_box=result.bounding_box or (0, 0, frame.shape[1], frame.shape[0]),
                    content=""
                )]
        except Exception as e:
            logger.debug(f"Text detector error: {e}")

        return []

    def _detect_faces(self, frame) -> List[Detection]:
        """
        Detect faces in frame.
        """
        detections = []

        try:
            # Use OpenCV's built-in face detector
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                detections.append(Detection(
                    type=DetectionType.FACE,
                    confidence=0.8,
                    bounding_box=(x, y, w, h),
                    content="Unknown face"
                ))

        except Exception as e:
            logger.debug(f"Face detection error: {e}")

        return detections

    def _detect_qr(self, frame) -> List[Detection]:
        """
        Detect and decode QR codes.
        """
        detections = []

        try:
            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(frame)

            if data:
                x, y = int(bbox[0][0][0]), int(bbox[0][0][1])
                w = int(bbox[0][2][0] - x)
                h = int(bbox[0][2][1] - y)

                detections.append(Detection(
                    type=DetectionType.QR_CODE,
                    confidence=1.0,
                    bounding_box=(x, y, w, h),
                    content=data
                ))

        except Exception as e:
            logger.debug(f"QR detection error: {e}")

        return detections

    def _should_notify(self, detection: Detection) -> bool:
        """
        Check if we should notify about this detection.

        Avoids spam by:
        - Checking cooldown since last similar detection
        - Requiring stare threshold for suggestions
        """
        # Check cooldown
        last_time = self._last_detections.get(detection.type, 0)
        if (time.time() - last_time) * 1000 < self.config.suggestion_cooldown_ms:
            return False

        # Check confidence threshold
        min_confidence = {
            DetectionType.EQUATION: 0.7,
            DetectionType.CODE: 0.7,
            DetectionType.TEXT: 0.6,
            DetectionType.FACE: 0.8,
            DetectionType.QR_CODE: 0.9,
            DetectionType.POKER_CARDS: 0.8,
            # Screen-based detection (Gemini)
            DetectionType.POKER_TABLE: 0.7,
            DetectionType.CODE_EDITOR: 0.7,
            DetectionType.MATH_PROBLEM: 0.7,
        }

        if detection.confidence < min_confidence.get(detection.type, 0.7):
            return False

        return True

    def track_stare(self, content_hash: str) -> bool:
        """
        Track staring at content.

        Returns True if user has been looking at content
        long enough to suggest assistance.
        """
        now = time.time()

        if content_hash not in self._stare_tracker:
            self._stare_tracker[content_hash] = now
            return False

        stare_time_ms = (now - self._stare_tracker[content_hash]) * 1000
        return stare_time_ms >= self.config.stare_threshold_ms

    def clear_stare_tracker(self):
        """Clear stare tracking (e.g., when user looks away)."""
        self._stare_tracker.clear()


# Suggestion generator
class SuggestionGenerator:
    """
    Generates helpful suggestions based on EDITH detections.
    """

    SUGGESTIONS = {
        DetectionType.EQUATION: [
            "I notice an equation. Would you like me to solve it?",
            "Math detected. Need help?",
        ],
        DetectionType.CODE: [
            "I see some code. Want me to explain or debug it?",
            "Code detected. Shall I analyze it?",
        ],
        DetectionType.TEXT: [
            "There's text here. Would you like a summary?",
        ],
        DetectionType.QR_CODE: [
            "QR code detected: {content}",
        ],
        DetectionType.POKER_CARDS: [
            "I see cards. Want GTO analysis?",
        ],
    }

    @classmethod
    def get_suggestion(cls, detection: Detection) -> str:
        """Get a suggestion for a detection."""
        import random

        templates = cls.SUGGESTIONS.get(detection.type, ["Detection found."])
        template = random.choice(templates)

        return template.format(
            content=detection.content[:50] if detection.content else ""
        )


# Test
async def test_edith():
    """Test EDITH scanner."""
    print("=== EDITH Scanner Test ===\n")

    async def mock_capture():
        # Return None (no camera in test)
        return None

    async def on_detect(detection):
        print(f"Detection: {detection.type.value} - {detection.confidence:.2f}")
        suggestion = SuggestionGenerator.get_suggestion(detection)
        print(f"Suggestion: {suggestion}")

    config = ScanConfig(
        enabled=True,
        scan_interval_seconds=2.0,
        detection_types=["equation", "code", "text", "face", "qr_code"]
    )

    edith = EdithScanner(
        config=config,
        capture_image=mock_capture,
        on_detection=on_detect
    )

    print("Starting EDITH (will run for 5 seconds)...")
    await edith.start()
    await asyncio.sleep(5)
    await edith.stop()
    print("EDITH stopped")


if __name__ == "__main__":
    asyncio.run(test_edith())
