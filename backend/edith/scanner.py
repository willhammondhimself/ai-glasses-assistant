"""
EDITH Core Scanner

The main proactive scanning engine that analyzes camera frames
and detects actionable content for AR assistance.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Set
from collections import deque

logger = logging.getLogger(__name__)


class DetectionType(Enum):
    """Types of content EDITH can detect."""
    EQUATION = "equation"
    CODE = "code"
    POKER_CARDS = "poker_cards"
    TEXT = "text"
    DOCUMENT = "document"
    QR_CODE = "qr_code"
    FACE = "face"
    OBJECT = "object"
    NONE = "none"


class ScanPriority(Enum):
    """Priority levels for scan processing."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class ScanConfig:
    """Configuration for EDITH scanning behavior."""
    # FPS settings
    base_fps: float = 2.0          # Idle scanning rate
    active_fps: float = 10.0       # When content detected
    max_fps: float = 15.0          # Maximum rate

    # Detection thresholds
    min_confidence: float = 0.7    # Minimum detection confidence
    stare_threshold_ms: int = 1500 # Time looking at content to trigger suggestion

    # Cooldown settings (prevent spam)
    suggestion_cooldown_ms: int = 10000  # 10s between same-type suggestions
    duplicate_cooldown_ms: int = 30000   # 30s for same exact content

    # Power management
    battery_save_threshold: float = 0.2  # Enable battery saver below 20%
    idle_timeout_s: int = 60             # Reduce scanning after inactivity

    # Detection toggles
    detect_equations: bool = True
    detect_code: bool = True
    detect_poker: bool = True
    detect_text: bool = True


@dataclass
class ScanResult:
    """Result of a single frame scan."""
    detection_type: DetectionType
    confidence: float
    content: str                    # Extracted content (equation, code, etc.)
    bounding_box: Optional[Dict[str, int]] = None  # x, y, width, height
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    frame_id: Optional[str] = None

    @property
    def is_actionable(self) -> bool:
        """Whether this detection warrants user notification."""
        return self.confidence >= 0.7 and self.detection_type != DetectionType.NONE


@dataclass
class StareTracker:
    """Tracks user's gaze/attention on detected content."""
    content_hash: str
    detection_type: DetectionType
    first_seen: datetime
    last_seen: datetime
    frame_count: int = 1

    @property
    def duration_ms(self) -> int:
        """How long user has been looking at this content."""
        return int((self.last_seen - self.first_seen).total_seconds() * 1000)


class EdithScanner:
    """
    EDITH Proactive Scanner Engine

    Continuously analyzes camera frames to detect:
    - Math equations (auto-solve suggestions)
    - Code snippets (debug/explain suggestions)
    - Poker hands (GTO analysis suggestions)
    - Text documents (summary/analysis suggestions)

    Uses adaptive power management to balance responsiveness
    with battery life on the glasses.
    """

    def __init__(self, config: Optional[ScanConfig] = None):
        self.config = config or ScanConfig()

        # Detection state
        self._running = False
        self._current_fps = self.config.base_fps
        self._last_detection: Optional[ScanResult] = None

        # Stare tracking
        self._stare_tracker: Optional[StareTracker] = None
        self._stare_threshold_reached = False

        # Cooldown tracking
        self._suggestion_cooldowns: Dict[DetectionType, datetime] = {}
        self._content_cooldowns: Dict[str, datetime] = {}

        # Recent detections for deduplication
        self._recent_detections: deque = deque(maxlen=50)

        # Callbacks
        self._on_detection: Optional[Callable[[ScanResult], None]] = None
        self._on_suggestion: Optional[Callable[[ScanResult, str], None]] = None

        # Detectors (lazy loaded)
        self._detectors: Dict[DetectionType, Any] = {}

        # Stats
        self._scan_count = 0
        self._detection_count = 0
        self._suggestion_count = 0

    def set_detection_callback(
        self,
        callback: Callable[[ScanResult], None]
    ) -> None:
        """Set callback for when content is detected."""
        self._on_detection = callback

    def set_suggestion_callback(
        self,
        callback: Callable[[ScanResult, str], None]
    ) -> None:
        """Set callback for when a suggestion should be shown."""
        self._on_suggestion = callback

    async def start(self) -> None:
        """Start the scanning loop."""
        self._running = True
        logger.info("EDITH scanner started")

    async def stop(self) -> None:
        """Stop the scanning loop."""
        self._running = False
        logger.info("EDITH scanner stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    async def process_frame(
        self,
        frame_data: bytes,
        frame_id: Optional[str] = None,
    ) -> Optional[ScanResult]:
        """
        Process a single camera frame.

        This is the main entry point called by the glasses client
        for each captured frame.

        Args:
            frame_data: Raw image bytes (JPEG or PNG)
            frame_id: Optional unique frame identifier

        Returns:
            ScanResult if content detected, None otherwise
        """
        if not self._running:
            return None

        self._scan_count += 1

        # Quick on-device classification first
        # (In production, this would use a lightweight model on the glasses)
        detection_type = await self._quick_classify(frame_data)

        if detection_type == DetectionType.NONE:
            # Nothing detected, reset stare tracker
            self._handle_no_detection()
            return None

        # Full detection with API call
        result = await self._full_detect(frame_data, detection_type, frame_id)

        if result and result.is_actionable:
            self._detection_count += 1
            await self._handle_detection(result)
            return result

        return None

    async def _quick_classify(self, frame_data: bytes) -> DetectionType:
        """
        Quick on-device classification.

        Uses lightweight heuristics before making expensive API calls:
        - Edge detection for equations/text
        - Color patterns for playing cards
        - Syntax highlighting patterns for code

        In production, this would use a small neural network on the glasses.
        """
        # Placeholder - would use actual CV here
        # For now, return NONE and let full detection handle it
        return DetectionType.NONE

    async def _full_detect(
        self,
        frame_data: bytes,
        hint_type: DetectionType,
        frame_id: Optional[str],
    ) -> Optional[ScanResult]:
        """
        Full detection using API/vision model.

        Args:
            frame_data: Image bytes
            hint_type: Quick classification hint
            frame_id: Frame identifier

        Returns:
            Detailed ScanResult
        """
        # This would call the appropriate detector based on hint
        # For now, return placeholder
        return None

    async def _handle_detection(self, result: ScanResult) -> None:
        """Handle a successful detection."""
        # Check if this is duplicate content
        content_hash = self._hash_content(result)
        if self._is_duplicate(content_hash):
            return

        # Update stare tracker
        self._update_stare_tracker(result, content_hash)

        # Fire detection callback
        if self._on_detection:
            self._on_detection(result)

        # Check if stare threshold reached for suggestion
        if self._should_suggest(result, content_hash):
            await self._make_suggestion(result)

        # Increase scan rate while content is present
        self._current_fps = self.config.active_fps

    def _handle_no_detection(self) -> None:
        """Handle frame with no detection."""
        # Decay back to base FPS
        if self._current_fps > self.config.base_fps:
            self._current_fps = max(
                self.config.base_fps,
                self._current_fps * 0.9
            )

        # Check if stare tracker should be reset
        if self._stare_tracker:
            elapsed = (datetime.now() - self._stare_tracker.last_seen).total_seconds()
            if elapsed > 0.5:  # Content out of view for 500ms
                self._stare_tracker = None
                self._stare_threshold_reached = False

    def _update_stare_tracker(
        self,
        result: ScanResult,
        content_hash: str
    ) -> None:
        """Update the stare/attention tracker."""
        now = datetime.now()

        if (self._stare_tracker and
            self._stare_tracker.content_hash == content_hash):
            # Same content, update tracker
            self._stare_tracker.last_seen = now
            self._stare_tracker.frame_count += 1
        else:
            # New content, reset tracker
            self._stare_tracker = StareTracker(
                content_hash=content_hash,
                detection_type=result.detection_type,
                first_seen=now,
                last_seen=now,
            )
            self._stare_threshold_reached = False

    def _should_suggest(self, result: ScanResult, content_hash: str) -> bool:
        """Check if we should make a suggestion for this detection."""
        if not self._stare_tracker:
            return False

        # Check stare duration
        if self._stare_tracker.duration_ms < self.config.stare_threshold_ms:
            return False

        # Already suggested for this stare session
        if self._stare_threshold_reached:
            return False

        # Check type cooldown
        if result.detection_type in self._suggestion_cooldowns:
            cooldown_end = self._suggestion_cooldowns[result.detection_type]
            if datetime.now() < cooldown_end:
                return False

        # Check content cooldown
        if content_hash in self._content_cooldowns:
            cooldown_end = self._content_cooldowns[content_hash]
            if datetime.now() < cooldown_end:
                return False

        return True

    async def _make_suggestion(self, result: ScanResult) -> None:
        """Make a proactive suggestion to the user."""
        self._stare_threshold_reached = True
        self._suggestion_count += 1

        # Set cooldowns
        self._suggestion_cooldowns[result.detection_type] = (
            datetime.now() + timedelta(milliseconds=self.config.suggestion_cooldown_ms)
        )
        content_hash = self._hash_content(result)
        self._content_cooldowns[content_hash] = (
            datetime.now() + timedelta(milliseconds=self.config.duplicate_cooldown_ms)
        )

        # Generate suggestion message
        suggestion = self._generate_suggestion(result)

        # Fire callback
        if self._on_suggestion:
            self._on_suggestion(result, suggestion)

        logger.info(f"EDITH suggestion: {result.detection_type.value} - {suggestion}")

    def _generate_suggestion(self, result: ScanResult) -> str:
        """Generate JARVIS-style suggestion message."""
        suggestions = {
            DetectionType.EQUATION: "I've detected an equation, sir. Shall I solve it?",
            DetectionType.CODE: "Code snippet detected. Would you like me to analyze it?",
            DetectionType.POKER_CARDS: "I see poker cards. Want a GTO analysis?",
            DetectionType.TEXT: "Text detected. Shall I summarize or analyze?",
            DetectionType.DOCUMENT: "Document in view. Would you like a summary?",
        }
        return suggestions.get(
            result.detection_type,
            "Content detected. How may I assist, sir?"
        )

    def _hash_content(self, result: ScanResult) -> str:
        """Generate hash for content deduplication."""
        # Simple hash based on content and type
        return f"{result.detection_type.value}:{hash(result.content)}"

    def _is_duplicate(self, content_hash: str) -> bool:
        """Check if this content was recently detected."""
        for recent in self._recent_detections:
            if recent == content_hash:
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get scanner statistics."""
        return {
            "running": self._running,
            "current_fps": self._current_fps,
            "scan_count": self._scan_count,
            "detection_count": self._detection_count,
            "suggestion_count": self._suggestion_count,
            "detection_rate": (
                self._detection_count / self._scan_count
                if self._scan_count > 0 else 0
            ),
        }

    def reset_cooldowns(self) -> None:
        """Reset all cooldowns (for testing/user override)."""
        self._suggestion_cooldowns.clear()
        self._content_cooldowns.clear()

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
