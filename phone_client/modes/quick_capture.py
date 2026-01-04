"""
Quick Capture - Voice and visual note-taking with intelligent tagging.
Capture thoughts, images, and context on-the-fly with minimal friction.
"""
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CaptureType(Enum):
    """Types of captures."""
    VOICE = "voice"          # Voice memo/note
    PHOTO = "photo"          # Camera capture
    SCREENSHOT = "screenshot"  # Screen/display capture
    TEXT = "text"            # Manual text entry
    LOCATION = "location"    # Location bookmark
    REMINDER = "reminder"    # Quick reminder
    IDEA = "idea"            # Idea/thought
    TODO = "todo"            # Action item


class CapturePriority(Enum):
    """Priority levels for captures."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Capture:
    """Single captured item."""
    id: str
    type: CaptureType
    content: str                           # Text content or file path
    created_at: datetime
    tags: List[str] = field(default_factory=list)
    priority: CapturePriority = CapturePriority.NORMAL
    context: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    reminder_time: Optional[datetime] = None
    location: Optional[str] = None
    attachments: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "priority": self.priority.value,
            "context": self.context,
            "processed": self.processed,
            "reminder_time": self.reminder_time.isoformat() if self.reminder_time else None,
            "location": self.location,
            "attachments": self.attachments
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Capture":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=CaptureType(data["type"]),
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            tags=data.get("tags", []),
            priority=CapturePriority(data.get("priority", 2)),
            context=data.get("context", {}),
            processed=data.get("processed", False),
            reminder_time=datetime.fromisoformat(data["reminder_time"]) if data.get("reminder_time") else None,
            location=data.get("location"),
            attachments=data.get("attachments", [])
        )


class QuickCapture:
    """
    Quick Capture system for WHAM.

    Provides frictionless capture of voice notes, photos, ideas, and reminders
    with intelligent auto-tagging and organization.
    """

    # Auto-tag keywords
    AUTO_TAGS = {
        "work": ["meeting", "project", "deadline", "client", "email", "task", "report"],
        "personal": ["buy", "call", "birthday", "dinner", "weekend", "family"],
        "idea": ["what if", "could we", "maybe", "think about", "consider"],
        "urgent": ["asap", "urgent", "immediately", "right now", "emergency"],
        "shopping": ["buy", "get", "pick up", "grocery", "store", "amazon"],
        "health": ["doctor", "appointment", "medicine", "workout", "gym"],
        "finance": ["pay", "bill", "money", "budget", "expense", "invoice"],
    }

    # Voice command triggers
    VOICE_TRIGGERS = {
        "note": CaptureType.VOICE,
        "capture": CaptureType.VOICE,
        "remember": CaptureType.VOICE,
        "remind me": CaptureType.REMINDER,
        "todo": CaptureType.TODO,
        "idea": CaptureType.IDEA,
        "photo": CaptureType.PHOTO,
        "screenshot": CaptureType.SCREENSHOT,
        "bookmark": CaptureType.LOCATION,
    }

    def __init__(self, config: dict, storage_dir: str = "./captures"):
        """
        Initialize Quick Capture.

        Args:
            config: Configuration dictionary
            storage_dir: Directory to store captures
        """
        self.config = config
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._captures: Dict[str, Capture] = {}
        self._pending_reminders: List[Capture] = []

        # Callbacks
        self._on_capture: List[Callable[[Capture], None]] = []

        # Load settings
        capture_config = config.get("quick_capture", {})
        self.auto_tag = capture_config.get("auto_tag", True)
        self.auto_categorize = capture_config.get("auto_categorize", True)
        self.max_voice_length_s = capture_config.get("max_voice_length_s", 60)
        self.default_reminder_delay_min = capture_config.get("default_reminder_delay_min", 30)

        # Load existing captures
        self._load_captures()

        logger.info(f"QuickCapture initialized (storage: {self.storage_dir})")

    def _load_captures(self):
        """Load captures from storage."""
        captures_file = self.storage_dir / "captures.json"
        if captures_file.exists():
            try:
                with open(captures_file, "r") as f:
                    data = json.load(f)
                    for item in data:
                        capture = Capture.from_dict(item)
                        self._captures[capture.id] = capture
                        if capture.reminder_time and not capture.processed:
                            self._pending_reminders.append(capture)
                logger.info(f"Loaded {len(self._captures)} captures")
            except Exception as e:
                logger.error(f"Failed to load captures: {e}")

    def _save_captures(self):
        """Save captures to storage."""
        captures_file = self.storage_dir / "captures.json"
        try:
            data = [c.to_dict() for c in self._captures.values()]
            with open(captures_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save captures: {e}")

    def capture_voice(self, transcript: str, context: Dict[str, Any] = None) -> Capture:
        """
        Capture a voice note.

        Args:
            transcript: Transcribed voice content
            context: Optional context (location, activity, etc.)

        Returns:
            Created Capture
        """
        # Detect capture type from trigger words
        capture_type = self._detect_type_from_content(transcript)

        # Extract reminder time if applicable
        reminder_time = None
        if capture_type == CaptureType.REMINDER:
            reminder_time, transcript = self._extract_reminder_time(transcript)

        # Auto-tag
        tags = self._auto_tag(transcript) if self.auto_tag else []

        # Detect priority
        priority = self._detect_priority(transcript)

        capture = Capture(
            id=str(uuid.uuid4())[:8],
            type=capture_type,
            content=transcript,
            created_at=datetime.now(),
            tags=tags,
            priority=priority,
            context=context or {},
            reminder_time=reminder_time
        )

        self._add_capture(capture)
        return capture

    def capture_photo(self, image_path: str, caption: str = "", context: Dict[str, Any] = None) -> Capture:
        """
        Capture a photo with optional caption.

        Args:
            image_path: Path to captured image
            caption: Optional text caption
            context: Optional context

        Returns:
            Created Capture
        """
        tags = self._auto_tag(caption) if caption and self.auto_tag else []

        capture = Capture(
            id=str(uuid.uuid4())[:8],
            type=CaptureType.PHOTO,
            content=caption or "Photo capture",
            created_at=datetime.now(),
            tags=tags,
            context=context or {},
            attachments=[image_path]
        )

        self._add_capture(capture)
        return capture

    def capture_text(self, text: str, capture_type: CaptureType = CaptureType.TEXT) -> Capture:
        """
        Capture text directly.

        Args:
            text: Text content
            capture_type: Type of capture

        Returns:
            Created Capture
        """
        tags = self._auto_tag(text) if self.auto_tag else []
        priority = self._detect_priority(text)

        capture = Capture(
            id=str(uuid.uuid4())[:8],
            type=capture_type,
            content=text,
            created_at=datetime.now(),
            tags=tags,
            priority=priority
        )

        self._add_capture(capture)
        return capture

    def capture_reminder(self, text: str, remind_at: datetime = None) -> Capture:
        """
        Create a reminder.

        Args:
            text: Reminder content
            remind_at: When to remind (defaults to configured delay)

        Returns:
            Created Capture
        """
        if remind_at is None:
            from datetime import timedelta
            remind_at = datetime.now() + timedelta(minutes=self.default_reminder_delay_min)

        tags = self._auto_tag(text) if self.auto_tag else []
        tags.append("reminder")

        capture = Capture(
            id=str(uuid.uuid4())[:8],
            type=CaptureType.REMINDER,
            content=text,
            created_at=datetime.now(),
            tags=tags,
            priority=CapturePriority.HIGH,
            reminder_time=remind_at
        )

        self._add_capture(capture)
        self._pending_reminders.append(capture)
        return capture

    def _add_capture(self, capture: Capture):
        """Add capture to storage and notify callbacks."""
        self._captures[capture.id] = capture
        self._save_captures()

        # Notify callbacks
        for callback in self._on_capture:
            try:
                callback(capture)
            except Exception as e:
                logger.error(f"Capture callback error: {e}")

        logger.info(f"Captured: [{capture.type.value}] {capture.content[:50]}...")

    def _detect_type_from_content(self, content: str) -> CaptureType:
        """Detect capture type from content keywords."""
        content_lower = content.lower()

        for trigger, capture_type in self.VOICE_TRIGGERS.items():
            if content_lower.startswith(trigger):
                return capture_type

        return CaptureType.VOICE

    def _auto_tag(self, content: str) -> List[str]:
        """Auto-generate tags from content."""
        content_lower = content.lower()
        tags = []

        for tag, keywords in self.AUTO_TAGS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    tags.append(tag)
                    break

        return list(set(tags))

    def _detect_priority(self, content: str) -> CapturePriority:
        """Detect priority from content."""
        content_lower = content.lower()

        urgent_words = ["urgent", "asap", "emergency", "immediately", "critical"]
        high_words = ["important", "priority", "don't forget", "must"]

        for word in urgent_words:
            if word in content_lower:
                return CapturePriority.URGENT

        for word in high_words:
            if word in content_lower:
                return CapturePriority.HIGH

        return CapturePriority.NORMAL

    def _extract_reminder_time(self, content: str) -> tuple:
        """
        Extract reminder time from content.

        Returns:
            Tuple of (reminder_datetime, cleaned_content)
        """
        from datetime import timedelta
        content_lower = content.lower()
        now = datetime.now()
        reminder_time = None
        cleaned = content

        # Simple time extraction patterns
        patterns = {
            "in 5 minutes": timedelta(minutes=5),
            "in 10 minutes": timedelta(minutes=10),
            "in 15 minutes": timedelta(minutes=15),
            "in 30 minutes": timedelta(minutes=30),
            "in an hour": timedelta(hours=1),
            "in 1 hour": timedelta(hours=1),
            "in 2 hours": timedelta(hours=2),
            "tomorrow": timedelta(days=1),
            "tonight": None,  # Special handling
            "this evening": None,
        }

        for pattern, delta in patterns.items():
            if pattern in content_lower:
                if delta:
                    reminder_time = now + delta
                else:
                    # Tonight = 7pm today
                    reminder_time = now.replace(hour=19, minute=0, second=0)
                cleaned = content_lower.replace(pattern, "").strip()
                break

        if not reminder_time:
            reminder_time = now + timedelta(minutes=self.default_reminder_delay_min)

        return reminder_time, cleaned

    def get_capture(self, capture_id: str) -> Optional[Capture]:
        """Get capture by ID."""
        return self._captures.get(capture_id)

    def get_recent(self, limit: int = 10) -> List[Capture]:
        """Get recent captures."""
        captures = sorted(
            self._captures.values(),
            key=lambda c: c.created_at,
            reverse=True
        )
        return captures[:limit]

    def get_by_tag(self, tag: str) -> List[Capture]:
        """Get captures by tag."""
        return [c for c in self._captures.values() if tag in c.tags]

    def get_by_type(self, capture_type: CaptureType) -> List[Capture]:
        """Get captures by type."""
        return [c for c in self._captures.values() if c.type == capture_type]

    def get_pending_reminders(self) -> List[Capture]:
        """Get reminders that are due."""
        now = datetime.now()
        due = []
        for capture in self._pending_reminders:
            if capture.reminder_time and capture.reminder_time <= now:
                due.append(capture)
        return due

    def mark_processed(self, capture_id: str):
        """Mark capture as processed."""
        if capture_id in self._captures:
            self._captures[capture_id].processed = True
            # Remove from pending reminders
            self._pending_reminders = [
                c for c in self._pending_reminders if c.id != capture_id
            ]
            self._save_captures()

    def delete_capture(self, capture_id: str) -> bool:
        """Delete a capture."""
        if capture_id in self._captures:
            del self._captures[capture_id]
            self._pending_reminders = [
                c for c in self._pending_reminders if c.id != capture_id
            ]
            self._save_captures()
            return True
        return False

    def search(self, query: str) -> List[Capture]:
        """Search captures by content."""
        query_lower = query.lower()
        results = []
        for capture in self._captures.values():
            if query_lower in capture.content.lower():
                results.append(capture)
            elif any(query_lower in tag for tag in capture.tags):
                results.append(capture)
        return results

    def get_stats(self) -> dict:
        """Get capture statistics."""
        total = len(self._captures)
        by_type = {}
        by_tag = {}

        for capture in self._captures.values():
            # Count by type
            type_name = capture.type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            # Count by tag
            for tag in capture.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1

        return {
            "total": total,
            "by_type": by_type,
            "by_tag": by_tag,
            "pending_reminders": len(self._pending_reminders),
            "unprocessed": len([c for c in self._captures.values() if not c.processed])
        }

    def on_capture(self, callback: Callable[[Capture], None]):
        """Register callback for new captures."""
        self._on_capture.append(callback)

    def format_for_display(self, capture: Capture) -> List[str]:
        """Format capture for HUD display."""
        lines = []

        # Type icon
        icons = {
            CaptureType.VOICE: "🎤",
            CaptureType.PHOTO: "📷",
            CaptureType.SCREENSHOT: "📱",
            CaptureType.TEXT: "📝",
            CaptureType.LOCATION: "📍",
            CaptureType.REMINDER: "⏰",
            CaptureType.IDEA: "💡",
            CaptureType.TODO: "☑️",
        }
        icon = icons.get(capture.type, "•")

        # Header
        lines.append(f"{icon} {capture.type.value.upper()}")
        lines.append(f"ID: {capture.id}")
        lines.append("")

        # Content
        content = capture.content
        if len(content) > 100:
            content = content[:97] + "..."
        lines.append(content)

        # Tags
        if capture.tags:
            lines.append("")
            lines.append(f"Tags: {', '.join(capture.tags)}")

        # Reminder time
        if capture.reminder_time:
            lines.append(f"Remind: {capture.reminder_time.strftime('%I:%M %p')}")

        return lines


# Test
def test_quick_capture():
    """Test quick capture functionality."""
    import tempfile

    print("=== Quick Capture Test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "quick_capture": {
                "auto_tag": True,
                "auto_categorize": True,
                "default_reminder_delay_min": 15
            }
        }

        qc = QuickCapture(config, storage_dir=tmpdir)

        # Test voice capture
        print("1. Voice capture:")
        capture1 = qc.capture_voice("Remember to call mom about the birthday dinner")
        print(f"   Content: {capture1.content}")
        print(f"   Tags: {capture1.tags}")
        print(f"   Priority: {capture1.priority.name}")
        print()

        # Test reminder
        print("2. Reminder capture:")
        capture2 = qc.capture_voice("Remind me in 30 minutes to check the oven")
        print(f"   Content: {capture2.content}")
        print(f"   Reminder: {capture2.reminder_time}")
        print()

        # Test todo
        print("3. Todo capture:")
        capture3 = qc.capture_voice("Todo urgent: submit the project report")
        print(f"   Type: {capture3.type.value}")
        print(f"   Priority: {capture3.priority.name}")
        print()

        # Test idea
        print("4. Idea capture:")
        capture4 = qc.capture_voice("Idea: what if we added voice shortcuts?")
        print(f"   Type: {capture4.type.value}")
        print(f"   Tags: {capture4.tags}")
        print()

        # Stats
        print("5. Stats:")
        stats = qc.get_stats()
        print(f"   Total: {stats['total']}")
        print(f"   By type: {stats['by_type']}")
        print(f"   By tag: {stats['by_tag']}")
        print()

        # Search
        print("6. Search 'birthday':")
        results = qc.search("birthday")
        for r in results:
            print(f"   [{r.type.value}] {r.content[:50]}...")
        print()

        # Display format
        print("7. Display format:")
        for line in qc.format_for_display(capture1):
            print(f"   {line}")


if __name__ == "__main__":
    test_quick_capture()
