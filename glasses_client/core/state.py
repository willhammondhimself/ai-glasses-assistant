"""
Session State Management

Manages state across the glasses client session including:
- Current mode and context
- User preferences
- Offline queue
- Performance metrics
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import json


class AppMode(Enum):
    """Application modes."""
    IDLE = "idle"
    MENTAL_MATH = "mental_math"
    CAMERA_SOLVE = "camera_solve"
    CODE_DEBUG = "code_debug"
    CONTENT = "content"
    MEETING = "meeting"


@dataclass
class UserPreferences:
    """User preferences for the glasses client."""
    # Voice settings
    voice_enabled: bool = True
    voice_speed: float = 1.0
    voice_volume: float = 0.8

    # Display settings
    auto_advance_slides: bool = False
    auto_advance_delay_ms: int = 3000
    show_hints: bool = True

    # Mental math settings
    default_difficulty: int = 2
    timer_sounds: bool = True

    # Offline settings
    pre_cache_problems: bool = True
    cache_size_mb: int = 100


@dataclass
class SessionMetrics:
    """Metrics for the current session."""
    started_at: datetime = field(default_factory=datetime.utcnow)
    problems_attempted: int = 0
    problems_correct: int = 0
    total_time_ms: int = 0
    current_streak: int = 0
    best_streak: int = 0
    api_calls: int = 0
    cache_hits: int = 0

    @property
    def accuracy(self) -> float:
        if self.problems_attempted == 0:
            return 0.0
        return self.problems_correct / self.problems_attempted

    @property
    def avg_time_ms(self) -> float:
        if self.problems_attempted == 0:
            return 0.0
        return self.total_time_ms / self.problems_attempted


@dataclass
class PendingSyncItem:
    """Item waiting to be synced with server."""
    id: str
    action: str
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    attempts: int = 0


class SessionState:
    """
    Manages all session state for the glasses client.

    Provides:
    - Mode tracking
    - User preferences
    - Session metrics
    - Offline queue management
    """

    def __init__(self):
        self.mode = AppMode.IDLE
        self.preferences = UserPreferences()
        self.metrics = SessionMetrics()
        self._context: Dict[str, Any] = {}
        self._pending_sync: List[PendingSyncItem] = []
        self._is_online = True

    # Mode management

    def set_mode(self, mode: AppMode):
        """Set the current application mode."""
        self.mode = mode
        self._context = {}  # Clear context on mode change

    def get_mode(self) -> AppMode:
        """Get the current application mode."""
        return self.mode

    # Context management (mode-specific state)

    def set_context(self, key: str, value: Any):
        """Set a context value for the current mode."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._context.get(key, default)

    def clear_context(self):
        """Clear all context."""
        self._context = {}

    # Metrics tracking

    def record_attempt(self, correct: bool, time_ms: int):
        """Record a problem attempt."""
        self.metrics.problems_attempted += 1
        self.metrics.total_time_ms += time_ms

        if correct:
            self.metrics.problems_correct += 1
            self.metrics.current_streak += 1
            self.metrics.best_streak = max(
                self.metrics.best_streak,
                self.metrics.current_streak
            )
        else:
            self.metrics.current_streak = 0

    def record_api_call(self, from_cache: bool = False):
        """Record an API call."""
        self.metrics.api_calls += 1
        if from_cache:
            self.metrics.cache_hits += 1

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the session."""
        duration = (datetime.utcnow() - self.metrics.started_at).total_seconds()

        return {
            "duration_seconds": duration,
            "problems_attempted": self.metrics.problems_attempted,
            "problems_correct": self.metrics.problems_correct,
            "accuracy": round(self.metrics.accuracy * 100, 1),
            "avg_time_ms": round(self.metrics.avg_time_ms),
            "best_streak": self.metrics.best_streak,
            "api_calls": self.metrics.api_calls,
            "cache_hit_rate": (
                round(self.metrics.cache_hits / self.metrics.api_calls * 100, 1)
                if self.metrics.api_calls > 0 else 0
            )
        }

    # Online/Offline state

    def set_online(self, is_online: bool):
        """Set online status."""
        self._is_online = is_online

    def is_online(self) -> bool:
        """Check if we're online."""
        return self._is_online

    # Pending sync queue (for offline mode)

    def queue_for_sync(self, action: str, data: Dict[str, Any]) -> str:
        """
        Queue an action for sync when back online.

        Args:
            action: Action type (e.g., "record_attempt")
            data: Action data

        Returns:
            ID of the queued item
        """
        import uuid
        item_id = str(uuid.uuid4())

        self._pending_sync.append(PendingSyncItem(
            id=item_id,
            action=action,
            data=data
        ))

        return item_id

    def get_pending_sync(self) -> List[PendingSyncItem]:
        """Get all items pending sync."""
        return self._pending_sync.copy()

    def mark_synced(self, item_id: str):
        """Mark an item as synced."""
        self._pending_sync = [
            item for item in self._pending_sync
            if item.id != item_id
        ]

    def get_pending_count(self) -> int:
        """Get count of items pending sync."""
        return len(self._pending_sync)

    # Persistence

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "mode": self.mode.value,
            "preferences": {
                "voice_enabled": self.preferences.voice_enabled,
                "voice_speed": self.preferences.voice_speed,
                "default_difficulty": self.preferences.default_difficulty,
                "auto_advance_slides": self.preferences.auto_advance_slides,
            },
            "metrics": {
                "problems_attempted": self.metrics.problems_attempted,
                "problems_correct": self.metrics.problems_correct,
                "best_streak": self.metrics.best_streak,
            },
            "context": self._context,
            "pending_sync_count": len(self._pending_sync)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Deserialize state from dictionary."""
        state = cls()

        if "mode" in data:
            try:
                state.mode = AppMode(data["mode"])
            except ValueError:
                pass

        if "preferences" in data:
            prefs = data["preferences"]
            state.preferences.voice_enabled = prefs.get("voice_enabled", True)
            state.preferences.voice_speed = prefs.get("voice_speed", 1.0)
            state.preferences.default_difficulty = prefs.get("default_difficulty", 2)

        if "context" in data:
            state._context = data["context"]

        return state
