"""
Notifications - Smart notification queue with context awareness.
Respects user attention and provides intelligent interruption management.
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class NotificationPriority(IntEnum):
    """
    Notification priority levels.

    Higher values = more urgent, more likely to interrupt.
    """
    LOW = 1        # Ambient - can be missed, expires quickly
    MEDIUM = 2     # Queue - show when idle
    HIGH = 3       # Important - show after current action
    CRITICAL = 4   # Urgent - interrupt anything


@dataclass
class Notification:
    """A notification to display on the HUD."""
    title: str
    message: str
    priority: NotificationPriority = NotificationPriority.MEDIUM
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    icon: str = ""  # Symbol or emoji
    color: Tuple[int, int, int] = None  # RGB color, or None for default
    duration_ms: int = 3000  # Display duration
    action: Optional[Callable] = None  # Callback on tap
    category: str = "general"  # For filtering/grouping
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # Auto-expire time
    sound: bool = False  # Play sound (usually silent on glasses)
    vibrate: bool = True  # Haptic feedback

    def is_expired(self) -> bool:
        """Check if notification has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class NotificationManager:
    """
    Context-aware notification queue with attention management.

    Features:
    - Priority-based queue with intelligent interruption
    - Context awareness (knows when user is busy)
    - Do-not-disturb mode
    - Expiration and auto-cleanup
    - Rate limiting to prevent notification spam
    """

    # Default colors for priority levels
    PRIORITY_COLORS = {
        NotificationPriority.LOW: (100, 100, 100),      # Gray
        NotificationPriority.MEDIUM: (0, 200, 200),     # Cyan
        NotificationPriority.HIGH: (255, 200, 0),       # Amber
        NotificationPriority.CRITICAL: (255, 80, 80),   # Red
    }

    # Icons for common categories
    CATEGORY_ICONS = {
        "battery": "🔋",
        "error": "⚠️",
        "success": "✓",
        "info": "ℹ",
        "poker": "♠",
        "math": "∑",
        "meeting": "📅",
        "cost": "$",
    }

    def __init__(
        self,
        config: dict = None,
        max_queue: int = 10,
        rate_limit_per_min: int = 30
    ):
        """
        Initialize notification manager.

        Args:
            config: Configuration dict with 'notifications' section
            max_queue: Maximum notifications in queue
            rate_limit_per_min: Max notifications per minute
        """
        self.config = config or {}
        notif_config = self.config.get("notifications", {})

        self.max_queue = notif_config.get("max_queue", max_queue)
        self.auto_dismiss_ms = notif_config.get("auto_dismiss_ms", 5000)
        self.enabled = notif_config.get("enabled", True)
        self.default_vibrate = notif_config.get("vibration", True)
        self.rate_limit = rate_limit_per_min

        # Queue state
        self._queue: List[Notification] = []
        self._current: Optional[Notification] = None
        self._display_callback: Optional[Callable] = None

        # Context state
        self._user_busy = False
        self._busy_context: str = ""
        self._do_not_disturb = False

        # Rate limiting
        self._notification_times: List[float] = []

        # Statistics
        self._total_sent = 0
        self._total_dismissed = 0
        self._total_expired = 0

        logger.info(f"NotificationManager initialized (enabled={self.enabled})")

    def set_display_callback(self, callback: Callable[[Notification], Any]):
        """
        Set callback for displaying notifications.

        Args:
            callback: Function(notification) called when notification should display
        """
        self._display_callback = callback

    def push(self, notification: Notification) -> bool:
        """
        Add notification to queue.

        Args:
            notification: Notification to add

        Returns:
            True if added, False if rejected (rate limit, disabled, etc.)
        """
        if not self.enabled:
            logger.debug("Notifications disabled, ignoring")
            return False

        # Rate limiting check
        now = time.time()
        self._notification_times = [t for t in self._notification_times if now - t < 60]
        if len(self._notification_times) >= self.rate_limit:
            logger.warning(f"Rate limit reached ({self.rate_limit}/min)")
            return False

        # Apply defaults
        if notification.color is None:
            notification.color = self.PRIORITY_COLORS.get(
                notification.priority,
                self.PRIORITY_COLORS[NotificationPriority.MEDIUM]
            )
        if not notification.icon and notification.category in self.CATEGORY_ICONS:
            notification.icon = self.CATEGORY_ICONS[notification.category]
        if notification.vibrate is None:
            notification.vibrate = self.default_vibrate

        # Set default expiration for low priority
        if notification.priority == NotificationPriority.LOW and notification.expires_at is None:
            notification.expires_at = now + 30  # 30 second expiry

        # Queue management
        self._cleanup_expired()

        if len(self._queue) >= self.max_queue:
            # Remove lowest priority notification
            self._queue.sort(key=lambda n: (n.priority, -n.created_at))
            removed = self._queue.pop(0)
            logger.debug(f"Queue full, removed: {removed.title}")

        # Insert by priority (higher priority at end for faster pop)
        self._queue.append(notification)
        self._queue.sort(key=lambda n: (n.priority, n.created_at))

        self._notification_times.append(now)
        self._total_sent += 1

        logger.debug(f"Notification queued: {notification.title} (priority={notification.priority.name})")

        # Check if we should display immediately
        self._try_display()

        return True

    def push_quick(
        self,
        title: str,
        message: str = "",
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        category: str = "general",
        duration_ms: int = 3000
    ) -> bool:
        """
        Quick helper to push a notification.

        Args:
            title: Notification title
            message: Notification message
            priority: Priority level
            category: Category for icon selection
            duration_ms: Display duration

        Returns:
            True if added successfully
        """
        notif = Notification(
            title=title,
            message=message,
            priority=priority,
            category=category,
            duration_ms=duration_ms
        )
        return self.push(notif)

    def pop(self) -> Optional[Notification]:
        """
        Get next notification to display respecting context.

        Returns:
            Next notification or None if queue empty or blocked
        """
        self._cleanup_expired()

        if not self._queue:
            return None

        # Check context blockers
        if self._do_not_disturb:
            # Only CRITICAL can interrupt DND
            critical = [n for n in self._queue if n.priority == NotificationPriority.CRITICAL]
            if not critical:
                return None
            # Get highest priority critical
            return self._queue.pop(self._queue.index(critical[-1]))

        if self._user_busy:
            # Only HIGH and CRITICAL can interrupt busy
            urgent = [n for n in self._queue if n.priority >= NotificationPriority.HIGH]
            if not urgent:
                return None
            return self._queue.pop(self._queue.index(urgent[-1]))

        # Normal mode: return highest priority
        return self._queue.pop()

    def peek(self) -> Optional[Notification]:
        """Peek at next notification without removing."""
        self._cleanup_expired()
        if not self._queue:
            return None
        return self._queue[-1]

    def _cleanup_expired(self):
        """Remove expired notifications."""
        now = time.time()
        before = len(self._queue)
        self._queue = [n for n in self._queue if not n.is_expired()]
        expired = before - len(self._queue)
        if expired > 0:
            self._total_expired += expired
            logger.debug(f"Cleaned up {expired} expired notifications")

    def _try_display(self):
        """Try to display next notification if conditions allow."""
        if self._current:
            # Already displaying
            return

        notification = self.peek()
        if not notification:
            return

        # Check if we should display based on priority and context
        can_display = False

        if notification.priority == NotificationPriority.CRITICAL:
            can_display = True
        elif notification.priority == NotificationPriority.HIGH:
            can_display = not self._do_not_disturb
        elif notification.priority >= NotificationPriority.MEDIUM:
            can_display = not self._user_busy and not self._do_not_disturb
        else:
            can_display = not self._user_busy and not self._do_not_disturb

        if can_display:
            self._current = self.pop()
            if self._current and self._display_callback:
                try:
                    self._display_callback(self._current)
                except Exception as e:
                    logger.error(f"Display callback error: {e}")

    def set_user_busy(self, busy: bool, context: str = ""):
        """
        Set whether user is busy (in active mode).

        Args:
            busy: Whether user is busy
            context: Context description (e.g., "poker_analysis", "mental_math")
        """
        self._user_busy = busy
        self._busy_context = context
        logger.debug(f"User busy: {busy} ({context})")

        if not busy:
            # User now idle, try to display pending notifications
            self._try_display()

    def set_do_not_disturb(self, enabled: bool):
        """
        Enable/disable do-not-disturb mode.

        Args:
            enabled: Whether DND is enabled
        """
        self._do_not_disturb = enabled
        logger.info(f"Do Not Disturb: {'enabled' if enabled else 'disabled'}")

    def dismiss_current(self):
        """Dismiss currently displayed notification."""
        if self._current:
            self._total_dismissed += 1
            logger.debug(f"Dismissed: {self._current.title}")
            self._current = None

        # Try to show next
        self._try_display()

    def clear_all(self):
        """Clear all pending notifications."""
        count = len(self._queue)
        self._queue.clear()
        self._current = None
        logger.info(f"Cleared {count} notifications")

    def clear_category(self, category: str):
        """Clear all notifications of a specific category."""
        before = len(self._queue)
        self._queue = [n for n in self._queue if n.category != category]
        removed = before - len(self._queue)
        logger.debug(f"Cleared {removed} notifications in category '{category}'")

    def get_queue_length(self) -> int:
        """Get current queue length."""
        return len(self._queue)

    def get_stats(self) -> dict:
        """Get notification statistics."""
        return {
            "queue_length": len(self._queue),
            "total_sent": self._total_sent,
            "total_dismissed": self._total_dismissed,
            "total_expired": self._total_expired,
            "user_busy": self._user_busy,
            "do_not_disturb": self._do_not_disturb,
            "rate_current": len(self._notification_times),
            "rate_limit": self.rate_limit,
        }

    # ============================================================
    # Common Notification Helpers
    # ============================================================

    def notify_battery_low(self, percent: int):
        """Send battery low warning."""
        priority = NotificationPriority.CRITICAL if percent < 10 else NotificationPriority.HIGH
        self.push(Notification(
            title=f"Battery Low: {percent}%",
            message="Consider saving your session",
            priority=priority,
            category="battery",
            color=(255, 80, 80) if percent < 10 else (255, 200, 0),
            duration_ms=5000
        ))

    def notify_cost_warning(self, current: float, limit: float):
        """Send API cost warning."""
        self.push(Notification(
            title=f"API Cost: ${current:.2f}",
            message=f"Approaching limit (${limit:.2f})",
            priority=NotificationPriority.HIGH,
            category="cost",
            color=(255, 200, 0),
            duration_ms=4000
        ))

    def notify_cost_limit(self, current: float, limit: float):
        """Send API cost limit reached."""
        self.push(Notification(
            title="Cost Limit Reached",
            message=f"${current:.2f} / ${limit:.2f}",
            priority=NotificationPriority.CRITICAL,
            category="cost",
            color=(255, 80, 80),
            duration_ms=6000
        ))

    def notify_error(self, title: str, message: str = ""):
        """Send error notification."""
        self.push(Notification(
            title=title,
            message=message,
            priority=NotificationPriority.HIGH,
            category="error",
            color=(255, 80, 80),
            duration_ms=5000
        ))

    def notify_success(self, title: str, message: str = ""):
        """Send success notification."""
        self.push(Notification(
            title=title,
            message=message,
            priority=NotificationPriority.MEDIUM,
            category="success",
            color=(0, 255, 128),
            duration_ms=2000
        ))

    def notify_info(self, title: str, message: str = ""):
        """Send info notification."""
        self.push(Notification(
            title=title,
            message=message,
            priority=NotificationPriority.LOW,
            category="info",
            color=(0, 200, 200),
            duration_ms=2000
        ))

    def notify_streak(self, streak: int):
        """Celebrate streak milestone."""
        messages = {
            3: "Getting warmed up!",
            5: "Solid streak going!",
            7: "You're on fire!",
            10: "Double digits!",
            15: "Incredible focus!",
            20: "LEGENDARY!",
            25: "Quarter century!",
            50: "UNSTOPPABLE!",
        }
        message = messages.get(streak, f"Keep it going!")

        colors = {
            3: (0, 200, 200),
            5: (0, 255, 128),
            10: (255, 200, 0),
            20: (200, 100, 255),
            50: (255, 128, 0),
        }
        # Find closest color threshold
        color_thresholds = sorted(colors.keys())
        color = (0, 200, 200)
        for threshold in color_thresholds:
            if streak >= threshold:
                color = colors[threshold]

        self.push(Notification(
            title=f"🔥 Streak: {streak}",
            message=message,
            priority=NotificationPriority.MEDIUM,
            category="math",
            color=color,
            duration_ms=2000
        ))


# Notification renderer helper
class NotificationRenderer:
    """Helper for rendering notifications to HUD."""

    def __init__(self, renderer):
        """
        Initialize notification renderer.

        Args:
            renderer: OLEDRenderer instance
        """
        self.renderer = renderer

    def render_notification(self, notification: Notification) -> str:
        """
        Render a notification to Lua code.

        Args:
            notification: Notification to render

        Returns:
            Lua code string
        """
        self.renderer.clear()

        # Toast-style notification at bottom
        toast_height = 80
        toast_y = self.renderer.height - toast_height

        # Background
        self.renderer.rect(0, toast_y, self.renderer.width, toast_height,
                          (30, 30, 35), filled=True)

        # Color accent bar on left
        self.renderer.rect(0, toast_y, 4, toast_height, notification.color, filled=True)

        # Icon if present
        text_x = 30
        if notification.icon:
            self.renderer.text(notification.icon, 20, toast_y + 20, notification.color, 24)
            text_x = 50

        # Title
        self.renderer.text(notification.title, text_x, toast_y + 20, notification.color, 24, "left")

        # Message (if present)
        if notification.message:
            # Truncate long messages
            msg = notification.message
            if len(msg) > 50:
                msg = msg[:47] + "..."
            self.renderer.text(msg, text_x, toast_y + 50, (180, 180, 180), 18, "left")

        self.renderer.show()
        return self.renderer.get_lua()

    def render_notification_badge(self, count: int, x: int, y: int) -> str:
        """
        Render a notification count badge.

        Args:
            count: Number of pending notifications
            x: X position
            y: Y position

        Returns:
            Lua code string
        """
        if count <= 0:
            return ""

        # Badge background
        badge_color = (255, 80, 80) if count > 5 else (255, 200, 0)
        self.renderer.rect(x - 12, y - 12, 24, 24, badge_color, filled=True)

        # Count text
        count_str = str(count) if count < 10 else "9+"
        self.renderer.text(count_str, x, y, (255, 255, 255), 16)

        return self.renderer.get_lua()


# Test
def test_notifications():
    """Test notification manager."""
    print("=== Notification Manager Test ===\n")

    manager = NotificationManager()

    # Track displayed notifications
    displayed = []
    manager.set_display_callback(lambda n: displayed.append(n))

    # Test basic push
    print("Adding notifications:")
    manager.push_quick("Test 1", "Low priority", NotificationPriority.LOW)
    manager.push_quick("Test 2", "Medium priority", NotificationPriority.MEDIUM)
    manager.push_quick("Test 3", "High priority", NotificationPriority.HIGH)
    print(f"  Queue length: {manager.get_queue_length()}")
    print(f"  Displayed: {len(displayed)}")
    print()

    # Test priority ordering
    print("Pop order (by priority):")
    while notif := manager.pop():
        print(f"  {notif.title} ({notif.priority.name})")
    print()

    # Reset for context tests
    displayed.clear()

    # Test busy mode
    print("Testing busy mode:")
    manager.push_quick("Low during busy", priority=NotificationPriority.LOW)
    manager.set_user_busy(True, "poker")

    result = manager.pop()
    print(f"  Low priority during busy: {result}")

    manager.push_quick("Critical during busy", priority=NotificationPriority.CRITICAL)
    result = manager.pop()
    print(f"  Critical during busy: {result.title if result else None}")

    manager.set_user_busy(False)
    print()

    # Test DND mode
    print("Testing Do Not Disturb:")
    manager.set_do_not_disturb(True)
    manager.push_quick("High during DND", priority=NotificationPriority.HIGH)
    result = manager.pop()
    print(f"  High during DND: {result}")

    manager.push_quick("Critical during DND", priority=NotificationPriority.CRITICAL)
    result = manager.pop()
    print(f"  Critical during DND: {result.title if result else None}")

    manager.set_do_not_disturb(False)
    print()

    # Test helpers
    print("Testing helper methods:")
    manager.notify_battery_low(15)
    manager.notify_success("Great job!", "Task completed")
    manager.notify_streak(10)
    print(f"  Queue after helpers: {manager.get_queue_length()}")
    print()

    # Stats
    print("Statistics:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_notifications()
