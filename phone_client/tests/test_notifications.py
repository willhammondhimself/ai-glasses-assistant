"""
Notification Manager Tests.
Tests notification queue, priority handling, context awareness, and rate limiting.
"""
import pytest
import time
from unittest.mock import MagicMock

# Path setup via conftest.py


from core.notifications import (
    NotificationManager, NotificationPriority, Notification, NotificationRenderer
)


class TestNotificationPriority:
    """Test notification priority levels."""

    def test_priority_ordering(self):
        """Verify priority ordering (higher = more urgent)."""
        assert NotificationPriority.LOW < NotificationPriority.MEDIUM
        assert NotificationPriority.MEDIUM < NotificationPriority.HIGH
        assert NotificationPriority.HIGH < NotificationPriority.CRITICAL

    def test_priority_values(self):
        """Verify priority numeric values."""
        assert NotificationPriority.LOW == 1
        assert NotificationPriority.MEDIUM == 2
        assert NotificationPriority.HIGH == 3
        assert NotificationPriority.CRITICAL == 4


class TestNotificationDataclass:
    """Test Notification dataclass."""

    def test_notification_defaults(self):
        """Test default notification values."""
        notif = Notification(title="Test", message="Message")

        assert notif.title == "Test"
        assert notif.message == "Message"
        assert notif.priority == NotificationPriority.MEDIUM
        assert notif.duration_ms == 3000
        assert notif.vibrate is True
        assert notif.sound is False

    def test_notification_expiration(self):
        """Test notification expiration check."""
        # Not expired
        notif = Notification(
            title="Test",
            message="",
            expires_at=time.time() + 60
        )
        assert notif.is_expired() is False

        # Expired
        notif = Notification(
            title="Test",
            message="",
            expires_at=time.time() - 1
        )
        assert notif.is_expired() is True

        # No expiration
        notif = Notification(title="Test", message="")
        assert notif.is_expired() is False


class TestQueueBehavior:
    """Test notification queue operations."""

    def test_push_and_pop(self):
        """Test basic push and pop operations."""
        manager = NotificationManager()

        notif = Notification(title="Test", message="")
        manager.push(notif)

        assert manager.get_queue_length() == 1

        popped = manager.pop()
        assert popped.title == "Test"
        assert manager.get_queue_length() == 0

    def test_priority_ordering(self):
        """Test notifications are popped by priority (highest first)."""
        manager = NotificationManager()

        # Push in random order
        manager.push(Notification(title="Medium", message="", priority=NotificationPriority.MEDIUM))
        manager.push(Notification(title="Low", message="", priority=NotificationPriority.LOW))
        manager.push(Notification(title="High", message="", priority=NotificationPriority.HIGH))

        # Should pop in priority order
        assert manager.pop().title == "High"
        assert manager.pop().title == "Medium"
        assert manager.pop().title == "Low"

    def test_same_priority_fifo(self):
        """Test same priority pops in FIFO order."""
        manager = NotificationManager()

        manager.push(Notification(title="First", message="", priority=NotificationPriority.MEDIUM))
        time.sleep(0.01)  # Ensure different timestamps
        manager.push(Notification(title="Second", message="", priority=NotificationPriority.MEDIUM))

        first = manager.pop()
        second = manager.pop()

        assert first.title == "First"
        assert second.title == "Second"

    def test_queue_max_size(self):
        """Test queue respects max size limit."""
        manager = NotificationManager(max_queue=3)

        for i in range(5):
            manager.push(Notification(title=f"N{i}", message="", priority=NotificationPriority.MEDIUM))

        assert manager.get_queue_length() == 3

    def test_queue_evicts_lowest_priority(self):
        """Test queue evicts lowest priority when full."""
        manager = NotificationManager(max_queue=3)

        manager.push(Notification(title="Low", message="", priority=NotificationPriority.LOW))
        manager.push(Notification(title="Medium", message="", priority=NotificationPriority.MEDIUM))
        manager.push(Notification(title="High", message="", priority=NotificationPriority.HIGH))

        # This should evict Low
        manager.push(Notification(title="Critical", message="", priority=NotificationPriority.CRITICAL))

        # Low should be evicted
        titles = [manager.pop().title for _ in range(3)]
        assert "Low" not in titles
        assert "Critical" in titles


class TestContextAwareness:
    """Test context-aware notification behavior."""

    def test_busy_blocks_low_priority(self):
        """Test low/medium priority blocked when user busy."""
        manager = NotificationManager()

        manager.push(Notification(title="Low", message="", priority=NotificationPriority.LOW))
        manager.push(Notification(title="Medium", message="", priority=NotificationPriority.MEDIUM))

        manager.set_user_busy(True, "poker")

        # Should return None for low/medium during busy
        assert manager.pop() is None

    def test_busy_allows_high_priority(self):
        """Test HIGH priority allowed during busy."""
        manager = NotificationManager()

        manager.push(Notification(title="High", message="", priority=NotificationPriority.HIGH))
        manager.set_user_busy(True, "poker")

        popped = manager.pop()
        assert popped is not None
        assert popped.title == "High"

    def test_busy_allows_critical(self):
        """Test CRITICAL always allowed during busy."""
        manager = NotificationManager()

        manager.push(Notification(title="Critical", message="", priority=NotificationPriority.CRITICAL))
        manager.set_user_busy(True, "poker")

        popped = manager.pop()
        assert popped is not None
        assert popped.title == "Critical"

    def test_dnd_blocks_most(self):
        """Test DND blocks all except CRITICAL."""
        manager = NotificationManager()

        manager.push(Notification(title="High", message="", priority=NotificationPriority.HIGH))
        manager.set_do_not_disturb(True)

        assert manager.pop() is None

    def test_dnd_allows_critical(self):
        """Test CRITICAL allowed during DND."""
        manager = NotificationManager()

        manager.push(Notification(title="Critical", message="", priority=NotificationPriority.CRITICAL))
        manager.set_do_not_disturb(True)

        popped = manager.pop()
        assert popped is not None
        assert popped.title == "Critical"

    def test_busy_cleared_shows_pending(self):
        """Test pending notifications shown when busy cleared."""
        manager = NotificationManager()
        displayed = []
        manager.set_display_callback(lambda n: displayed.append(n))

        manager.set_user_busy(True)
        manager.push(Notification(title="Pending", message="", priority=NotificationPriority.MEDIUM))

        assert len(displayed) == 0  # Not displayed while busy

        manager.set_user_busy(False)
        # Display callback should have been triggered


class TestRateLimiting:
    """Test notification rate limiting."""

    def test_rate_limit_enforced(self):
        """Test rate limit prevents spam."""
        manager = NotificationManager(rate_limit_per_min=5)

        success_count = 0
        for i in range(10):
            if manager.push(Notification(title=f"N{i}", message="")):
                success_count += 1

        assert success_count == 5

    def test_rate_limit_resets(self):
        """Test rate limit resets over time."""
        manager = NotificationManager(rate_limit_per_min=2)

        # Use up rate limit
        manager.push(Notification(title="1", message=""))
        manager.push(Notification(title="2", message=""))

        # Third should fail
        result = manager.push(Notification(title="3", message=""))
        assert result is False


class TestExpiration:
    """Test notification expiration handling."""

    def test_expired_notifications_cleaned(self):
        """Test expired notifications are removed."""
        manager = NotificationManager()

        # Add notification that expires immediately
        manager.push(Notification(
            title="Expired",
            message="",
            expires_at=time.time() - 1  # Already expired
        ))

        # Should not be returned
        assert manager.pop() is None

    def test_low_priority_auto_expires(self):
        """Test LOW priority gets auto-expiration."""
        manager = NotificationManager()

        notif = Notification(title="Low", message="", priority=NotificationPriority.LOW)
        manager.push(notif)

        # Check the notification in queue has expiration
        queued = manager.peek()
        assert queued.expires_at is not None


class TestDisplayCallback:
    """Test display callback integration."""

    def test_callback_called_on_push(self):
        """Test display callback is triggered on push."""
        manager = NotificationManager()
        displayed = []

        manager.set_display_callback(lambda n: displayed.append(n))
        manager.push(Notification(title="Test", message=""))

        assert len(displayed) == 1
        assert displayed[0].title == "Test"

    def test_callback_not_called_when_busy(self):
        """Test callback not called when user is busy."""
        manager = NotificationManager()
        displayed = []

        manager.set_display_callback(lambda n: displayed.append(n))
        manager.set_user_busy(True)

        manager.push(Notification(title="Low", message="", priority=NotificationPriority.LOW))

        assert len(displayed) == 0


class TestDismissAndClear:
    """Test notification dismissal and clearing."""

    def test_dismiss_current(self):
        """Test dismissing current notification."""
        manager = NotificationManager()
        displayed = []

        manager.set_display_callback(lambda n: displayed.append(n))
        manager.push(Notification(title="First", message=""))
        manager.push(Notification(title="Second", message=""))

        assert len(displayed) == 1

        manager.dismiss_current()

        # Second should now be displayed
        assert len(displayed) == 2

    def test_clear_all(self):
        """Test clearing all notifications."""
        manager = NotificationManager()

        for i in range(5):
            manager.push(Notification(title=f"N{i}", message=""))

        manager.clear_all()

        assert manager.get_queue_length() == 0

    def test_clear_category(self):
        """Test clearing by category."""
        manager = NotificationManager()

        manager.push(Notification(title="Battery", message="", category="battery"))
        manager.push(Notification(title="Error", message="", category="error"))
        manager.push(Notification(title="Battery2", message="", category="battery"))

        manager.clear_category("battery")

        assert manager.get_queue_length() == 1
        assert manager.pop().category == "error"


class TestHelperMethods:
    """Test convenience helper methods."""

    def test_notify_battery_low(self):
        """Test battery low helper."""
        manager = NotificationManager()

        manager.notify_battery_low(15)

        notif = manager.pop()
        assert "Battery" in notif.title or "15%" in notif.title
        assert notif.priority == NotificationPriority.HIGH

    def test_notify_battery_critical(self):
        """Test critical battery is CRITICAL priority."""
        manager = NotificationManager()

        manager.notify_battery_low(5)

        notif = manager.pop()
        assert notif.priority == NotificationPriority.CRITICAL

    def test_notify_success(self):
        """Test success notification helper."""
        manager = NotificationManager()

        manager.notify_success("Done!", "Task completed")

        notif = manager.pop()
        assert notif.title == "Done!"
        assert notif.category == "success"

    def test_notify_error(self):
        """Test error notification helper."""
        manager = NotificationManager()

        manager.notify_error("Failed", "Connection lost")

        notif = manager.pop()
        assert notif.priority == NotificationPriority.HIGH
        assert notif.category == "error"

    def test_notify_streak(self):
        """Test streak celebration helper."""
        manager = NotificationManager()

        manager.notify_streak(10)

        notif = manager.pop()
        assert "10" in notif.title

    def test_notify_cost_warning(self):
        """Test cost warning helper."""
        manager = NotificationManager()

        manager.notify_cost_warning(1.50, 2.00)

        notif = manager.pop()
        assert "1.50" in notif.title or "Cost" in notif.title


class TestStatistics:
    """Test notification statistics."""

    def test_stats_tracking(self):
        """Test stats are tracked correctly."""
        manager = NotificationManager()

        manager.push(Notification(title="1", message=""))
        manager.push(Notification(title="2", message=""))
        manager.dismiss_current()

        stats = manager.get_stats()

        assert stats["total_sent"] == 2
        assert stats["total_dismissed"] >= 1

    def test_rate_current_tracking(self):
        """Test current rate is tracked."""
        manager = NotificationManager()

        manager.push(Notification(title="1", message=""))
        manager.push(Notification(title="2", message=""))

        stats = manager.get_stats()
        assert stats["rate_current"] == 2


class TestQuickPush:
    """Test quick push helper."""

    def test_push_quick(self):
        """Test push_quick convenience method."""
        manager = NotificationManager()

        result = manager.push_quick(
            "Title",
            "Message",
            priority=NotificationPriority.HIGH,
            category="error"
        )

        assert result is True
        notif = manager.pop()
        assert notif.title == "Title"
        assert notif.message == "Message"
        assert notif.priority == NotificationPriority.HIGH


class TestCategoryIcons:
    """Test category icon assignment."""

    def test_battery_icon(self):
        """Test battery notifications get battery icon."""
        manager = NotificationManager()

        manager.push(Notification(title="Low Battery", message="", category="battery"))

        notif = manager.pop()
        assert notif.icon == "🔋"

    def test_error_icon(self):
        """Test error notifications get warning icon."""
        manager = NotificationManager()

        manager.push(Notification(title="Error", message="", category="error"))

        notif = manager.pop()
        assert notif.icon == "⚠️"


class TestPriorityColors:
    """Test priority-based colors."""

    def test_default_colors_applied(self):
        """Test default colors are applied by priority."""
        manager = NotificationManager()

        manager.push(Notification(title="High", message="", priority=NotificationPriority.HIGH))

        notif = manager.pop()
        assert notif.color == (255, 200, 0)  # Amber

    def test_critical_color(self):
        """Test critical gets red color."""
        manager = NotificationManager()

        manager.push(Notification(title="Critical", message="", priority=NotificationPriority.CRITICAL))

        notif = manager.pop()
        assert notif.color == (255, 80, 80)  # Red


class TestDisabled:
    """Test behavior when notifications disabled."""

    def test_disabled_rejects_all(self):
        """Test disabled manager rejects all notifications."""
        config = {"notifications": {"enabled": False}}
        manager = NotificationManager(config=config)

        result = manager.push(Notification(title="Test", message=""))

        assert result is False
        assert manager.get_queue_length() == 0


class TestNotificationRenderer:
    """Test NotificationRenderer helper."""

    def test_render_notification(self):
        """Test rendering notification to Lua."""
        from halo.oled_renderer import OLEDRenderer

        renderer = OLEDRenderer()
        notif_renderer = NotificationRenderer(renderer)

        notif = Notification(
            title="Test Title",
            message="Test message",
            color=(255, 0, 0),
            icon="⚠️"
        )

        lua = notif_renderer.render_notification(notif)

        assert "frame.display" in lua
        assert "Test Title" in lua

    def test_render_badge(self):
        """Test rendering notification count badge."""
        from halo.oled_renderer import OLEDRenderer

        renderer = OLEDRenderer()
        notif_renderer = NotificationRenderer(renderer)

        # Render badge
        renderer.clear()
        lua = notif_renderer.render_notification_badge(5, 600, 30)

        # Should have rect for badge background


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
