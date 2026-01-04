"""
Pytest configuration for phone_client tests.
Sets up import paths for test discovery.
"""
import sys
import os

# Add phone_client directory to path for imports
_phone_client_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _phone_client_dir not in sys.path:
    sys.path.insert(0, _phone_client_dir)

# Pytest fixtures can be added here
import pytest


@pytest.fixture
def power_manager():
    """Create a PowerManager instance for testing."""
    from core.power_manager import PowerManager
    return PowerManager()


@pytest.fixture
def oled_renderer():
    """Create an OLEDRenderer instance for testing."""
    from halo.oled_renderer import OLEDRenderer
    return OLEDRenderer()


@pytest.fixture
def animation_engine():
    """Create an AnimationEngine instance for testing."""
    from halo.animations import AnimationEngine
    return AnimationEngine()


@pytest.fixture
def notification_manager():
    """Create a NotificationManager instance for testing."""
    from core.notifications import NotificationManager
    return NotificationManager()
