"""Core components for Halo Frame client."""

from .display import DisplayController, DisplaySlide, HALO_COLORS
from .connection import ConnectionManager
from .state import SessionState

__all__ = [
    "DisplayController",
    "DisplaySlide",
    "HALO_COLORS",
    "ConnectionManager",
    "SessionState",
]
