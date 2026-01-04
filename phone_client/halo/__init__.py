"""Halo AR Glasses Bluetooth LE Connection Module."""
from .connection import HaloConnection
from .oled_renderer import OLEDRenderer, OLEDColors, create_oled_renderer
from .animations import AnimationEngine, AnimationTimings, create_animation_engine
from .haptics import HapticPatterns, HapticManager
from .themes import ColorTheme, ContextualThemes

__all__ = [
    "HaloConnection",
    "OLEDRenderer",
    "OLEDColors",
    "create_oled_renderer",
    "AnimationEngine",
    "AnimationTimings",
    "create_animation_engine",
    "HapticPatterns",
    "HapticManager",
    "ColorTheme",
    "ContextualThemes",
]
