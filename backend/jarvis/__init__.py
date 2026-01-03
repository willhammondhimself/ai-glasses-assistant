"""
JARVIS Personality Layer

Tony Stark-style AI personality for AR glasses coach.
Transforms dry responses into engaging, witty interactions.
"""

from .personality import JarvisPersonality
from .templates import ResponseTemplates, GREETINGS, SPEED_FEEDBACK, STREAK_MILESTONES
from .context import JarvisContext, SessionStats
from .performance import PerformanceAnalyzer

__all__ = [
    "JarvisPersonality",
    "ResponseTemplates",
    "JarvisContext",
    "SessionStats",
    "PerformanceAnalyzer",
    "GREETINGS",
    "SPEED_FEEDBACK",
    "STREAK_MILESTONES",
]
