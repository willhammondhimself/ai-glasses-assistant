"""
WHAM Personality Layer

Will's Helpful Assistant Module - AI personality for AR glasses coach.
Transforms dry responses into engaging, witty interactions.
"""

from .personality import WHAMPersonality
from .templates import ResponseTemplates, GREETINGS, SPEED_FEEDBACK, STREAK_MILESTONES
from .context import WHAMContext, SessionStats
from .performance import PerformanceAnalyzer

__all__ = [
    "WHAMPersonality",
    "ResponseTemplates",
    "WHAMContext",
    "SessionStats",
    "PerformanceAnalyzer",
    "GREETINGS",
    "SPEED_FEEDBACK",
    "STREAK_MILESTONES",
]
