"""Voice tools for WHAM assistant."""
from .base import VoiceTool, VoiceToolResult
from .router import ToolRouter, ToolMatch
from .calendar_tool import CalendarVoiceTool
from .memory_tool import MemoryVoiceTool
from .code_tool import CodeVoiceTool
from .poker_tool import PokerVoiceTool
from .physics_tool import PhysicsVoiceTool
from .news_tool import PerplexityNewsTool

# JARVIS features - Sprint 1
from .email_tool import EmailVoiceTool
from .music_tool import MusicVoiceTool
from .package_tool import PackageVoiceTool
from .briefing_tool import BriefingVoiceTool

# JARVIS features - Sprint 2
from .vision_assistant_tool import VisionAssistantTool
from .screenshot_tool import ScreenshotVoiceTool
from .context_tool import ContextVoiceTool

# JARVIS features - Sprint 3
from .notes_tool import NotesVoiceTool
from .smart_home_tool import SmartHomeVoiceTool
from .fitness_tool import FitnessVoiceTool
from .travel_tool import TravelVoiceTool

# JARVIS features - Sprint 4
from .learning_tool import LearningVoiceTool
from .alert_tool import AlertVoiceTool

__all__ = [
    "VoiceTool",
    "VoiceToolResult",
    "ToolRouter",
    "ToolMatch",
    "CalendarVoiceTool",
    "MemoryVoiceTool",
    "CodeVoiceTool",
    "PokerVoiceTool",
    "PhysicsVoiceTool",
    "PerplexityNewsTool",
    # JARVIS features - Sprint 1
    "EmailVoiceTool",
    "MusicVoiceTool",
    "PackageVoiceTool",
    "BriefingVoiceTool",
    # JARVIS features - Sprint 2
    "VisionAssistantTool",
    "ScreenshotVoiceTool",
    "ContextVoiceTool",
    # JARVIS features - Sprint 3
    "NotesVoiceTool",
    "SmartHomeVoiceTool",
    "FitnessVoiceTool",
    "TravelVoiceTool",
    # JARVIS features - Sprint 4
    "LearningVoiceTool",
    "AlertVoiceTool",
]
