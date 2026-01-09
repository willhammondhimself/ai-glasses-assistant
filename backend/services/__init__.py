"""Backend services for WHAM assistant."""
from .gmail_service import GmailService, get_gmail_service
from .spotify_service import SpotifyService, get_spotify_service
from .package_tracker import PackageTrackerService, get_package_tracker
from .morning_briefing import MorningBriefingService, get_briefing_service
from .context_engine import ContextEngine, get_context_engine, UserContext, UserMode

# Sprint 3 services
from .notes_service import NotesService, get_notes_service, Note
from .home_assistant_service import HomeAssistantService, get_home_assistant, HomeDevice
from .healthkit_service import HealthKitService, get_healthkit_service, HealthMetrics
from .travel_service import TravelService, get_travel_service, Flight, Trip, Hotel

# Sprint 4 services
from .learning_service import LearningService, get_learning_service, Flashcard, LearningStats
from .proactive_engine import ProactiveEngine, get_proactive_engine, Alert, AlertType

# Sprint 5 services
from .answer_evaluator import AnswerEvaluator, get_answer_evaluator

__all__ = [
    # Sprint 1 services
    "GmailService",
    "get_gmail_service",
    "SpotifyService",
    "get_spotify_service",
    "PackageTrackerService",
    "get_package_tracker",
    "MorningBriefingService",
    "get_briefing_service",
    # Sprint 2 services
    "ContextEngine",
    "get_context_engine",
    "UserContext",
    "UserMode",
    # Sprint 3 services
    "NotesService",
    "get_notes_service",
    "Note",
    "HomeAssistantService",
    "get_home_assistant",
    "HomeDevice",
    "HealthKitService",
    "get_healthkit_service",
    "HealthMetrics",
    "TravelService",
    "get_travel_service",
    "Flight",
    "Trip",
    "Hotel",
    # Sprint 4 services
    "LearningService",
    "get_learning_service",
    "Flashcard",
    "LearningStats",
    "ProactiveEngine",
    "get_proactive_engine",
    "Alert",
    "AlertType",
    # Sprint 5 services
    "AnswerEvaluator",
    "get_answer_evaluator",
]
