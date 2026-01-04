"""
WHAM Web Dashboard - Backend API Module.
Provides web access to WHAM features without Halo glasses.
"""
from .api import router as dashboard_router
from .models import (
    CaptureCreate,
    CaptureUpdate,
    CaptureResponse,
    MemoryCreate,
    MemoryResponse,
    ChallengeProgress,
    SessionStats,
    CostBreakdown,
    OpponentProfile,
    HealthStatus,
)

__all__ = [
    "dashboard_router",
    "CaptureCreate",
    "CaptureUpdate",
    "CaptureResponse",
    "MemoryCreate",
    "MemoryResponse",
    "ChallengeProgress",
    "SessionStats",
    "CostBreakdown",
    "OpponentProfile",
    "HealthStatus",
]
