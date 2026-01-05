"""
WHAM Dashboard - Pydantic Models for API requests and responses.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================
# Enums
# ============================================================

class CaptureType(str, Enum):
    NOTE = "note"
    TODO = "todo"
    IDEA = "idea"
    REMINDER = "reminder"
    VOICE = "voice"
    PHOTO = "photo"


class CapturePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MemoryCategory(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    CONTACT = "contact"
    LOCATION = "location"
    HABIT = "habit"
    RELATIONSHIP = "relationship"


class ChallengeCategory(str, Enum):
    MENTAL_MATH = "mental_math"
    FOCUS = "focus"
    LEARNING = "learning"
    PRODUCTIVITY = "productivity"
    WELLNESS = "wellness"


# ============================================================
# Capture Models
# ============================================================

class CaptureCreate(BaseModel):
    """Request to create a new capture."""
    text: str = Field(..., min_length=1, max_length=2000)
    type: CaptureType = CaptureType.NOTE
    priority: CapturePriority = CapturePriority.NORMAL
    tags: List[str] = Field(default_factory=list)
    reminder_time: Optional[datetime] = None


class CaptureUpdate(BaseModel):
    """Request to update an existing capture."""
    text: Optional[str] = Field(None, min_length=1, max_length=2000)
    type: Optional[CaptureType] = None
    priority: Optional[CapturePriority] = None
    tags: Optional[List[str]] = None
    processed: Optional[bool] = None


class CaptureResponse(BaseModel):
    """Response for a single capture."""
    id: str
    text: str
    type: CaptureType
    priority: CapturePriority
    tags: List[str]
    created_at: datetime
    processed: bool = False
    reminder_time: Optional[datetime] = None


class CaptureListResponse(BaseModel):
    """Response for list of captures."""
    captures: List[CaptureResponse]
    total: int
    page: int = 1
    per_page: int = 20


# ============================================================
# Memory Models
# ============================================================

class MemoryCreate(BaseModel):
    """Request to create a new memory."""
    key: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1, max_length=2000)
    category: MemoryCategory = MemoryCategory.FACT
    entity: Optional[str] = None


class MemoryResponse(BaseModel):
    """Response for a single memory."""
    id: str
    key: str
    value: str
    category: MemoryCategory
    entity: Optional[str]
    created_at: datetime
    last_accessed: datetime
    access_count: int


class MemorySearchResponse(BaseModel):
    """Response for memory search."""
    memories: List[MemoryResponse]
    total: int
    query: str


class MemoryStats(BaseModel):
    """Memory statistics."""
    total_memories: int
    total_entities: int
    by_category: Dict[str, int]
    indexed_keywords: int


# ============================================================
# Challenge Models
# ============================================================

class ChallengeResponse(BaseModel):
    """Response for a single challenge."""
    id: str
    title: str
    description: str
    category: ChallengeCategory
    difficulty: str
    xp_reward: int
    target: int
    progress: int
    completed: bool
    expires_at: Optional[datetime]


class ChallengeProgress(BaseModel):
    """Request to update challenge progress."""
    category: ChallengeCategory
    increment: int = 1


class XPResponse(BaseModel):
    """Response for XP and level info."""
    total_xp: int
    level: int
    xp_to_next_level: int
    current_streak: int
    longest_streak: int
    achievements_unlocked: int


class ChallengeListResponse(BaseModel):
    """Response for today's challenges."""
    challenges: List[ChallengeResponse]
    xp: XPResponse


# ============================================================
# Session Models
# ============================================================

class ActivityEvent(BaseModel):
    """Single activity event in timeline."""
    timestamp: datetime
    type: str  # poker, focus, capture, homework, etc.
    description: str
    duration_seconds: Optional[int] = None
    cost: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionStats(BaseModel):
    """Session statistics."""
    total_cost: float
    total_xp_earned: int
    focus_minutes: int
    captures_count: int
    poker_hands: int
    homework_problems: int
    active_minutes: int


class TodayResponse(BaseModel):
    """Response for today's activities."""
    date: str
    activities: List[ActivityEvent]
    stats: SessionStats
    challenges: List[ChallengeResponse]
    recent_captures: List[CaptureResponse]


class HistoryDay(BaseModel):
    """Single day in history."""
    date: str
    stats: SessionStats
    highlight: Optional[str] = None


class HistoryResponse(BaseModel):
    """Response for session history."""
    days: List[HistoryDay]
    total_days: int


# ============================================================
# Poker Models
# ============================================================

class OpponentProfile(BaseModel):
    """Poker opponent profile."""
    name: str
    hands_observed: int
    vpip: float  # Voluntarily Put $ In Pot (%)
    pfr: float   # Pre-Flop Raise (%)
    aggression: float
    archetype: str  # TAG, LAG, Nit, Fish, etc.
    notes: List[str] = Field(default_factory=list)
    last_seen: Optional[datetime] = None


class OpponentListResponse(BaseModel):
    """Response for list of opponents."""
    opponents: List[OpponentProfile]
    total: int


class PokerSession(BaseModel):
    """Single poker session."""
    id: str
    date: datetime
    duration_minutes: int
    hands_played: int
    profit_bb: float
    cost: float
    mistakes: int
    big_hands: List[str] = Field(default_factory=list)


class PokerStats(BaseModel):
    """Overall poker statistics."""
    total_sessions: int
    total_hands: int
    win_rate_bb_100: float
    vpip: float
    pfr: float
    total_profit_bb: float
    total_cost: float
    biggest_pot_won: float
    biggest_pot_lost: float


class PokerSessionListResponse(BaseModel):
    """Response for poker sessions."""
    sessions: List[PokerSession]
    stats: PokerStats


# ============================================================
# Cost Models
# ============================================================

class CostItem(BaseModel):
    """Single cost item."""
    category: str
    cost: float
    requests: int


class CostBreakdown(BaseModel):
    """Cost breakdown for a period."""
    period: str  # "today", "weekly", "monthly"
    total: float
    budget: float
    remaining: float
    by_category: List[CostItem]


class CostProjection(BaseModel):
    """Cost projection."""
    projected_monthly: float
    current_daily_avg: float
    trend: str  # "increasing", "stable", "decreasing"
    recommendation: Optional[str] = None


# ============================================================
# Health & System Models
# ============================================================

class ServiceStatus(BaseModel):
    """Status of a single service."""
    name: str
    status: str  # "online", "offline", "degraded"
    latency_ms: Optional[int] = None
    last_check: datetime


class HealthStatus(BaseModel):
    """Overall system health."""
    status: str
    uptime_seconds: int
    database_size_mb: float
    captures_count: int
    memories_count: int
    services: List[ServiceStatus]
    version: str = "1.0.0"


# ============================================================
# Vision Detection Models (Backend EDITH)
# ============================================================

class MathDetectionResult(BaseModel):
    """Math equation detection result from vision."""
    equation: str = Field(..., description="Canonical equation string")
    problem_type: str = Field(..., description="Type: arithmetic, algebra, polynomial, derivative, integral, trigonometry, word_problem, proof, unknown")
    variables: List[str] = Field(default_factory=list, description="Variables detected (e.g., ['x', 'y'])")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence 0.0-1.0")
    raw_response: Optional[str] = Field(None, description="Raw Gemini response for debugging")


class CodeDetectionResult(BaseModel):
    """Code extraction result from vision."""
    code: str = Field(..., description="Extracted code with indentation preserved")
    language: str = Field(..., description="Lowercase language: python, javascript, unknown")
    error_visible: bool = Field(..., description="Whether error message visible in screenshot")
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_response: Optional[str] = None


class PokerDetectionResult(BaseModel):
    """Poker cards and game state detection from vision."""
    hole_cards: List[str] = Field(..., min_length=2, max_length=2, description='Hero hole cards like ["Ah", "Kc"] or ["?", "?"]')
    board: List[str] = Field(default_factory=list, max_length=5, description="Community cards 0-5")
    pot_size_bb: float = Field(..., ge=0.0, description="Pot size in big blinds")
    bet_facing_bb: float = Field(..., ge=0.0, description="Current bet to call in big blinds")
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_response: Optional[str] = None


class TextDetectionResult(BaseModel):
    """OCR text extraction result from vision."""
    text: str = Field(..., description="Extracted text content")
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_response: Optional[str] = None


class VisionError(BaseModel):
    """Vision detection error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


# ============================================================
# Generic Response Models
# ============================================================

class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = True
    message: str = "Operation completed successfully"


class ErrorResponse(BaseModel):
    """Generic error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None
