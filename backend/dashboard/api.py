"""
WHAM Dashboard - FastAPI Routes.
Provides web API access to WHAM features.
"""
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

# Add phone_client to path for imports
PHONE_CLIENT_PATH = Path(__file__).parent.parent.parent / "phone_client"
sys.path.insert(0, str(PHONE_CLIENT_PATH))

from .models import (
    # Capture models
    CaptureCreate,
    CaptureUpdate,
    CaptureResponse,
    CaptureListResponse,
    CaptureType as APICaptureType,
    CapturePriority as APICapturePriority,
    # Memory models
    MemoryCreate,
    MemoryResponse,
    MemorySearchResponse,
    MemoryStats,
    MemoryCategory,
    # Challenge models
    ChallengeResponse,
    ChallengeProgress,
    ChallengeListResponse,
    XPResponse,
    ChallengeCategory as APIChallengeCategory,
    # Session models
    ActivityEvent,
    SessionStats,
    TodayResponse,
    HistoryDay,
    HistoryResponse,
    # Poker models
    OpponentProfile,
    OpponentListResponse,
    PokerSession,
    PokerStats,
    PokerSessionListResponse,
    # Cost models
    CostItem,
    CostBreakdown,
    CostProjection,
    # Health models
    ServiceStatus,
    HealthStatus,
    # Generic
    SuccessResponse,
    ErrorResponse,
)

# Initialize router
router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

# Global start time for uptime tracking
_start_time = time.time()

# Lazy-loaded data stores (initialized on first use)
_quick_capture = None
_context_memory = None
_daily_challenges = None
_focus_mode = None

def _get_config() -> dict:
    """Load config from phone_client."""
    import yaml
    config_path = PHONE_CLIENT_PATH / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

def _get_quick_capture():
    """Lazy-load QuickCapture instance."""
    global _quick_capture
    if _quick_capture is None:
        from modes.quick_capture import QuickCapture
        _quick_capture = QuickCapture(_get_config(), storage_dir=str(PHONE_CLIENT_PATH / "captures"))
    return _quick_capture

def _get_context_memory():
    """Lazy-load ContextMemory instance."""
    global _context_memory
    if _context_memory is None:
        from core.context_memory import ContextMemory
        _context_memory = ContextMemory(_get_config(), storage_dir=str(PHONE_CLIENT_PATH / "memory"))
    return _context_memory

def _get_daily_challenges():
    """Lazy-load DailyChallenges instance."""
    global _daily_challenges
    if _daily_challenges is None:
        from core.daily_challenge import DailyChallenges
        _daily_challenges = DailyChallenges(_get_config(), storage_dir=str(PHONE_CLIENT_PATH / "challenges"))
    return _daily_challenges

def _get_focus_mode():
    """Lazy-load FocusMode instance."""
    global _focus_mode
    if _focus_mode is None:
        from modes.focus_mode import FocusMode
        _focus_mode = FocusMode(_get_config())
    return _focus_mode


# ============================================================
# Health & System Routes
# ============================================================

@router.get("/api/health", response_model=HealthStatus)
async def get_health():
    """Get system health status."""
    uptime = int(time.time() - _start_time)

    # Calculate database sizes
    captures_dir = PHONE_CLIENT_PATH / "captures"
    memory_dir = PHONE_CLIENT_PATH / "memory"
    challenges_dir = PHONE_CLIENT_PATH / "challenges"

    db_size_mb = 0.0
    for dir_path in [captures_dir, memory_dir, challenges_dir]:
        if dir_path.exists():
            for f in dir_path.glob("*.json"):
                db_size_mb += f.stat().st_size / (1024 * 1024)

    # Get counts
    qc = _get_quick_capture()
    cm = _get_context_memory()

    services = [
        ServiceStatus(
            name="QuickCapture",
            status="online",
            latency_ms=1,
            last_check=datetime.now()
        ),
        ServiceStatus(
            name="ContextMemory",
            status="online",
            latency_ms=1,
            last_check=datetime.now()
        ),
        ServiceStatus(
            name="DailyChallenges",
            status="online",
            latency_ms=1,
            last_check=datetime.now()
        ),
    ]

    return HealthStatus(
        status="healthy",
        uptime_seconds=uptime,
        database_size_mb=round(db_size_mb, 2),
        captures_count=len(qc._captures),
        memories_count=len(cm._memories),
        services=services,
        version="1.0.0"
    )


# ============================================================
# Quick Capture Routes
# ============================================================

@router.post("/api/captures", response_model=CaptureResponse)
async def create_capture(capture: CaptureCreate):
    """Create a new capture."""
    qc = _get_quick_capture()

    # Map API types to internal types
    from modes.quick_capture import CaptureType as InternalCaptureType

    type_map = {
        APICaptureType.NOTE: InternalCaptureType.VOICE,
        APICaptureType.TODO: InternalCaptureType.TODO,
        APICaptureType.IDEA: InternalCaptureType.IDEA,
        APICaptureType.REMINDER: InternalCaptureType.REMINDER,
        APICaptureType.VOICE: InternalCaptureType.VOICE,
        APICaptureType.PHOTO: InternalCaptureType.PHOTO,
    }

    if capture.reminder_time:
        result = qc.capture_reminder(capture.text, capture.reminder_time)
    else:
        result = qc.capture_text(capture.text, type_map.get(capture.type, InternalCaptureType.VOICE))

    # Record progress for challenges
    dc = _get_daily_challenges()
    from core.daily_challenge import ChallengeCategory
    dc.record_progress(ChallengeCategory.PRODUCTIVITY, 1)

    return CaptureResponse(
        id=result.id,
        text=result.content,
        type=capture.type,
        priority=capture.priority,
        tags=result.tags,
        created_at=result.created_at,
        processed=result.processed,
        reminder_time=result.reminder_time
    )


@router.get("/api/captures", response_model=CaptureListResponse)
async def list_captures(
    category: Optional[str] = None,
    days: int = Query(default=7, ge=1, le=90),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100)
):
    """List captures with optional filtering."""
    qc = _get_quick_capture()

    # Filter by date
    cutoff = datetime.now() - timedelta(days=days)
    captures = [c for c in qc._captures.values() if c.created_at > cutoff]

    # Filter by category/tag
    if category:
        captures = [c for c in captures if category.lower() in [t.lower() for t in c.tags]]

    # Sort by date descending
    captures.sort(key=lambda x: x.created_at, reverse=True)

    # Paginate
    total = len(captures)
    start = (page - 1) * per_page
    end = start + per_page
    captures = captures[start:end]

    return CaptureListResponse(
        captures=[
            CaptureResponse(
                id=c.id,
                text=c.content,
                type=APICaptureType.NOTE,  # Default
                priority=APICapturePriority.NORMAL,
                tags=c.tags,
                created_at=c.created_at,
                processed=c.processed,
                reminder_time=c.reminder_time
            )
            for c in captures
        ],
        total=total,
        page=page,
        per_page=per_page
    )


@router.get("/api/captures/{capture_id}", response_model=CaptureResponse)
async def get_capture(capture_id: str):
    """Get a specific capture."""
    qc = _get_quick_capture()
    capture = qc.get_capture(capture_id)

    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")

    return CaptureResponse(
        id=capture.id,
        text=capture.content,
        type=APICaptureType.NOTE,
        priority=APICapturePriority.NORMAL,
        tags=capture.tags,
        created_at=capture.created_at,
        processed=capture.processed,
        reminder_time=capture.reminder_time
    )


@router.put("/api/captures/{capture_id}", response_model=CaptureResponse)
async def update_capture(capture_id: str, update: CaptureUpdate):
    """Update an existing capture."""
    qc = _get_quick_capture()
    capture = qc.get_capture(capture_id)

    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")

    # Update fields
    if update.text is not None:
        capture.content = update.text
    if update.tags is not None:
        capture.tags = update.tags
    if update.processed is not None:
        capture.processed = update.processed

    qc._save_captures()

    return CaptureResponse(
        id=capture.id,
        text=capture.content,
        type=APICaptureType.NOTE,
        priority=APICapturePriority.NORMAL,
        tags=capture.tags,
        created_at=capture.created_at,
        processed=capture.processed,
        reminder_time=capture.reminder_time
    )


@router.delete("/api/captures/{capture_id}", response_model=SuccessResponse)
async def delete_capture(capture_id: str):
    """Delete a capture."""
    qc = _get_quick_capture()

    if not qc.delete_capture(capture_id):
        raise HTTPException(status_code=404, detail="Capture not found")

    return SuccessResponse(message=f"Capture {capture_id} deleted")


# ============================================================
# Context Memory Routes
# ============================================================

@router.get("/api/memory/search", response_model=MemorySearchResponse)
async def search_memory(q: str = Query(..., min_length=1)):
    """Search memories."""
    cm = _get_context_memory()
    memories = cm.recall(q, limit=20)

    return MemorySearchResponse(
        memories=[
            MemoryResponse(
                id=m.id,
                key=m.keywords[0] if m.keywords else m.entity or "unknown",
                value=m.content,
                category=MemoryCategory.FACT,
                entity=m.entity,
                created_at=m.created_at,
                last_accessed=m.last_accessed,
                access_count=m.access_count
            )
            for m in memories
        ],
        total=len(memories),
        query=q
    )


@router.post("/api/memory", response_model=MemoryResponse)
async def create_memory(memory: MemoryCreate):
    """Store a new memory."""
    cm = _get_context_memory()

    # Map category to memory type
    from core.context_memory import MemoryType
    type_map = {
        MemoryCategory.FACT: MemoryType.FACT,
        MemoryCategory.PREFERENCE: MemoryType.PREFERENCE,
        MemoryCategory.CONTACT: MemoryType.RELATIONSHIP,
        MemoryCategory.LOCATION: MemoryType.LOCATION,
        MemoryCategory.HABIT: MemoryType.HABIT,
        MemoryCategory.RELATIONSHIP: MemoryType.RELATIONSHIP,
    }

    result = cm.remember(
        content=memory.value,
        memory_type=type_map.get(memory.category, MemoryType.FACT),
        keywords=[memory.key],
        entity=memory.entity
    )

    # Record for challenges
    dc = _get_daily_challenges()
    from core.daily_challenge import ChallengeCategory
    dc.record_progress(ChallengeCategory.LEARNING, 1)

    return MemoryResponse(
        id=result.id,
        key=memory.key,
        value=result.content,
        category=memory.category,
        entity=result.entity,
        created_at=result.created_at,
        last_accessed=result.last_accessed,
        access_count=result.access_count
    )


@router.get("/api/memory/stats", response_model=MemoryStats)
async def get_memory_stats():
    """Get memory statistics."""
    cm = _get_context_memory()
    stats = cm.get_stats()

    return MemoryStats(
        total_memories=stats["total_memories"],
        total_entities=stats["total_entities"],
        by_category=stats["by_type"],
        indexed_keywords=stats["indexed_keywords"]
    )


# ============================================================
# Challenge Routes
# ============================================================

@router.get("/api/challenges/today", response_model=ChallengeListResponse)
async def get_today_challenges():
    """Get today's challenges."""
    dc = _get_daily_challenges()
    challenges = dc.get_daily_challenges()
    progress = dc.get_progress()

    return ChallengeListResponse(
        challenges=[
            ChallengeResponse(
                id=c.id,
                title=c.title,
                description=c.description,
                category=APIChallengeCategory(c.category.value),
                difficulty=c.difficulty.name,
                xp_reward=c.xp_reward,
                target=c.target,
                progress=c.progress,
                completed=c.is_completed,
                expires_at=c.expires_at
            )
            for c in challenges
        ],
        xp=XPResponse(
            total_xp=progress.total_xp,
            level=progress.level,
            xp_to_next_level=dc.get_xp_to_next_level(),
            current_streak=progress.current_streak,
            longest_streak=progress.longest_streak,
            achievements_unlocked=len(progress.achievements_unlocked)
        )
    )


@router.post("/api/challenges/progress", response_model=SuccessResponse)
async def update_challenge_progress(progress: ChallengeProgress):
    """Manually update challenge progress."""
    dc = _get_daily_challenges()

    from core.daily_challenge import ChallengeCategory
    category_map = {
        APIChallengeCategory.MENTAL_MATH: ChallengeCategory.MENTAL_MATH,
        APIChallengeCategory.FOCUS: ChallengeCategory.FOCUS,
        APIChallengeCategory.LEARNING: ChallengeCategory.LEARNING,
        APIChallengeCategory.PRODUCTIVITY: ChallengeCategory.PRODUCTIVITY,
        APIChallengeCategory.WELLNESS: ChallengeCategory.WELLNESS,
    }

    dc.record_progress(category_map[progress.category], progress.increment)

    return SuccessResponse(message=f"Progress updated: +{progress.increment} for {progress.category.value}")


@router.get("/api/xp", response_model=XPResponse)
async def get_xp():
    """Get current XP and level."""
    dc = _get_daily_challenges()
    progress = dc.get_progress()

    return XPResponse(
        total_xp=progress.total_xp,
        level=progress.level,
        xp_to_next_level=dc.get_xp_to_next_level(),
        current_streak=progress.current_streak,
        longest_streak=progress.longest_streak,
        achievements_unlocked=len(progress.achievements_unlocked)
    )


# ============================================================
# Session Routes
# ============================================================

@router.get("/api/sessions/today", response_model=TodayResponse)
async def get_today_sessions():
    """Get today's activities."""
    qc = _get_quick_capture()
    dc = _get_daily_challenges()
    fm = _get_focus_mode()

    today = datetime.now().date()

    # Gather activities
    activities = []

    # Add captures from today
    for capture in qc._captures.values():
        if capture.created_at.date() == today:
            activities.append(ActivityEvent(
                timestamp=capture.created_at,
                type="capture",
                description=f"Captured: {capture.content[:50]}...",
                metadata={"tags": capture.tags}
            ))

    # Add focus sessions
    for session in fm._sessions_history:
        if session.started_at.date() == today:
            activities.append(ActivityEvent(
                timestamp=session.started_at,
                type="focus",
                description=f"Focus: {session.task}",
                duration_seconds=session.total_focus_time_s,
                metadata={"pomodoros": session.completed_pomodoros}
            ))

    # Sort by timestamp
    activities.sort(key=lambda x: x.timestamp, reverse=True)

    # Calculate stats
    focus_time = sum(s.total_focus_time_s for s in fm._sessions_history if s.started_at.date() == today)
    captures_today = len([c for c in qc._captures.values() if c.created_at.date() == today])

    stats = SessionStats(
        total_cost=0.0,  # Would need cost tracking
        total_xp_earned=dc.get_progress().total_xp,
        focus_minutes=focus_time // 60,
        captures_count=captures_today,
        poker_hands=0,  # Would need poker integration
        homework_problems=0,
        active_minutes=focus_time // 60 + captures_today * 2
    )

    # Get challenges
    challenges = dc.get_daily_challenges()
    challenge_responses = [
        ChallengeResponse(
            id=c.id,
            title=c.title,
            description=c.description,
            category=APIChallengeCategory(c.category.value),
            difficulty=c.difficulty.name,
            xp_reward=c.xp_reward,
            target=c.target,
            progress=c.progress,
            completed=c.is_completed,
            expires_at=c.expires_at
        )
        for c in challenges
    ]

    # Recent captures
    recent = qc.get_recent(5)
    recent_captures = [
        CaptureResponse(
            id=c.id,
            text=c.content,
            type=APICaptureType.NOTE,
            priority=APICapturePriority.NORMAL,
            tags=c.tags,
            created_at=c.created_at,
            processed=c.processed,
            reminder_time=c.reminder_time
        )
        for c in recent
    ]

    return TodayResponse(
        date=today.isoformat(),
        activities=activities[:20],
        stats=stats,
        challenges=challenge_responses,
        recent_captures=recent_captures
    )


@router.get("/api/sessions/history", response_model=HistoryResponse)
async def get_session_history(days: int = Query(default=7, ge=1, le=90)):
    """Get session history for past N days."""
    qc = _get_quick_capture()
    fm = _get_focus_mode()

    history = []
    today = datetime.now().date()

    for i in range(days):
        date = today - timedelta(days=i)

        # Calculate stats for this day
        focus_time = sum(
            s.total_focus_time_s
            for s in fm._sessions_history
            if s.started_at.date() == date
        )
        captures = len([
            c for c in qc._captures.values()
            if c.created_at.date() == date
        ])

        stats = SessionStats(
            total_cost=0.0,
            total_xp_earned=0,
            focus_minutes=focus_time // 60,
            captures_count=captures,
            poker_hands=0,
            homework_problems=0,
            active_minutes=focus_time // 60 + captures * 2
        )

        # Generate highlight
        highlight = None
        if focus_time > 3600:
            highlight = f"Deep focus: {focus_time // 3600}h+ focused"
        elif captures > 10:
            highlight = f"Productive: {captures} captures"

        history.append(HistoryDay(
            date=date.isoformat(),
            stats=stats,
            highlight=highlight
        ))

    return HistoryResponse(days=history, total_days=days)


@router.get("/api/sessions/stats", response_model=SessionStats)
async def get_session_stats():
    """Get aggregate session statistics."""
    qc = _get_quick_capture()
    dc = _get_daily_challenges()
    fm = _get_focus_mode()

    # Calculate totals
    total_focus = sum(s.total_focus_time_s for s in fm._sessions_history)

    return SessionStats(
        total_cost=0.0,
        total_xp_earned=dc.get_progress().total_xp,
        focus_minutes=total_focus // 60,
        captures_count=len(qc._captures),
        poker_hands=0,
        homework_problems=0,
        active_minutes=total_focus // 60
    )


# ============================================================
# Poker Routes (Stubs - would need poker integration)
# ============================================================

@router.get("/api/poker/opponents", response_model=OpponentListResponse)
async def get_opponents():
    """Get all poker opponent profiles."""
    # Stub - would load from poker tracker
    return OpponentListResponse(opponents=[], total=0)


@router.get("/api/poker/opponent/{name}", response_model=OpponentProfile)
async def get_opponent(name: str):
    """Get specific opponent profile."""
    raise HTTPException(status_code=404, detail="Opponent not found")


@router.get("/api/poker/sessions", response_model=PokerSessionListResponse)
async def get_poker_sessions():
    """Get poker session history."""
    return PokerSessionListResponse(
        sessions=[],
        stats=PokerStats(
            total_sessions=0,
            total_hands=0,
            win_rate_bb_100=0.0,
            vpip=0.0,
            pfr=0.0,
            total_profit_bb=0.0,
            total_cost=0.0,
            biggest_pot_won=0.0,
            biggest_pot_lost=0.0
        )
    )


@router.get("/api/poker/stats", response_model=PokerStats)
async def get_poker_stats():
    """Get poker statistics."""
    return PokerStats(
        total_sessions=0,
        total_hands=0,
        win_rate_bb_100=0.0,
        vpip=0.0,
        pfr=0.0,
        total_profit_bb=0.0,
        total_cost=0.0,
        biggest_pot_won=0.0,
        biggest_pot_lost=0.0
    )


# ============================================================
# Cost Tracking Routes
# ============================================================

@router.get("/api/cost/today", response_model=CostBreakdown)
async def get_today_cost():
    """Get today's cost breakdown."""
    # Stub - would integrate with cost tracking
    return CostBreakdown(
        period="today",
        total=0.0,
        budget=5.0,
        remaining=5.0,
        by_category=[
            CostItem(category="poker", cost=0.0, requests=0),
            CostItem(category="homework", cost=0.0, requests=0),
            CostItem(category="debug", cost=0.0, requests=0),
        ]
    )


@router.get("/api/cost/weekly", response_model=CostBreakdown)
async def get_weekly_cost():
    """Get weekly cost breakdown."""
    return CostBreakdown(
        period="weekly",
        total=0.0,
        budget=35.0,
        remaining=35.0,
        by_category=[]
    )


@router.get("/api/cost/projections", response_model=CostProjection)
async def get_cost_projections():
    """Get cost projections."""
    return CostProjection(
        projected_monthly=0.0,
        current_daily_avg=0.0,
        trend="stable",
        recommendation="Cost tracking not yet integrated"
    )


# ============================================================
# Dashboard HTML Route
# ============================================================

@router.get("/", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the main dashboard HTML page."""
    # Check if frontend exists
    frontend_path = Path(__file__).parent.parent.parent / "frontend" / "dashboard.html"

    if frontend_path.exists():
        with open(frontend_path) as f:
            return f.read()

    # Return a placeholder
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WHAM Dashboard</title>
        <style>
            body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .status { color: green; }
            code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>WHAM Dashboard API</h1>
        <p class="status">✓ API is running</p>
        <h2>Available Endpoints</h2>
        <ul>
            <li><code>GET /dashboard/api/health</code> - System health</li>
            <li><code>GET /dashboard/api/sessions/today</code> - Today's activities</li>
            <li><code>GET /dashboard/api/captures</code> - List captures</li>
            <li><code>POST /dashboard/api/captures</code> - Create capture</li>
            <li><code>GET /dashboard/api/memory/search?q=...</code> - Search memories</li>
            <li><code>GET /dashboard/api/challenges/today</code> - Today's challenges</li>
            <li><code>GET /dashboard/api/xp</code> - XP and level</li>
        </ul>
        <p>Full frontend coming soon. See <code>/docs</code> for API documentation.</p>
    </body>
    </html>
    """
