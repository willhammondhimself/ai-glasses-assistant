"""
WHAM Dashboard - FastAPI Routes.
Provides web API access to WHAM features.
"""
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, UploadFile, File, Form
from fastapi.responses import HTMLResponse

# Import WebSocket manager for live updates
from backend.websocket.manager import ws_manager, WebSocketHandler

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
    # Vision models
    MathDetectionResult,
    CodeDetectionResult,
    PokerDetectionResult,
    TextDetectionResult,
    VisionError,
    # Homework models
    HomeworkSolution,
    HomeworkHistoryEntry,
    HomeworkHistoryQuery,
    HomeworkHistoryResponse,
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
        from phone_client.modes.quick_capture import QuickCapture
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
        from phone_client.modes.focus_mode import FocusMode
        _focus_mode = FocusMode(_get_config())
    return _focus_mode


# Lazy-loaded vision detectors
_math_detector = None
_code_detector = None
_poker_detector = None
_text_detector = None


def _get_math_detector():
    """Get singleton EquationDetector."""
    global _math_detector
    if _math_detector is None:
        from backend.vision.detector import EquationDetector
        _math_detector = EquationDetector()
    return _math_detector


def _get_code_detector():
    """Get singleton CodeDetector."""
    global _code_detector
    if _code_detector is None:
        from backend.vision.detector import CodeDetector
        _code_detector = CodeDetector()
    return _code_detector


def _get_poker_detector():
    """Get singleton PokerDetector."""
    global _poker_detector
    if _poker_detector is None:
        from backend.vision.detector import PokerDetector
        _poker_detector = PokerDetector()
    return _poker_detector


def _get_text_detector():
    """Get singleton TextDetector."""
    global _text_detector
    if _text_detector is None:
        from backend.vision.detector import TextDetector
        _text_detector = TextDetector()
    return _text_detector


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
    from phone_client.modes.quick_capture import CaptureType as InternalCaptureType

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
# RAG Document Routes
# ============================================================

# Lazy-loaded RAG store
_rag_store = None


def _get_rag_store():
    """Get singleton RAGStore instance."""
    global _rag_store
    if _rag_store is None:
        from backend.rag import get_rag_store
        _rag_store = get_rag_store()
    return _rag_store


@router.post("/api/rag/upload")
async def upload_rag_document(
    file: UploadFile = File(...),
    category: str = Form("general")
):
    """
    Upload and index a document for RAG retrieval.

    Accepts: PDF, TXT files (max 10MB)
    Returns: Document ID and character count
    """
    # Validate file type
    allowed_types = ["text/plain", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"Only TXT and PDF supported, got {file.content_type}")

    # Read file
    content_bytes = await file.read()
    if len(content_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10MB)")

    # Extract text based on type
    if file.content_type == "application/pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            content = "\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception as e:
            raise HTTPException(400, f"PDF extraction failed: {str(e)}")
    else:
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "Invalid text encoding (expected UTF-8)")

    if not content.strip():
        raise HTTPException(400, "Document is empty")

    # Add to RAG store
    store = _get_rag_store()
    doc_id = await store.add_document(content, file.filename, category)

    return {
        "success": True,
        "doc_id": doc_id,
        "filename": file.filename,
        "chars": len(content),
        "category": category
    }


@router.get("/api/rag/query")
async def query_rag_documents(
    q: str = Query(..., min_length=1, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    top_k: int = Query(3, ge=1, le=10, description="Number of results")
):
    """
    Semantic search across uploaded documents.

    Returns relevant document chunks with AI-generated answer.
    """
    store = _get_rag_store()
    results = await store.query(q, top_k=top_k, category=category)

    if not results:
        return {
            "query": q,
            "answer": "No relevant documents found. Try uploading some documents first.",
            "sources": []
        }

    # Build context from top results
    context = "\n\n---\n\n".join(
        f"[{r['metadata'].get('filename', 'unknown')}]:\n{r['content'][:2000]}"
        for r in results
    )

    # Generate answer using Gemini
    from phone_client.api_clients.gemini_client import GeminiClient
    gemini = GeminiClient()
    prompt = f"""Based on the following documents, answer this question: {q}

Documents:
{context}

Provide a concise, helpful answer based only on the information in the documents above."""

    try:
        answer = await gemini.quick_query(prompt)
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    return {
        "query": q,
        "answer": answer,
        "sources": [
            {
                "filename": r["metadata"].get("filename", "unknown"),
                "score": round(r["score"], 3),
                "snippet": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"]
            }
            for r in results
        ]
    }


@router.get("/api/rag/stats")
async def get_rag_stats():
    """Get RAG storage statistics."""
    store = _get_rag_store()
    return store.get_stats()


@router.get("/api/rag/documents")
async def list_rag_documents(limit: int = Query(50, ge=1, le=100)):
    """List all uploaded documents."""
    store = _get_rag_store()
    return {"documents": store.list_documents(limit=limit)}


# ============================================================
# Agent Routes (Phase 9)
# ============================================================

# Lazy-loaded agent
_agent_executor = None


def _get_agent():
    """Get singleton AgentExecutor instance."""
    global _agent_executor
    if _agent_executor is None:
        from backend.agent import get_agent
        _agent_executor = get_agent()
    return _agent_executor


@router.post("/api/agent/chat")
async def agent_chat(request: dict):
    """
    Send message to agent for tool execution.

    Request body:
        {"message": "Clear my afternoon"}

    Returns:
        {
            "response": "I've cleared 2 events from your afternoon.",
            "tool_used": "calendar",
            "tool_result": "Cleared 2 events",
            "success": true
        }
    """
    message = request.get("message", "")

    if not message:
        raise HTTPException(400, "Message required")

    agent = _get_agent()
    result = await agent.run(message)

    # Broadcast via WebSocket for real-time UI updates
    try:
        await broadcast_dashboard_update("agent_response", result)
    except Exception:
        pass  # WebSocket optional

    # Broadcast to HUD clients for AR glasses overlay
    try:
        from backend.websocket.hud_handler import broadcast_to_hud
        # Determine mode based on tool used
        mode = result.get("tool_used") or "all"
        await broadcast_to_hud(result, mode=mode)
    except Exception:
        pass  # HUD optional

    return result


@router.get("/api/agent/tools")
async def list_agent_tools():
    """List available agent tools."""
    agent = _get_agent()
    return {
        "tools": [
            {"name": t.name, "description": t.description}
            for t in agent.tools.values()
        ]
    }


@router.get("/api/agent/calendar/status")
async def calendar_auth_status():
    """Check calendar authentication status."""
    from backend.agent.tools.calendar import CalendarTool
    cal = CalendarTool()
    return {
        "authenticated": cal.is_configured(),
        "message": "Apple Calendar connected" if cal.is_configured() else "Calendar in mock mode (set APPLE_ID & APPLE_APP_PASSWORD)"
    }


@router.post("/api/agent/reset")
async def reset_agent_chat():
    """Reset agent conversation history."""
    agent = _get_agent()
    agent.reset_chat()
    return {"success": True, "message": "Agent conversation reset"}


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
# Poker Routes
# ============================================================

# Lazy-load poker analyzer
_poker_analyzer: Optional['PokerAnalyzer'] = None

def get_poker_analyzer():
    """Get singleton PokerAnalyzer instance."""
    global _poker_analyzer
    if _poker_analyzer is None:
        try:
            from phone_client.intelligence.poker_analyzer import PokerAnalyzer
            _poker_analyzer = PokerAnalyzer()
        except Exception as e:
            import logging
            logging.error(f"Failed to initialize PokerAnalyzer: {e}")
            _poker_analyzer = None
    return _poker_analyzer


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
# Poker Analysis Routes (Phase 5)
# ============================================================

@router.get("/api/poker/files")
async def list_poker_sessions():
    """List all available poker session files."""
    import json

    analyzer = get_poker_analyzer()
    if not analyzer:
        raise HTTPException(500, "PokerAnalyzer not available")

    session_files = []
    try:
        for session_file in analyzer.sessions_dir.glob("*.json"):
            if session_file.stem in ["reviews", "profiles"]:
                continue

            with open(session_file) as f:
                data = json.load(f)

            session_files.append({
                "session_id": session_file.stem,
                "date": data.get("start_time", 0),
                "hands": len(data.get("hands", [])),
                "profit_bb": data.get("stats", {}).get("total_profit_bb", 0.0),
                "reviewed": (analyzer.reviews_dir / f"{session_file.stem}.json").exists()
            })

        return sorted(session_files, key=lambda x: x["date"], reverse=True)
    except Exception as e:
        raise HTTPException(500, f"Failed to list sessions: {e}")


@router.post("/api/poker/analyze/{session_id}")
async def analyze_poker_session(session_id: str, force_refresh: bool = False):
    """
    Deep analysis of a poker session.

    Returns:
        Session review with hand-by-hand GTO analysis
    """
    analyzer = get_poker_analyzer()
    if not analyzer:
        raise HTTPException(500, "PokerAnalyzer not available")

    try:
        result = await analyzer.analyze_session(session_id, force_refresh)
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        import logging
        logging.error(f"Session analysis error: {e}")
        raise HTTPException(500, f"Analysis failed: {e}")


@router.get("/api/poker/profile/{player_name}")
async def profile_poker_opponent(player_name: str, force_refresh: bool = False):
    """
    Build opponent profile with exploitation strategy.

    Returns:
        OpponentProfile with stats and exploits
    """
    analyzer = get_poker_analyzer()
    if not analyzer:
        raise HTTPException(500, "PokerAnalyzer not available")

    try:
        profile = await analyzer.profile_opponent(player_name, force_refresh)
        return asdict(profile)
    except Exception as e:
        import logging
        logging.error(f"Opponent profiling error: {e}")
        raise HTTPException(500, f"Profiling failed: {e}")


@router.get("/api/poker/leaks")
async def analyze_poker_leaks(last_n_sessions: int = 10):
    """
    Identify recurring mistakes across sessions.

    Returns:
        LeakReport with prioritized fixes
    """
    analyzer = get_poker_analyzer()
    if not analyzer:
        raise HTTPException(500, "PokerAnalyzer not available")

    try:
        report = await analyzer.find_leaks(last_n_sessions)
        return asdict(report)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        import logging
        logging.error(f"Leak analysis error: {e}")
        raise HTTPException(500, f"Analysis failed: {e}")


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
# Suggestions Routes
# ============================================================

@router.get("/api/suggestions")
async def get_suggestions():
    """
    Get proactive suggestions based on usage patterns.

    Returns list of AI-powered suggestions:
    - Time patterns: "You usually study at 2pm"
    - Usage anomalies: "You haven't captured anything today"
    - Cost patterns: "Poker costs high on Sundays, enable budget mode?"

    Returns:
        List[dict]: Up to 3 suggestions sorted by priority and confidence
    """
    from phone_client.intelligence.predictor import PredictiveEngine

    predictor = PredictiveEngine()
    suggestions = await predictor.get_suggestions()

    return [
        {
            'type': s.type,
            'title': s.title,
            'message': s.message,
            'confidence': s.confidence,
            'action': s.action,
            'priority': s.priority
        }
        for s in suggestions
    ]


# ============================================================
# Phase 4: Research & Skills Routes
# ============================================================

@router.get("/api/research/history")
async def get_research_history(days: int = Query(7, ge=1, le=90)):
    """
    Get research query history from Phase 4.

    Args:
        days: Number of days to look back (1-90)

    Returns:
        Dict with queries, total_queries, total_cost, period_days
    """
    from phone_client.intelligence.pattern_analyzer import PatternAnalyzer

    analyzer = PatternAnalyzer()
    sessions = analyzer._load_sessions(days)

    # Extract all research queries from general stats
    queries = []
    for session in sessions:
        # Research queries are in general.research_queries
        general_stats = session.get('general', {})
        for query in general_stats.get('research_queries', []):
            queries.append(query)

    # Sort by timestamp descending
    queries.sort(key=lambda q: q['timestamp'], reverse=True)

    return {
        'queries': queries,
        'total_queries': len(queries),
        'total_cost': sum(q['cost'] for q in queries),
        'period_days': days
    }


@router.get("/api/skills")
async def get_skill_metrics(days: int = Query(30, ge=7, le=90)):
    """
    Get skill progression metrics from Phase 4.

    Args:
        days: Number of days to analyze (7-90)

    Returns:
        Dict with skill metrics for poker, homework, focus, and overall consistency
    """
    from phone_client.intelligence.pattern_analyzer import PatternAnalyzer

    analyzer = PatternAnalyzer()
    sessions = analyzer._load_sessions(days)

    if len(sessions) < 3:
        raise HTTPException(400, "Need at least 3 days of data for skill analysis")

    metrics = analyzer.calculate_skill_metrics(sessions, days)

    return {
        'success': True,
        'days_analyzed': len(sessions),
        'metrics': metrics,
        'updated_at': datetime.now().isoformat()
    }


@router.post("/api/research/verify")
async def verify_claim(request: dict):
    """
    Fact-check a claim using Perplexity (Phase 4).

    Request body:
        {
            "claim": str,
            "domains": List[str] (optional)
        }

    Returns:
        Dict with verdict, confidence, explanation, sources, cost
    """
    from api_clients.perplexity_client import PerplexityClient

    perplexity = PerplexityClient()

    if not perplexity.is_available:
        raise HTTPException(503, "Perplexity API not available")

    result = await perplexity.verify_claim(
        claim=request['claim'],
        search_domains=request.get('domains')
    )

    return result


# ============================================================
# Phase 4B: Academic Intelligence Routes
# ============================================================

@router.post("/api/academic/concept-bridge")
async def concept_bridge(request: dict):
    """
    Build concept bridge across subjects (Phase 4B).

    Request body:
        {
            "concept": str,
            "connect_to": List[str],
            "level": str (optional, default: "undergraduate")
        }

    Returns:
        ConceptBridge dataclass as dict
    """
    from phone_client.intelligence.academic_assistant import AcademicAssistant

    assistant = AcademicAssistant()

    concept = request.get('concept')
    connect_to = request.get('connect_to', [])
    level = request.get('level', 'undergraduate')

    if not concept or not connect_to:
        raise HTTPException(400, "Missing concept or connect_to")

    result = await assistant.build_concept_bridge(concept, connect_to, level)

    return asdict(result)


@router.post("/api/academic/derivation")
async def derivation(request: dict):
    """
    Explain mathematical derivation step-by-step (Phase 4B).

    Request body:
        {
            "equation": str,
            "context": str (optional),
            "show_all_steps": bool (optional, default: True)
        }

    Returns:
        DerivationSteps dataclass as dict
    """
    from phone_client.intelligence.academic_assistant import AcademicAssistant

    assistant = AcademicAssistant()

    equation = request.get('equation')
    context = request.get('context', '')
    show_all_steps = request.get('show_all_steps', True)

    if not equation:
        raise HTTPException(400, "Missing equation")

    result = await assistant.explain_derivation(equation, context, show_all_steps)

    return asdict(result)


@router.post("/api/academic/problem-strategy")
async def problem_strategy(request: dict):
    """
    Generate problem-solving strategy (Phase 4B).

    Request body:
        {
            "problem_type": str,
            "include_examples": bool (optional, default: True)
        }

    Returns:
        ProblemStrategy dataclass as dict
    """
    from phone_client.intelligence.academic_assistant import AcademicAssistant

    assistant = AcademicAssistant()

    problem_type = request.get('problem_type')
    include_examples = request.get('include_examples', True)

    if not problem_type:
        raise HTTPException(400, "Missing problem_type")

    result = await assistant.generate_problem_strategy(problem_type, include_examples)

    return asdict(result)


@router.post("/api/academic/exam-pattern")
async def exam_pattern(request: dict):
    """
    Analyze exam patterns for a course (Phase 4B).

    Request body:
        {
            "course": str,
            "exam_type": str (optional, default: "final"),
            "past_exams_context": str (optional)
        }

    Returns:
        ExamPattern dataclass as dict
    """
    from phone_client.intelligence.academic_assistant import AcademicAssistant

    assistant = AcademicAssistant()

    course = request.get('course')
    exam_type = request.get('exam_type', 'final')
    past_exams_context = request.get('past_exams_context', '')

    if not course:
        raise HTTPException(400, "Missing course")

    result = await assistant.analyze_exam_patterns(course, exam_type, past_exams_context)

    return asdict(result)


@router.post("/api/academic/notation")
async def notation(request: dict):
    """
    Decode mathematical notation (Phase 4B).

    Request body:
        {
            "notation": str,
            "subject_context": str (optional)
        }

    Returns:
        NotationExplanation dataclass as dict
    """
    from phone_client.intelligence.academic_assistant import AcademicAssistant

    assistant = AcademicAssistant()

    notation = request.get('notation')
    subject_context = request.get('subject_context', '')

    if not notation:
        raise HTTPException(400, "Missing notation")

    result = await assistant.decode_notation(notation, subject_context)

    return asdict(result)


@router.post("/api/academic/visual-intuition")
async def visual_intuition(request: dict):
    """
    Find visual intuition for a concept (Phase 4C).

    Request body:
        {
            "concept": str,
            "subject_area": str (optional)
        }

    Returns:
        VisualIntuition dataclass as dict
    """
    from phone_client.intelligence.academic_assistant import AcademicAssistant

    assistant = AcademicAssistant()

    concept = request.get('concept')
    if not concept:
        raise HTTPException(400, "Missing concept")

    result = await assistant.find_visual_intuition(
        concept,
        request.get('subject_area', '')
    )

    return asdict(result)


@router.post("/api/academic/formula-sheet")
async def formula_sheet(request: dict):
    """
    Generate formula sheet (Phase 4C).

    Request body:
        {
            "topic": str
        }

    Returns:
        FormulaSheet dataclass as dict
    """
    from phone_client.intelligence.academic_assistant import AcademicAssistant

    assistant = AcademicAssistant()

    topic = request.get('topic')
    if not topic:
        raise HTTPException(400, "Missing topic")

    result = await assistant.generate_formula_sheet(topic)

    return asdict(result)


@router.post("/api/academic/paper-summary")
async def paper_summary(request: dict):
    """
    Pre-read academic paper (Phase 4C).

    Request body:
        {
            "title": str,
            "authors": str (optional),
            "year": str (optional)
        }

    Returns:
        PaperSummary dataclass as dict
    """
    from phone_client.intelligence.academic_assistant import AcademicAssistant

    assistant = AcademicAssistant()

    title = request.get('title')
    if not title:
        raise HTTPException(400, "Missing paper title")

    result = await assistant.preread_paper(
        title,
        request.get('authors', ''),
        request.get('year', '')
    )

    return asdict(result)


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


# =============================================================================
# VISION DETECTION ROUTES (WHAM Vision)
# =============================================================================

import logging
logger = logging.getLogger(__name__)


@router.post("/api/vision/math", response_model=MathDetectionResult)
async def detect_math(file: UploadFile = File(...)):
    """
    Extract math equation from uploaded image.

    Accepts: image/jpeg, image/png (max 10MB)
    Returns: MathDetectionResult with equation and type
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, detail="Invalid file type. Use JPEG or PNG")

    # Read image bytes (limit to 10MB)
    MAX_SIZE = 10 * 1024 * 1024
    image_data = await file.read()
    if len(image_data) > MAX_SIZE:
        raise HTTPException(400, detail="File too large. Max 10MB")

    # Detect using EquationDetector
    detector = _get_math_detector()
    if not detector.is_available:
        raise HTTPException(503, detail="Gemini vision not available")

    try:
        scan_result = await detector.detect(image_data)

        if not scan_result:
            # No equation found
            return MathDetectionResult(
                equation="",
                problem_type="unknown",
                variables=[],
                confidence=0.0
            )

        return MathDetectionResult(**scan_result.metadata)

    except Exception as e:
        logger.error(f"Math detection failed: {e}")
        raise HTTPException(500, detail=f"Detection failed: {str(e)}")


@router.post("/api/vision/code", response_model=CodeDetectionResult)
async def detect_code(file: UploadFile = File(...)):
    """
    Extract code from uploaded screenshot.

    Accepts: image/jpeg, image/png (max 10MB)
    Returns: CodeDetectionResult with code and language
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, detail="Invalid file type. Use JPEG or PNG")

    MAX_SIZE = 10 * 1024 * 1024
    image_data = await file.read()
    if len(image_data) > MAX_SIZE:
        raise HTTPException(400, detail="File too large. Max 10MB")

    detector = _get_code_detector()
    if not detector.is_available:
        raise HTTPException(503, detail="Gemini vision not available")

    try:
        scan_result = await detector.detect(image_data)

        if not scan_result:
            return CodeDetectionResult(
                code="",
                language="unknown",
                error_visible=False,
                confidence=0.0
            )

        return CodeDetectionResult(**scan_result.metadata)

    except Exception as e:
        logger.error(f"Code detection failed: {e}")
        raise HTTPException(500, detail=f"Detection failed: {str(e)}")


@router.post("/api/vision/poker", response_model=PokerDetectionResult)
async def detect_poker(file: UploadFile = File(...)):
    """
    Extract poker cards and game state from uploaded image.

    Accepts: image/jpeg, image/png (max 10MB)
    Returns: PokerDetectionResult with cards and pot info
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, detail="Invalid file type. Use JPEG or PNG")

    MAX_SIZE = 10 * 1024 * 1024
    image_data = await file.read()
    if len(image_data) > MAX_SIZE:
        raise HTTPException(400, detail="File too large. Max 10MB")

    detector = _get_poker_detector()
    if not detector.is_available:
        raise HTTPException(503, detail="Gemini vision not available")

    try:
        scan_result = await detector.detect(image_data)

        if not scan_result:
            return PokerDetectionResult(
                hole_cards=["?", "?"],
                board=[],
                pot_size_bb=0.0,
                bet_facing_bb=0.0,
                confidence=0.0
            )

        return PokerDetectionResult(**scan_result.metadata)

    except Exception as e:
        logger.error(f"Poker detection failed: {e}")
        raise HTTPException(500, detail=f"Detection failed: {str(e)}")


@router.post("/api/vision/text", response_model=TextDetectionResult)
async def detect_text(file: UploadFile = File(...)):
    """
    Extract text from uploaded image via OCR.

    Accepts: image/jpeg, image/png (max 10MB)
    Returns: TextDetectionResult with extracted text
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, detail="Invalid file type. Use JPEG or PNG")

    MAX_SIZE = 10 * 1024 * 1024
    image_data = await file.read()
    if len(image_data) > MAX_SIZE:
        raise HTTPException(400, detail="File too large. Max 10MB")

    detector = _get_text_detector()
    if not detector.is_available:
        raise HTTPException(503, detail="Gemini vision not available")

    try:
        scan_result = await detector.detect(image_data)

        if not scan_result:
            return TextDetectionResult(
                text="",
                confidence=0.0
            )

        return TextDetectionResult(**scan_result.metadata)

    except Exception as e:
        logger.error(f"Text detection failed: {e}")
        raise HTTPException(500, detail=f"Detection failed: {str(e)}")


@router.get("/api/vision/health")
async def vision_health():
    """Check vision detector availability."""
    detector = _get_math_detector()  # Use any detector to check Gemini
    return {
        "gemini_available": detector.is_available,
        "detectors": ["math", "code", "poker", "text"]
    }


# =============================================================================
# HOMEWORK VISION COPILOT (Phase 7)
# =============================================================================

# Import homework history module
from backend.homework import history as homework_history


async def _solve_with_academic_tools(
    equation: str,
    problem_type: str
) -> dict:
    """
    Route to appropriate academic tool based on problem type.

    Args:
        equation: The math equation to solve
        problem_type: Type from vision detection (algebra, calculus, etc.)

    Returns:
        dict with solution_steps, concept_explanation, tool_used
    """
    from phone_client.intelligence.academic_assistant import AcademicAssistant

    assistant = AcademicAssistant()

    # Route based on problem type
    if problem_type in ["calculus_derivative", "calculus_integral", "limit"]:
        # Use derivation explainer for calculus
        # Returns DerivationSteps with: steps (List[Dict]), final_result, context
        result = await assistant.explain_derivation(equation, context=problem_type)
        # Extract step explanations from the dict structure
        step_texts = []
        for step in (result.steps or []):
            if isinstance(step, dict):
                expr = step.get('expression', '')
                expl = step.get('explanation', '')
                step_texts.append(f"{expr}: {expl}" if expr else expl)
            else:
                step_texts.append(str(step) if step else "")
        return {
            "solution_steps": step_texts,
            "concept_explanation": result.final_result or result.context or "",
            "tool_used": "derivation_explainer"
        }

    elif problem_type in ["series", "polynomial", "algebra"]:
        # Use problem strategy for algebraic problems
        result = await assistant.generate_problem_strategy(f"{problem_type}: {equation}")
        return {
            "solution_steps": result.approach_steps,
            "concept_explanation": result.key_insights[0] if result.key_insights else "",
            "tool_used": "problem_strategy"
        }

    elif problem_type in ["trigonometry", "logarithm"]:
        # Use concept bridge for trig/log
        result = await assistant.build_concept_bridge(
            equation,
            connect_to=["algebra", "calculus"],
            level="undergraduate"
        )
        return {
            "solution_steps": [f"{r['target']}: {r['relationship']}" for r in result.relationships],
            "concept_explanation": result.importance or "",
            "tool_used": "concept_bridge"
        }

    elif problem_type == "arithmetic":
        # Simple arithmetic - compute directly, give concise answer
        import re

        # Clean up the equation for evaluation
        expr = equation.replace('=', '').strip()
        expr = expr.replace('÷', '/').replace('×', '*').replace('−', '-')
        # Handle implicit multiplication: 2(3) -> 2*(3)
        expr = re.sub(r'(\d)\(', r'\1*(', expr)

        try:
            # Safely evaluate the arithmetic expression
            # Only allow basic math operations
            allowed_chars = set('0123456789+-*/().^ ')
            if all(c in allowed_chars for c in expr):
                expr_for_eval = expr.replace('^', '**')
                result_value = eval(expr_for_eval)

                # Format as integer if whole number
                if isinstance(result_value, float) and result_value.is_integer():
                    result_value = int(result_value)

                return {
                    "solution_steps": [
                        f"Expression: {equation}",
                        f"Evaluate following order of operations (PEMDAS)",
                        f"Answer: {result_value}"
                    ],
                    "concept_explanation": f"The answer is {result_value}",
                    "tool_used": "arithmetic_calculator"
                }
        except Exception:
            pass  # Fall through to default handler if evaluation fails

        # If direct evaluation fails, use a simple prompt
        result = await assistant.explain_derivation(equation, context="simple arithmetic - give a SHORT 2-3 step solution with a clear numerical answer")
        step_texts = []
        for step in (result.steps or [])[:3]:  # Limit to 3 steps
            if isinstance(step, dict):
                expr = step.get('expression', '')
                expl = step.get('explanation', '')
                step_texts.append(f"{expr}: {expl}" if expr else expl)
            else:
                step_texts.append(str(step) if step else "")
        return {
            "solution_steps": step_texts,
            "concept_explanation": result.final_result or "",
            "tool_used": "arithmetic_simplified"
        }

    else:
        # Default: use derivation explainer
        # Returns DerivationSteps with: steps (List[Dict]), final_result, context
        result = await assistant.explain_derivation(equation, context="general mathematics")
        # Extract step explanations from the dict structure
        step_texts = []
        for step in (result.steps or []):
            if isinstance(step, dict):
                expr = step.get('expression', '')
                expl = step.get('explanation', '')
                step_texts.append(f"{expr}: {expl}" if expr else expl)
            else:
                step_texts.append(str(step) if step else "")
        return {
            "solution_steps": step_texts,
            "concept_explanation": result.final_result or result.context or "",
            "tool_used": "derivation_explainer"
        }


@router.post("/api/homework/solve", response_model=HomeworkSolution)
async def solve_homework(file: UploadFile = File(...)):
    """
    Complete homework vision copilot pipeline.

    1. Extracts math equation from image (Gemini Vision)
    2. Routes to appropriate Academic Intelligence tool
    3. Returns structured solution with steps
    4. Logs to homework history

    Accepts: image/jpeg, image/png (max 10MB)
    Returns: HomeworkSolution with problem, steps, explanation
    """
    start_time = time.time()

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, detail="Invalid file type. Use JPEG or PNG")

    MAX_SIZE = 10 * 1024 * 1024
    image_data = await file.read()
    if len(image_data) > MAX_SIZE:
        raise HTTPException(400, detail="File too large. Max 10MB")

    # Step 1: Detect math equation
    detector = _get_math_detector()
    if not detector.is_available:
        raise HTTPException(503, detail="Gemini vision not available")

    try:
        scan_result = await detector.detect(image_data)

        if not scan_result or not scan_result.content:
            return HomeworkSolution(
                problem=MathDetectionResult(
                    equation="",
                    problem_type="unknown",
                    variables=[],
                    confidence=0.0
                ),
                solution_steps=[],
                concept_explanation="No math equation detected in image.",
                tool_used="none",
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message="No math equation detected"
            )

        # Build problem result
        problem = MathDetectionResult(**scan_result.metadata)

        # Step 2: Solve with academic tools
        try:
            solution = await _solve_with_academic_tools(
                problem.equation,
                problem.problem_type
            )
        except Exception as e:
            logger.error(f"Academic tool error: {e}")
            solution = {
                "solution_steps": [f"Error: {str(e)}"],
                "concept_explanation": "",
                "tool_used": "error"
            }

        execution_time_ms = (time.time() - start_time) * 1000

        # Build response
        result = HomeworkSolution(
            problem=problem,
            solution_steps=solution["solution_steps"],
            concept_explanation=solution["concept_explanation"],
            tool_used=solution["tool_used"],
            timestamp=time.time(),
            execution_time_ms=execution_time_ms,
            success=True
        )

        # Step 3: Log to homework history
        try:
            homework_history.append_entry({
                "problem_latex": problem.equation,
                "problem_type": problem.problem_type,
                "solution_summary": solution["solution_steps"][0] if solution["solution_steps"] else "No solution",
                "full_solution": result.model_dump()
            })
        except Exception as e:
            logger.warning(f"Failed to log homework history: {e}")

        # Broadcast full solution to dashboard for real-time display
        try:
            await broadcast_dashboard_update("homework_solution", {
                "problem_latex": problem.equation,
                "problem_type": problem.problem_type,
                "solution_steps": solution["solution_steps"],
                "concept_explanation": solution["concept_explanation"],
                "tool_used": solution["tool_used"],
                "execution_time_ms": execution_time_ms,
                "success": True,
                "confidence": problem.confidence
            })
        except Exception:
            pass  # Non-critical

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Homework solve failed: {e}")
        raise HTTPException(500, detail=f"Solution failed: {str(e)}")


@router.get("/api/homework/history", response_model=HomeworkHistoryResponse)
async def get_homework_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    problem_type: Optional[str] = None,
    starred_only: bool = False
):
    """
    Query homework history with filters.

    Args:
        limit: Maximum entries to return (1-100)
        offset: Pagination offset
        problem_type: Filter by problem type
        starred_only: Only return starred entries

    Returns:
        HomeworkHistoryResponse with entries and total count
    """
    try:
        result = homework_history.query_history(
            limit=limit,
            offset=offset,
            problem_type=problem_type,
            starred_only=starred_only
        )

        return HomeworkHistoryResponse(
            entries=[HomeworkHistoryEntry(**e) for e in result["entries"]],
            total=result["total"]
        )
    except Exception as e:
        logger.error(f"Homework history query failed: {e}")
        raise HTTPException(500, detail=f"Query failed: {str(e)}")


@router.get("/api/homework/history/{entry_id}")
async def get_homework_entry(entry_id: str):
    """Get a single homework history entry by ID."""
    entry = homework_history.get_entry(entry_id)
    if not entry:
        raise HTTPException(404, detail="Entry not found")
    return HomeworkHistoryEntry(**entry)


@router.patch("/api/homework/history/{entry_id}")
async def update_homework_entry(entry_id: str, updates: dict):
    """
    Update homework entry (star/unstar, add tags).

    Request body:
        {
            "starred": bool (optional),
            "tags": List[str] (optional)
        }
    """
    if not homework_history.update_entry(entry_id, updates):
        raise HTTPException(404, detail="Entry not found")
    return {"success": True, "message": "Entry updated"}


@router.delete("/api/homework/history/{entry_id}")
async def delete_homework_entry(entry_id: str):
    """Delete a homework history entry."""
    if not homework_history.delete_entry(entry_id):
        raise HTTPException(404, detail="Entry not found")
    return {"success": True, "message": "Entry deleted"}


@router.get("/api/homework/stats")
async def get_homework_stats():
    """Get homework history statistics."""
    return homework_history.get_stats()


# =============================================================================
# WEBSOCKET LIVE UPDATES
# =============================================================================

class DashboardWebSocketHandler(WebSocketHandler):
    """Handler for dashboard real-time updates."""

    async def handle_message(self, websocket, topic, data, conn_info):
        """Handle incoming WebSocket messages."""
        if data.get("type") == "ping":
            await websocket.send_json({"type": "pong"})
        elif data.get("type") == "subscribe":
            # Future: handle subscription to specific update types
            pass


# Global handler instance
_dashboard_ws_handler = DashboardWebSocketHandler()


@router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for live dashboard updates.

    Sends real-time notifications for:
    - session_update: New session activity
    - challenge_complete: Challenge completion with XP
    - cost_update: Real-time cost tracking
    - capture_created: New capture saved

    Client should send periodic ping messages to maintain connection.
    """
    await _dashboard_ws_handler.handle_connection(
        websocket,
        topic="dashboard",
        metadata={"type": "web_dashboard", "connected_at": datetime.now().isoformat()}
    )


# =============================================================================
# BROADCAST FUNCTIONS (call from other modules to push updates)
# =============================================================================

async def broadcast_dashboard_update(update_type: str, data: dict):
    """Broadcast update to all connected dashboard clients."""
    await ws_manager.broadcast("dashboard", {
        "type": update_type,
        "data": data,
        "timestamp": datetime.now().isoformat()
    })


async def broadcast_session_update(session_type: str, details: dict):
    """Broadcast new session activity."""
    await broadcast_dashboard_update("session_update", {
        "session_type": session_type,
        **details
    })


async def broadcast_challenge_complete(challenge: str, xp_earned: int):
    """Broadcast challenge completion."""
    await broadcast_dashboard_update("challenge_complete", {
        "challenge": challenge,
        "xp": xp_earned
    })


async def broadcast_cost_update(cost_today: float, projected_monthly: float):
    """Broadcast cost tracking update."""
    await broadcast_dashboard_update("cost_update", {
        "today": cost_today,
        "projected_monthly": projected_monthly
    })


async def broadcast_capture_created(capture_preview: str, capture_type: str = "note"):
    """Broadcast new capture notification."""
    await broadcast_dashboard_update("capture_created", {
        "preview": capture_preview[:100],
        "type": capture_type
    })
