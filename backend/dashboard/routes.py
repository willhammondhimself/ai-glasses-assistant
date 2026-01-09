"""Dashboard routes - FastAPI endpoints for WHAM web UI."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Templates will be initialized when the app starts
templates: Optional[Jinja2Templates] = None


def init_templates(template_dir: Path = None):
    """Initialize Jinja2 templates."""
    global templates
    if template_dir is None:
        template_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(template_dir))
    logger.info(f"Dashboard templates initialized from {template_dir}")


# Helper functions to get data from services

async def get_learning_stats():
    """Get flashcard learning statistics."""
    try:
        from backend.services.learning_service import get_learning_service
        service = get_learning_service()
        stats = await service.get_stats()
        return stats.to_dict()
    except Exception as e:
        logger.error(f"Error getting learning stats: {e}")
        return {"error": str(e)}


async def get_flashcards(limit: int = 50):
    """Get recent flashcards."""
    try:
        from backend.services.learning_service import get_learning_service
        service = get_learning_service()
        cards = await service.get_all_cards(limit=limit)
        return [c.to_dict() for c in cards]
    except Exception as e:
        logger.error(f"Error getting flashcards: {e}")
        return []


def get_recent_alerts(limit: int = 20):
    """Get recent alerts from proactive engine."""
    try:
        from backend.services.proactive_engine import get_proactive_engine
        engine = get_proactive_engine()
        alerts = engine.get_recent_alerts(limit=limit)
        return [a.to_dict() for a in alerts]
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return []


def get_alert_preferences():
    """Get alert preferences."""
    try:
        from backend.services.proactive_engine import get_proactive_engine
        engine = get_proactive_engine()
        return engine.get_preferences()
    except Exception as e:
        logger.error(f"Error getting alert preferences: {e}")
        return {}


async def get_reminders():
    """Get active reminders/timers."""
    try:
        from backend.voice.tools.reminders import RemindersTool
        tool = RemindersTool()
        result = await tool._list_reminders()
        return result.data.get("reminders", [])
    except Exception as e:
        logger.error(f"Error getting reminders: {e}")
        return []


def get_service_health():
    """Get health status of all services."""
    import os
    services = []
    
    # Check API keys
    api_keys = [
        ("Gemini", "GEMINI_API_KEY"),
        ("Anthropic", "ANTHROPIC_API_KEY"),
        ("Perplexity", "PERPLEXITY_API_KEY"),
        ("LiveKit", "LIVEKIT_API_KEY"),
        ("Spotify", "SPOTIFY_CLIENT_ID"),
        ("Home Assistant", "HOME_ASSISTANT_TOKEN"),
    ]
    
    for name, env_var in api_keys:
        configured = bool(os.getenv(env_var))
        services.append({
            "name": name,
            "status": "configured" if configured else "not_configured",
            "icon": "✅" if configured else "⚠️"
        })
    
    return services


def get_cost_summary():
    """Get cost tracking summary."""
    try:
        from backend.router.cost_tracker import CostTracker
        tracker = CostTracker()
        return {
            "today": tracker.get_daily_cost(),
            "month": tracker.get_monthly_cost(),
            "budget": tracker.monthly_budget,
            "remaining": tracker.get_budget_remaining()
        }
    except Exception as e:
        logger.debug(f"Cost tracker not available: {e}")
        return {"today": 0, "month": 0, "budget": 50.0, "remaining": 50.0}


# Routes

@router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page."""
    if templates is None:
        init_templates()
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "learning_stats": await get_learning_stats(),
        "alerts": get_recent_alerts(limit=5),
        "services": get_service_health(),
        "costs": get_cost_summary(),
        "now": datetime.now()
    })


@router.get("/learning", response_class=HTMLResponse)
async def learning_page(request: Request):
    """Flashcard learning management page."""
    if templates is None:
        init_templates()
    
    return templates.TemplateResponse("learning.html", {
        "request": request,
        "stats": await get_learning_stats(),
        "cards": await get_flashcards(limit=50),
        "now": datetime.now()
    })


@router.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request):
    """Alert history and preferences page."""
    if templates is None:
        init_templates()
    
    return templates.TemplateResponse("alerts.html", {
        "request": request,
        "alerts": get_recent_alerts(limit=50),
        "preferences": get_alert_preferences(),
        "now": datetime.now()
    })


@router.get("/reminders", response_class=HTMLResponse)
async def reminders_page(request: Request):
    """Reminders and timers management page."""
    if templates is None:
        init_templates()
    
    return templates.TemplateResponse("reminders.html", {
        "request": request,
        "reminders": await get_reminders(),
        "now": datetime.now()
    })


# API endpoints for HTMX

@router.post("/api/flashcards")
async def create_flashcard(
    front: str = Form(...),
    back: str = Form(...),
    tags: str = Form("")
):
    """Create a new flashcard."""
    try:
        from backend.services.learning_service import get_learning_service
        service = get_learning_service()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        card = await service.create_card(front=front, back=back, tags=tag_list or None)
        return {"success": True, "card": card.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/flashcards/{card_id}")
async def delete_flashcard(card_id: int):
    """Delete a flashcard."""
    try:
        from backend.services.learning_service import get_learning_service
        service = get_learning_service()
        success = await service.delete_card(card_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/reminders")
async def create_reminder(
    message: str = Form(...),
    minutes: int = Form(...)
):
    """Create a new reminder."""
    try:
        from backend.voice.tools.reminders import RemindersTool
        tool = RemindersTool()
        result = await tool._set_reminder(f"remind me in {minutes} minutes to {message}")
        return {"success": result.success, "data": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/reminders/{reminder_id}")
async def cancel_reminder(reminder_id: int):
    """Cancel a reminder."""
    try:
        from backend.voice.tools.reminders import RemindersTool
        tool = RemindersTool()
        await tool.mark_completed(reminder_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/alerts/toggle/{alert_type}")
async def toggle_alert_type(alert_type: str, enabled: bool = Form(...)):
    """Enable or disable an alert type."""
    try:
        from backend.services.proactive_engine import get_proactive_engine, AlertType
        engine = get_proactive_engine()
        type_enum = AlertType(alert_type)
        
        if enabled:
            engine.enable_alert_type(type_enum)
        else:
            engine.disable_alert_type(type_enum)
        
        return {"success": True, "enabled": enabled}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
