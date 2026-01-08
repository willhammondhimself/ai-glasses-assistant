"""
AI Glasses Coach - FastAPI Backend Server v3.0.0

Main server that routes requests to specialized engines:
- Math: SymPy (free) → Claude fallback
- Vision: OpenRouter/GPT-4o for equation extraction
- CS: Claude for code explanation and debugging
- Poker: Claude for GTO-based hand analysis
- Chemistry: SymPy + Claude for chemistry problems
- Biology: Genetics calculations + Claude explanations
- Statistics: scipy/numpy + Claude for interpretation
- Quant: Interview prep for Jane Street, Citadel, etc.
"""

import time
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Import engines
from backend.math_solver import MathEngine
from backend.vision import VisionEngine
from backend.cs import CSEngine
from backend.poker import PokerEngine
from backend.chemistry import ChemistryEngine
from backend.biology import BiologyEngine
from backend.statistics import StatisticsEngine

# Import quant engines
from backend.quant import (
    MentalMathEngine,
    ProbabilityEngine,
    OptionsEngine,
    MarketMakingEngine,
    FermiEngine,
    InterviewModeEngine
)

# Import LeetCode and Flashcard engines
from backend.leetcode import LeetCodeEngine
from backend.flashcards import FlashcardEngine

# Import cache and history
from backend.cache.redis_cache import get_cache
from backend.history.tracker import get_tracker

# Import intelligent router
from backend.router import (
    IntelligentRouter,
    get_router,
    init_router,
    ProblemClassifier,
    BudgetZone,
    get_cost_tracker,
)

# Import WebSocket manager and handlers
from backend.websocket import ws_manager, MentalMathHandler
from backend.websocket.hud_handler import hud_handler

# Import meeting mode handler
from backend.meeting import meeting_handler

# Import voice status handler
from backend.voice.status import voice_status, voice_status_handler

# Import rate limiter middleware
from backend.middleware import RateLimitMiddleware, rate_limiter

# Import AR response models and formatters
from backend.models import PaginatedResponse, ColorScheme, Slide
from backend.formatters import MathSlideBuilder, QuantSlideBuilder

# Import dashboard API
from backend.dashboard import dashboard_router

# LiveKit token generation
import os
try:
    from livekit import api as livekit_api
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="AI Glasses Coach API",
    description="Backend API for Brilliant Labs Halo AR glasses - Math, Vision, CS, Poker, Chemistry, Biology, Statistics, and Quant Finance",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Include dashboard router
app.include_router(dashboard_router)

# Mount static files for dashboard
FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "static"
if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")


# Serve dashboard at multiple URL patterns
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve dashboard at root URL."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "dashboard.html"
    if frontend_path.exists():
        with open(frontend_path, "r", encoding="utf-8") as f:
            return f.read()
    return RedirectResponse(url="/docs")

@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard_html():
    """Serve dashboard as direct HTML file."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "dashboard.html"
    if frontend_path.exists():
        with open(frontend_path, "r", encoding="utf-8") as f:
            return f.read()
    return RedirectResponse(url="/dashboard/")

@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests to prevent 404 errors."""
    return Response(status_code=204)


# ==================== Timing Middleware ====================

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start_time) * 1000)
    response.headers["X-Duration-Ms"] = str(duration_ms)
    return response


# ==================== Router Initialization ====================

@app.on_event("startup")
async def startup_event():
    """Initialize the intelligent router with adapters."""
    try:
        # Lazy import to avoid circular dependencies
        from backend.router import MathEngineAdapter

        # Get the math engine and create adapter
        math_engine = get_engine("math")
        math_adapter = MathEngineAdapter(engine=math_engine)

        # Initialize router with adapters
        init_router(adapters=[math_adapter])
        print("✅ Intelligent Router initialized with MathEngineAdapter")
    except Exception as e:
        print(f"⚠️  Router initialization failed: {e}")
        print("   Falling back to direct engine calls")


# ==================== Engine Initialization ====================

_engines: Dict[str, Any] = {}


def get_engine(name: str):
    if name not in _engines:
        classes = {
            "math": MathEngine,
            "vision": VisionEngine,
            "cs": CSEngine,
            "poker": PokerEngine,
            "chemistry": ChemistryEngine,
            "biology": BiologyEngine,
            "statistics": StatisticsEngine,
            "mental_math": MentalMathEngine,
            "probability": ProbabilityEngine,
            "options": OptionsEngine,
            "market_making": MarketMakingEngine,
            "fermi": FermiEngine,
            "interview": InterviewModeEngine,
            "leetcode": LeetCodeEngine,
            "flashcards": FlashcardEngine,
        }
        if name in classes:
            _engines[name] = classes[name]()
    return _engines.get(name)


# ==================== Request/Response Models ====================

# Math
class MathSolveRequest(BaseModel):
    problem: str = Field(..., description="Math problem to solve")

class MathSolveResponse(BaseModel):
    solution: Optional[str] = None
    method: str = ""
    steps: Optional[str] = None
    error: Optional[str] = None
    cached: bool = False

class MathLatexRequest(BaseModel):
    latex: str

# Vision
class VisionExtractRequest(BaseModel):
    image_base64: str

class VisionExtractResponse(BaseModel):
    latex: Optional[str] = None
    raw_text: Optional[str] = None
    confidence: float = 0
    error: Optional[str] = None

class VisionTextResponse(BaseModel):
    text: Optional[str] = None
    confidence: float = 0
    error: Optional[str] = None

class VisionDiagramRequest(BaseModel):
    image_base64: str
    context: str = ""

class VisionDiagramResponse(BaseModel):
    description: Optional[str] = None
    elements: List[str] = []
    relationships: List[str] = []
    error: Optional[str] = None

# CS
class CSExplainRequest(BaseModel):
    code: str
    language: str = "auto"

class CSExplainResponse(BaseModel):
    explanation: Optional[str] = None
    key_concepts: List[str] = []
    complexity: Optional[str] = None
    suggestions: List[str] = []
    error: Optional[str] = None
    cached: bool = False

class CSDebugRequest(BaseModel):
    code: str
    error: str
    language: str = "auto"

class CSDebugResponse(BaseModel):
    diagnosis: Optional[str] = None
    fix: Optional[str] = None
    fixed_code: Optional[str] = None
    explanation: Optional[str] = None
    error: Optional[str] = None

class CSAlgorithmRequest(BaseModel):
    algorithm: str

class CSAlgorithmResponse(BaseModel):
    explanation: Optional[str] = None
    how_it_works: List[str] = []
    complexity: Optional[str] = None
    use_cases: List[str] = []
    pseudocode: Optional[str] = None
    error: Optional[str] = None

class CSCompareRequest(BaseModel):
    problem: str
    approaches: List[str]

class CSCompareResponse(BaseModel):
    comparison: Optional[str] = None
    tradeoffs: List[str] = []
    recommendation: Optional[str] = None
    error: Optional[str] = None

# Poker
class PokerHandRequest(BaseModel):
    hero_cards: str
    board: str = ""
    position: str = ""
    villain_position: str = ""
    pot_size: float = 0
    stack_size: float = 100
    villain_stack: float = 0
    action: str = ""
    game_type: str = "cash 6max"
    villain_tendencies: str = ""

class PokerHandResponse(BaseModel):
    recommendation: Optional[str] = None
    reasoning: Optional[str] = None
    ev_estimate: Optional[str] = None
    range_analysis: Optional[str] = None
    error: Optional[str] = None
    cached: bool = False

class PokerRangeRequest(BaseModel):
    position: str
    action: str
    game_type: str = "cash 6max"

class PokerOddsRequest(BaseModel):
    pot_size: float
    bet_size: float
    stack_size: Optional[float] = None

class PokerSizingRequest(BaseModel):
    hero_cards: str
    board: str = ""
    pot_size: float
    position: str = ""
    action_type: str = "bet"

# Chemistry
class ChemistryBalanceRequest(BaseModel):
    equation: str

class ChemistryBalanceResponse(BaseModel):
    balanced: Optional[str] = None
    coefficients: Optional[Dict] = None
    method: Optional[str] = None
    error: Optional[str] = None
    cached: bool = False

class ChemistryMolecularWeightRequest(BaseModel):
    formula: str

class ChemistryMolarityRequest(BaseModel):
    solute_moles: float
    volume_liters: float

class ChemistrySolveRequest(BaseModel):
    problem: str

# Biology
class BiologyPunnettRequest(BaseModel):
    parent1: str
    parent2: str

class BiologyPunnettResponse(BaseModel):
    grid: Optional[List[List[str]]] = None
    genotype_ratios: Optional[Dict] = None
    phenotype_ratios: Optional[Dict] = None
    genotype_probabilities: Optional[Dict] = None
    phenotype_probabilities: Optional[Dict] = None
    error: Optional[str] = None

class BiologyExplainRequest(BaseModel):
    concept: str

class BiologyGeneticsRequest(BaseModel):
    problem: str

# Statistics
class StatsDescriptiveRequest(BaseModel):
    data: List[float]

class StatsHypothesisRequest(BaseModel):
    test_type: str
    data: Dict[str, Any]

class StatsCorrelationRequest(BaseModel):
    x: List[float]
    y: List[float]

class StatsRegressionRequest(BaseModel):
    x: List[float]
    y: List[float]

class StatsProbabilityRequest(BaseModel):
    problem: str

class StatsExplainRequest(BaseModel):
    concept: str


# Quant - Mental Math
class MentalMathGenerateRequest(BaseModel):
    problem_type: Optional[str] = None
    difficulty: int = 2

class MentalMathCheckRequest(BaseModel):
    problem_id: str
    answer: str
    time_ms: int

# Quant - Probability
class ProbabilityGenerateRequest(BaseModel):
    problem_type: Optional[str] = None
    difficulty: int = 2

class ProbabilityCheckRequest(BaseModel):
    problem_id: str
    answer: str

# Quant - Options
class OptionsBlackScholesRequest(BaseModel):
    S: float = Field(..., description="Current stock price")
    K: float = Field(..., description="Strike price")
    r: float = Field(..., description="Risk-free rate (decimal)")
    sigma: float = Field(..., description="Volatility (decimal)")
    T: float = Field(..., description="Time to expiry (years)")
    option_type: str = "call"

class OptionsGreeksRequest(BaseModel):
    S: float
    K: float
    r: float
    sigma: float
    T: float
    option_type: str = "call"

class OptionsImpliedVolRequest(BaseModel):
    market_price: float
    S: float
    K: float
    r: float
    T: float
    option_type: str = "call"

class OptionsParityRequest(BaseModel):
    call_price: float
    put_price: float
    S: float
    K: float
    r: float
    T: float

# Quant - Market Making
class MarketScenarioRequest(BaseModel):
    scenario_type: Optional[str] = None
    difficulty: int = 2

class MarketEdgeRequest(BaseModel):
    prob_win: float
    payout_win: float
    payout_lose: float

class MarketKellyRequest(BaseModel):
    prob_win: float
    odds: float
    bankroll: float = 10000

class MarketSharpeRequest(BaseModel):
    returns: List[float]
    risk_free_rate: float = 0.02
    periods_per_year: int = 252

# Quant - Fermi
class FermiGenerateRequest(BaseModel):
    category: Optional[str] = None

class FermiHintRequest(BaseModel):
    problem_id: str
    hint_level: int = 1

class FermiEvaluateRequest(BaseModel):
    problem_id: str
    estimate: float

# Quant - Interview
class InterviewStartRequest(BaseModel):
    duration_min: int = 30
    firm_style: str = "general"
    difficulty: int = 2

class InterviewNextRequest(BaseModel):
    session_id: str
    prev_answer: Optional[str] = None
    prev_time_ms: Optional[int] = None


# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    cache = get_cache()
    return {
        "status": "healthy",
        "version": "3.0.0",
        "engines": {
            "math": "available",
            "vision": "available",
            "cs": "available",
            "poker": "available",
            "chemistry": "available",
            "biology": "available",
            "statistics": "available",
            "meeting": "available",
            "quant": {
                "mental_math": "available",
                "probability": "available",
                "options": "available",
                "market_making": "available",
                "fermi": "available",
                "interview": "available"
            }
        },
        "cache": cache.stats(),
        "voice": "available" if LIVEKIT_AVAILABLE else "not_installed",
        "rate_limiting": rate_limiter.enabled
    }


@app.get("/rate-limit/stats")
async def rate_limit_stats():
    """Get rate limiter statistics and configuration."""
    return rate_limiter.get_stats()


# ==================== LiveKit Voice Endpoints ====================

class VoiceTokenRequest(BaseModel):
    room_name: str = "wham-voice"
    participant_name: str = "user"
    voice: str = "Puck"  # Gemini voices: Puck, Charon, Kore, Fenrir, Aoede


@app.post("/voice/token")
async def get_voice_token(request: VoiceTokenRequest):
    """
    Generate a LiveKit room token for voice agent connection.

    Requires LIVEKIT_API_KEY and LIVEKIT_API_SECRET environment variables.
    """
    if not LIVEKIT_AVAILABLE:
        return {"error": "LiveKit SDK not installed. Run: pip install livekit-api"}

    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        return {
            "error": "LiveKit credentials not configured",
            "help": "Set LIVEKIT_API_KEY and LIVEKIT_API_SECRET environment variables"
        }

    # Create access token
    token = livekit_api.AccessToken(api_key, api_secret)
    token.with_identity(request.participant_name)
    token.with_name(request.participant_name)
    token.with_grants(livekit_api.VideoGrants(
        room_join=True,
        room=request.room_name,
        can_publish=True,
        can_subscribe=True,
    ))

    jwt_token = token.to_jwt()
    livekit_url = os.getenv("LIVEKIT_URL", "wss://your-project.livekit.cloud")

    # Dispatch agent to the room so it can handle voice interactions
    try:
        from livekit.api import CreateAgentDispatchRequest
        lk_api = livekit_api.LiveKitAPI(
            url=livekit_url.replace("wss://", "https://"),
            api_key=api_key,
            api_secret=api_secret,
        )
        # Request agent dispatch with voice metadata
        dispatch_request = CreateAgentDispatchRequest(
            room=request.room_name,
            agent_name="",
            metadata=f'{{"voice": "{request.voice}"}}',  # Pass voice to agent
        )
        await lk_api.agent_dispatch.create_dispatch(dispatch_request)
        await lk_api.aclose()
    except Exception as e:
        # Log but don't fail - user can still connect, agent may already be dispatched
        import logging
        logging.warning(f"Agent dispatch failed (may already be dispatched): {e}")

    return {
        "token": jwt_token,
        "url": livekit_url,
        "room": request.room_name,
        "participant": request.participant_name,
        "voice": request.voice
    }


@app.get("/voice/status")
async def voice_status_endpoint():
    """Check voice agent availability and real-time status."""
    return {
        "livekit_sdk": "installed" if LIVEKIT_AVAILABLE else "not_installed",
        "api_key_set": bool(os.getenv("LIVEKIT_API_KEY")),
        "api_secret_set": bool(os.getenv("LIVEKIT_API_SECRET")),
        "url": os.getenv("LIVEKIT_URL", "not_configured"),
        "help": "Visit https://livekit.io/cloud to get credentials",
        "agent_status": voice_status.get_status()
    }


@app.get("/voice/agent-status")
async def voice_agent_status():
    """Get detailed voice agent real-time status."""
    return voice_status.get_status()


@app.get("/voice/agent-history")
async def voice_agent_history(limit: int = 50):
    """Get voice agent state change history."""
    return {"history": voice_status.get_history(limit)}


@app.websocket("/ws/voice-status")
async def voice_status_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice agent status updates.

    Clients receive automatic broadcasts when voice agent state changes.

    Protocol:
    Client sends:
    - {"type": "get_status"}   - Get current status
    - {"type": "get_history", "limit": N}  - Get state history
    - {"type": "ping"}         - Keep-alive

    Server sends:
    - {"type": "voice_status", "state": "listening|thinking|speaking|...", ...}
    - {"type": "status", ...}  - Response to get_status
    - {"type": "history", ...} - Response to get_history
    - {"type": "pong"}         - Response to ping
    """
    await voice_status_handler.handle_connection(
        websocket=websocket,
        topic="voice:status",
        metadata={"subscriber": True}
    )


# ==================== Voice Vision Endpoints ====================

class VoiceVisionRequest(BaseModel):
    image: str  # Base64-encoded image
    context: Optional[str] = None  # Optional voice context
    analyze_type: str = "general"  # general, text, objects, scene


@app.post("/voice/vision")
async def voice_vision_analyze(request: VoiceVisionRequest):
    """
    Analyze an image and return a voice-friendly description.

    This endpoint is designed for injecting visual context into voice conversations.
    The response is optimized for text-to-speech output.
    """
    from backend.voice.vision import analyze_for_voice

    result = await analyze_for_voice(
        image_base64=request.image,
        context=request.context or "",
        analyze_type=request.analyze_type
    )

    return {
        "success": result.success,
        "description": result.description,
        "details": result.details
    }


@app.post("/voice/vision/auto")
async def voice_vision_auto(request: VoiceVisionRequest):
    """
    Analyze image for auto-detection mode (glasses always-on recording).

    Only returns a result if something noteworthy is detected.
    Returns {"detected": false} for unremarkable scenes.
    """
    from backend.voice.vision import analyze_for_auto_detection

    result = await analyze_for_auto_detection(request.image)

    if result is None:
        return {"detected": False, "description": None}

    return {
        "detected": True,
        "description": result.description,
        "details": result.details
    }


@app.get("/voice/tools")
async def list_voice_tools():
    """List available voice tools."""
    from backend.voice.tools.router import get_router, register_default_tools

    router = get_router()

    # Register tools if not already done
    if not router.tools:
        try:
            register_default_tools(router)
        except Exception as e:
            return {"tools": [], "error": str(e)}

    return {"tools": router.list_tools()}


# ==================== Math Endpoints ====================

@app.post("/math/solve", response_model=MathSolveResponse)
async def math_solve(request: MathSolveRequest):
    cache = get_cache()
    tracker = get_tracker()
    start = time.time()

    cache_key = cache.generate_key("math", request.problem)
    cached = cache.get(cache_key)
    if cached:
        tracker.log_query("math", request.problem, cached, 0, 0, True)
        return MathSolveResponse(**cached, cached=True)

    engine = get_engine("math")
    result = engine.solve(request.problem)

    duration = int((time.time() - start) * 1000)
    cost = 0.001 if result.get("method") == "claude" else 0
    cache.set(cache_key, result)
    tracker.log_query("math", request.problem, result, cost, duration, False)

    return MathSolveResponse(**result, cached=False)


@app.post("/math/solve-latex", response_model=MathSolveResponse)
async def math_solve_latex(request: MathLatexRequest):
    engine = get_engine("math")
    result = engine.solve_latex(request.latex)
    return MathSolveResponse(**result)


# ==================== Vision Endpoints ====================

@app.post("/vision/extract", response_model=VisionExtractResponse)
async def vision_extract(request: VisionExtractRequest):
    tracker = get_tracker()
    start = time.time()
    engine = get_engine("vision")
    result = engine.extract_equation(request.image_base64)
    duration = int((time.time() - start) * 1000)
    tracker.log_query("vision", "extract", result, 0.005, duration, False)
    return VisionExtractResponse(**result)


@app.post("/vision/ocr", response_model=VisionTextResponse)
async def vision_ocr(request: VisionExtractRequest):
    engine = get_engine("vision")
    result = engine.extract_text(request.image_base64)
    return VisionTextResponse(**result)


@app.post("/vision/diagram", response_model=VisionDiagramResponse)
async def vision_diagram(request: VisionDiagramRequest):
    engine = get_engine("vision")
    result = engine.analyze_diagram(request.image_base64, request.context)
    return VisionDiagramResponse(**result)


# ==================== CS Endpoints ====================

@app.post("/cs/explain", response_model=CSExplainResponse)
async def cs_explain(request: CSExplainRequest):
    cache = get_cache()
    tracker = get_tracker()
    start = time.time()

    cache_key = cache.generate_key("cs", {"code": request.code[:500], "lang": request.language})
    cached = cache.get(cache_key)
    if cached:
        return CSExplainResponse(**cached, cached=True)

    engine = get_engine("cs")
    result = engine.explain_code(request.code, request.language)

    duration = int((time.time() - start) * 1000)
    cache.set(cache_key, result)
    tracker.log_query("cs", request.code[:100], result, 0.002, duration, False)

    return CSExplainResponse(**result, cached=False)


@app.post("/cs/debug", response_model=CSDebugResponse)
async def cs_debug(request: CSDebugRequest):
    engine = get_engine("cs")
    result = engine.debug_code(request.code, request.error, request.language)
    return CSDebugResponse(**result)


@app.post("/cs/algorithm", response_model=CSAlgorithmResponse)
async def cs_algorithm(request: CSAlgorithmRequest):
    cache = get_cache()
    cache_key = cache.generate_key("cs_algo", request.algorithm)
    cached = cache.get(cache_key)
    if cached:
        return CSAlgorithmResponse(**cached)

    engine = get_engine("cs")
    result = engine.explain_algorithm(request.algorithm)
    cache.set(cache_key, result)
    return CSAlgorithmResponse(**result)


@app.post("/cs/compare", response_model=CSCompareResponse)
async def cs_compare(request: CSCompareRequest):
    engine = get_engine("cs")
    result = engine.compare_approaches(request.problem, request.approaches)
    return CSCompareResponse(**result)


# ==================== Poker Endpoints ====================

@app.post("/poker/analyze", response_model=PokerHandResponse)
async def poker_analyze(request: PokerHandRequest):
    cache = get_cache()
    tracker = get_tracker()
    start = time.time()

    hand_info = request.model_dump()
    cache_key = cache.generate_key("poker", hand_info)
    cached = cache.get(cache_key)
    if cached:
        return PokerHandResponse(**cached, cached=True)

    engine = get_engine("poker")
    result = engine.analyze_hand(hand_info)

    duration = int((time.time() - start) * 1000)
    cache.set(cache_key, result)
    tracker.log_query("poker", str(hand_info)[:100], result, 0.002, duration, False)

    return PokerHandResponse(**result, cached=False)


@app.post("/poker/range")
async def poker_range(request: PokerRangeRequest):
    engine = get_engine("poker")
    return engine.get_preflop_range(request.position, request.action, request.game_type)


@app.post("/poker/odds")
async def poker_odds(request: PokerOddsRequest):
    engine = get_engine("poker")
    return engine.calculate_pot_odds(request.pot_size, request.bet_size, request.stack_size)


@app.post("/poker/sizing")
async def poker_sizing(request: PokerSizingRequest):
    engine = get_engine("poker")
    situation = request.model_dump()
    return engine.analyze_bet_sizing(situation)


# ==================== Chemistry Endpoints ====================

@app.post("/chemistry/balance", response_model=ChemistryBalanceResponse)
async def chemistry_balance(request: ChemistryBalanceRequest):
    cache = get_cache()
    tracker = get_tracker()
    start = time.time()

    cache_key = cache.generate_key("chemistry", request.equation)
    cached = cache.get(cache_key)
    if cached:
        return ChemistryBalanceResponse(**cached, cached=True)

    engine = get_engine("chemistry")
    result = engine.balance_equation(request.equation)

    duration = int((time.time() - start) * 1000)
    cost = 0.001 if result.get("method") == "claude" else 0
    cache.set(cache_key, result)
    tracker.log_query("chemistry", request.equation, result, cost, duration, False)

    return ChemistryBalanceResponse(**result, cached=False)


@app.post("/chemistry/molecular-weight")
async def chemistry_molecular_weight(request: ChemistryMolecularWeightRequest):
    engine = get_engine("chemistry")
    return engine.calculate_molecular_weight(request.formula)


@app.post("/chemistry/molarity")
async def chemistry_molarity(request: ChemistryMolarityRequest):
    engine = get_engine("chemistry")
    return engine.calculate_molarity(request.solute_moles, request.volume_liters)


@app.post("/chemistry/solve")
async def chemistry_solve(request: ChemistrySolveRequest):
    tracker = get_tracker()
    start = time.time()
    engine = get_engine("chemistry")
    result = engine.solve_chemistry_problem(request.problem)
    duration = int((time.time() - start) * 1000)
    tracker.log_query("chemistry", request.problem[:100], result, 0.002, duration, False)
    return result


# ==================== Biology Endpoints ====================

@app.post("/biology/punnett", response_model=BiologyPunnettResponse)
async def biology_punnett(request: BiologyPunnettRequest):
    cache = get_cache()
    cache_key = cache.generate_key("biology", f"{request.parent1}x{request.parent2}")
    cached = cache.get(cache_key)
    if cached:
        return BiologyPunnettResponse(**cached)

    engine = get_engine("biology")
    result = engine.punnett_square(request.parent1, request.parent2)
    cache.set(cache_key, result)
    return BiologyPunnettResponse(**result)


@app.post("/biology/explain")
async def biology_explain(request: BiologyExplainRequest):
    cache = get_cache()
    tracker = get_tracker()
    start = time.time()

    cache_key = cache.generate_key("biology", request.concept)
    cached = cache.get(cache_key)
    if cached:
        return cached

    engine = get_engine("biology")
    result = engine.explain_concept(request.concept)

    duration = int((time.time() - start) * 1000)
    cache.set(cache_key, result)
    tracker.log_query("biology", request.concept, result, 0.002, duration, False)
    return result


@app.post("/biology/genetics")
async def biology_genetics(request: BiologyGeneticsRequest):
    tracker = get_tracker()
    start = time.time()
    engine = get_engine("biology")
    result = engine.solve_genetics_problem(request.problem)
    duration = int((time.time() - start) * 1000)
    tracker.log_query("biology", request.problem[:100], result, 0.002, duration, False)
    return result


# ==================== Statistics Endpoints ====================

@app.post("/statistics/descriptive")
async def stats_descriptive(request: StatsDescriptiveRequest):
    engine = get_engine("statistics")
    return engine.descriptive_stats(request.data)


@app.post("/statistics/hypothesis-test")
async def stats_hypothesis(request: StatsHypothesisRequest):
    engine = get_engine("statistics")
    return engine.hypothesis_test(request.test_type, request.data)


@app.post("/statistics/correlation")
async def stats_correlation(request: StatsCorrelationRequest):
    engine = get_engine("statistics")
    return engine.correlation(request.x, request.y)


@app.post("/statistics/regression")
async def stats_regression(request: StatsRegressionRequest):
    engine = get_engine("statistics")
    return engine.regression(request.x, request.y)


@app.post("/statistics/probability")
async def stats_probability(request: StatsProbabilityRequest):
    tracker = get_tracker()
    start = time.time()
    engine = get_engine("statistics")
    result = engine.probability_calculation(request.problem)
    duration = int((time.time() - start) * 1000)
    tracker.log_query("statistics", request.problem[:100], result, 0.002, duration, False)
    return result


@app.post("/statistics/explain")
async def stats_explain(request: StatsExplainRequest):
    cache = get_cache()
    tracker = get_tracker()
    start = time.time()

    cache_key = cache.generate_key("statistics", request.concept)
    cached = cache.get(cache_key)
    if cached:
        return cached

    engine = get_engine("statistics")
    result = engine.explain_stats_concept(request.concept)

    duration = int((time.time() - start) * 1000)
    cache.set(cache_key, result)
    tracker.log_query("statistics", request.concept, result, 0.002, duration, False)
    return result


# ==================== Quant Finance Endpoints ====================

# ----- Mental Math -----

@app.post("/quant/mental-math/generate")
async def quant_mental_math_generate(request: MentalMathGenerateRequest):
    engine = get_engine("mental_math")
    return engine.generate_problem(request.problem_type, request.difficulty)


@app.post("/quant/mental-math/check")
async def quant_mental_math_check(request: MentalMathCheckRequest):
    tracker = get_tracker()
    engine = get_engine("mental_math")
    result = engine.check_answer(request.problem_id, request.answer, request.time_ms)
    tracker.log_query("mental_math", request.answer, result, 0, request.time_ms, False)
    return result


@app.get("/quant/mental-math/types")
async def quant_mental_math_types():
    engine = get_engine("mental_math")
    return {
        "problem_types": engine.get_problem_types(),
        "difficulty_levels": engine.get_difficulty_info()
    }


# ----- Probability -----

@app.post("/quant/probability/generate")
async def quant_probability_generate(request: ProbabilityGenerateRequest):
    engine = get_engine("probability")
    return engine.generate_problem(request.problem_type, request.difficulty)


@app.post("/quant/probability/card")
async def quant_probability_card(request: ProbabilityGenerateRequest):
    engine = get_engine("probability")
    return engine.generate_card_problem(request.difficulty)


@app.post("/quant/probability/dice")
async def quant_probability_dice(request: ProbabilityGenerateRequest):
    engine = get_engine("probability")
    return engine.generate_dice_problem(request.difficulty)


@app.post("/quant/probability/ev")
async def quant_probability_ev(request: ProbabilityGenerateRequest):
    engine = get_engine("probability")
    return engine.generate_ev_problem(request.difficulty)


@app.post("/quant/probability/check")
async def quant_probability_check(request: ProbabilityCheckRequest):
    tracker = get_tracker()
    engine = get_engine("probability")
    result = engine.check_answer(request.problem_id, request.answer)
    tracker.log_query("probability", request.answer, result, 0, 0, False)
    return result


@app.get("/quant/probability/monty-hall")
async def quant_monty_hall(iterations: int = 10000):
    engine = get_engine("probability")
    return engine.monty_hall_simulation(iterations)


@app.get("/quant/probability/birthday")
async def quant_birthday(n_people: int = 23):
    engine = get_engine("probability")
    return engine.birthday_paradox(n_people)


# ----- Options -----

@app.post("/quant/options/black-scholes")
async def quant_options_black_scholes(request: OptionsBlackScholesRequest):
    engine = get_engine("options")
    return engine.black_scholes(
        request.S, request.K, request.r, request.sigma, request.T, request.option_type
    )


@app.post("/quant/options/greeks")
async def quant_options_greeks(request: OptionsGreeksRequest):
    engine = get_engine("options")
    return engine.greeks(
        request.S, request.K, request.r, request.sigma, request.T, request.option_type
    )


@app.post("/quant/options/implied-vol")
async def quant_options_implied_vol(request: OptionsImpliedVolRequest):
    engine = get_engine("options")
    return engine.implied_volatility(
        request.market_price, request.S, request.K, request.r, request.T, request.option_type
    )


@app.post("/quant/options/parity")
async def quant_options_parity(request: OptionsParityRequest):
    engine = get_engine("options")
    return engine.parity_check(
        request.call_price, request.put_price, request.S, request.K, request.r, request.T
    )


@app.get("/quant/options/formulas")
async def quant_options_formulas():
    engine = get_engine("options")
    return engine.get_formulas()


# ----- Market Making -----

@app.post("/quant/market/scenario")
async def quant_market_scenario(request: MarketScenarioRequest):
    engine = get_engine("market_making")
    return engine.generate_scenario(request.scenario_type, request.difficulty)


@app.post("/quant/market/edge")
async def quant_market_edge(request: MarketEdgeRequest):
    engine = get_engine("market_making")
    return engine.calculate_edge(request.prob_win, request.payout_win, request.payout_lose)


@app.post("/quant/market/kelly")
async def quant_market_kelly(request: MarketKellyRequest):
    engine = get_engine("market_making")
    return engine.kelly_criterion(request.prob_win, request.odds, request.bankroll)


@app.post("/quant/market/sharpe")
async def quant_market_sharpe(request: MarketSharpeRequest):
    engine = get_engine("market_making")
    return engine.sharpe_ratio(request.returns, request.risk_free_rate, request.periods_per_year)


@app.get("/quant/market/formulas")
async def quant_market_formulas():
    engine = get_engine("market_making")
    return engine.get_formulas()


# ----- Fermi -----

@app.post("/quant/fermi/generate")
async def quant_fermi_generate(request: FermiGenerateRequest):
    engine = get_engine("fermi")
    return engine.generate_problem(request.category)


@app.post("/quant/fermi/hint")
async def quant_fermi_hint(request: FermiHintRequest):
    engine = get_engine("fermi")
    return engine.get_hints(request.problem_id, request.hint_level)


@app.post("/quant/fermi/evaluate")
async def quant_fermi_evaluate(request: FermiEvaluateRequest):
    tracker = get_tracker()
    engine = get_engine("fermi")
    result = engine.evaluate_estimate(request.problem_id, request.estimate)
    tracker.log_query("fermi", str(request.estimate), result, 0, 0, False)
    return result


@app.get("/quant/fermi/categories")
async def quant_fermi_categories():
    engine = get_engine("fermi")
    return {"categories": engine.get_categories()}


@app.get("/quant/fermi/approach")
async def quant_fermi_approach():
    engine = get_engine("fermi")
    return engine.get_approach_template()


# ----- Interview Mode -----

@app.post("/quant/interview/start")
async def quant_interview_start(request: InterviewStartRequest):
    engine = get_engine("interview")
    return engine.start_session(request.duration_min, request.firm_style, request.difficulty)


@app.post("/quant/interview/next")
async def quant_interview_next(request: InterviewNextRequest):
    engine = get_engine("interview")
    return engine.get_next_problem(request.session_id, request.prev_answer, request.prev_time_ms)


@app.post("/quant/interview/end")
async def quant_interview_end(session_id: str):
    engine = get_engine("interview")
    return engine.end_session(session_id)


@app.get("/quant/interview/firms")
async def quant_interview_firms():
    engine = get_engine("interview")
    return engine.get_firm_styles()


@app.get("/quant/progress")
async def quant_progress(include_history: bool = False):
    engine = get_engine("interview")
    return engine.get_progress(include_history)


# ==================== History & Stats Endpoints ====================

@app.get("/history")
async def get_history(limit: int = 50, engine: Optional[str] = None):
    tracker = get_tracker()
    queries = tracker.get_history(limit=limit, engine=engine)
    return {"queries": queries, "total": len(queries)}


@app.get("/stats")
async def get_usage_stats():
    tracker = get_tracker()
    return tracker.get_stats()


@app.delete("/cache")
async def clear_cache():
    cache = get_cache()
    count = cache.clear()
    return {"cleared": count}


@app.get("/cache/stats")
async def cache_stats():
    cache = get_cache()
    return cache.stats()


# ==================== LeetCode Endpoints ====================

class LeetCodeSolutionRequest(BaseModel):
    problem_id: int
    code: str
    language: str
    passed: bool = False
    time_taken_ms: Optional[int] = None
    hints_used: int = 0


@app.get("/leetcode/problems")
async def list_leetcode_problems(
    difficulty: Optional[str] = None,
    topic: Optional[str] = None,
    jane_street_mode: bool = False,
    limit: int = 50,
    offset: int = 0
):
    """Search cached LeetCode problems with filters."""
    engine = get_engine("leetcode")
    topics = [topic] if topic else None
    return engine.search_problems(difficulty, topics, jane_street_mode, limit, offset)


@app.get("/leetcode/problem/{slug}")
async def get_leetcode_problem(slug: str):
    """Fetch a LeetCode problem by slug (caches result)."""
    engine = get_engine("leetcode")
    return await engine.fetch_problem(slug)


@app.get("/leetcode/hints/{problem_id}/{level}")
async def get_leetcode_hint(problem_id: int, level: int):
    """Get hint at specified level (0-2)."""
    engine = get_engine("leetcode")
    return engine.get_hints(problem_id, level)


@app.post("/leetcode/solution")
async def save_leetcode_solution(request: LeetCodeSolutionRequest):
    """Save user's solution for a problem."""
    engine = get_engine("leetcode")
    return engine.save_solution(
        request.problem_id,
        request.code,
        request.language,
        request.passed,
        request.time_taken_ms,
        request.hints_used
    )


@app.get("/leetcode/solutions/{problem_id}")
async def get_leetcode_solutions(problem_id: int):
    """Get all user solutions for a problem."""
    engine = get_engine("leetcode")
    return {"solutions": engine.get_solutions(problem_id)}


@app.get("/leetcode/progress")
async def get_leetcode_progress():
    """Get user's LeetCode progress stats."""
    engine = get_engine("leetcode")
    return engine.get_progress()


# ==================== Flashcard Endpoints ====================

class FlashcardCreateRequest(BaseModel):
    front: str
    back: str
    source_type: str = "manual"
    source_id: Optional[str] = None
    tags: Optional[List[str]] = None


class FlashcardReviewRequest(BaseModel):
    card_id: int
    quality: int = Field(..., ge=0, le=5)
    time_ms: Optional[int] = None


class FlashcardGenerateRequest(BaseModel):
    source_type: str
    problem_data: Dict[str, Any]


class FlashcardSyncRequest(BaseModel):
    reviews: List[Dict[str, Any]]


@app.get("/flashcards/due")
async def get_due_flashcards(
    limit: int = 20,
    source_type: Optional[str] = None
):
    """Get cards due for review."""
    engine = get_engine("flashcards")
    return {"cards": engine.get_due_cards(limit, source_type)}


@app.get("/flashcards/all")
async def get_all_flashcards(
    limit: int = 100,
    offset: int = 0,
    source_type: Optional[str] = None
):
    """Get all cards with pagination."""
    engine = get_engine("flashcards")
    return {"cards": engine.get_all_cards(limit, offset, source_type)}


@app.post("/flashcards")
async def create_flashcard(request: FlashcardCreateRequest):
    """Create a new flashcard."""
    engine = get_engine("flashcards")
    return engine.create_card(
        request.front,
        request.back,
        request.source_type,
        request.source_id,
        request.tags
    )


@app.post("/flashcards/review")
async def review_flashcard(request: FlashcardReviewRequest):
    """Record a flashcard review and update SM-2 parameters."""
    engine = get_engine("flashcards")
    return engine.review_card(request.card_id, request.quality, request.time_ms)


@app.post("/flashcards/generate")
async def generate_flashcard(request: FlashcardGenerateRequest):
    """Auto-generate a flashcard from a solved problem."""
    engine = get_engine("flashcards")
    return engine.generate_from_problem(request.source_type, request.problem_data)


@app.post("/flashcards/sync")
async def sync_flashcard_reviews(request: FlashcardSyncRequest):
    """Sync multiple reviews (for offline mode)."""
    engine = get_engine("flashcards")
    return engine.sync_reviews(request.reviews)


@app.delete("/flashcards/{card_id}")
async def delete_flashcard(card_id: int):
    """Delete a flashcard."""
    engine = get_engine("flashcards")
    return engine.delete_card(card_id)


@app.get("/flashcards/stats")
async def get_flashcard_stats():
    """Get flashcard statistics."""
    engine = get_engine("flashcards")
    return engine.get_stats()


@app.get("/flashcards/heatmap")
async def get_flashcard_heatmap(year: Optional[int] = None):
    """Get daily activity data for heatmap visualization."""
    engine = get_engine("flashcards")
    return {"data": engine.get_heatmap_data(year)}


@app.get("/flashcards/forecast")
async def get_flashcard_forecast(days: int = 7):
    """Get forecast of cards due in the next N days."""
    engine = get_engine("flashcards")
    return {"forecast": engine.get_review_forecast(days)}


# ==================== Router Endpoints ====================

class RouterSolveResponse(BaseModel):
    """Response from the intelligent router."""
    solution: Optional[str] = None
    method: str = ""
    steps: Optional[str] = None
    error: Optional[str] = None
    cached: bool = False
    cache_type: Optional[str] = None
    engine_used: str = ""
    latency_ms: float = 0
    cost: float = 0
    budget_zone: str = ""


@app.post("/v2/math/solve", response_model=RouterSolveResponse)
async def math_solve_routed(request: MathSolveRequest):
    """
    Solve math problem using intelligent router.

    The router automatically:
    - Checks semantic cache first
    - Routes to cheapest engine (SymPy before Claude)
    - Tracks costs against monthly budget
    - Falls back gracefully on errors
    """
    router = get_router()

    if router is None:
        # Fall back to direct engine call
        engine = get_engine("math")
        result = engine.solve(request.problem)
        return RouterSolveResponse(
            solution=result.get("solution"),
            method=result.get("method", ""),
            steps=result.get("steps"),
            error=result.get("error"),
            cached=False,
            engine_used="math_direct",
            latency_ms=0,
            cost=0.001 if result.get("method") == "claude" else 0,
            budget_zone="unknown"
        )

    # Use the intelligent router
    routing_result = await router.route(request.problem)

    return RouterSolveResponse(
        solution=routing_result.data.get("solution") if routing_result.data else None,
        method=routing_result.method,
        steps=routing_result.data.get("steps") if routing_result.data else None,
        error=routing_result.data.get("error") if routing_result.data else None,
        cached=routing_result.method == "cached",
        cache_type=routing_result.cache_type,
        engine_used=routing_result.metadata.get("engine", "unknown"),
        latency_ms=routing_result.latency_ms,
        cost=routing_result.cost,
        budget_zone=routing_result.metadata.get("budget_zone", "unknown")
    )


@app.get("/router/status")
async def router_status():
    """Get intelligent router status and statistics."""
    router = get_router()

    if router is None:
        return {
            "status": "not_initialized",
            "message": "Router not initialized. Using direct engine calls."
        }

    cost_tracker = get_cost_tracker()
    metrics = cost_tracker.get_current_metrics()

    return {
        "status": "active",
        "adapters": [adapter.name for adapter in router._adapters],
        "budget": {
            "monthly_limit": cost_tracker.monthly_budget,
            "current_spend": round(metrics.total_cost, 4),
            "remaining": round(metrics.budget_remaining, 4),
            "zone": metrics.budget_zone.value,
            "usage_percent": round(metrics.budget_usage_percent, 1)
        },
        "stats": {
            "total_requests": metrics.total_requests,
            "local_solves": metrics.local_solves,
            "claude_calls": metrics.claude_calls,
            "cache_hits": metrics.cache_hits,
            "avg_latency_ms": round(metrics.avg_latency_ms, 1)
        }
    }


@app.post("/router/reset-stats")
async def router_reset_stats():
    """Reset router statistics (for testing)."""
    cost_tracker = get_cost_tracker()
    cost_tracker.reset_period_stats()
    return {"status": "ok", "message": "Statistics reset"}


# ==================== AR Glasses Endpoints ====================

# Initialize handlers
_mental_math_handler = MentalMathHandler()
_math_slide_builder = MathSlideBuilder()
_quant_slide_builder = QuantSlideBuilder()


@app.post("/ar/math/solve")
async def ar_math_solve(request: MathSolveRequest):
    """
    Solve math problem and return paginated AR slides.

    Optimized for Halo Frame display (640x400, 20 deg FOV).
    Returns slides with title, content, color scheme, and voice narration.
    """
    import time
    start = time.time()

    # Use the router if available, otherwise direct engine
    router = get_router()

    if router:
        routing_result = await router.route(request.problem)
        engine_result = routing_result.data or {}
        engine_result["problem"] = request.problem
        method = routing_result.method
        cached = routing_result.method == "cached"
    else:
        engine = get_engine("math")
        engine_result = engine.solve(request.problem)
        engine_result["problem"] = request.problem
        method = engine_result.get("method", "unknown")
        cached = False

    latency_ms = (time.time() - start) * 1000

    # Build paginated response
    response = _math_slide_builder.build_response(
        engine_result=engine_result,
        method=method,
        cached=cached,
        latency_ms=latency_ms
    )

    return response.dict()


@app.websocket("/ws/mental-math/{session_id}")
async def mental_math_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for mental math speed run mode.

    Protocol:
    Client sends:
    - {"action": "start", "difficulty": 1-5}  - Start/next problem
    - {"action": "answer", "answer": <value>} - Submit answer
    - {"action": "skip"}                       - Skip current problem
    - {"action": "stats"}                      - Get session stats
    - {"action": "end"}                        - End session

    Server sends:
    - {"type": "problem", "problem": "47 × 83", ...}
    - {"type": "result", "correct": true, "time_ms": 2340, ...}
    - {"type": "timer", "remaining_ms": 5000}
    - {"type": "stats", ...}
    """
    topic = f"mental-math:{session_id}"
    await _mental_math_handler.handle_connection(
        websocket=websocket,
        topic=topic,
        metadata={"session_id": session_id}
    )


@app.websocket("/ws/meeting/{session_id}")
async def meeting_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for WHAM meeting assistant mode.

    Protocol:
    Client sends:
    - {"type": "meeting_start", "config": {...}}  - Start meeting session
    - {"type": "meeting_end"}                      - End meeting session
    - {"type": "audio_chunk", "audio": "base64", ...} - Audio for transcription
    - {"type": "double_tap"}                       - Request quick suggestion
    - {"type": "voice_command", "query": "...", "audio": "base64"} - Detailed help

    Server sends:
    - {"type": "status", "status": "meeting_started|processing|meeting_ended", ...}
    - {"type": "transcript_update", "segment": {...}}
    - {"type": "suggestion", "trigger": "double_tap|voice_command", "suggestion": {...}}
    - {"type": "error", "message": "..."}
    """
    topic = f"meeting:{session_id}"
    await meeting_handler.handle_connection(
        websocket=websocket,
        topic=topic,
        metadata={"session_id": session_id}
    )


@app.websocket("/ws/hud")
async def hud_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for HUD streaming to AR glasses.

    Streams agent outputs, poker recommendations, and code debug results.

    Protocol:
    Client sends:
    - {"type": "subscribe_mode", "mode": "poker|homework|code|all"}
    - {"type": "agent_request", "message": "..."}
    - {"type": "ping"}

    Server sends:
    - {"type": "subscribed", "mode": "...", "message": "..."}
    - {"type": "agent_response", "data": {...}}
    - {"type": "hud_update", "mode": "...", "data": {...}}
    - {"type": "error", "message": "..."}
    - {"type": "pong"}
    """
    await hud_handler.handle_connection(websocket, "hud:global")


@app.get("/ws/status")
async def websocket_status():
    """Get WebSocket connection status."""
    return {
        "total_connections": ws_manager.get_connection_count(),
        "topics": ws_manager.get_topics(),
        "topics_detail": {
            topic: ws_manager.get_connection_count(topic)
            for topic in ws_manager.get_topics()
        }
    }


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
