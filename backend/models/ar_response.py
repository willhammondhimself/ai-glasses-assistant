"""
AR Response Models for Halo Frame Glasses

Paginated, color-coded slide-based responses optimized for:
- 640x400 OLED display
- 20 degree FOV
- 45 chars/line, 4 lines max per slide
"""

from enum import Enum
from typing import Optional, List, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ColorScheme(str, Enum):
    """Color schemes for different content types on Halo display."""

    # Math & Physics
    MATH = "math"                    # Blue (#4CC9F0) - equations, steps
    RESULT_GREEN = "result_green"    # Green (#00FF88) - correct answers
    RESULT_RED = "result_red"        # Red (#FF4444) - errors, wrong answers

    # Code
    CODE = "code"                    # Light gray (#D4D4D4) - code display
    CODE_ERROR = "code_error"        # Red highlight - bug location
    CODE_FIX = "code_fix"            # Green highlight - fix suggestion

    # Quant/Mental Math
    QUANT_TIMER = "quant_timer"      # Orange (#FF8C00) - timer display
    QUANT_STREAK = "quant_streak"    # Gold (#FFD700) - streak indicator

    # General
    INFO = "info"                    # Gray (#C8C8C8) - general info
    WARNING = "warning"              # Yellow (#FFD93D) - warnings
    SUCCESS = "success"              # Green (#00FF88) - success states


class Slide(BaseModel):
    """
    Single slide for Halo Frame display.

    Constraints (640x400 @ 20 deg FOV):
    - Title: max 30 chars
    - Content: max 45 chars/line, 3 lines (title takes 1)
    """

    title: str = Field(..., max_length=30, description="Slide title, max 30 chars")
    content: str = Field(..., description="Main content, will be word-wrapped")
    color_scheme: ColorScheme = ColorScheme.INFO
    voice_narration: Optional[str] = None  # TTS text for audio output

    # Display hints
    centered: bool = False           # Center content on screen
    large_font: bool = False         # Use larger font (for single values)
    show_progress: bool = True       # Show slide progress indicator

    class Config:
        use_enum_values = True


class PaginatedResponse(BaseModel):
    """
    Paginated response for AR glasses display.

    Replaces raw text responses with structured slides
    optimized for the 640x400 display.
    """

    # Response metadata
    type: str = Field(..., description="Response type: math, cs, poker, quant")
    is_paginated: bool = True
    total_slides: int
    slides: List[Slide]

    # Quick display options
    summary: Optional[str] = None    # One-liner for quick display mode

    # Backward compatibility
    raw_solution: Optional[str] = None  # Original text solution
    method: str = ""                 # Engine method used (sympy, claude, etc)
    cached: bool = False             # Whether from cache

    # Error handling
    error: Optional[str] = None

    # Timing
    latency_ms: Optional[float] = None

    class Config:
        use_enum_values = True

    @classmethod
    def from_error(cls, error_message: str, response_type: str = "error") -> "PaginatedResponse":
        """Create an error response."""
        return cls(
            type=response_type,
            total_slides=1,
            slides=[
                Slide(
                    title="Error",
                    content=error_message[:135],  # 45 * 3 max
                    color_scheme=ColorScheme.RESULT_RED,
                    voice_narration=f"Error: {error_message}"
                )
            ],
            error=error_message
        )


# Mental Math specific models

class TimerState(str, Enum):
    """Timer states for mental math mode."""
    WAITING = "waiting"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


class MentalMathProblem(BaseModel):
    """Problem sent to glasses for mental math mode."""

    problem_id: str
    problem: str                     # "47 × 83"
    difficulty: int = Field(ge=1, le=5, description="1-5 difficulty scale")
    time_target_ms: int              # Target time in milliseconds
    category: str = "arithmetic"     # arithmetic, algebra, probability, etc.

    # Display hints
    display_large: bool = True       # Large centered display
    color_scheme: ColorScheme = ColorScheme.MATH


class MentalMathResult(BaseModel):
    """Result of a mental math attempt."""

    problem_id: str
    correct: bool
    user_answer: Any                 # What user answered
    correct_answer: Any              # Actual answer
    time_ms: int                     # Time taken

    # Performance feedback
    within_target: bool              # Beat the time target
    percentile: Optional[float] = None  # Performance vs others

    # Streak tracking
    current_streak: int = 0
    best_streak: int = 0

    # Display
    color_scheme: ColorScheme = ColorScheme.RESULT_GREEN if True else ColorScheme.RESULT_RED

    def __init__(self, **data):
        super().__init__(**data)
        # Set color based on correctness
        object.__setattr__(
            self,
            'color_scheme',
            ColorScheme.RESULT_GREEN if self.correct else ColorScheme.RESULT_RED
        )


# WebSocket message types for real-time modes

class WSMessageType(str, Enum):
    """WebSocket message types for AR modes."""

    # Mental Math
    PROBLEM = "problem"
    ANSWER = "answer"
    RESULT = "result"
    TIMER_UPDATE = "timer_update"
    STREAK_UPDATE = "streak_update"

    # General
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


class WSMessage(BaseModel):
    """WebSocket message wrapper."""

    type: WSMessageType
    data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
