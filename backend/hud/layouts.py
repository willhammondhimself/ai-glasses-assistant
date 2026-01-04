"""
Iron Man HUD Layouts

Mode-specific screen layouts combining HUD components.
Each layout optimizes component placement for its use case.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from .colors import WHAM_COLORS, rgb_to_hex
from .components import (
    HUDComponent,
    Position,
    Size,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
    ProgressBar,
    TimerDisplay,
    StreakCounter,
    AccuracyMeter,
    DifficultyIndicator,
    ProblemDisplay,
    ResultOverlay,
    SessionSummary,
    WHAMMessage,
)


class HUDLayout(ABC):
    """Base class for HUD layouts."""

    def __init__(self):
        self.components: Dict[str, HUDComponent] = {}

    @abstractmethod
    def build(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build layout with current data."""
        pass

    def render_all(self) -> Dict[str, Any]:
        """Render all components."""
        return {
            name: comp.to_dict()
            for name, comp in self.components.items()
            if comp.visible
        }


class MentalMathLayout(HUDLayout):
    """
    Layout for Mental Math speed run mode.

    Screen regions:
    ┌─────────────────────────────────────┐
    │  [D2]        Timer        [Streak]  │  <- Top bar
    │                                     │
    │                                     │
    │            47 × 83 = ?              │  <- Problem (center)
    │                                     │
    │                                     │
    │  [Accuracy]              [Progress] │  <- Bottom bar
    │  "WHAM message here..."           │  <- WHAM feedback
    └─────────────────────────────────────┘
    """

    # Layout constants
    MARGIN = 20
    TOP_BAR_Y = 15
    PROBLEM_Y = 150
    BOTTOM_BAR_Y = 340
    WHAM_Y = 370

    def __init__(self):
        super().__init__()
        self._init_components()

    def _init_components(self):
        """Initialize all layout components."""
        # Top bar - left: difficulty
        self.components["difficulty"] = DifficultyIndicator(
            position=Position(self.MARGIN, self.TOP_BAR_Y),
            difficulty=2,
            style="dots"
        )

        # Top bar - center: timer
        self.components["timer"] = TimerDisplay(
            position=Position(DISPLAY_WIDTH // 2 - 50, self.TOP_BAR_Y),
            target_ms=4000,
            show_arc=True,
            size_preset="medium"
        )

        # Top bar - right: streak
        self.components["streak"] = StreakCounter(
            position=Position(DISPLAY_WIDTH - 100, self.TOP_BAR_Y),
            streak=0
        )

        # Center: problem display
        self.components["problem"] = ProblemDisplay(
            position=Position(self.MARGIN, self.PROBLEM_Y),
            font_size="xlarge"
        )

        # Bottom bar - left: accuracy
        self.components["accuracy"] = AccuracyMeter(
            position=Position(self.MARGIN, self.BOTTOM_BAR_Y),
            show_fraction=True,
            show_bar=False
        )

        # Bottom bar - right: session progress
        self.components["progress"] = ProgressBar(
            position=Position(DISPLAY_WIDTH - 220, self.BOTTOM_BAR_Y),
            width=200,
            height=6,
            show_segments=True,
            segment_count=10
        )

        # WHAM message at bottom
        self.components["wham"] = WHAMMessage(
            position=Position(self.MARGIN, self.WHAM_Y),
            message=""
        )

        # Result overlay (hidden by default)
        self.components["result"] = ResultOverlay(
            position=Position(DISPLAY_WIDTH // 2 - 150, DISPLAY_HEIGHT // 2 - 60),
            correct=True,
            time_ms=0
        )
        self.components["result"].visible = False

    def build(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build layout with current state data.

        Expected data keys:
        - difficulty: int
        - elapsed_ms: int
        - target_ms: int
        - streak: int
        - correct: int
        - total: int
        - problem_text: str
        - problem_number: int
        - wham_message: str
        - show_result: bool
        - result_correct: bool
        - result_time_ms: int
        - result_speed_tier: str
        """
        # Update difficulty
        if "difficulty" in data:
            self.components["difficulty"].difficulty = data["difficulty"]

        # Update timer
        if "elapsed_ms" in data:
            self.components["timer"].elapsed_ms = data["elapsed_ms"]
        if "target_ms" in data:
            self.components["timer"].target_ms = data["target_ms"]

        # Update streak
        if "streak" in data:
            self.components["streak"].streak = data["streak"]

        # Update accuracy
        if "correct" in data:
            self.components["accuracy"].correct = data["correct"]
        if "total" in data:
            self.components["accuracy"].total = data["total"]

        # Update problem
        if "problem_text" in data:
            self.components["problem"].problem_text = data["problem_text"]
        if "problem_number" in data:
            self.components["problem"].problem_number = data["problem_number"]

        # Update progress bar (problems done / target)
        target_problems = data.get("target_problems", 10)
        current = data.get("total", 0)
        self.components["progress"].progress = min(1, current / target_problems)

        # Update WHAM message
        if "wham_message" in data:
            self.components["wham"].message = data["wham_message"]
            self.components["wham"].visible = bool(data["wham_message"])

        # Update result overlay
        if data.get("show_result"):
            self.components["result"].correct = data.get("result_correct", True)
            self.components["result"].time_ms = data.get("result_time_ms", 0)
            self.components["result"].speed_tier = data.get("result_speed_tier", "good")
            self.components["result"].message = data.get("result_message", "")
            self.components["result"].visible = True
        else:
            self.components["result"].visible = False

        return self.render_all()

    def show_problem_phase(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build layout for problem display phase."""
        data["show_result"] = False
        return self.build(data)

    def show_result_phase(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build layout for result display phase."""
        data["show_result"] = True
        return self.build(data)


class PokerLayout(HUDLayout):
    """
    Layout for Poker analysis mode.

    Screen regions:
    ┌─────────────────────────────────────┐
    │  [Hand]      Pot Odds    [Position] │
    │                                     │
    │   Your Hand: A♠ K♠                  │
    │   Board: 10♠ J♠ 2♣ 5♦              │
    │                                     │
    │  Equity: 65%    |    Action: RAISE  │
    │  "WHAM analysis..."               │
    └─────────────────────────────────────┘
    """

    MARGIN = 20
    TOP_BAR_Y = 15
    HAND_Y = 100
    BOARD_Y = 180
    ANALYSIS_Y = 280
    WHAM_Y = 350

    def __init__(self):
        super().__init__()
        self._init_components()

    def _init_components(self):
        """Initialize poker-specific components."""
        # Top bar elements would go here
        # Hand display, board display, equity meter, etc.

        # WHAM message
        self.components["wham"] = WHAMMessage(
            position=Position(self.MARGIN, self.WHAM_Y),
            message=""
        )

    def build(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build poker layout with analysis data."""
        # Update components based on data
        if "wham_message" in data:
            self.components["wham"].message = data["wham_message"]

        return self.render_all()


class CodeDebugLayout(HUDLayout):
    """
    Layout for Code debugging mode.

    Screen regions:
    ┌─────────────────────────────────────┐
    │  [Language]   Line #    [Severity]  │
    │                                     │
    │   Error: TypeError                  │
    │   cannot read property 'x' of null  │
    │                                     │
    │  Suggested Fix:                     │
    │  Check if obj exists before access  │
    │  "WHAM suggestion..."             │
    └─────────────────────────────────────┘
    """

    MARGIN = 20

    def __init__(self):
        super().__init__()
        self._init_components()

    def _init_components(self):
        """Initialize code debug components."""
        self.components["wham"] = WHAMMessage(
            position=Position(self.MARGIN, 350),
            message=""
        )

    def build(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build code debug layout."""
        if "wham_message" in data:
            self.components["wham"].message = data["wham_message"]

        return self.render_all()


class HomeworkLayout(HUDLayout):
    """
    Layout for Homework help mode.

    Shows camera view overlay with solution annotations.
    """

    MARGIN = 20

    def __init__(self):
        super().__init__()
        self._init_components()

    def _init_components(self):
        """Initialize homework help components."""
        self.components["wham"] = WHAMMessage(
            position=Position(self.MARGIN, 350),
            message="Point at a problem for analysis, sir."
        )

    def build(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build homework help layout."""
        if "wham_message" in data:
            self.components["wham"].message = data["wham_message"]

        return self.render_all()


# Layout factory
def get_layout(mode: str) -> HUDLayout:
    """Get the appropriate layout for a mode."""
    layouts = {
        "mental_math": MentalMathLayout,
        "poker": PokerLayout,
        "code_debug": CodeDebugLayout,
        "homework": HomeworkLayout,
    }

    layout_class = layouts.get(mode, MentalMathLayout)
    return layout_class()
