"""
Homework HUD Display - Math solution display for Halo glasses.
Renders solutions, steps, and hints to the 640x400 OLED display.
"""
from typing import Optional, List
from .colors import Colors
from .renderer import HUDRenderer, COLORS, rgb_to_lua, FONT_LARGE, FONT_MEDIUM, FONT_SMALL, FONT_TINY, CENTER_X, CENTER_Y


class HomeworkHUD:
    """
    Homework mode HUD with solution display.

    Layout (640x400):
    ┌─────────────────────────────────┐
    │ MATH MODE           [SOLVING...] │ ← Status
    │                                  │
    │ Problem: 2x² + 5x - 3 = 0       │ ← Original problem
    │                                  │
    │ ANSWER: x = 0.5, x = -3         │ ← Big answer
    │                                  │
    │ Steps:                          │
    │ 1. Factor: (2x - 1)(x + 3)     │
    │ 2. Set each factor to 0         │
    │                                  │
    │ Method: local_sympy | $0.00     │ ← Cost tracking
    └─────────────────────────────────┘
    """

    def __init__(self, width: int = 640, height: int = 400):
        self.width = width
        self.height = height
        self.renderer = HUDRenderer(width, height)

    def render_thinking(
        self,
        problem: str = "",
        elapsed_s: float = 0,
        method: str = "analyzing"
    ) -> str:
        """
        Render thinking screen while solving.

        Args:
            problem: The math problem being solved
            elapsed_s: Seconds elapsed
            method: Current solving method

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            "MATH MODE",
            40, 30,
            COLORS.primary, FONT_SMALL, "left"
        )

        # Thinking animation
        dots = "." * (int(elapsed_s) % 4)
        self.renderer.text(
            f"SOLVING{dots}",
            self.width - 40, 30,
            COLORS.warning, FONT_SMALL, "right"
        )

        # Timer
        self.renderer.text(
            f"{elapsed_s:.1f}s",
            self.width - 40, 60,
            COLORS.text_dim, FONT_TINY, "right"
        )

        # Problem display (truncate if too long)
        if len(problem) > 40:
            problem = problem[:37] + "..."
        self.renderer.text(
            problem,
            CENTER_X, CENTER_Y - 40,
            COLORS.text_primary, FONT_MEDIUM, "center"
        )

        # Method indicator
        self.renderer.text(
            f"Method: {method}",
            CENTER_X, CENTER_Y + 60,
            COLORS.text_dim, FONT_TINY, "center"
        )

        # Pulsing indicator
        self.renderer.text(
            "Analyzing...",
            CENTER_X, self.height - 60,
            COLORS.primary, FONT_SMALL, "center"
        )

        self.renderer.show()
        return self.renderer.get_lua()

    def render_solution(
        self,
        problem: str,
        answer: str,
        steps: List[str] = None,
        method: str = "local_sympy",
        cost: float = 0,
        latency_ms: float = 0
    ) -> str:
        """
        Render solution display.

        Args:
            problem: Original problem
            answer: The solution
            steps: Step-by-step solution
            method: Solving method used
            cost: API cost
            latency_ms: Time to solve

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            "MATH MODE",
            40, 25,
            COLORS.primary, FONT_SMALL, "left"
        )

        # Latency
        latency_str = f"{latency_ms:.0f}ms" if latency_ms < 1000 else f"{latency_ms/1000:.1f}s"
        self.renderer.text(
            latency_str,
            self.width - 40, 25,
            COLORS.text_dim, FONT_TINY, "right"
        )

        # Problem (truncate if needed)
        if len(problem) > 45:
            problem = problem[:42] + "..."
        self.renderer.text(
            problem,
            CENTER_X, 70,
            COLORS.text_secondary, FONT_SMALL, "center"
        )

        # Big answer display
        if len(answer) > 30:
            answer = answer[:27] + "..."
        self.renderer.text(
            answer,
            CENTER_X, 130,
            COLORS.success, FONT_LARGE, "center"
        )

        # Steps section (show up to 3 steps)
        if steps:
            y_start = 190
            self.renderer.text(
                "Steps:",
                40, y_start,
                COLORS.accent, FONT_TINY, "left"
            )

            for i, step in enumerate(steps[:3]):
                if len(step) > 55:
                    step = step[:52] + "..."
                self.renderer.text(
                    step,
                    60, y_start + 30 + (i * 30),
                    COLORS.text_secondary, FONT_TINY, "left"
                )

            if len(steps) > 3:
                self.renderer.text(
                    f"... +{len(steps) - 3} more steps",
                    60, y_start + 30 + (3 * 30),
                    COLORS.text_dim, FONT_TINY, "left"
                )

        # Footer - method and cost
        method_color = COLORS.success if method == "local_sympy" else COLORS.accent
        cost_str = f"${cost:.3f}" if cost > 0 else "FREE"
        self.renderer.text(
            f"Method: {method} | {cost_str}",
            CENTER_X, self.height - 25,
            method_color if cost == 0 else COLORS.text_dim,
            FONT_TINY, "center"
        )

        self.renderer.show()
        return self.renderer.get_lua()

    def render_hint(
        self,
        problem: str,
        hint: str,
        cost: float = 0
    ) -> str:
        """
        Render hint display.

        Args:
            problem: The math problem
            hint: The hint text
            cost: API cost for hint

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            "HINT",
            CENTER_X, 40,
            COLORS.warning, FONT_MEDIUM, "center"
        )

        # Problem
        if len(problem) > 40:
            problem = problem[:37] + "..."
        self.renderer.text(
            problem,
            CENTER_X, 90,
            COLORS.text_secondary, FONT_SMALL, "center"
        )

        # Hint (wrap if needed)
        hint_lines = self._wrap_text(hint, 50)
        y_start = 160
        for i, line in enumerate(hint_lines[:4]):
            self.renderer.text(
                line,
                CENTER_X, y_start + (i * 35),
                COLORS.text_primary, FONT_SMALL, "center"
            )

        # Cost
        if cost > 0:
            self.renderer.text(
                f"Hint cost: ${cost:.3f}",
                CENTER_X, self.height - 30,
                COLORS.text_dim, FONT_TINY, "center"
            )

        self.renderer.show()
        return self.renderer.get_lua()

    def render_step_explanation(
        self,
        step: str,
        explanation: str,
        step_number: int = 1,
        total_steps: int = 1
    ) -> str:
        """
        Render detailed step explanation.

        Args:
            step: The step being explained
            explanation: Detailed explanation
            step_number: Current step number
            total_steps: Total number of steps

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            f"Step {step_number}/{total_steps}",
            CENTER_X, 35,
            COLORS.accent, FONT_MEDIUM, "center"
        )

        # Step
        if len(step) > 50:
            step = step[:47] + "..."
        self.renderer.text(
            step,
            CENTER_X, 90,
            COLORS.primary, FONT_SMALL, "center"
        )

        # Divider line
        self.renderer.rect(100, 120, self.width - 200, 2, COLORS.text_dim, True)

        # Explanation (wrapped)
        explanation_lines = self._wrap_text(explanation, 55)
        y_start = 150
        for i, line in enumerate(explanation_lines[:6]):
            self.renderer.text(
                line,
                40, y_start + (i * 30),
                COLORS.text_secondary, FONT_TINY, "left"
            )

        self.renderer.show()
        return self.renderer.get_lua()

    def render_failed(
        self,
        problem: str,
        error: str = "Could not solve"
    ) -> str:
        """
        Render failed solution display.

        Args:
            problem: The math problem
            error: Error message

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            "MATH MODE",
            CENTER_X, 40,
            COLORS.error, FONT_MEDIUM, "center"
        )

        # Problem
        if len(problem) > 40:
            problem = problem[:37] + "..."
        self.renderer.text(
            problem,
            CENTER_X, 100,
            COLORS.text_secondary, FONT_SMALL, "center"
        )

        # Error
        self.renderer.text(
            error,
            CENTER_X, CENTER_Y + 20,
            COLORS.warning, FONT_MEDIUM, "center"
        )

        # Suggestion
        self.renderer.text(
            "Try rephrasing or showing a clearer image",
            CENTER_X, self.height - 50,
            COLORS.text_dim, FONT_TINY, "center"
        )

        self.renderer.show()
        return self.renderer.get_lua()

    def render_session_stats(
        self,
        problems_solved: int,
        local_solves: int,
        cloud_solves: int,
        total_cost: float,
        cost_saved: float
    ) -> str:
        """
        Render session statistics.

        Args:
            problems_solved: Total problems solved
            local_solves: Problems solved locally (free)
            cloud_solves: Problems requiring cloud
            total_cost: Total API cost
            cost_saved: Estimated savings vs cloud-only

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            "HOMEWORK SESSION",
            CENTER_X, 35,
            COLORS.primary, FONT_MEDIUM, "center"
        )

        # Stats
        stats = [
            f"Problems Solved: {problems_solved}",
            f"Local (free): {local_solves}",
            f"Cloud: {cloud_solves}",
            f"Total Cost: ${total_cost:.3f}",
            f"Estimated Savings: ${cost_saved:.3f}",
        ]

        y_start = 100
        for i, stat in enumerate(stats):
            color = COLORS.success if i == 1 else COLORS.text_secondary
            self.renderer.text(
                stat,
                CENTER_X, y_start + (i * 45),
                color, FONT_SMALL, "center"
            )

        # Efficiency percentage
        if problems_solved > 0:
            efficiency = (local_solves / problems_solved) * 100
            eff_color = COLORS.success if efficiency >= 70 else COLORS.warning
            self.renderer.text(
                f"Local efficiency: {efficiency:.0f}%",
                CENTER_X, self.height - 40,
                eff_color, FONT_SMALL, "center"
            )

        self.renderer.show()
        return self.renderer.get_lua()

    def _wrap_text(self, text: str, max_chars: int) -> List[str]:
        """Wrap text to specified character width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= max_chars:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return lines


# Test
def test_homework_hud():
    """Test homework HUD rendering."""
    print("=== Homework HUD Test ===\n")

    hud = HomeworkHUD()

    # Test thinking screen
    print("Thinking screen:")
    print("-" * 40)
    lua = hud.render_thinking(
        problem="2x² + 5x - 3 = 0",
        elapsed_s=1.5,
        method="sympy"
    )
    print(lua[:500] + "...")
    print()

    # Test solution screen
    print("Solution screen:")
    print("-" * 40)
    lua = hud.render_solution(
        problem="2x² + 5x - 3 = 0",
        answer="x = 0.5, x = -3",
        steps=[
            "1. Factor: (2x - 1)(x + 3) = 0",
            "2. Set each factor to 0",
            "3. Solve: x = 1/2, x = -3"
        ],
        method="local_sympy",
        cost=0,
        latency_ms=45
    )
    print(lua[:500] + "...")


if __name__ == "__main__":
    test_homework_hud()
