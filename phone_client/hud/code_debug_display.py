"""
Code Debug HUD Display - Syntax error and debugging display for Halo glasses.
Renders errors, suggestions, and fixes to the 640x400 OLED display.
"""
from typing import Optional, List
from .colors import Colors
from .renderer import HUDRenderer, COLORS, rgb_to_lua, FONT_LARGE, FONT_MEDIUM, FONT_SMALL, FONT_TINY, CENTER_X, CENTER_Y


class CodeDebugHUD:
    """
    Code debug mode HUD with error display.

    Layout (640x400):
    ┌─────────────────────────────────┐
    │ DEBUG MODE  [Python]  [45ms]    │ ← Status + language
    │                                  │
    │ ✓ SYNTAX OK                     │ ← OR error display
    │                                  │
    │ ─────── OR ───────              │
    │                                  │
    │ ✗ Line 12: SyntaxError          │ ← Error location
    │ Missing closing parenthesis      │ ← Error message
    │                                  │
    │ Suggestion:                      │
    │ Add ')' at end of line 12       │
    │                                  │
    │ Method: local_ast | $0.00        │ ← Cost tracking
    └─────────────────────────────────┘
    """

    # Language colors
    LANG_COLORS = {
        'python': (255, 200, 50),     # Python yellow
        'javascript': (240, 220, 60),  # JS yellow
        'js': (240, 220, 60),
        'typescript': (50, 150, 255),  # TS blue
        'ts': (50, 150, 255),
        'unknown': COLORS.text_dim,
    }

    def __init__(self, width: int = 640, height: int = 400):
        self.width = width
        self.height = height
        self.renderer = HUDRenderer(width, height)

    def render_analyzing(
        self,
        elapsed_s: float = 0,
        language: str = "detecting"
    ) -> str:
        """
        Render analyzing screen while checking code.

        Args:
            elapsed_s: Seconds elapsed
            language: Detected/detecting language

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            "DEBUG MODE",
            40, 30,
            COLORS.primary, FONT_SMALL, "left"
        )

        # Language badge
        lang_color = self.LANG_COLORS.get(language.lower(), COLORS.text_dim)
        self.renderer.text(
            f"[{language.title()}]",
            CENTER_X, 30,
            lang_color, FONT_SMALL, "center"
        )

        # Thinking animation
        dots = "." * (int(elapsed_s) % 4)
        self.renderer.text(
            f"CHECKING{dots}",
            self.width - 40, 30,
            COLORS.warning, FONT_SMALL, "right"
        )

        # Scanning animation
        self.renderer.text(
            "Scanning code...",
            CENTER_X, CENTER_Y,
            COLORS.primary, FONT_MEDIUM, "center"
        )

        # Progress indicator
        self.renderer.text(
            f"{elapsed_s:.1f}s",
            CENTER_X, CENTER_Y + 50,
            COLORS.text_dim, FONT_TINY, "center"
        )

        self.renderer.show()
        return self.renderer.get_lua()

    def render_syntax_ok(
        self,
        language: str,
        latency_ms: float = 0
    ) -> str:
        """
        Render syntax OK display.

        Args:
            language: Programming language
            latency_ms: Time to check

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            "DEBUG MODE",
            40, 30,
            COLORS.primary, FONT_SMALL, "left"
        )

        # Language badge
        lang_color = self.LANG_COLORS.get(language.lower(), COLORS.text_dim)
        self.renderer.text(
            f"[{language.title()}]",
            CENTER_X, 30,
            lang_color, FONT_SMALL, "center"
        )

        # Latency
        self.renderer.text(
            f"{latency_ms:.0f}ms",
            self.width - 40, 30,
            COLORS.text_dim, FONT_TINY, "right"
        )

        # Big checkmark and OK message
        self.renderer.text(
            "SYNTAX OK",
            CENTER_X, CENTER_Y - 20,
            COLORS.success, FONT_LARGE, "center"
        )

        # Subtitle
        self.renderer.text(
            "No errors detected",
            CENTER_X, CENTER_Y + 40,
            COLORS.text_secondary, FONT_SMALL, "center"
        )

        # Method
        self.renderer.text(
            f"Method: local_{language.lower()} | FREE",
            CENTER_X, self.height - 30,
            COLORS.success, FONT_TINY, "center"
        )

        self.renderer.show()
        return self.renderer.get_lua()

    def render_errors(
        self,
        language: str,
        errors: List[dict],
        latency_ms: float = 0,
        cost: float = 0
    ) -> str:
        """
        Render syntax errors display.

        Args:
            language: Programming language
            errors: List of error dicts with 'line', 'message', 'suggestion'
            latency_ms: Time to check
            cost: API cost

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            "DEBUG MODE",
            40, 25,
            COLORS.primary, FONT_SMALL, "left"
        )

        # Language badge
        lang_color = self.LANG_COLORS.get(language.lower(), COLORS.text_dim)
        self.renderer.text(
            f"[{language.title()}]",
            CENTER_X, 25,
            lang_color, FONT_SMALL, "center"
        )

        # Latency
        self.renderer.text(
            f"{latency_ms:.0f}ms",
            self.width - 40, 25,
            COLORS.text_dim, FONT_TINY, "right"
        )

        # Error count
        error_count = len(errors)
        count_text = f"{error_count} ERROR{'S' if error_count > 1 else ''}"
        self.renderer.text(
            count_text,
            CENTER_X, 70,
            COLORS.error, FONT_MEDIUM, "center"
        )

        # Show first error details
        if errors:
            error = errors[0]
            line = error.get('line', '?')
            message = error.get('message', 'Unknown error')
            suggestion = error.get('suggestion', '')

            # Line number
            self.renderer.text(
                f"Line {line}:",
                40, 120,
                COLORS.error, FONT_SMALL, "left"
            )

            # Error message (truncate if needed)
            if len(message) > 50:
                message = message[:47] + "..."
            self.renderer.text(
                message,
                40, 155,
                COLORS.text_primary, FONT_SMALL, "left"
            )

            # Suggestion
            if suggestion:
                self.renderer.text(
                    "Suggestion:",
                    40, 210,
                    COLORS.accent, FONT_TINY, "left"
                )
                if len(suggestion) > 55:
                    suggestion = suggestion[:52] + "..."
                self.renderer.text(
                    suggestion,
                    40, 240,
                    COLORS.text_secondary, FONT_TINY, "left"
                )

            # More errors indicator
            if error_count > 1:
                self.renderer.text(
                    f"+{error_count - 1} more error{'s' if error_count > 2 else ''}",
                    CENTER_X, 300,
                    COLORS.warning, FONT_TINY, "center"
                )

        # Footer - method and cost
        method = f"local_{language.lower()}" if cost == 0 else "deepseek"
        cost_str = f"${cost:.3f}" if cost > 0 else "FREE"
        self.renderer.text(
            f"Method: {method} | {cost_str}",
            CENTER_X, self.height - 25,
            COLORS.success if cost == 0 else COLORS.text_dim,
            FONT_TINY, "center"
        )

        self.renderer.show()
        return self.renderer.get_lua()

    def render_explanation(
        self,
        error: dict,
        explanation: str,
        cost: float = 0
    ) -> str:
        """
        Render detailed error explanation.

        Args:
            error: Error dict with 'line', 'message'
            explanation: Detailed explanation from DeepSeek
            cost: API cost

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            "ERROR EXPLAINED",
            CENTER_X, 30,
            COLORS.accent, FONT_MEDIUM, "center"
        )

        # Error summary
        line = error.get('line', '?')
        message = error.get('message', 'Unknown error')
        if len(message) > 45:
            message = message[:42] + "..."
        self.renderer.text(
            f"Line {line}: {message}",
            CENTER_X, 75,
            COLORS.error, FONT_SMALL, "center"
        )

        # Divider
        self.renderer.rect(80, 105, self.width - 160, 2, COLORS.text_dim, True)

        # Explanation (wrapped)
        explanation_lines = self._wrap_text(explanation, 55)
        y_start = 130
        for i, line_text in enumerate(explanation_lines[:7]):
            self.renderer.text(
                line_text,
                40, y_start + (i * 28),
                COLORS.text_secondary, FONT_TINY, "left"
            )

        # Cost
        self.renderer.text(
            f"Explanation cost: ${cost:.3f}",
            CENTER_X, self.height - 25,
            COLORS.text_dim, FONT_TINY, "center"
        )

        self.renderer.show()
        return self.renderer.get_lua()

    def render_fix_suggestion(
        self,
        original_line: str,
        fixed_line: str,
        line_number: int,
        cost: float = 0
    ) -> str:
        """
        Render suggested fix display.

        Args:
            original_line: The erroneous code line
            fixed_line: The suggested fix
            line_number: Line number
            cost: API cost

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            "SUGGESTED FIX",
            CENTER_X, 30,
            COLORS.success, FONT_MEDIUM, "center"
        )

        # Line number
        self.renderer.text(
            f"Line {line_number}",
            CENTER_X, 75,
            COLORS.text_secondary, FONT_SMALL, "center"
        )

        # Original (error)
        self.renderer.text(
            "Before:",
            40, 120,
            COLORS.error, FONT_TINY, "left"
        )
        if len(original_line) > 55:
            original_line = original_line[:52] + "..."
        self.renderer.text(
            original_line,
            40, 150,
            COLORS.text_dim, FONT_SMALL, "left"
        )

        # Fixed (success)
        self.renderer.text(
            "After:",
            40, 210,
            COLORS.success, FONT_TINY, "left"
        )
        if len(fixed_line) > 55:
            fixed_line = fixed_line[:52] + "..."
        self.renderer.text(
            fixed_line,
            40, 240,
            COLORS.text_primary, FONT_SMALL, "left"
        )

        # Cost
        self.renderer.text(
            f"Fix suggestion cost: ${cost:.3f}",
            CENTER_X, self.height - 25,
            COLORS.text_dim, FONT_TINY, "center"
        )

        self.renderer.show()
        return self.renderer.get_lua()

    def render_runtime_error(
        self,
        error_type: str,
        error_message: str,
        traceback_lines: List[str] = None,
        suggestion: str = ""
    ) -> str:
        """
        Render runtime error analysis.

        Args:
            error_type: Type of error (TypeError, ValueError, etc.)
            error_message: Error message
            traceback_lines: Relevant traceback lines
            suggestion: Fix suggestion

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            "RUNTIME ERROR",
            CENTER_X, 30,
            COLORS.error, FONT_MEDIUM, "center"
        )

        # Error type
        self.renderer.text(
            error_type,
            CENTER_X, 80,
            COLORS.warning, FONT_SMALL, "center"
        )

        # Error message
        if len(error_message) > 50:
            error_message = error_message[:47] + "..."
        self.renderer.text(
            error_message,
            CENTER_X, 120,
            COLORS.text_primary, FONT_SMALL, "center"
        )

        # Traceback (if provided)
        if traceback_lines:
            y_start = 170
            for i, line in enumerate(traceback_lines[:3]):
                if len(line) > 55:
                    line = line[:52] + "..."
                self.renderer.text(
                    line,
                    40, y_start + (i * 25),
                    COLORS.text_dim, FONT_TINY, "left"
                )

        # Suggestion
        if suggestion:
            self.renderer.text(
                "Suggestion:",
                40, 280,
                COLORS.accent, FONT_TINY, "left"
            )
            if len(suggestion) > 55:
                suggestion = suggestion[:52] + "..."
            self.renderer.text(
                suggestion,
                40, 310,
                COLORS.text_secondary, FONT_TINY, "left"
            )

        self.renderer.show()
        return self.renderer.get_lua()

    def render_session_stats(
        self,
        code_checked: int,
        local_checks: int,
        cloud_explains: int,
        total_cost: float,
        cost_saved: float
    ) -> str:
        """
        Render debug session statistics.

        Args:
            code_checked: Total code snippets checked
            local_checks: Checks done locally (free)
            cloud_explains: Explanations from cloud
            total_cost: Total API cost
            cost_saved: Estimated savings

        Returns:
            Lua code for display
        """
        self.renderer.clear()

        # Header
        self.renderer.text(
            "DEBUG SESSION",
            CENTER_X, 35,
            COLORS.primary, FONT_MEDIUM, "center"
        )

        # Stats
        stats = [
            f"Code Checked: {code_checked}",
            f"Local (free): {local_checks}",
            f"Cloud Explains: {cloud_explains}",
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

        # Efficiency
        if code_checked > 0:
            efficiency = (local_checks / code_checked) * 100
            eff_color = COLORS.success if efficiency >= 60 else COLORS.warning
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
def test_code_debug_hud():
    """Test code debug HUD rendering."""
    print("=== Code Debug HUD Test ===\n")

    hud = CodeDebugHUD()

    # Test syntax OK
    print("Syntax OK screen:")
    print("-" * 40)
    lua = hud.render_syntax_ok(
        language="python",
        latency_ms=12
    )
    print(lua[:500] + "...")
    print()

    # Test error display
    print("Error display:")
    print("-" * 40)
    lua = hud.render_errors(
        language="python",
        errors=[
            {
                'line': 12,
                'message': "SyntaxError: Missing closing parenthesis",
                'suggestion': "Add ')' at end of line 12"
            },
            {
                'line': 15,
                'message': "IndentationError: unexpected indent"
            }
        ],
        latency_ms=8,
        cost=0
    )
    print(lua[:500] + "...")


if __name__ == "__main__":
    test_code_debug_hud()
