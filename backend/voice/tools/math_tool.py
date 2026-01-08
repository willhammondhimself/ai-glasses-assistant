"""Math calculation tool for voice queries."""
import logging
import re
from typing import Optional
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class MathTool(VoiceTool):
    """Perform calculations and solve math problems."""

    name = "math"
    description = "Calculate math expressions and solve equations"

    keywords = [
        r"\bcalculate\b",
        r"\bwhat\s+is\s+\d+",
        r"\d+\s*[\+\-\*\/\^]\s*\d+",
        r"\bcompute\b",
        r"\bsolve\b",
        r"\bconvert\b.*\bto\b",
        r"\bsquare\s+root\b",
        r"\b\d+\s*(plus|minus|times|divided|multiplied)\b",
        r"\bpercentage\b",
        r"\b\d+\s*%\s*of\b",
    ]

    priority = 15  # High priority for math expressions

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Calculate math expressions.

        Args:
            query: The user's voice query

        Returns:
            VoiceToolResult with calculation result
        """
        try:
            # Try to extract and evaluate the expression
            expression, result = self._evaluate_expression(query)

            if result is not None:
                # Format result nicely for voice
                result_str = self._format_number(result)

                return VoiceToolResult(
                    success=True,
                    message=f"{result_str}",
                    data={"expression": expression, "result": result}
                )
            else:
                return VoiceToolResult(
                    success=False,
                    message="I couldn't understand that math expression."
                )
        except Exception as e:
            logger.error(f"Math calculation failed: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I couldn't calculate that."
            )

    def _evaluate_expression(self, query: str) -> tuple[Optional[str], Optional[float]]:
        """Extract and evaluate a math expression from the query.

        Returns:
            Tuple of (expression_string, result) or (None, None)
        """
        query_lower = query.lower()

        # Replace word operators with symbols
        replacements = [
            (r'\bplus\b', '+'),
            (r'\bminus\b', '-'),
            (r'\btimes\b', '*'),
            (r'\bmultiplied\s+by\b', '*'),
            (r'\bdivided\s+by\b', '/'),
            (r'\bover\b', '/'),
            (r'\bto\s+the\s+power\s+of\b', '**'),
            (r'\bsquared\b', '**2'),
            (r'\bcubed\b', '**3'),
            (r'\bx\b', '*'),  # "2 x 3" -> "2 * 3"
        ]

        expr = query_lower
        for pattern, replacement in replacements:
            expr = re.sub(pattern, replacement, expr)

        # Handle percentage calculations: "X% of Y"
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)', expr)
        if pct_match:
            pct = float(pct_match.group(1))
            value = float(pct_match.group(2))
            result = (pct / 100) * value
            return f"{pct}% of {value}", result

        # Handle square root
        sqrt_match = re.search(r'square\s+root\s+(?:of\s+)?(\d+(?:\.\d+)?)', expr)
        if sqrt_match:
            value = float(sqrt_match.group(1))
            result = value ** 0.5
            return f"√{value}", result

        # Extract numeric expression
        # Remove common prefixes
        expr = re.sub(r'^(?:what\s+is|calculate|compute|solve)\s*', '', expr)
        expr = re.sub(r'\?$', '', expr)

        # Extract just the math part
        math_match = re.search(r'([\d\.\s\+\-\*\/\^\(\)]+)', expr)
        if math_match:
            math_expr = math_match.group(1).strip()

            # Clean up
            math_expr = re.sub(r'\s+', '', math_expr)
            math_expr = math_expr.replace('^', '**')

            # Safety check - only allow math operations
            if not re.match(r'^[\d\.\+\-\*\/\(\)\s\*]+$', math_expr):
                return None, None

            try:
                # Use eval with restricted builtins for safety
                result = eval(math_expr, {"__builtins__": {}}, {})
                return math_expr, float(result)
            except:
                return None, None

        return None, None

    def _format_number(self, num: float) -> str:
        """Format a number nicely for voice output."""
        if num == int(num):
            return f"{int(num):,}"
        elif abs(num) < 0.01:
            return f"{num:.6f}"
        elif abs(num) < 1:
            return f"{num:.4f}"
        else:
            return f"{num:,.2f}"
