"""
CS/Code Slide Builder

Formats code debugging and explanation results for AR display.
Handles syntax highlighting hints, error locations, and fixes.
"""

import re
from typing import List, Optional, Tuple

from backend.models.ar_response import Slide, ColorScheme
from .base import BaseSlideBuilder


class CSSlideBuilder(BaseSlideBuilder):
    """
    Builds slides for code-related content.

    Slide structure for debugging:
    1. Error Location (red)
    2. Error Explanation (info)
    3. Fixed Code (green)
    4. Explanation (info)

    Slide structure for explanations:
    1-N. Explanation slides (info)
    """

    def __init__(self):
        super().__init__()
        self.response_type = "cs"

    def build_slides(self, engine_result: dict) -> List[Slide]:
        """
        Build slides from CS engine result.

        Args:
            engine_result: Dict with 'code', 'error', 'fix', 'explanation', etc.

        Returns:
            List of formatted slides
        """
        slides = []

        # Determine if this is a debug result or explanation
        if engine_result.get("error") or engine_result.get("bug_location"):
            slides = self._build_debug_slides(engine_result)
        else:
            slides = self._build_explanation_slides(engine_result)

        return slides if slides else [Slide(
            title="Code",
            content=str(engine_result.get("solution", "No content")),
            color_scheme=ColorScheme.CODE
        )]

    def _build_debug_slides(self, result: dict) -> List[Slide]:
        """Build slides for code debugging flow."""
        slides = []

        # Slide 1: Error Location
        error_line = result.get("error_line", result.get("bug_location", ""))
        error_msg = result.get("error_message", result.get("error", ""))
        line_number = result.get("line_number", "")

        if error_line or error_msg:
            title = f"Error @ Line {line_number}" if line_number else "Error Found"
            content = self._format_code_snippet(error_line) if error_line else error_msg

            slides.append(Slide(
                title=self._truncate_title(title),
                content=content,
                color_scheme=ColorScheme.CODE_ERROR,
                voice_narration=f"Found error: {error_msg}" if error_msg else "Error in code"
            ))

        # Slide 2: Error Explanation
        if error_msg and error_line:
            slides.append(Slide(
                title="Issue",
                content=self._wrap_error_message(error_msg),
                color_scheme=ColorScheme.INFO,
                voice_narration=error_msg
            ))

        # Slide 3: Fixed Code
        fixed_code = result.get("fixed_code", result.get("fix", ""))
        if fixed_code:
            slides.append(Slide(
                title="Fix",
                content=self._format_code_snippet(fixed_code),
                color_scheme=ColorScheme.CODE_FIX,
                voice_narration="Here's the fix"
            ))

        # Slide 4: Explanation of fix
        explanation = result.get("explanation", result.get("fix_explanation", ""))
        if explanation:
            slides.extend(self._create_content_slides(
                content=explanation,
                title="Why",
                color_scheme=ColorScheme.INFO,
                voice_prefix=""
            ))

        return slides

    def _build_explanation_slides(self, result: dict) -> List[Slide]:
        """Build slides for code explanation."""
        slides = []

        # Main explanation/solution
        content = result.get("explanation", result.get("solution", ""))
        if content:
            slides.extend(self._create_content_slides(
                content=content,
                title="Explanation",
                color_scheme=ColorScheme.CODE,
                voice_prefix=""
            ))

        # Code example if present
        code_example = result.get("code_example", result.get("code", ""))
        if code_example:
            slides.extend(self._format_code_slides(code_example))

        return slides

    def _format_code_snippet(self, code: str) -> str:
        """
        Format code snippet for AR display.

        Optimizes for 45-char line width:
        - Truncates long lines
        - Preserves key indentation
        - Adds line numbers if multi-line
        """
        lines = code.strip().split('\n')
        formatted_lines = []

        for i, line in enumerate(lines[:self.MAX_CONTENT_LINES]):
            # Preserve 2-space indent, truncate rest
            indent = len(line) - len(line.lstrip())
            indent_str = "  " * min(indent // 2, 2)  # Max 4 spaces

            content = line.strip()
            max_content = self.MAX_LINE_CHARS - len(indent_str) - 4  # Room for line num

            if len(content) > max_content:
                content = content[:max_content - 3] + "..."

            formatted_lines.append(f"{indent_str}{content}")

        return '\n'.join(formatted_lines)

    def _format_code_slides(self, code: str) -> List[Slide]:
        """
        Format longer code into multiple slides.

        Each slide shows up to 3 lines of code.
        """
        slides = []
        lines = code.strip().split('\n')

        for i in range(0, len(lines), 3):
            chunk = lines[i:i + 3]
            formatted = '\n'.join(
                self._truncate_code_line(line)
                for line in chunk
            )

            slide_num = (i // 3) + 1
            total = (len(lines) + 2) // 3

            slides.append(Slide(
                title=f"Code {slide_num}/{total}" if total > 1 else "Code",
                content=formatted,
                color_scheme=ColorScheme.CODE
            ))

        return slides

    def _truncate_code_line(self, line: str) -> str:
        """Truncate a single line of code to fit display."""
        if len(line) <= self.MAX_LINE_CHARS:
            return line

        # Preserve indentation
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        max_content = self.MAX_LINE_CHARS - len(indent) - 3

        return indent + stripped[:max_content] + "..."

    def _wrap_error_message(self, error: str) -> str:
        """Wrap error message to fit display, preserving meaning."""
        # Common error message simplifications
        simplified = error
        simplifications = [
            (r"TypeError: ", "Type error: "),
            (r"NameError: ", "Name error: "),
            (r"SyntaxError: ", "Syntax error: "),
            (r"IndexError: ", "Index error: "),
            (r"KeyError: ", "Key error: "),
            (r"AttributeError: ", "Attr error: "),
            (r"ValueError: ", "Value error: "),
        ]

        for pattern, replacement in simplifications:
            simplified = re.sub(pattern, replacement, simplified)

        wrapped = self._wrap_text(simplified)
        return '\n'.join(wrapped[:self.MAX_CONTENT_LINES])

    def _generate_summary(self, engine_result: dict) -> Optional[str]:
        """Generate summary for code result."""
        if engine_result.get("error") or engine_result.get("bug_location"):
            error = engine_result.get("error_message", engine_result.get("error", ""))
            if error:
                return f"Bug: {error}"[:self.MAX_LINE_CHARS]
            return "Bug found - tap to see fix"

        return super()._generate_summary(engine_result)
