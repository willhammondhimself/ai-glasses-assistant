"""
Local Syntax Checker for Python and JavaScript.
Zero API costs for basic syntax validation.
"""
import ast
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy import esprima
esprima = None


def _load_esprima():
    """Lazy load esprima."""
    global esprima
    if esprima is None:
        try:
            import esprima as _esprima
            esprima = _esprima
            logger.info("esprima loaded successfully")
        except ImportError:
            logger.warning("esprima not installed. Run: pip install esprima")
    return esprima is not None


@dataclass
class SyntaxError:
    """Syntax error details."""
    line: int
    column: int
    message: str
    suggestion: Optional[str] = None


@dataclass
class SyntaxResult:
    """Result of syntax check."""
    valid: bool
    errors: List[SyntaxError]
    language: str


class LocalSyntaxChecker:
    """
    Local syntax checking for Python and JavaScript.

    Features:
    - Python: Uses built-in ast module
    - JavaScript: Uses esprima library
    - Error suggestions for common mistakes
    """

    # Common Python syntax error patterns and fixes
    PYTHON_FIXES = {
        "expected ':'": "Add a colon (:) at the end of the line",
        "invalid syntax": "Check for missing parentheses, brackets, or quotes",
        "unexpected EOF": "Check for unclosed brackets, parentheses, or strings",
        "unexpected indent": "Fix indentation - use consistent spaces or tabs",
        "expected an indented block": "Add indented code after the colon",
        "unmatched ')'": "Add a matching opening parenthesis",
        "unmatched ']'": "Add a matching opening bracket",
        "unmatched '}'": "Add a matching opening brace",
    }

    # Common JavaScript syntax error patterns
    JS_FIXES = {
        "unexpected token": "Check for missing semicolons or brackets",
        "unexpected end of input": "Check for unclosed brackets or strings",
        "missing )": "Add a closing parenthesis",
        "missing }": "Add a closing brace",
        "missing ]": "Add a closing bracket",
    }

    def __init__(self):
        self._esprima_available = _load_esprima()

    def check_python(self, code: str) -> SyntaxResult:
        """
        Check Python syntax.

        Args:
            code: Python code string

        Returns:
            SyntaxResult with validity and errors
        """
        try:
            ast.parse(code)
            return SyntaxResult(valid=True, errors=[], language="python")

        except SyntaxError as e:
            error = self._parse_python_error(e)
            return SyntaxResult(valid=False, errors=[error], language="python")

    def _parse_python_error(self, e: SyntaxError) -> SyntaxError:
        """Parse Python SyntaxError into our format."""
        message = str(e.msg) if hasattr(e, 'msg') else str(e)
        suggestion = None

        # Find matching fix suggestion
        for pattern, fix in self.PYTHON_FIXES.items():
            if pattern.lower() in message.lower():
                suggestion = fix
                break

        return SyntaxError(
            line=e.lineno or 1,
            column=e.offset or 1,
            message=message,
            suggestion=suggestion
        )

    def check_javascript(self, code: str) -> SyntaxResult:
        """
        Check JavaScript syntax.

        Args:
            code: JavaScript code string

        Returns:
            SyntaxResult with validity and errors
        """
        if not self._esprima_available:
            # Fallback: basic check
            return self._basic_js_check(code)

        try:
            esprima.parseScript(code, tolerant=False)
            return SyntaxResult(valid=True, errors=[], language="javascript")

        except esprima.Error as e:
            error = self._parse_js_error(e)
            return SyntaxResult(valid=False, errors=[error], language="javascript")

        except Exception as e:
            return SyntaxResult(
                valid=False,
                errors=[SyntaxError(1, 1, str(e))],
                language="javascript"
            )

    def _parse_js_error(self, e) -> SyntaxError:
        """Parse esprima error into our format."""
        message = str(e)
        suggestion = None

        # Extract line/column if available
        line = getattr(e, 'lineNumber', 1)
        column = getattr(e, 'column', 1)

        # Find matching fix suggestion
        for pattern, fix in self.JS_FIXES.items():
            if pattern.lower() in message.lower():
                suggestion = fix
                break

        return SyntaxError(
            line=line,
            column=column,
            message=message,
            suggestion=suggestion
        )

    def _basic_js_check(self, code: str) -> SyntaxResult:
        """Basic JavaScript bracket matching without esprima."""
        errors = []

        # Check bracket matching
        stack = []
        brackets = {'(': ')', '[': ']', '{': '}'}

        for i, char in enumerate(code):
            if char in brackets:
                stack.append((char, i))
            elif char in brackets.values():
                if not stack:
                    errors.append(SyntaxError(
                        line=code[:i].count('\n') + 1,
                        column=i - code[:i].rfind('\n'),
                        message=f"Unmatched '{char}'",
                        suggestion=f"Add a matching opening bracket"
                    ))
                else:
                    opener, _ = stack.pop()
                    if brackets[opener] != char:
                        errors.append(SyntaxError(
                            line=code[:i].count('\n') + 1,
                            column=i - code[:i].rfind('\n'),
                            message=f"Expected '{brackets[opener]}' but found '{char}'",
                            suggestion="Check bracket matching"
                        ))

        # Check for unclosed brackets
        for opener, pos in stack:
            errors.append(SyntaxError(
                line=code[:pos].count('\n') + 1,
                column=pos - code[:pos].rfind('\n'),
                message=f"Unclosed '{opener}'",
                suggestion=f"Add a matching '{brackets[opener]}'"
            ))

        return SyntaxResult(
            valid=len(errors) == 0,
            errors=errors,
            language="javascript"
        )

    def detect_language(self, code: str) -> str:
        """
        Auto-detect language from code patterns.

        Args:
            code: Code string

        Returns:
            Language name: "python", "javascript", or "unknown"
        """
        code_lower = code.lower()

        # Python indicators
        python_indicators = [
            'def ', 'import ', 'from ', 'class ',
            'if __name__', ':', 'elif ', 'async def',
            'print(', 'self.', 'None', 'True', 'False'
        ]
        python_score = sum(1 for p in python_indicators if p in code)

        # JavaScript indicators
        js_indicators = [
            'function ', 'const ', 'let ', 'var ',
            '=>', 'async ', 'await ', 'console.log',
            'null', 'undefined', 'true', 'false',
            'document.', 'window.'
        ]
        js_score = sum(1 for j in js_indicators if j in code_lower)

        if python_score > js_score:
            return "python"
        elif js_score > python_score:
            return "javascript"
        elif ':' in code and 'function' not in code_lower:
            return "python"
        elif ';' in code or '{' in code:
            return "javascript"
        else:
            return "unknown"

    def check(self, code: str, language: str = None) -> SyntaxResult:
        """
        Check syntax with auto-detection.

        Args:
            code: Code string
            language: Optional language override

        Returns:
            SyntaxResult
        """
        if language is None:
            language = self.detect_language(code)

        if language == "python":
            return self.check_python(code)
        elif language == "javascript":
            return self.check_javascript(code)
        else:
            return SyntaxResult(
                valid=True,
                errors=[],
                language="unknown"
            )

    def format_errors(self, result: SyntaxResult) -> str:
        """Format errors for display."""
        if result.valid:
            return f"✓ {result.language.title()} syntax OK"

        lines = [f"✗ {result.language.title()} syntax errors:"]
        for error in result.errors:
            lines.append(f"  Line {error.line}: {error.message}")
            if error.suggestion:
                lines.append(f"    → {error.suggestion}")

        return "\n".join(lines)


# Test
if __name__ == "__main__":
    print("=== Local Syntax Checker Test ===\n")

    checker = LocalSyntaxChecker()

    # Test Python - valid
    print("Test 1: Valid Python")
    code = "def hello():\n    print('world')"
    result = checker.check_python(code)
    print(checker.format_errors(result))
    print()

    # Test Python - invalid
    print("Test 2: Invalid Python (missing colon)")
    code = "def hello()\n    print('world')"
    result = checker.check_python(code)
    print(checker.format_errors(result))
    print()

    # Test JavaScript - valid
    print("Test 3: Valid JavaScript")
    code = "function hello() { console.log('world'); }"
    result = checker.check_javascript(code)
    print(checker.format_errors(result))
    print()

    # Test JavaScript - invalid
    print("Test 4: Invalid JavaScript (missing brace)")
    code = "function hello() { console.log('world');"
    result = checker.check_javascript(code)
    print(checker.format_errors(result))
    print()

    # Test auto-detection
    print("Test 5: Auto-detect language")
    codes = [
        "def main():\n    pass",
        "const x = 10;",
        "import numpy as np",
        "console.log('hello');",
    ]
    for code in codes:
        lang = checker.detect_language(code)
        print(f"  '{code[:30]}...' → {lang}")
