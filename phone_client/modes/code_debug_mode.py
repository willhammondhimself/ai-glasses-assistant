"""
Code Debug Mode - Local-first syntax checking with cloud explanations.

Flow:
1. OCR via Gemini Flash (~1s, $0.002)
2. Local syntax check with ast/esprima (~instant, $0)
3. If explanation needed → DeepSeek (~4s, $0.007)

Cost savings: 60-80% vs cloud-only (most errors are syntax)
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable, List

from ..api_clients import GeminiClient, DeepSeekClient
from ..local_engines import LocalSyntaxChecker, SyntaxResult, SyntaxError
from ..core import CostTracker
from ..hud.renderer import HUDRenderer

logger = logging.getLogger(__name__)


@dataclass
class DebugConfig:
    """Configuration for Code Debug Mode."""
    speak_errors: bool = True
    auto_explain: bool = False  # Auto-explain first error
    show_fix: bool = True
    cloud_explain: bool = True


@dataclass
class DebugResult:
    """Result from code debugging."""
    code: str
    language: str
    valid: bool
    errors: List[SyntaxError]
    explanation: Optional[str]
    fix: Optional[str]
    method: str  # "local_ast", "local_esprima", "deepseek"
    cost: float
    latency_ms: float


class CodeDebugMode:
    """
    Code debugging with local-first syntax checking.

    Features:
    - Auto language detection (Python, JavaScript)
    - Local syntax checking with ast (Python) and esprima (JS)
    - Error suggestions for common mistakes
    - DeepSeek explanations for complex errors
    - Cost tracking
    """

    def __init__(
        self,
        config: DebugConfig = None,
        wham=None,
        renderer: HUDRenderer = None,
        send_display: Callable[[str], Awaitable[bool]] = None,
        speak: Callable[[str], Awaitable[bool]] = None,
    ):
        self.config = config or DebugConfig()
        self.wham = wham
        self.renderer = renderer
        self.send_display = send_display or self._noop_send
        self.speak = speak or self._noop_speak

        # Components
        self.syntax_checker = LocalSyntaxChecker()
        self.gemini = GeminiClient()
        self.deepseek = DeepSeekClient()
        self.cost_tracker = CostTracker()

        # State
        self.is_active = False
        self.current_code: Optional[str] = None
        self.current_result: Optional[DebugResult] = None

    async def _noop_send(self, lua: str) -> bool:
        return True

    async def _noop_speak(self, text: str) -> bool:
        logger.info(f"[SPEAK] {text}")
        return True

    async def start(self):
        """Start code debug mode."""
        self.is_active = True
        self.cost_tracker.reset()

        greeting = "Debug mode active. Show me some code."
        if self.config.speak_errors:
            await self.speak(greeting)

        logger.info("Code debug mode started")

    async def stop(self) -> dict:
        """Stop debug mode and return summary."""
        self.is_active = False

        summary = {
            "code_checked": self.cost_tracker.hand_count,
            "total_cost": self.cost_tracker.session_cost,
            "local_checks": sum(1 for e in self.cost_tracker.entries if e.model == "local"),
            "cloud_explains": sum(1 for e in self.cost_tracker.entries if e.model != "local"),
        }

        logger.info(f"Code debug mode stopped: {summary}")
        return summary

    async def analyze(self, image_data: bytes) -> DebugResult:
        """
        Analyze image and check code for errors.

        Args:
            image_data: JPEG image bytes

        Returns:
            DebugResult with errors and suggestions
        """
        import time
        start = time.perf_counter()

        # Step 1: OCR via Gemini
        ocr_result = await self._extract_code(image_data)
        if not ocr_result or not ocr_result.code:
            return DebugResult(
                code="",
                language="unknown",
                valid=False,
                errors=[SyntaxError(1, 1, "Could not extract code from image")],
                explanation=None,
                fix=None,
                method="failed",
                cost=0.002,
                latency_ms=(time.perf_counter() - start) * 1000
            )

        self.cost_tracker.add("gemini_flash", context="code_ocr")
        self.current_code = ocr_result.code

        # Step 2: Local syntax check
        result = await self._check_syntax(ocr_result.code, ocr_result.language)
        self.cost_tracker.add("local", 0.0, context="syntax_check")
        self.cost_tracker.hand_count += 1

        latency = (time.perf_counter() - start) * 1000

        debug_result = DebugResult(
            code=ocr_result.code,
            language=result.language,
            valid=result.valid,
            errors=result.errors,
            explanation=None,
            fix=None,
            method=f"local_{result.language}",
            cost=0.002,
            latency_ms=latency
        )

        # Step 3: Auto-explain if configured and errors found
        if self.config.auto_explain and not result.valid and result.errors:
            explanation = await self.explain_error(0)
            debug_result.explanation = explanation
            debug_result.cost += 0.007

        await self._display_result(debug_result)
        self.current_result = debug_result
        return debug_result

    async def check_text(self, code: str, language: str = None) -> DebugResult:
        """
        Check code text for syntax errors.

        Args:
            code: Code string
            language: Optional language override

        Returns:
            DebugResult
        """
        import time
        start = time.perf_counter()

        self.current_code = code

        result = await self._check_syntax(code, language)
        self.cost_tracker.add("local", 0.0, context="syntax_check")
        self.cost_tracker.hand_count += 1

        debug_result = DebugResult(
            code=code,
            language=result.language,
            valid=result.valid,
            errors=result.errors,
            explanation=None,
            fix=None,
            method=f"local_{result.language}",
            cost=0,
            latency_ms=(time.perf_counter() - start) * 1000
        )

        await self._display_result(debug_result)
        self.current_result = debug_result
        return debug_result

    async def _extract_code(self, image_data: bytes):
        """Extract code from image via Gemini OCR."""
        if not self.gemini.is_available:
            logger.error("Gemini not available for OCR")
            return None

        return await self.gemini.extract_code(image_data)

    async def _check_syntax(self, code: str, language: str = None) -> SyntaxResult:
        """Check syntax locally."""
        if language is None:
            language = self.syntax_checker.detect_language(code)

        return self.syntax_checker.check(code, language)

    async def _display_result(self, result: DebugResult):
        """Display result on HUD and speak."""
        if result.valid:
            message = f"{result.language.title()} syntax OK"
            logger.info(message)
            if self.config.speak_errors:
                await self.speak("Code looks good. No syntax errors.")
        else:
            error = result.errors[0] if result.errors else None
            if error:
                message = f"Error on line {error.line}: {error.message}"
                logger.info(message)
                if self.config.speak_errors:
                    speech = f"Line {error.line}: {error.message}"
                    if error.suggestion:
                        speech += f". Try: {error.suggestion}"
                    await self.speak(speech)

    async def explain_error(self, error_index: int = 0) -> str:
        """
        Get detailed explanation via DeepSeek.

        Args:
            error_index: Which error to explain (0-based)

        Returns:
            Explanation string
        """
        if not self.current_code:
            return "No code loaded."

        if not self.current_result or self.current_result.valid:
            return "No errors to explain."

        if error_index >= len(self.current_result.errors):
            return "Error not found."

        error = self.current_result.errors[error_index]

        if not self.deepseek.is_available:
            return "Explanation service not available."

        explanation = await self.deepseek.explain_code(
            self.current_code,
            f"Line {error.line}: {error.message}"
        )
        self.cost_tracker.add("deepseek_v3.1", context="explain_error")

        return explanation

    async def suggest_fix(self) -> str:
        """
        Get suggested fix for current errors.

        Returns:
            Fixed code or explanation
        """
        if not self.current_code or not self.current_result:
            return "No code loaded."

        if self.current_result.valid:
            return "No errors to fix."

        if not self.deepseek.is_available:
            return "Fix service not available."

        prompt = f"""Fix this {self.current_result.language} code:

```
{self.current_code}
```

Errors:
{chr(10).join(f'Line {e.line}: {e.message}' for e in self.current_result.errors)}

Provide ONLY the corrected code, no explanations."""

        response = await self.deepseek.live_analysis(prompt)
        self.cost_tracker.add("deepseek_v3.1", context="suggest_fix")

        # Extract code from response
        if "```" in response:
            # Extract code block
            parts = response.split("```")
            if len(parts) >= 2:
                code_block = parts[1]
                # Remove language identifier if present
                lines = code_block.split("\n")
                if lines[0].strip() in ["python", "javascript", "js"]:
                    code_block = "\n".join(lines[1:])
                return code_block.strip()

        return response

    async def analyze_runtime_error(self, code: str, error: str) -> str:
        """
        Analyze a runtime error.

        Args:
            code: The code that produced the error
            error: The error message

        Returns:
            Explanation and fix suggestion
        """
        if not self.deepseek.is_available:
            return "Error analysis not available."

        explanation = await self.deepseek.explain_code(code, error)
        self.cost_tracker.add("deepseek_v3.1", context="runtime_error")

        return explanation

    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "code_checked": self.cost_tracker.hand_count,
            "total_cost": self.cost_tracker.session_cost,
            "cost_saved": self.cost_tracker.hand_count * 0.007 - self.cost_tracker.session_cost,
            "breakdown": self.cost_tracker.model_breakdown,
        }
