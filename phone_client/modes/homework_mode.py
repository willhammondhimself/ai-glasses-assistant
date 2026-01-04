"""
Homework Mode - Local-first math solving with cloud fallback.

Flow:
1. OCR via Gemini Flash (~1s, $0.002)
2. Try local SymPy solve (~instant, $0)
3. If complex → DeepSeek fallback (~4s, $0.007)

Cost savings: 70-80% vs cloud-only (most homework is solvable locally)
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable

from ..api_clients import GeminiClient, DeepSeekClient
from ..local_engines import LocalMathSolver, MathSolution
from ..core import CostTracker
from ..hud.renderer import HUDRenderer

logger = logging.getLogger(__name__)


@dataclass
class HomeworkConfig:
    """Configuration for Homework Mode."""
    speak_solution: bool = True
    show_steps: bool = True
    auto_advance: bool = True
    cloud_fallback: bool = True


@dataclass
class HomeworkSolution:
    """Solution from homework mode."""
    problem: str
    answer: str
    steps: list
    method: str  # "local_sympy", "deepseek", "failed"
    cost: float
    latency_ms: float


class HomeworkMode:
    """
    Homework assistance with local-first solving.

    Features:
    - Automatic problem type detection
    - Local SymPy solving for algebra, calculus, polynomials
    - DeepSeek fallback for word problems and proofs
    - Step-by-step solution display
    - Cost tracking
    """

    def __init__(
        self,
        config: HomeworkConfig = None,
        wham=None,
        renderer: HUDRenderer = None,
        send_display: Callable[[str], Awaitable[bool]] = None,
        speak: Callable[[str], Awaitable[bool]] = None,
    ):
        self.config = config or HomeworkConfig()
        self.wham = wham
        self.renderer = renderer
        self.send_display = send_display or self._noop_send
        self.speak = speak or self._noop_speak

        # Components
        self.local_solver = LocalMathSolver()
        self.gemini = GeminiClient()
        self.deepseek = DeepSeekClient()
        self.cost_tracker = CostTracker()

        # State
        self.is_active = False
        self.current_problem: Optional[str] = None
        self.current_solution: Optional[HomeworkSolution] = None

    async def _noop_send(self, lua: str) -> bool:
        return True

    async def _noop_speak(self, text: str) -> bool:
        logger.info(f"[SPEAK] {text}")
        return True

    async def start(self):
        """Start homework mode."""
        self.is_active = True
        self.cost_tracker.reset()

        greeting = "Homework mode active. Show me a problem."
        if self.config.speak_solution:
            await self.speak(greeting)

        logger.info("Homework mode started")

    async def stop(self) -> dict:
        """Stop homework mode and return summary."""
        self.is_active = False

        summary = {
            "problems_solved": self.cost_tracker.hand_count,
            "total_cost": self.cost_tracker.session_cost,
            "local_solves": sum(1 for e in self.cost_tracker.entries if e.model == "local"),
            "cloud_solves": sum(1 for e in self.cost_tracker.entries if e.model != "local"),
        }

        logger.info(f"Homework mode stopped: {summary}")
        return summary

    async def analyze(self, image_data: bytes) -> HomeworkSolution:
        """
        Analyze image and solve math problem.

        Args:
            image_data: JPEG image bytes

        Returns:
            HomeworkSolution with answer and steps
        """
        import time
        start = time.perf_counter()

        # Step 1: OCR via Gemini
        ocr_result = await self._extract_math(image_data)
        if not ocr_result or not ocr_result.equation:
            return HomeworkSolution(
                problem="Could not extract problem",
                answer="OCR failed",
                steps=[],
                method="failed",
                cost=0.002,
                latency_ms=(time.perf_counter() - start) * 1000
            )

        self.cost_tracker.add("gemini_flash", context="math_ocr")
        self.current_problem = ocr_result.equation

        # Step 2: Try local solve first
        solution = await self._solve_local(ocr_result.equation)

        if solution:
            # Solved locally - FREE!
            self.cost_tracker.add("local", 0.0, context="math_solve")
            self.cost_tracker.hand_count += 1

            latency = (time.perf_counter() - start) * 1000
            result = HomeworkSolution(
                problem=ocr_result.equation,
                answer=solution.answer,
                steps=solution.steps,
                method="local_sympy",
                cost=0.002,  # Only OCR cost
                latency_ms=latency
            )

            await self._display_solution(result)
            self.current_solution = result
            return result

        # Step 3: Fallback to DeepSeek
        if self.config.cloud_fallback:
            solution = await self._solve_cloud(ocr_result.equation)
            self.cost_tracker.add("deepseek_v3.1", context="math_solve")
            self.cost_tracker.hand_count += 1

            latency = (time.perf_counter() - start) * 1000
            result = HomeworkSolution(
                problem=ocr_result.equation,
                answer=solution.get("answer", ""),
                steps=solution.get("steps", []),
                method="deepseek",
                cost=0.009,  # OCR + DeepSeek
                latency_ms=latency
            )

            await self._display_solution(result)
            self.current_solution = result
            return result

        # No solution
        return HomeworkSolution(
            problem=ocr_result.equation,
            answer="Could not solve",
            steps=["Problem requires manual solving"],
            method="failed",
            cost=0.002,
            latency_ms=(time.perf_counter() - start) * 1000
        )

    async def solve_text(self, problem: str) -> HomeworkSolution:
        """
        Solve a text-based math problem.

        Args:
            problem: Math problem as text

        Returns:
            HomeworkSolution
        """
        import time
        start = time.perf_counter()

        self.current_problem = problem

        # Try local first
        solution = await self._solve_local(problem)

        if solution:
            self.cost_tracker.add("local", 0.0, context="math_solve")
            self.cost_tracker.hand_count += 1

            result = HomeworkSolution(
                problem=problem,
                answer=solution.answer,
                steps=solution.steps,
                method="local_sympy",
                cost=0,
                latency_ms=(time.perf_counter() - start) * 1000
            )

            await self._display_solution(result)
            return result

        # Cloud fallback
        if self.config.cloud_fallback:
            cloud_solution = await self._solve_cloud(problem)
            self.cost_tracker.add("deepseek_v3.1", context="math_solve")
            self.cost_tracker.hand_count += 1

            result = HomeworkSolution(
                problem=problem,
                answer=cloud_solution.get("answer", ""),
                steps=cloud_solution.get("steps", []),
                method="deepseek",
                cost=0.007,
                latency_ms=(time.perf_counter() - start) * 1000
            )

            await self._display_solution(result)
            return result

        return HomeworkSolution(
            problem=problem,
            answer="Could not solve",
            steps=[],
            method="failed",
            cost=0,
            latency_ms=(time.perf_counter() - start) * 1000
        )

    async def _extract_math(self, image_data: bytes):
        """Extract math from image via Gemini OCR."""
        if not self.gemini.is_available:
            logger.error("Gemini not available for OCR")
            return None

        return await self.gemini.extract_math(image_data)

    async def _solve_local(self, problem: str) -> Optional[MathSolution]:
        """Try to solve locally with SymPy."""
        if not self.local_solver.is_available:
            return None

        if not self.local_solver.can_solve_locally(problem):
            return None

        return await self.local_solver.solve(problem)

    async def _solve_cloud(self, problem: str) -> dict:
        """Solve using DeepSeek."""
        if not self.deepseek.is_available:
            return {"answer": "DeepSeek not available", "steps": []}

        response = await self.deepseek.solve_math(problem)

        # Parse response
        answer = ""
        steps = []

        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Answer:") or line.startswith("Final answer:"):
                answer = line.split(":", 1)[1].strip()
            elif line.startswith("Step") or line.startswith(("1.", "2.", "3.")):
                steps.append(line)

        if not answer and lines:
            # Use last non-empty line as answer
            answer = lines[-1] if lines[-1] else lines[-2] if len(lines) > 1 else ""

        return {"answer": answer, "steps": steps}

    async def _display_solution(self, solution: HomeworkSolution):
        """Display solution on HUD and speak it."""
        # Build display (placeholder - would use HUD renderer)
        display_text = f"Problem: {solution.problem}\n\nAnswer: {solution.answer}"
        if self.config.show_steps and solution.steps:
            display_text += "\n\nSteps:\n" + "\n".join(solution.steps[:3])

        logger.info(f"Solution: {solution.answer} (via {solution.method})")

        # Speak answer
        if self.config.speak_solution:
            speech = f"The answer is {solution.answer}"
            await self.speak(speech)

    async def get_hint(self) -> str:
        """Get a hint for the current problem."""
        if not self.current_problem:
            return "No problem loaded."

        if not self.deepseek.is_available:
            return "Hint service not available."

        prompt = f"""Give a brief hint (1-2 sentences) for this math problem:

{self.current_problem}

Don't solve it - just point in the right direction."""

        response = await self.deepseek.live_analysis(prompt)
        self.cost_tracker.add("deepseek_v3.1", context="hint")

        return response[:200]  # Truncate

    async def explain_step(self, step_index: int = 0) -> str:
        """Explain a specific step in more detail."""
        if not self.current_solution or not self.current_solution.steps:
            return "No solution to explain."

        if step_index >= len(self.current_solution.steps):
            return "Step not found."

        step = self.current_solution.steps[step_index]

        prompt = f"""Explain this math step in simple terms:

Problem: {self.current_problem}
Step: {step}

Explain why we do this step and what it accomplishes."""

        response = await self.deepseek.live_analysis(prompt)
        self.cost_tracker.add("deepseek_v3.1", context="explain")

        return response[:300]

    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "problems_solved": self.cost_tracker.hand_count,
            "total_cost": self.cost_tracker.session_cost,
            "cost_saved": self.cost_tracker.hand_count * 0.007 - self.cost_tracker.session_cost,
            "breakdown": self.cost_tracker.model_breakdown,
        }
