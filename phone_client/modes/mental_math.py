"""
Mental Math Drill Mode.
High-speed arithmetic training with millisecond timing.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
from enum import Enum

from ..local_engines.mental_math import MentalMathEngine, MathProblem, AnswerResult
from ..wham.personality import WHAMPersonality
from ..hud.renderer import HUDRenderer


class DrillState(Enum):
    """Current state of the drill."""
    IDLE = "idle"
    SHOWING_PROBLEM = "showing_problem"
    AWAITING_ANSWER = "awaiting_answer"
    SHOWING_RESULT = "showing_result"
    PAUSED = "paused"
    ENDED = "ended"


@dataclass
class DrillConfig:
    """Configuration for mental math drill."""
    difficulty: int = 2
    problem_count: Optional[int] = None  # None = infinite
    show_result_ms: int = 1500           # How long to show result
    auto_advance: bool = True            # Auto-advance to next problem
    speak_feedback: bool = True          # TTS feedback


class MentalMathMode:
    """
    Mental Math Drill Mode.

    Runs a high-speed mental math training session with:
    - Instant problem generation (local, no API)
    - Millisecond-precision timing
    - WHAM personality feedback
    - Real-time HUD updates
    """

    def __init__(
        self,
        config: DrillConfig,
        wham: WHAMPersonality,
        renderer: HUDRenderer,
        send_display: Callable[[str], Awaitable[bool]],
        speak: Callable[[str], Awaitable[bool]],
    ):
        self.config = config
        self.wham = wham
        self.renderer = renderer
        self.send_display = send_display
        self.speak = speak

        self.engine = MentalMathEngine(difficulty=config.difficulty)
        self.state = DrillState.IDLE

        self._current_problem: Optional[MathProblem] = None
        self._problem_start_time: float = 0
        self._problems_completed: int = 0
        self._timer_task: Optional[asyncio.Task] = None

    async def start(self) -> str:
        """
        Start the mental math drill.
        Returns the activation message.
        """
        self.wham.reset_session()
        self.state = DrillState.IDLE
        self._problems_completed = 0

        # Activation message
        message = self.wham.get_mode_activation("mental_math", self.config.difficulty)

        if self.config.speak_feedback:
            await self.speak(message)

        # Show first problem
        await self._next_problem()

        return message

    async def stop(self) -> dict:
        """
        Stop the drill and return session summary.
        """
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

        self.state = DrillState.ENDED

        # Get session summary
        summary = self.wham.get_session_summary()

        # Show summary on HUD
        lua = self.renderer.render_session_summary(
            total=summary["total"],
            correct=summary["correct"],
            accuracy=summary["accuracy"],
            avg_time_ms=summary["avg_time_ms"],
            best_streak=summary["best_streak"],
            grade=summary["grade"]
        )
        await self.send_display(lua)

        # Speak summary
        if self.config.speak_feedback:
            await self.speak(summary["commentary"])

        return summary

    async def pause(self):
        """Pause the drill."""
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

        self.state = DrillState.PAUSED

    async def resume(self):
        """Resume the drill."""
        if self.state == DrillState.PAUSED:
            await self._next_problem()

    async def submit_answer(self, answer_text: str) -> dict:
        """
        Submit an answer to the current problem.

        Args:
            answer_text: The answer (spoken or typed)

        Returns:
            Result dict with feedback
        """
        if self.state != DrillState.AWAITING_ANSWER or self._current_problem is None:
            return {"error": "No active problem"}

        # Stop timer
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

        # Calculate time
        elapsed_ms = (time.perf_counter() - self._problem_start_time) * 1000

        # Parse answer
        parsed_answer = self.engine.parse_spoken_answer(answer_text)
        if parsed_answer is None:
            # Couldn't parse - treat as wrong
            return await self._handle_parse_error(answer_text, elapsed_ms)

        # Check answer
        result = self.engine.check_answer(
            self._current_problem,
            parsed_answer,
            elapsed_ms
        )

        # Get WHAM feedback
        feedback = self.wham.get_full_feedback(
            correct=result.correct,
            time_ms=elapsed_ms,
            target_ms=self._current_problem.time_target_ms,
            expected_answer=self._current_problem.answer if not result.correct else None
        )

        # Update display
        self.state = DrillState.SHOWING_RESULT
        lua = self.renderer.render_result(
            correct=result.correct,
            time_ms=elapsed_ms,
            feedback=feedback,
            streak=self.wham.current_streak,
            answer=self._current_problem.answer if not result.correct else None
        )
        await self.send_display(lua)

        # Speak feedback
        if self.config.speak_feedback:
            await self.speak(feedback)

        self._problems_completed += 1

        # Auto-advance to next problem
        if self.config.auto_advance:
            await asyncio.sleep(self.config.show_result_ms / 1000)

            # Check if we should continue
            if self.config.problem_count and self._problems_completed >= self.config.problem_count:
                await self.stop()
            else:
                await self._next_problem()

        return {
            "correct": result.correct,
            "time_ms": elapsed_ms,
            "expected": result.expected,
            "given": result.given,
            "feedback": feedback,
            "streak": self.wham.current_streak,
            "within_target": result.within_target,
        }

    async def _handle_parse_error(self, answer_text: str, elapsed_ms: float) -> dict:
        """Handle unparseable answer."""
        feedback = f"Couldn't understand '{answer_text}'. Skipping."

        self.wham.stats.record_incorrect()

        self.state = DrillState.SHOWING_RESULT
        lua = self.renderer.render_result(
            correct=False,
            time_ms=elapsed_ms,
            feedback=feedback,
            streak=0,
            answer=self._current_problem.answer
        )
        await self.send_display(lua)

        if self.config.speak_feedback:
            await self.speak(feedback)

        if self.config.auto_advance:
            await asyncio.sleep(self.config.show_result_ms / 1000)
            await self._next_problem()

        return {
            "correct": False,
            "time_ms": elapsed_ms,
            "parse_error": True,
            "feedback": feedback,
        }

    async def _next_problem(self):
        """Generate and display the next problem."""
        # Generate problem
        self._current_problem = self.engine.generate_problem(
            difficulty=self.config.difficulty
        )

        # Reset timer
        self._problem_start_time = time.perf_counter()
        self.state = DrillState.AWAITING_ANSWER

        # Show on HUD
        lua = self.renderer.render_math_problem(
            problem_text=self._current_problem.problem_text,
            difficulty=self.config.difficulty,
            elapsed_ms=0,
            target_ms=self._current_problem.time_target_ms,
            streak=self.wham.current_streak
        )
        await self.send_display(lua)

        # Start timer update task
        self._timer_task = asyncio.create_task(self._update_timer())

    async def _update_timer(self):
        """Continuously update the timer display."""
        try:
            while self.state == DrillState.AWAITING_ANSWER:
                elapsed_ms = (time.perf_counter() - self._problem_start_time) * 1000

                # Update display
                lua = self.renderer.render_math_problem(
                    problem_text=self._current_problem.problem_text,
                    difficulty=self.config.difficulty,
                    elapsed_ms=elapsed_ms,
                    target_ms=self._current_problem.time_target_ms,
                    streak=self.wham.current_streak
                )
                await self.send_display(lua)

                # Update every 100ms for smooth timer
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            pass

    async def skip_problem(self):
        """Skip the current problem (counts as wrong)."""
        if self.state != DrillState.AWAITING_ANSWER:
            return

        elapsed_ms = (time.perf_counter() - self._problem_start_time) * 1000
        self.wham.stats.record_incorrect()

        feedback = f"Skipped. Answer was {self._current_problem.answer:g}."

        self.state = DrillState.SHOWING_RESULT
        lua = self.renderer.render_result(
            correct=False,
            time_ms=elapsed_ms,
            feedback=feedback,
            streak=0,
            answer=self._current_problem.answer
        )
        await self.send_display(lua)

        if self.config.auto_advance:
            await asyncio.sleep(self.config.show_result_ms / 1000)
            await self._next_problem()

    def set_difficulty(self, difficulty: int):
        """Change difficulty level."""
        self.config.difficulty = max(1, min(5, difficulty))
        self.engine.difficulty = self.config.difficulty

    @property
    def is_active(self) -> bool:
        """Check if drill is currently active."""
        return self.state in [DrillState.AWAITING_ANSWER, DrillState.SHOWING_PROBLEM]

    @property
    def current_problem_text(self) -> Optional[str]:
        """Get current problem text."""
        if self._current_problem:
            return self._current_problem.problem_text
        return None


# Standalone test
async def test_drill():
    """Test the mental math mode."""
    print("=== Mental Math Mode Test ===\n")

    # Mock functions
    async def mock_send(lua: str) -> bool:
        print(f"[DISPLAY] {len(lua)} chars")
        return True

    async def mock_speak(text: str) -> bool:
        print(f"[SPEAK] {text}")
        return True

    wham = WHAMPersonality()
    renderer = HUDRenderer()
    config = DrillConfig(difficulty=2, problem_count=3, auto_advance=False)

    mode = MentalMathMode(
        config=config,
        wham=wham,
        renderer=renderer,
        send_display=mock_send,
        speak=mock_speak
    )

    # Start
    msg = await mode.start()
    print(f"\nActivation: {msg}\n")

    # Simulate answering problems
    for i in range(3):
        print(f"\n--- Problem {i+1}: {mode.current_problem_text} ---")
        await asyncio.sleep(0.5)

        # Submit correct answer (cheating for test)
        answer = str(int(mode._current_problem.answer))
        result = await mode.submit_answer(answer)
        print(f"Result: {result}")

        if config.auto_advance:
            await asyncio.sleep(0.1)
        else:
            await mode._next_problem()

    # End
    summary = await mode.stop()
    print(f"\n=== Session Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(test_drill())
