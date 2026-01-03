"""
Mental Math Mode

Speed run mode for mental math practice.
Connects to backend via WebSocket for real-time problem delivery.
"""

import asyncio
import logging
import uuid
from typing import Optional, Dict, Any

from glasses_client.core.state import AppMode
from glasses_client.core.display import HALO_COLORS
from glasses_client.input.voice import VoiceCommand
from .base import BaseMode

logger = logging.getLogger(__name__)


class MentalMathMode(BaseMode):
    """
    Mental math speed run mode.

    Features:
    - Real-time problem generation via WebSocket
    - Voice input for answers
    - Timer with color-coded urgency
    - Streak tracking
    - Session statistics
    """

    mode_type = AppMode.MENTAL_MATH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_id = str(uuid.uuid4())[:8]
        self._current_problem: Optional[Dict[str, Any]] = None
        self._difficulty = 2
        self._streak = 0
        self._best_streak = 0
        self._problems_attempted = 0
        self._problems_correct = 0

    async def run(
        self,
        difficulty: int = 2,
        category: str = "arithmetic"
    ):
        """
        Run the mental math mode.

        Args:
            difficulty: Starting difficulty (1-5)
            category: Problem category
        """
        self._difficulty = difficulty

        # Connect to WebSocket
        ws_path = f"/ws/mental-math/{self._session_id}"
        ws = await self.connect_websocket(ws_path)

        # Register message handlers
        self._setup_handlers(ws)

        # Show welcome
        await self._show_welcome()

        # Main loop
        while self._running:
            # Wait for start command
            command = await self._wait_for_start()

            if command == "exit":
                break

            # Request problem
            await ws.start_problem(self._difficulty, category)

            # Wait for problem
            problem_data = await self._wait_for_message("problem", timeout=5.0)

            if not problem_data:
                await self.display.show_error("Failed to get problem")
                continue

            self._current_problem = problem_data

            # Display problem
            await self._show_problem(problem_data)

            # Wait for answer (with timer updates)
            answer = await self._wait_for_answer(
                time_target_ms=problem_data.get("time_target_ms", 15000)
            )

            if answer is None:
                # Timed out or skipped
                if not self._running:
                    break
                continue

            # Submit answer
            await ws.submit_answer(answer)

            # Wait for result
            result = await self._wait_for_message("result", timeout=5.0)

            if result:
                await self._show_result(result)

                # Update stats
                self._update_stats(result)

                # Brief pause before next
                await asyncio.sleep(1.5)

        # Show session summary
        await self._show_summary()

    def _setup_handlers(self, ws):
        """Set up WebSocket message handlers."""

        @ws.on_message("timer")
        async def handle_timer(data):
            if self._current_problem:
                await self.display.show_timer(
                    self._current_problem.get("problem", ""),
                    data.get("remaining_ms", 0),
                    data.get("target_ms", 15000)
                )

        @ws.on_message("timeout")
        async def handle_timeout(data):
            await self.display.show_result(
                correct=False,
                answer=str(data.get("correct_answer", "?")),
                time_ms=data.get("target_ms", 15000),
                streak=0
            )
            self._streak = 0
            self._problems_attempted += 1

        @ws.on_message("error")
        async def handle_error(data):
            logger.error(f"WebSocket error: {data.get('message')}")
            await self.display.show_error(data.get("message", "Unknown error"))

    async def _wait_for_message(
        self,
        message_type: str,
        timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """Wait for a specific message type."""
        if not self._ws:
            return None

        start = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start) < timeout:
            msg = await self._ws.receive(timeout=0.1)

            if msg and msg.get("type") == message_type:
                return msg

        return None

    async def _show_welcome(self):
        """Show welcome screen."""
        from glasses_client.core.display import DisplaySlide

        slide = DisplaySlide(
            lines=[
                "[MENTAL MATH]",
                f"Difficulty: D{self._difficulty}",
                "",
                "Say 'start' to begin"
            ],
            color=HALO_COLORS["math"],
            centered=True
        )
        await self.display.render_slide(slide)

    async def _wait_for_start(self) -> str:
        """Wait for start command."""
        while self._running:
            result = await self.voice.listen(timeout_ms=10000)

            if result:
                if result.command in [VoiceCommand.START, VoiceCommand.NEXT]:
                    return "start"
                elif result.command == VoiceCommand.EXIT:
                    return "exit"

            # Check for tap (single tap = start)
            tap = await self.tap.wait_for_tap(timeout_ms=100)
            if tap:
                return "start"

        return "exit"

    async def _show_problem(self, problem_data: Dict[str, Any]):
        """Display a problem."""
        await self.display.show_problem(
            problem=problem_data.get("problem", "?"),
            difficulty=self._difficulty,
            category=problem_data.get("category", "MATH").upper()
        )

    async def _wait_for_answer(self, time_target_ms: int) -> Optional[Any]:
        """
        Wait for user's answer with timer.

        Returns:
            The answer value, or None if skipped/timeout
        """
        # Start listening for voice answer
        start_time = asyncio.get_event_loop().time()
        timeout_seconds = time_target_ms / 1000

        while self._running:
            remaining_time = timeout_seconds - (asyncio.get_event_loop().time() - start_time)

            if remaining_time <= 0:
                return None

            # Listen for voice (short timeout)
            result = await self.voice.listen(timeout_ms=min(1000, int(remaining_time * 1000)))

            if result:
                # Check for number
                if result.command == VoiceCommand.NUMBER:
                    return result.number_value

                # Check for skip
                if result.command == VoiceCommand.SKIP:
                    return None

                # Check for exit
                if result.command == VoiceCommand.EXIT:
                    self._running = False
                    return None

                # Try to parse raw text as number
                number = self.voice._parse_number(result.raw_text)
                if number is not None:
                    return number

        return None

    async def _show_result(self, result: Dict[str, Any]):
        """Display the result."""
        correct = result.get("correct", False)
        await self.display.show_result(
            correct=correct,
            answer=str(result.get("correct_answer", "?")),
            time_ms=result.get("time_ms", 0),
            streak=result.get("streak", 0)
        )

    def _update_stats(self, result: Dict[str, Any]):
        """Update session statistics."""
        self._problems_attempted += 1

        if result.get("correct"):
            self._problems_correct += 1
            self._streak = result.get("streak", self._streak + 1)
            self._best_streak = max(self._best_streak, self._streak)
        else:
            self._streak = 0

        # Record in state
        self.state.record_attempt(
            correct=result.get("correct", False),
            time_ms=result.get("time_ms", 0)
        )

    async def _show_summary(self):
        """Show session summary."""
        from glasses_client.core.display import DisplaySlide

        accuracy = (
            self._problems_correct / self._problems_attempted * 100
            if self._problems_attempted > 0 else 0
        )

        slide = DisplaySlide(
            lines=[
                "Session Complete",
                f"Score: {self._problems_correct}/{self._problems_attempted}",
                f"Accuracy: {accuracy:.0f}%",
                f"Best Streak: {self._best_streak}"
            ],
            color=HALO_COLORS["result_green"] if accuracy >= 70 else HALO_COLORS["info"],
            centered=True
        )
        await self.display.render_slide(slide)

        # Wait for dismiss
        await asyncio.sleep(3)
