"""
WebSocket Handlers for AR Glasses Modes

Specialized handlers for different AR modes:
- Mental Math speed run
- Camera solve (future)
- Code debug (future)
"""

import asyncio
import time
import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from fastapi import WebSocket

from .manager import WebSocketHandler, ConnectionInfo, ws_manager
from backend.formatters.quant_formatter import (
    QuantSlideBuilder,
    format_problem_for_ws,
    format_result_for_ws,
)

logger = logging.getLogger(__name__)


@dataclass
class MentalMathSession:
    """Tracks state for a mental math session."""
    session_id: str
    difficulty: int = 2
    current_problem: Optional[dict] = None
    problem_start_time: Optional[float] = None
    correct_count: int = 0
    total_count: int = 0
    current_streak: int = 0
    best_streak: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)


class MentalMathHandler(WebSocketHandler):
    """
    Handler for mental math speed run mode.

    Protocol:
    Client sends:
    - {"action": "start", "difficulty": 1-5}  - Start/next problem
    - {"action": "answer", "answer": <value>} - Submit answer
    - {"action": "skip"}                       - Skip current problem
    - {"action": "stats"}                      - Get session stats
    - {"action": "end"}                        - End session

    Server sends:
    - {"type": "problem", "problem": "47 × 83", ...}
    - {"type": "result", "correct": true, "time_ms": 2340, ...}
    - {"type": "timer", "remaining_ms": 5000}
    - {"type": "stats", ...}
    """

    def __init__(self):
        super().__init__()
        self.sessions: Dict[str, MentalMathSession] = {}
        self.timer_tasks: Dict[str, asyncio.Task] = {}
        self.slide_builder = QuantSlideBuilder()
        self._engine = None  # Lazy load

    def _get_engine(self):
        """Lazy load the mental math engine."""
        if self._engine is None:
            try:
                from backend.quant.mental_math import MentalMathEngine
                self._engine = MentalMathEngine()
            except ImportError:
                logger.warning("MentalMathEngine not available, using mock")
                self._engine = MockMentalMathEngine()
        return self._engine

    async def handle_message(
        self,
        websocket: WebSocket,
        topic: str,
        data: dict,
        conn_info: ConnectionInfo
    ):
        """Handle incoming mental math messages."""
        action = data.get("action", "")
        session_id = topic.split(":")[-1] if ":" in topic else topic

        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = MentalMathSession(session_id=session_id)

        session = self.sessions[session_id]

        try:
            if action == "start":
                await self._handle_start(websocket, session, data)
            elif action == "answer":
                await self._handle_answer(websocket, session, data)
            elif action == "skip":
                await self._handle_skip(websocket, session)
            elif action == "stats":
                await self._send_stats(websocket, session)
            elif action == "end":
                await self._handle_end(websocket, session)
            elif action == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}"
                })
        except Exception as e:
            logger.error(f"Error handling action {action}: {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

    async def _handle_start(
        self,
        websocket: WebSocket,
        session: MentalMathSession,
        data: dict
    ):
        """Start a new problem."""
        # Update difficulty if provided
        if "difficulty" in data:
            session.difficulty = max(1, min(5, data["difficulty"]))

        # Cancel any existing timer
        await self._cancel_timer(session.session_id)

        # Generate new problem
        engine = self._get_engine()
        problem_data = engine.generate_problem(
            difficulty=session.difficulty,
            category=data.get("category", "arithmetic")
        )

        session.current_problem = problem_data
        session.problem_start_time = time.time()

        # Send problem to client
        ws_problem = format_problem_for_ws(
            problem=problem_data["problem"],
            difficulty=session.difficulty,
            category=problem_data.get("category", "arithmetic")
        )
        ws_problem["problem_id"] = problem_data.get("id", str(uuid.uuid4()))

        await websocket.send_json(ws_problem)

        # Start timer task
        time_target = self.slide_builder.TIME_TARGETS.get(session.difficulty, 15000)
        self.timer_tasks[session.session_id] = asyncio.create_task(
            self._timer_loop(websocket, session, time_target)
        )

    async def _handle_answer(
        self,
        websocket: WebSocket,
        session: MentalMathSession,
        data: dict
    ):
        """Handle answer submission."""
        if not session.current_problem:
            await websocket.send_json({
                "type": "error",
                "message": "No active problem"
            })
            return

        # Cancel timer
        await self._cancel_timer(session.session_id)

        # Calculate time taken
        time_ms = int((time.time() - session.problem_start_time) * 1000)

        # Check answer
        user_answer = data.get("answer")
        correct_answer = session.current_problem.get("answer")

        # Normalize answers for comparison
        correct = self._check_answer(user_answer, correct_answer)

        # Update session stats
        session.total_count += 1
        if correct:
            session.correct_count += 1
            session.current_streak += 1
            session.best_streak = max(session.best_streak, session.current_streak)
        else:
            session.current_streak = 0

        # Send result
        result = format_result_for_ws(
            correct=correct,
            user_answer=user_answer,
            correct_answer=correct_answer,
            time_ms=time_ms,
            streak=session.current_streak
        )
        result["best_streak"] = session.best_streak
        result["accuracy"] = (
            session.correct_count / session.total_count
            if session.total_count > 0 else 0
        )

        await websocket.send_json(result)

        # Clear current problem
        session.current_problem = None
        session.problem_start_time = None

    async def _handle_skip(self, websocket: WebSocket, session: MentalMathSession):
        """Handle skipping current problem."""
        await self._cancel_timer(session.session_id)

        if session.current_problem:
            # Skipping breaks streak
            session.current_streak = 0
            session.total_count += 1

            await websocket.send_json({
                "type": "skipped",
                "correct_answer": session.current_problem.get("answer"),
                "streak": 0
            })

            session.current_problem = None
            session.problem_start_time = None

    async def _send_stats(self, websocket: WebSocket, session: MentalMathSession):
        """Send session statistics."""
        await websocket.send_json({
            "type": "stats",
            "correct": session.correct_count,
            "total": session.total_count,
            "accuracy": (
                session.correct_count / session.total_count
                if session.total_count > 0 else 0
            ),
            "current_streak": session.current_streak,
            "best_streak": session.best_streak,
            "difficulty": session.difficulty,
            "duration_seconds": (datetime.utcnow() - session.started_at).total_seconds()
        })

    async def _handle_end(self, websocket: WebSocket, session: MentalMathSession):
        """End the session."""
        await self._cancel_timer(session.session_id)

        # Send final stats
        await websocket.send_json({
            "type": "session_end",
            "correct": session.correct_count,
            "total": session.total_count,
            "accuracy": (
                session.correct_count / session.total_count
                if session.total_count > 0 else 0
            ),
            "best_streak": session.best_streak,
            "duration_seconds": (datetime.utcnow() - session.started_at).total_seconds()
        })

        # Clean up session
        if session.session_id in self.sessions:
            del self.sessions[session.session_id]

    async def _timer_loop(
        self,
        websocket: WebSocket,
        session: MentalMathSession,
        time_target_ms: int
    ):
        """Background task to send timer updates."""
        start = time.time()
        update_interval = 1.0  # Send updates every second

        try:
            while True:
                elapsed = (time.time() - start) * 1000
                remaining = max(0, time_target_ms - elapsed)

                await websocket.send_json({
                    "type": "timer",
                    "remaining_ms": int(remaining),
                    "elapsed_ms": int(elapsed),
                    "target_ms": time_target_ms
                })

                if remaining <= 0:
                    # Time's up!
                    await websocket.send_json({
                        "type": "timeout",
                        "correct_answer": session.current_problem.get("answer")
                        if session.current_problem else None
                    })
                    # Reset streak on timeout
                    session.current_streak = 0
                    session.total_count += 1
                    session.current_problem = None
                    break

                await asyncio.sleep(update_interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Timer error: {e}")

    async def _cancel_timer(self, session_id: str):
        """Cancel the timer task for a session."""
        if session_id in self.timer_tasks:
            self.timer_tasks[session_id].cancel()
            try:
                await self.timer_tasks[session_id]
            except asyncio.CancelledError:
                pass
            del self.timer_tasks[session_id]

    def _check_answer(self, user_answer: Any, correct_answer: Any) -> bool:
        """Check if user's answer matches the correct answer."""
        try:
            # Convert both to floats for numeric comparison
            user_num = float(str(user_answer).replace(',', ''))
            correct_num = float(str(correct_answer).replace(',', ''))
            return abs(user_num - correct_num) < 0.001
        except (ValueError, TypeError):
            # Fall back to string comparison
            return str(user_answer).strip().lower() == str(correct_answer).strip().lower()

    async def on_disconnect(
        self,
        websocket: WebSocket,
        topic: str,
        conn_info: ConnectionInfo
    ):
        """Clean up on disconnect."""
        session_id = topic.split(":")[-1] if ":" in topic else topic
        await self._cancel_timer(session_id)
        if session_id in self.sessions:
            del self.sessions[session_id]


class MockMentalMathEngine:
    """Mock engine for testing when real engine not available."""

    import random

    def generate_problem(self, difficulty: int = 2, category: str = "arithmetic") -> dict:
        """Generate a simple arithmetic problem."""
        import random

        if difficulty <= 2:
            a = random.randint(10, 99)
            b = random.randint(2, 9)
            op = random.choice(['+', '-', '×'])
        else:
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            op = random.choice(['+', '-', '×'])

        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        else:
            answer = a * b

        return {
            "id": str(uuid.uuid4()),
            "problem": f"{a} {op} {b}",
            "answer": answer,
            "category": category,
            "difficulty": difficulty
        }


# Singleton handler instance
mental_math_handler = MentalMathHandler()
