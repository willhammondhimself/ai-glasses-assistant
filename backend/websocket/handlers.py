"""
WebSocket Handlers for AR Glasses Modes

Specialized handlers for different AR modes:
- Mental Math speed run with JARVIS personality
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
from backend.jarvis import JarvisPersonality

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
    Handler for mental math speed run mode with JARVIS personality.

    Protocol:
    Client sends:
    - {"action": "start", "difficulty": 1-5}  - Start/next problem
    - {"action": "answer", "answer": <value>} - Submit answer
    - {"action": "skip"}                       - Skip current problem
    - {"action": "stats"}                      - Get session stats
    - {"action": "end"}                        - End session

    Server sends:
    - {"type": "problem", "problem": "47 × 83", ...}
    - {"type": "result", "correct": true, "time_ms": 2340, "jarvis": {...}}
    - {"type": "timer", "remaining_ms": 5000}
    - {"type": "stats", ...}
    """

    def __init__(self):
        super().__init__()
        self.sessions: Dict[str, MentalMathSession] = {}
        self.timer_tasks: Dict[str, asyncio.Task] = {}
        self.jarvis_instances: Dict[str, JarvisPersonality] = {}
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

        # Get or create session and JARVIS instance
        if session_id not in self.sessions:
            self.sessions[session_id] = MentalMathSession(session_id=session_id)
            self.jarvis_instances[session_id] = JarvisPersonality(
                user_name="Will",
                preferred_address="sir"
            )

        session = self.sessions[session_id]
        jarvis = self.jarvis_instances.get(session_id)

        try:
            if action == "start":
                await self._handle_start(websocket, session, jarvis, data)
            elif action == "answer":
                await self._handle_answer(websocket, session, jarvis, data)
            elif action == "skip":
                await self._handle_skip(websocket, session, jarvis)
            elif action == "stats":
                await self._send_stats(websocket, session, jarvis)
            elif action == "end":
                await self._handle_end(websocket, session, jarvis)
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
        jarvis: JarvisPersonality,
        data: dict
    ):
        """Start a new problem with JARVIS greeting on first problem."""
        # Update difficulty if provided
        if "difficulty" in data:
            session.difficulty = max(1, min(5, data["difficulty"]))
            if jarvis:
                jarvis.context.current_difficulty = session.difficulty

        # Cancel any existing timer
        await self._cancel_timer(session.session_id)

        # Send greeting on first problem
        if session.total_count == 0 and jarvis:
            greeting = jarvis.greet()
            mode_msg = jarvis.start_mode("mental_math")
            await websocket.send_json({
                "type": "jarvis",
                "message": greeting,
                "mode_activation": mode_msg,
            })

        # Generate new problem
        engine = self._get_engine()
        problem_data = engine.generate_problem(
            difficulty=session.difficulty,
            category=data.get("category", "arithmetic")
        )

        session.current_problem = problem_data
        session.problem_start_time = time.time()

        # Track in JARVIS context
        if jarvis:
            jarvis.context.start_problem(problem_data)

        # Send problem to client
        ws_problem = format_problem_for_ws(
            problem=problem_data["problem"],
            difficulty=session.difficulty,
            category=problem_data.get("category", "arithmetic")
        )
        ws_problem["problem_id"] = problem_data.get("id", str(uuid.uuid4()))
        ws_problem["problem_number"] = session.total_count + 1

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
        jarvis: JarvisPersonality,
        data: dict
    ):
        """Handle answer submission with JARVIS personality feedback."""
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

        # Get JARVIS feedback
        jarvis_response = None
        if jarvis:
            jarvis_response = jarvis.process_answer(
                user_answer=str(user_answer),
                correct_answer=str(correct_answer),
                difficulty=session.difficulty,
                category=session.current_problem.get("category", "mental_math")
            )

        # Send result with JARVIS enhancement
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

        # Add JARVIS personality data
        if jarvis_response:
            result["jarvis"] = {
                "feedback": jarvis_response.get("feedback"),
                "speed_tier": jarvis_response.get("speed_tier"),
                "streak_message": jarvis_response.get("streak_message"),
                "milestone_message": jarvis_response.get("milestone_message"),
                "suggestions": jarvis_response.get("suggestions", []),
            }

        await websocket.send_json(result)

        # Clear current problem
        session.current_problem = None
        session.problem_start_time = None

    async def _handle_skip(
        self,
        websocket: WebSocket,
        session: MentalMathSession,
        jarvis: JarvisPersonality
    ):
        """Handle skipping current problem."""
        await self._cancel_timer(session.session_id)

        if session.current_problem:
            # Skipping breaks streak
            session.current_streak = 0
            session.total_count += 1

            # Record in JARVIS as incorrect (skip = fail)
            if jarvis:
                jarvis.context.record_answer(
                    correct=False,
                    difficulty=session.difficulty,
                    category="mental_math"
                )

            await websocket.send_json({
                "type": "skipped",
                "correct_answer": session.current_problem.get("answer"),
                "streak": 0,
                "jarvis": {
                    "feedback": f"Skipped. The answer was {session.current_problem.get('answer')}. Onward, sir."
                }
            })

            session.current_problem = None
            session.problem_start_time = None

    async def _send_stats(
        self,
        websocket: WebSocket,
        session: MentalMathSession,
        jarvis: JarvisPersonality
    ):
        """Send session statistics with JARVIS insights."""
        stats_data = {
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
        }

        # Add JARVIS insights
        if jarvis:
            insights = jarvis.get_performance_insights()
            phase_message = jarvis.get_phase_aware_message()
            stats_data["jarvis"] = {
                "phase_message": phase_message,
                "insights": insights,
                "current_stats": jarvis.get_current_stats()
            }

        await websocket.send_json(stats_data)

    async def _handle_end(
        self,
        websocket: WebSocket,
        session: MentalMathSession,
        jarvis: JarvisPersonality
    ):
        """End the session with JARVIS summary."""
        await self._cancel_timer(session.session_id)

        # Get JARVIS session summary
        jarvis_summary = None
        if jarvis:
            jarvis_summary = jarvis.end_session()

        # Send final stats
        end_data = {
            "type": "session_end",
            "correct": session.correct_count,
            "total": session.total_count,
            "accuracy": (
                session.correct_count / session.total_count
                if session.total_count > 0 else 0
            ),
            "best_streak": session.best_streak,
            "duration_seconds": (datetime.utcnow() - session.started_at).total_seconds()
        }

        if jarvis_summary:
            end_data["jarvis"] = {
                "message": jarvis_summary.get("message"),
                "summary": jarvis_summary.get("summary"),
                "motivation": jarvis.get_jane_street_motivation()
            }

        await websocket.send_json(end_data)

        # Clean up session and JARVIS instance
        if session.session_id in self.sessions:
            del self.sessions[session.session_id]
        if session.session_id in self.jarvis_instances:
            del self.jarvis_instances[session.session_id]

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
        if session_id in self.jarvis_instances:
            del self.jarvis_instances[session_id]


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


class PokerHandler(WebSocketHandler):
    """
    Handler for Poker GTO analysis mode with JARVIS personality.

    Protocol:
    Client sends:
    - {"action": "analyze", "hand_info": {...}}    - Analyze hand
    - {"action": "pot_odds", "pot": X, "bet": Y}   - Calculate pot odds
    - {"action": "range", "position": "BTN", ...}  - Get preflop range
    - {"action": "sizing", "situation": {...}}     - Get bet sizing advice

    Server sends:
    - {"type": "analysis", "recommendation": ..., "jarvis": {...}}
    - {"type": "pot_odds", "odds": ..., "jarvis": {...}}
    - {"type": "range", "range": ..., "jarvis": {...}}
    """

    def __init__(self):
        super().__init__()
        self.jarvis_instances: Dict[str, JarvisPersonality] = {}
        self._engine = None

    def _get_engine(self):
        """Lazy load the poker engine."""
        if self._engine is None:
            try:
                from backend.poker.engine import PokerEngine
                self._engine = PokerEngine()
            except ImportError:
                logger.warning("PokerEngine not available")
                self._engine = None
        return self._engine

    async def handle_message(
        self,
        websocket: WebSocket,
        topic: str,
        data: dict,
        conn_info: ConnectionInfo
    ):
        """Handle incoming poker analysis messages."""
        action = data.get("action", "")
        session_id = topic.split(":")[-1] if ":" in topic else topic

        # Get or create JARVIS instance
        if session_id not in self.jarvis_instances:
            self.jarvis_instances[session_id] = JarvisPersonality(
                user_name="Will",
                preferred_address="sir"
            )
        jarvis = self.jarvis_instances[session_id]

        try:
            if action == "analyze":
                await self._handle_analyze(websocket, jarvis, data)
            elif action == "pot_odds":
                await self._handle_pot_odds(websocket, jarvis, data)
            elif action == "range":
                await self._handle_range(websocket, jarvis, data)
            elif action == "sizing":
                await self._handle_sizing(websocket, jarvis, data)
            elif action == "start":
                # Send JARVIS greeting for poker mode
                greeting = jarvis.greet()
                mode_msg = jarvis.start_mode("poker")
                await websocket.send_json({
                    "type": "jarvis",
                    "message": greeting,
                    "mode_activation": mode_msg,
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}"
                })
        except Exception as e:
            logger.error(f"Poker handler error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

    async def _handle_analyze(
        self,
        websocket: WebSocket,
        jarvis: JarvisPersonality,
        data: dict
    ):
        """Handle hand analysis request."""
        engine = self._get_engine()
        if not engine:
            await websocket.send_json({
                "type": "error",
                "message": "Poker engine not available"
            })
            return

        hand_info = data.get("hand_info", {})
        result = engine.analyze_hand(hand_info)

        # Add JARVIS commentary
        jarvis_comment = self._generate_poker_jarvis_comment(result)

        await websocket.send_json({
            "type": "analysis",
            "recommendation": result.get("recommendation"),
            "reasoning": result.get("reasoning"),
            "ev_estimate": result.get("ev_estimate"),
            "range_analysis": result.get("range_analysis"),
            "error": result.get("error"),
            "jarvis": {
                "message": jarvis_comment,
            }
        })

    async def _handle_pot_odds(
        self,
        websocket: WebSocket,
        jarvis: JarvisPersonality,
        data: dict
    ):
        """Handle pot odds calculation."""
        engine = self._get_engine()
        if not engine:
            await websocket.send_json({
                "type": "error",
                "message": "Poker engine not available"
            })
            return

        pot_size = data.get("pot", 0)
        bet_size = data.get("bet", 0)
        stack_size = data.get("stack")

        result = engine.calculate_pot_odds(pot_size, bet_size, stack_size)

        await websocket.send_json({
            "type": "pot_odds",
            **result,
            "jarvis": {
                "message": f"The math is clear, sir. {result.get('explanation', '')}",
            }
        })

    async def _handle_range(
        self,
        websocket: WebSocket,
        jarvis: JarvisPersonality,
        data: dict
    ):
        """Handle preflop range request."""
        engine = self._get_engine()
        if not engine:
            await websocket.send_json({
                "type": "error",
                "message": "Poker engine not available"
            })
            return

        position = data.get("position", "BTN")
        action = data.get("action_type", "RFI")
        game_type = data.get("game_type", "cash 6max")

        result = engine.get_preflop_range(position, action, game_type)

        await websocket.send_json({
            "type": "range",
            **result,
            "jarvis": {
                "message": f"GTO range for {position}, sir. {result.get('description', '')}",
            }
        })

    async def _handle_sizing(
        self,
        websocket: WebSocket,
        jarvis: JarvisPersonality,
        data: dict
    ):
        """Handle bet sizing analysis."""
        engine = self._get_engine()
        if not engine:
            await websocket.send_json({
                "type": "error",
                "message": "Poker engine not available"
            })
            return

        situation = data.get("situation", {})
        result = engine.analyze_bet_sizing(situation)

        await websocket.send_json({
            "type": "sizing",
            **result,
            "jarvis": {
                "message": f"Optimal sizing calculated, sir. {result.get('sizing_rationale', '')}",
            }
        })

    def _generate_poker_jarvis_comment(self, analysis: dict) -> str:
        """Generate JARVIS-style comment for poker analysis."""
        rec = analysis.get("recommendation", "").lower() if analysis.get("recommendation") else ""
        ev = analysis.get("ev_estimate", "").lower() if analysis.get("ev_estimate") else ""

        if analysis.get("error"):
            return "I'm having difficulty analyzing this hand, sir."

        if "+ev" in ev or "positive" in ev:
            if "raise" in rec or "bet" in rec:
                return "The numbers favor aggression here, sir. Apply pressure."
            elif "call" in rec:
                return "A profitable call, sir. The implied odds justify it."
            else:
                return "Positive expected value detected, sir."
        elif "-ev" in ev or "negative" in ev:
            if "fold" in rec:
                return "Discretion is the better part of valor here, sir. Fold."
            else:
                return "Marginal spot, sir. Proceed with caution."
        else:
            return "Analysis complete, sir. Execute accordingly."

    async def on_disconnect(
        self,
        websocket: WebSocket,
        topic: str,
        conn_info: ConnectionInfo
    ):
        """Clean up on disconnect."""
        session_id = topic.split(":")[-1] if ":" in topic else topic
        if session_id in self.jarvis_instances:
            del self.jarvis_instances[session_id]


class CodeDebugHandler(WebSocketHandler):
    """
    Handler for Code Debug mode with JARVIS personality.

    Protocol:
    Client sends:
    - {"action": "analyze", "code": "...", "language": "python"}
    - {"action": "explain", "code": "...", "language": "python"}
    - {"action": "fix", "error": "...", "code": "..."}

    Server sends:
    - {"type": "analysis", "issues": [...], "jarvis": {...}}
    - {"type": "explanation", "explanation": "...", "jarvis": {...}}
    - {"type": "fix", "suggestion": "...", "jarvis": {...}}
    """

    def __init__(self):
        super().__init__()
        self.jarvis_instances: Dict[str, JarvisPersonality] = {}

    async def handle_message(
        self,
        websocket: WebSocket,
        topic: str,
        data: dict,
        conn_info: ConnectionInfo
    ):
        """Handle incoming code debug messages."""
        action = data.get("action", "")
        session_id = topic.split(":")[-1] if ":" in topic else topic

        # Get or create JARVIS instance
        if session_id not in self.jarvis_instances:
            self.jarvis_instances[session_id] = JarvisPersonality(
                user_name="Will",
                preferred_address="sir"
            )
        jarvis = self.jarvis_instances[session_id]

        try:
            if action == "start":
                greeting = jarvis.greet()
                mode_msg = jarvis.start_mode("debug")
                await websocket.send_json({
                    "type": "jarvis",
                    "message": greeting,
                    "mode_activation": mode_msg,
                })
            elif action == "analyze":
                await self._handle_analyze(websocket, jarvis, data)
            elif action == "explain":
                await self._handle_explain(websocket, jarvis, data)
            elif action == "fix":
                await self._handle_fix(websocket, jarvis, data)
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}"
                })
        except Exception as e:
            logger.error(f"Code debug handler error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

    async def _handle_analyze(
        self,
        websocket: WebSocket,
        jarvis: JarvisPersonality,
        data: dict
    ):
        """Analyze code for issues."""
        code = data.get("code", "")
        language = data.get("language", "python")

        # Quick static analysis
        issues = self._quick_analyze(code, language)

        await websocket.send_json({
            "type": "analysis",
            "issues": issues,
            "language": language,
            "jarvis": {
                "message": self._get_analysis_comment(issues),
            }
        })

    async def _handle_explain(
        self,
        websocket: WebSocket,
        jarvis: JarvisPersonality,
        data: dict
    ):
        """Explain code functionality."""
        code = data.get("code", "")
        language = data.get("language", "python")

        # This would call an LLM in production
        await websocket.send_json({
            "type": "explanation",
            "code": code,
            "language": language,
            "jarvis": {
                "message": "Code analysis in progress, sir. Point at the specific section you'd like explained.",
            }
        })

    async def _handle_fix(
        self,
        websocket: WebSocket,
        jarvis: JarvisPersonality,
        data: dict
    ):
        """Suggest fix for error."""
        error = data.get("error", "")
        code = data.get("code", "")

        # Quick error pattern matching
        suggestion = self._suggest_fix(error, code)

        await websocket.send_json({
            "type": "fix",
            "error": error,
            "suggestion": suggestion,
            "jarvis": {
                "message": f"I've identified the issue, sir. {suggestion.get('brief', '')}",
            }
        })

    def _quick_analyze(self, code: str, language: str) -> list:
        """Quick static analysis without external tools."""
        issues = []

        # Common Python issues
        if language == "python":
            if "except:" in code and "except Exception" not in code:
                issues.append({
                    "type": "warning",
                    "message": "Bare except clause - catches all exceptions including KeyboardInterrupt",
                    "suggestion": "Use 'except Exception:' instead"
                })
            if "import *" in code:
                issues.append({
                    "type": "warning",
                    "message": "Wildcard import detected",
                    "suggestion": "Import specific names instead"
                })
            if "eval(" in code:
                issues.append({
                    "type": "security",
                    "message": "eval() is a security risk",
                    "suggestion": "Use ast.literal_eval() for safe evaluation"
                })

        # Common JavaScript issues
        if language in ["javascript", "typescript"]:
            if "var " in code:
                issues.append({
                    "type": "style",
                    "message": "Use of 'var' keyword",
                    "suggestion": "Use 'let' or 'const' instead"
                })
            if "== " in code and "=== " not in code:
                issues.append({
                    "type": "warning",
                    "message": "Loose equality comparison",
                    "suggestion": "Use strict equality '===' instead"
                })

        return issues

    def _suggest_fix(self, error: str, code: str) -> dict:
        """Suggest fix based on error pattern."""
        error_lower = error.lower()

        if "undefined" in error_lower or "is not defined" in error_lower:
            return {
                "brief": "Variable not defined before use.",
                "detailed": "Ensure the variable is declared before accessing it.",
                "common_cause": "Typo in variable name or missing import/declaration"
            }
        elif "typeerror" in error_lower and "null" in error_lower:
            return {
                "brief": "Null reference error.",
                "detailed": "You're trying to access a property on null or undefined.",
                "common_cause": "Missing null check or async timing issue"
            }
        elif "syntaxerror" in error_lower:
            return {
                "brief": "Syntax error detected.",
                "detailed": "Check for missing brackets, quotes, or semicolons.",
                "common_cause": "Unclosed bracket or string"
            }
        elif "indentationerror" in error_lower:
            return {
                "brief": "Python indentation error.",
                "detailed": "Check that indentation is consistent (spaces vs tabs).",
                "common_cause": "Mixed tabs and spaces"
            }
        else:
            return {
                "brief": "Error detected.",
                "detailed": "Review the error message and stack trace.",
                "common_cause": "Various possible causes"
            }

    def _get_analysis_comment(self, issues: list) -> str:
        """Generate JARVIS comment for analysis results."""
        if not issues:
            return "No obvious issues detected, sir. The code appears clean."

        security_issues = [i for i in issues if i.get("type") == "security"]
        if security_issues:
            return f"Security concern detected, sir. {security_issues[0].get('message', '')}"

        warnings = [i for i in issues if i.get("type") == "warning"]
        if warnings:
            return f"I've found {len(issues)} potential issues, sir. Starting with: {warnings[0].get('message', '')}"

        return f"Analysis complete. {len(issues)} items flagged for review, sir."

    async def on_disconnect(
        self,
        websocket: WebSocket,
        topic: str,
        conn_info: ConnectionInfo
    ):
        """Clean up on disconnect."""
        session_id = topic.split(":")[-1] if ":" in topic else topic
        if session_id in self.jarvis_instances:
            del self.jarvis_instances[session_id]


# Singleton handler instances
poker_handler = PokerHandler()
code_debug_handler = CodeDebugHandler()
