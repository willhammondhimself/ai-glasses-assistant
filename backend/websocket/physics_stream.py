"""Physics stream WebSocket handler for real-time equation OCR and solving."""

import json
import asyncio
import logging
import time
from typing import Dict, Set, Optional, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

from backend.physics.engine import PhysicsEngine, PhysicsSolution

logger = logging.getLogger(__name__)


class PhysicsStreamHandler:
    """Handles WebSocket connections for real-time physics HUD streaming.

    Protocol:
    Client sends:
    - {"type": "solve", "problem": "integral x^2"} - Solve equation
    - {"type": "ocr_frame", "image": "<base64>"} - Send camera frame for OCR
    - {"type": "graph", "function": "x^2"} - Request function graph
    - {"type": "formula", "name": "kinetic_energy"} - Look up formula
    - {"type": "ping"} - Heartbeat

    Server sends:
    - {"type": "connected", "message": "Physics HUD ready"}
    - {"type": "solution", "problem": "...", "solution": "...", "latex": "...", "steps": [...]}
    - {"type": "ocr_result", "equation": "...", "boxes": [...]}
    - {"type": "graph", "image": "<base64>"}
    - {"type": "formula", "name": "...", "formula": "..."}
    - {"type": "latency", "ms": ...}
    - {"type": "error", "message": "..."}
    """

    def __init__(self):
        self.connections: Set[WebSocket] = set()
        self.engine = PhysicsEngine()
        self._ocr_available = False

        # Try to import OCR module
        try:
            from backend.vision.physics_ocr import physics_ocr
            self._ocr_module = physics_ocr
            self._ocr_available = True
        except ImportError:
            logger.warning("Physics OCR module not available")
            self._ocr_module = None

    async def handle_connection(self, websocket: WebSocket):
        """Handle a new WebSocket connection."""
        await websocket.accept()
        self.connections.add(websocket)

        try:
            # Send welcome message
            await websocket.send_json({
                "type": "connected",
                "message": "Physics HUD ready. Send equations or camera frames.",
                "ocr_available": self._ocr_available,
                "formulas_available": True
            })

            # Handle messages
            while True:
                try:
                    data = await websocket.receive_json()
                    await self._handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON"
                    })

        except WebSocketDisconnect:
            logger.info("Physics stream client disconnected")
        except Exception as e:
            logger.error(f"Physics stream error: {e}")
        finally:
            self.connections.discard(websocket)

    async def _handle_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type", "")
        timestamp = data.get("timestamp", time.time() * 1000)

        if msg_type == "solve":
            # Solve equation
            problem = data.get("problem", "")
            if problem:
                await self._solve_problem(websocket, problem, timestamp)
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "No problem provided"
                })

        elif msg_type == "ocr_frame":
            # Process camera frame for equation OCR
            image_b64 = data.get("image", "")
            if image_b64:
                await self._process_ocr_frame(websocket, image_b64, timestamp)
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "No image data in frame"
                })

        elif msg_type == "graph":
            # Generate function graph
            function = data.get("function", "")
            x_range = data.get("x_range", (-10, 10))
            if function:
                await self._generate_graph(websocket, function, x_range)
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "No function provided"
                })

        elif msg_type == "formula":
            # Look up physics formula
            name = data.get("name", "")
            category = data.get("category")
            await self._lookup_formula(websocket, name, category)

        elif msg_type == "list_formulas":
            # List available formulas
            category = data.get("category")
            formulas = self.engine.list_formulas(category)
            await websocket.send_json({
                "type": "formula_list",
                "formulas": formulas
            })

        elif msg_type == "ping":
            await websocket.send_json({"type": "pong"})

        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            })

    async def _solve_problem(self, websocket: WebSocket, problem: str, timestamp: float):
        """Solve a physics/math problem and send results."""
        start_time = time.time()

        try:
            # Solve using physics engine
            solution = await self.engine.solve(problem)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Send solution
            await websocket.send_json({
                "type": "solution",
                "problem": solution.problem,
                "problem_type": solution.problem_type.value,
                "solution": solution.solution,
                "latex": solution.solution_latex,
                "steps": solution.steps,
                "numeric_value": solution.numeric_value,
                "method": solution.method,
                "formula_used": solution.formula_used,
                "error": solution.error
            })

            # Send latency
            await websocket.send_json({
                "type": "latency",
                "ms": latency_ms
            })

        except Exception as e:
            logger.error(f"Physics solve error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

    async def _process_ocr_frame(self, websocket: WebSocket, image_b64: str, timestamp: float):
        """Process camera frame for equation detection."""
        if not self._ocr_available or not self._ocr_module:
            await websocket.send_json({
                "type": "error",
                "message": "OCR not available. Use manual input."
            })
            return

        start_time = time.time()

        try:
            # Run OCR
            result = await self._ocr_module.detect_equation(image_b64)

            if result.equation:
                # Send OCR result
                await websocket.send_json({
                    "type": "ocr_result",
                    "equation": result.equation,
                    "confidence": result.confidence,
                    "boxes": result.boxes
                })

                # Auto-solve if confidence is high enough
                if result.confidence >= 0.7:
                    await self._solve_problem(websocket, result.equation, timestamp)
            else:
                await websocket.send_json({
                    "type": "ocr_result",
                    "equation": None,
                    "message": "No equation detected"
                })

            # Send latency
            latency_ms = int((time.time() - start_time) * 1000)
            await websocket.send_json({
                "type": "latency",
                "ms": latency_ms
            })

        except Exception as e:
            logger.error(f"Physics OCR error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"OCR failed: {str(e)}"
            })

    async def _generate_graph(self, websocket: WebSocket, function: str, x_range: tuple):
        """Generate function graph and send as base64 image."""
        try:
            from backend.physics.grapher import plot_function

            # Generate plot
            image_bytes = plot_function(function, x_range)

            import base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            await websocket.send_json({
                "type": "graph",
                "function": function,
                "x_range": x_range,
                "image": image_b64
            })

        except ImportError:
            await websocket.send_json({
                "type": "error",
                "message": "Grapher not available"
            })
        except Exception as e:
            logger.error(f"Graph generation error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Graph failed: {str(e)}"
            })

    async def _lookup_formula(self, websocket: WebSocket, name: str, category: Optional[str] = None):
        """Look up a physics formula."""
        # Search by name
        if category:
            formula = self.engine.get_formula(category, name)
            if formula:
                await websocket.send_json({
                    "type": "formula",
                    "category": category,
                    "name": name,
                    "formula": formula
                })
                return

        # Search all categories
        for cat, formulas in self.engine.PHYSICS_FORMULAS.items():
            for fname, formula in formulas.items():
                if name.lower() in fname.lower() or fname.lower() in name.lower():
                    await websocket.send_json({
                        "type": "formula",
                        "category": cat,
                        "name": fname,
                        "formula": formula
                    })
                    return

        # Not found
        await websocket.send_json({
            "type": "error",
            "message": f"Formula '{name}' not found"
        })

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        disconnected = set()
        for websocket in self.connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)

        # Remove disconnected clients
        self.connections -= disconnected


# Global instance
physics_stream_handler = PhysicsStreamHandler()
