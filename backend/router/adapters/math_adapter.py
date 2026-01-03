"""
Math Engine Adapter

Wraps the existing MathEngine for use with the intelligent router.
"""

import re
import time
from typing import List

from .base import IEngineAdapter, EngineResult, ClaudeAdapter

# Import the existing math engine
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from math_solver.engine import MathEngine


class MathEngineAdapter(IEngineAdapter):
    """
    Adapter for the MathEngine.

    Provides unified interface for math problem solving with:
    - SymPy for local equation solving
    - Claude fallback for word problems
    """

    # Keywords that indicate math problems
    MATH_INDICATORS = [
        "=", "+", "-", "*", "/", "^",
        "solve", "calculate", "evaluate", "simplify",
        "derivative", "integral", "factor", "expand",
        "equation", "expression", "formula"
    ]

    def __init__(self, engine: MathEngine = None):
        """
        Initialize adapter.

        Args:
            engine: Existing MathEngine instance (creates new if not provided)
        """
        self.engine = engine or MathEngine()

    @property
    def name(self) -> str:
        return "math"

    @property
    def capabilities(self) -> List[str]:
        return ["solve", "calculus", "algebra", "simplify", "evaluate"]

    @property
    def is_local(self) -> bool:
        # MathEngine tries SymPy first, so it CAN be local
        # But it also falls back to Claude
        return True  # Report as local since it tries local first

    @property
    def priority(self) -> int:
        return 10  # High priority - try math engine early

    def can_handle(self, problem: str, **kwargs) -> float:
        """
        Determine confidence for handling this problem.

        Args:
            problem: The problem text

        Returns:
            Confidence score 0.0-1.0
        """
        problem_lower = problem.lower()
        score = 0.0

        # Check for math indicators
        for indicator in self.MATH_INDICATORS:
            if indicator in problem_lower:
                score += 0.15

        # Check for numeric patterns
        if re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', problem):
            score += 0.3

        # Check for variable patterns (x, y, z with operators)
        if re.search(r'[xyz]\s*[\+\-\*\/\^=]\s*', problem_lower):
            score += 0.2

        # Check for equation pattern
        if '=' in problem:
            score += 0.2

        # Cap at 1.0
        return min(1.0, score)

    async def solve(self, problem: str, **kwargs) -> EngineResult:
        """
        Solve a math problem.

        Args:
            problem: The math problem to solve

        Returns:
            EngineResult with solution
        """
        start = time.time()

        try:
            # Call the existing MathEngine
            result = self.engine.solve(problem)

            latency = self._measure_time(start)

            if result.get("error"):
                return EngineResult(
                    success=False,
                    data=result,
                    method="error",
                    latency_ms=latency
                )

            # Determine if it used local (SymPy) or Claude
            method = result.get("method", "local")
            is_local = method == "sympy"

            return EngineResult(
                success=True,
                data=result,
                method=method,
                tokens_used=0 if is_local else 500,  # Estimate for Claude
                latency_ms=latency,
                cost=0.0 if is_local else 0.001,  # Rough estimate
                metadata={
                    "steps": result.get("steps"),
                    "local": is_local
                }
            )

        except Exception as e:
            return EngineResult(
                success=False,
                data={"error": str(e)},
                method="error",
                latency_ms=self._measure_time(start)
            )


class SymPyOnlyAdapter(IEngineAdapter):
    """
    Adapter that ONLY uses SymPy (no Claude fallback).
    Useful when you want to guarantee free/local execution.
    """

    def __init__(self, engine: MathEngine = None):
        self.engine = engine or MathEngine()

    @property
    def name(self) -> str:
        return "sympy"

    @property
    def capabilities(self) -> List[str]:
        return ["solve", "algebra", "simplify", "evaluate"]

    @property
    def is_local(self) -> bool:
        return True

    @property
    def priority(self) -> int:
        return 5  # Very high priority - try first

    def can_handle(self, problem: str, **kwargs) -> float:
        """Only handle pure equations that SymPy can parse."""
        # Use the engine's detection
        if self.engine._is_pure_equation(problem):
            return 0.8
        return 0.0  # Don't handle word problems

    async def solve(self, problem: str, **kwargs) -> EngineResult:
        """Solve using SymPy only."""
        start = time.time()

        try:
            # Directly call SymPy methods
            result = self.engine._try_sympy(problem)

            if result is None:
                return EngineResult(
                    success=False,
                    data={"error": "SymPy could not solve this problem"},
                    method="error",
                    latency_ms=self._measure_time(start)
                )

            return EngineResult(
                success=True,
                data={
                    "solution": result["solution"],
                    "method": "sympy",
                    "steps": result.get("steps"),
                    "error": None
                },
                method="sympy",
                tokens_used=0,
                latency_ms=self._measure_time(start),
                cost=0.0
            )

        except Exception as e:
            return EngineResult(
                success=False,
                data={"error": str(e)},
                method="error",
                latency_ms=self._measure_time(start)
            )


class ClaudeMathAdapter(ClaudeAdapter):
    """
    Adapter that ONLY uses Claude for math.
    Useful for complex word problems.
    """

    def __init__(self, engine: MathEngine = None):
        self.engine = engine or MathEngine()

    @property
    def name(self) -> str:
        return "claude_math"

    @property
    def capabilities(self) -> List[str]:
        return ["solve", "word_problems", "calculus", "proofs"]

    def can_handle(self, problem: str, **kwargs) -> float:
        """Claude can handle any math problem."""
        # Check if it looks like math
        math_indicators = ["+", "-", "*", "/", "=", "solve", "calculate", "find"]
        matches = sum(1 for i in math_indicators if i in problem.lower())

        if matches > 0:
            return min(0.9, 0.4 + matches * 0.1)
        return 0.3  # Can still try

    async def solve(self, problem: str, **kwargs) -> EngineResult:
        """Solve using Claude only."""
        start = time.time()

        try:
            # Use the engine's Claude method
            result = self.engine._solve_with_claude(problem)

            latency = self._measure_time(start)

            if result.get("error"):
                return EngineResult(
                    success=False,
                    data=result,
                    method="error",
                    latency_ms=latency
                )

            return EngineResult(
                success=True,
                data=result,
                method="claude",
                tokens_used=500,  # Estimate
                latency_ms=latency,
                cost=0.001  # Rough estimate
            )

        except Exception as e:
            return EngineResult(
                success=False,
                data={"error": str(e)},
                method="error",
                latency_ms=self._measure_time(start)
            )
