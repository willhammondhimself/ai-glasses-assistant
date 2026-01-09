"""
Physics Module for WHAM AI Glasses Assistant.

Provides hybrid SymPy + Wolfram Alpha physics problem solving:
- Offline calculus (integrals, derivatives, limits)
- Circuit analysis via Wolfram Alpha API
- Vector operations
- Step-by-step derivations for AR overlay
"""

from .engine import PhysicsEngine, PhysicsSolution, ProblemType
from .wolfram_client import WolframClient, WolframResult, wolfram_client

__all__ = [
    "PhysicsEngine",
    "PhysicsSolution",
    "ProblemType",
    "WolframClient",
    "WolframResult",
    "wolfram_client",
]
