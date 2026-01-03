"""
Quant Finance Module - Interview preparation for Jane Street, PDT, Citadel.

Engines:
- MentalMathEngine: Speed arithmetic, fractions, percentages
- ProbabilityEngine: Cards, dice, expected value
- OptionsEngine: Black-Scholes, Greeks, implied volatility
- MarketMakingEngine: Bid/ask, Kelly criterion, Sharpe ratio
- FermiEngine: Estimation problems
- InterviewModeEngine: Mixed problem sessions with adaptive difficulty
"""

from .mental_math import MentalMathEngine
from .probability import ProbabilityEngine
from .options import OptionsEngine
from .market_making import MarketMakingEngine
from .fermi import FermiEngine
from .interview import InterviewModeEngine

__all__ = [
    'MentalMathEngine',
    'ProbabilityEngine',
    'OptionsEngine',
    'MarketMakingEngine',
    'FermiEngine',
    'InterviewModeEngine'
]
