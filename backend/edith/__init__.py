"""
EDITH Proactive Scanner

"Even Dead, I'm The Hero" - Peter Parker's AI glasses system.
Background environmental scanning for AR glasses.

Capabilities:
- Equation detection and auto-solve suggestions
- Code pattern recognition
- Poker hand detection
- Text/document analysis
- Adaptive power management
"""

from .scanner import (
    EdithScanner,
    ScanResult,
    DetectionType,
    ScanConfig,
)
from .detector import (
    ContentDetector,
    EquationDetector,
    CodeDetector,
    PokerDetector,
    TextDetector,
)
from .power import (
    PowerManager,
    PowerProfile,
    AdaptiveFPS,
)

__all__ = [
    # Core scanner
    "EdithScanner",
    "ScanResult",
    "DetectionType",
    "ScanConfig",
    # Detectors
    "ContentDetector",
    "EquationDetector",
    "CodeDetector",
    "PokerDetector",
    "TextDetector",
    # Power management
    "PowerManager",
    "PowerProfile",
    "AdaptiveFPS",
]
