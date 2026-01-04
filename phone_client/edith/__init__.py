"""EDITH - Background environmental scanning."""
from .scanner import EdithScanner, ScanConfig, Detection, DetectionType, SuggestionGenerator
from .detectors import EquationDetector, CodeDetector, TextDetector, create_detectors

__all__ = [
    "EdithScanner",
    "ScanConfig",
    "Detection",
    "DetectionType",
    "SuggestionGenerator",
    "EquationDetector",
    "CodeDetector",
    "TextDetector",
    "create_detectors",
]
