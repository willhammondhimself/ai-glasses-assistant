"""Engine Adapters - Unified interface for all solution engines."""

from .base import IEngineAdapter, EngineResult, ClaudeAdapter

# Lazy imports for adapters that depend on external engines
_math_adapters_loaded = False
_MathEngineAdapter = None
_SymPyOnlyAdapter = None
_ClaudeMathAdapter = None


def _load_math_adapters():
    """Lazy load math adapters to avoid import errors when dependencies unavailable."""
    global _math_adapters_loaded, _MathEngineAdapter, _SymPyOnlyAdapter, _ClaudeMathAdapter
    if not _math_adapters_loaded:
        from .math_adapter import MathEngineAdapter, SymPyOnlyAdapter, ClaudeMathAdapter
        _MathEngineAdapter = MathEngineAdapter
        _SymPyOnlyAdapter = SymPyOnlyAdapter
        _ClaudeMathAdapter = ClaudeMathAdapter
        _math_adapters_loaded = True
    return _MathEngineAdapter, _SymPyOnlyAdapter, _ClaudeMathAdapter


def __getattr__(name):
    """Lazy attribute access for adapters."""
    if name == 'MathEngineAdapter':
        adapters = _load_math_adapters()
        return adapters[0]
    elif name == 'SymPyOnlyAdapter':
        adapters = _load_math_adapters()
        return adapters[1]
    elif name == 'ClaudeMathAdapter':
        adapters = _load_math_adapters()
        return adapters[2]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'IEngineAdapter',
    'EngineResult',
    'ClaudeAdapter',
    'MathEngineAdapter',
    'SymPyOnlyAdapter',
    'ClaudeMathAdapter',
]
