"""
Optional Torch gradient engine and its component model API.

Importing this namespace does not require PyTorch. Each public name is
imported on first access.
"""

from importlib import import_module

_MODULES = {
    "GradientEngine": ".engine",
    "PtychographyModel": ".engine",
    "SharedProbe": ".engine",
    "Phase": ".phases",
    "blind_recipe": ".phases",
    "Constraint": ".constraints",
    "ProbeCentering": ".constraints",
    "ObjectAmplitudeClamp": ".constraints",
    "PhaseRampRemoval": ".constraints",
    "ProbeModeOrthogonalization": ".constraints",
    "AbsorbingProbeBoundary": ".constraints",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # The engine provides installation guidance when PyTorch is unavailable.
    import_module(".engine", __name__)
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
