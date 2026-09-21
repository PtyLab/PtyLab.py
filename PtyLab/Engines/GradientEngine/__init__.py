"""Optional Torch gradient engine and its component model API.

Importing this namespace does not require PyTorch.
"""

from importlib import import_module

__all__ = ["GradientEngine", "PtychographyModel", "SharedProbe"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # The engine provides installation guidance when PyTorch is unavailable.
    engine = import_module(".engine", __name__)
    value = getattr(engine, name)
    globals()[name] = value
    return value
