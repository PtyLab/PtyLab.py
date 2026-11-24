"""
Optional Torch gradient engine and its component model API.

Importing this namespace does not require PyTorch. The submodules are
imported when a public name is first used.
"""

from importlib import import_module
from typing import TYPE_CHECKING

# The editor reads these imports (Ctrl+click works). They never run, so
# importing this package does not load torch.
if TYPE_CHECKING:
    from .constraints import (
        AbsorbingObject,
        AbsorbingProbeBoundary,
        Constraint,
        ObjectAmplitudeClamp,
        ObjectStateOrthogonalization,
        OPRProjection,
        PhaseRampRemoval,
        ProbeCentering,
        ProbeModeOrthogonalization,
    )
    from .engine import GradientEngine
    from .losses import (
        amplitude_loss,
        intensity_loss,
        mixed_poisson_gaussian_loss,
        poisson_loss,
    )
    from .models import OPRProbe, PtychographyModel, SharedProbe, probe_scale
    from .phases import Phase, blind_recipe
    from .propagation import KernelPropagator
    from .regularizers import object_smoothness, object_tv

# Public names. A written-out list, so linters see the imports above as used.
__all__ = [
    "AbsorbingObject",
    "AbsorbingProbeBoundary",
    "Constraint",
    "GradientEngine",
    "KernelPropagator",
    "OPRProbe",
    "OPRProjection",
    "ObjectAmplitudeClamp",
    "ObjectStateOrthogonalization",
    "Phase",
    "PhaseRampRemoval",
    "ProbeCentering",
    "ProbeModeOrthogonalization",
    "PtychographyModel",
    "SharedProbe",
    "amplitude_loss",
    "blind_recipe",
    "intensity_loss",
    "mixed_poisson_gaussian_loss",
    "object_smoothness",
    "object_tv",
    "poisson_loss",
    "probe_scale",
]


# engine.py comes first: it raises an install hint when PyTorch is missing.
_SUBMODULES = (
    ".engine",
    ".models",
    ".propagation",
    ".phases",
    ".constraints",
    ".losses",
    ".regularizers",
)


def __getattr__(name):
    # Runs only for names not yet in this namespace, i.e. on first access.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    modules = [import_module(m, __name__) for m in _SUBMODULES]
    for public in __all__:
        globals()[public] = next(
            getattr(m, public) for m in modules if hasattr(m, public)
        )
    return globals()[name]
