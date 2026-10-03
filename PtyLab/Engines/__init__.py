# from . import ePIE_reconstructor, mPIE_reconstructor, pSD_reconstructor
# Engines available by default
from .aPIE import aPIE

# # for other Engines (like one you are developing but which is too specific) you can always import PtyLab.Engines.<your_engine_filename>.<your_class>
from .BaseEngine import BaseEngine

__all__ = [
    "aPIE",
    "BaseEngine",
    "e3PIE",
    "ePIE",
    "mPIE",
    "mqNewton",
    "multiPIE",
    "OPR",
    "qNewton",
    "zPIE",
    "purityPIE",
]
from .e3PIE import e3PIE
from .ePIE import ePIE
from .mPIE import mPIE, pcPIE
from .mqNewton import mqNewton
from .multiPIE import multiPIE
from .OPR import OPR
from .qNewton import qNewton
from .zPIE import zPIE
from .purityPIE import purityPIE


import warnings

warnings.warn(
    "`pcPIE` is deprecated. Use `mPIE` with "
    "`params.positionCorrectionSwitch = True` instead.",
    DeprecationWarning,
    stacklevel=2,
)
