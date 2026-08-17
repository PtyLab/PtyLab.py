# This file contains utilities that enable the use of a GPU while allowing to run the toolbox without one
import logging
from typing import Any

import numpy as np


def _import_cupy() -> Any:
    """Return the cupy module, or None when it is not installed."""
    try:
        import cupy

        return cupy
    except ImportError:
        return None


_cupy = _import_cupy()
CP_AVAILABLE = _cupy is not None

# Falls back to numpy so that `cp.` references still resolve on a machine
# without cupy; every cupy-only call is guarded by CP_AVAILABLE. Annotated Any
# because the two modules differ in exactly the attributes those guards select
# -- asnumpy and get_array_module exist only on the cupy branch, so a checker
# that sees the union rejects calls that are unreachable without cupy.
cp: Any = _cupy if CP_AVAILABLE else np


def getArrayModule(*args, **kwargs):
    """
    Return a numerical array processing module based on wether the array lives on the CPU or on the GPU.

    See cupy.getArrayModule for details.
    :param args:
    :param kwargs:
    :return:
    """
    if CP_AVAILABLE:
        return cp.get_array_module(*args, **kwargs)
    else:
        return np


def isGpuArray(ary):
    return getArrayModule(ary) is not np


def asNumpyArray(ary) -> np.ndarray:
    """
    Return a numpy.ndarray version of `ary`.

    :param ary: numpy or cupy ndarray
    :return: cpu-version of ary

    """
    if CP_AVAILABLE:
        return cp.asnumpy(ary)
    else:
        return ary


def asCupyArray(field: np.ndarray, dtype="auto"):
    if dtype == "auto":
        if np.isrealobj(field):
            dtype = np.float32
        elif np.iscomplexobj(field):
            dtype = np.complex64
        else:
            raise NotImplementedError(f"Dtype {field.dtype} is not supported.")
    return cp.array(field, copy=False, dtype=dtype)


def transfer_fields_to_gpu(
    self: object, fields: list[str], logger: logging.Logger, dtype="auto"
):
    """
    Move any fields defined in fields to the CPU. Fields has to be a list of strings with field names
    :param self:
    :param fields:
    :param logger:
    :param dtype: data type. If 'auto', will be set to np.float32 for real-valued data and np.complex64 for complex
    :return:
    """
    for field in fields:
        if hasattr(self, field):  # This field is defined
            # move it to the CPU
            attribute = getattr(self, field)
            try:
                setattr(self, field, asCupyArray(attribute, dtype=dtype))
            except AttributeError:
                logger.error(f"Cannot set attribute {field}")
                raise
            logger.debug(f"Moved {field} to GPU")
        else:
            logger.debug(f"Skipped {field} as it is not defined")


def transfer_fields_to_cpu(self: object, fields: list[str], logger: logging.Logger):
    """
    Move any fields defined in fields to the CPU. Fields has to be a list of strings with field names
    :param self:
    :param fields:
    :param logger:
    :return:
    """
    for field in fields:
        if hasattr(self, field):  # This field is defined
            # move it to the CPU
            attribute = getattr(self, field)
            setattr(self, field, asNumpyArray(attribute))
            logger.debug(f"Moved {field} to CPU")
        else:
            logger.debug(f"Skipped {field} as it is not defined")
