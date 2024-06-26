# This file contains utilities that enable the use of a GPU while allowing to run the toolbox without one
import logging
from typing import List

import numpy as np

try:
    import cupy as cp

    CP_AVAILABLE = True
except ImportError:
    CP_AVAILABLE = False
    cp = np


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
    if getArrayModule(ary) is np:
        return False
    else:
        return True


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
    self: object, fields: List[str], logger: logging.Logger, dtype="auto"
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
                setattr(self, field, asCupyArray(attribute))
            except AttributeError:
                self.logger.error(f"Cannot set attribute {field}")
                raise
            self.logger.debug(f"Moved {field} to GPU")
        else:
            self.logger.debug(f"Skipped {field} as it is not defined")


def transfer_fields_to_cpu(self: object, fields: List[str], logger: logging.Logger):
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
            self.logger.debug(f"Moved {field} to CPU")
        else:
            self.logger.debug(f"Skipped {field} as it is not defined")
