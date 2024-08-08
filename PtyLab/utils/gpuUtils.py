# This file contains utilities that enable the use of a GPU while allowing to run the toolbox without one
import logging
from typing import List

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GPU")

def check_gpu_availability(verbose=True):
    """Check if GPU and cupy are available."""
    try:
        import cupy

        if cupy.cuda.is_available():
            if verbose:
                logger.info("cupy and CUDA available, switching to GPU")
            return True

    except AttributeError:
        if verbose:
            logger.info("CUDA is unavailable, switching to CPU")
        return False

    except ImportError:
        if verbose:
            logger.info("cupy is unavailable, switching to CPU")
        return False


CP_AVAILABLE = True if check_gpu_availability() else False

if CP_AVAILABLE:
    import cupy as cp
else:
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
            self.logger.debug(f"Skipped {field} as it is not defined")
            self.logger.debug(f"Skipped {field} as it is not defined")


def check_array_type(arr):
    """ Checks if the array if a numpy or a cupy array. Useful for debugging"""
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            print("This is a CuPy array")
        elif isinstance(arr, np.ndarray):
            print("This is a NumPy array")
        else:
            print("This is neither a NumPy nor a CuPy array")
    except ImportError:
        if isinstance(arr, np.ndarray):
            print("This is a NumPy array")
        else:
            print("This is not a NumPy array (CuPy not available)")