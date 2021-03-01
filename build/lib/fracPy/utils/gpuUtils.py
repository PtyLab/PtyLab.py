# This file contains utilities that enable the use of a GPU while allowing to run the toolbox without one
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
    raise NotImplementedError()



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

