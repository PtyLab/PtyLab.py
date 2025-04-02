from PtyLab.utils.gpuUtils import getArrayModule

try:
    import cupy as cp
except ImportError:
    pass

import numpy as np
from scipy import signal


def complexexp(angle):
    """
    Faster way of implementing np.exp(1j*something_unitary)

    Parameters
    ----------
    angle: np.ndarray
        Angle of the exponent

    Returns
    -------

    cos(angle) + 1j * sin(angle)

    """
    xp = getArrayModule(angle)
    return xp.cos(angle) + 1j * xp.sin(angle)


def iterate_6d_fields(fields):
    """
    Iterate over the first four dimensions of a 6D field array. (nlambda, nosm, npsm, nslice, Np, Np)
    corresponding to multi-wavelengths, object modes, probe modes, multislice, diffraction pattern (2d)
    """
    for idx in np.ndindex(fields.shape[:4]):
        yield idx


# creating a bandpass filter
def convolve2d(in1, in2, on_gpu, mode="same"):
    """Using the convolve2d function based on whether on GPU of not"""
    if on_gpu:
        return _fft_convolve2d(in1, in2, on_gpu, mode=mode)
    else:
        return signal.convolve2d(in1, in2, mode=mode)


def gaussian2D(n, std, on_gpu):
    """Creates a 2D gaussian"""
    xp = cp if on_gpu else np
    # create the grid of (x,y) values
    n = (n - 1) // 2
    x, y = xp.meshgrid(xp.arange(-n, n + 1), xp.arange(-n, n + 1))
    # analytic function
    h = xp.exp(-(x**2 + y**2) / (2 * std**2))
    # truncate very small values to zero
    mask = h < xp.finfo(float).eps * xp.max(h)
    h *= 1 - mask
    # normalize filter to unit L1 energy
    sumh = xp.sum(h)
    if sumh != 0:
        h = h / sumh
    return h


def _fft_convolve2d(x, y, on_gpu, mode="same"):
    """
    2D convolution using FFT, for CuPy arrays.
    """
    xp = cp if on_gpu else np

    s1 = x.shape
    s2 = y.shape
    size = s1[0] + s2[0] - 1, s1[1] + s2[1] - 1
    fx = xp.fft.fft2(x, size)
    fy = xp.fft.fft2(y, size)
    result = xp.fft.ifft2(fx * fy)

    if mode == "same":
        return _centered(result, s1, on_gpu)
    elif mode == "valid":
        return _centered(result, (s1[0] - s2[0] + 1, s1[1] - s2[1] + 1), on_gpu)
    else:  # 'full'
        return result


def _centered(arr, newsize, on_gpu):
    xp = cp if on_gpu else np

    # Return the center newsize portion of the array.
    newsize = xp.asarray(newsize)
    currsize = xp.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]
