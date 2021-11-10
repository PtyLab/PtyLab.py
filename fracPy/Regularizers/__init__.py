import numpy as np
from typing import List

from fracPy.Operators.Operators import aspw
from fracPy.utils.gpuUtils import getArrayModule, isGpuArray, asNumpyArray
from fracPy.utils.utils import fft2c


def TV_at(object_estimate, dz, dx, wavelength, ss=slice(None, None))-> np.ndarray:
    """
    Return the total variation of an object on all possible distances.

    :param object_estimate:
    :param dz:
    :param dx:
    :param wavelength:
    :param ss: Subset. Either slice object or list of two ints.
    :return:
    """
    if isinstance(ss, list):
        N = object_estimate.shape[-1]
        ss = slice(int(ss[0]*N), int(ss[1]*N))
        print(ss)
    xp = getArrayModule(object_estimate)
    if isGpuArray(dz):
        dz = dz.get()
    OE_ff = fft2c(object_estimate)
    return np.array([TV(
        aspw(xp.squeeze(OE_ff), z=float(z),
                   wavelength=float(wavelength),
                   L=dx*object_estimate.shape[-1],
                   bandlimit=False, is_FT=True)[0][...,ss,ss]) for z in dz])


def TV(field, aleph=1e-2):
    """
    Return the Total Variation of a field.

    :param field:
    :return:
    """
    xp = getArrayModule(field)

    grad_x = xp.roll(field, -1, axis=-1) - xp.roll(field, 1, axis=-1)
    grad_y = xp.roll(field, -1, axis=-2) - xp.roll(field, 1, axis=-1)
    value = xp.sum(xp.sqrt(abs(grad_x*grad_x.conj()) + abs(grad_y*grad_y.conj())+aleph))
    value = float(asNumpyArray(value))
    return value