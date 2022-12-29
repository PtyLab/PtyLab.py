import numpy as np
from typing import List, Union, Tuple

from PtyLab.Operators.Operators import aspw
from PtyLab.utils.gpuUtils import getArrayModule, isGpuArray, asNumpyArray
from PtyLab.utils.utils import fft2c


def std(field, aleph=1e-2):
    """
    Return the standard deviation of a field.

    Parameters
    ----------
    field: np.ndarray
    aleph: float
        Ignored

    Returns
    -------
    standard deviation of field as implemented in numpy

    """
    xp = getArrayModule(field)
    return asNumpyArray(xp.std(field))


def min_std(*args, **kwargs):
    """
    Return minus the standard deviation of a field. See ``Ptylab.Regularizers.std''  for more info

    Parameters
    ----------
    field: np.ndarray
        field
    aleph: float
        Ignored

    Returns
    -------

    minus the standard deviation of the field

    """
    return -std(*args, **kwargs)


def TV(field, aleph=1e-3):
    """
    Calculate Total Variation of a field.

    Parameters
    ----------
    field: np.ndarray
        Optical field to process
    aleph: float
        Tiny constant to avoid dividing by zero

    Returns
    -------
    TV_value: float
        Total variation of the field.
    """

    xp = getArrayModule(field)

    grad_x = xp.roll(field, -1, axis=-1) - xp.roll(field, 1, axis=-1)
    grad_y = xp.roll(field, -1, axis=-2) - xp.roll(field, 1, axis=-1)
    value = xp.sum(
        xp.sqrt(abs(grad_x * grad_x.conj()) + abs(grad_y * grad_y.conj()) + aleph)
    )
    value = float(asNumpyArray(value))
    return value


def metric_at(
    object_estimate,
    dz,
    dx,
    wavelength,
    ss=(slice(None, None), slice(None, None)),
    intensity_only=False,
    return_propagated=False,
    average_by_power=True,
    metric: str = TV,
        savemem=True,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Return the value of a metric function over a range of distances given by dz.

    Note on savemem:
        When savemem == False, the entire field is propagated and only afterwards a slice is extracted.
        This is the right way to do it for larger propagation distances. However, for small propagation distances
        a lot of time can be saved by only propagating the sliced area in the original, and for autofocusing typically only small
        amounts of propagation are required.

    Parameters
    ----------
    object_estimate: np.ndarray
        the field that has to be propagated
    dz: np.ndarray
        Distances to propagate to
    dx: float
        Pixel size of the field
    wavelength: float
        Wavelength to be propagated at
    ss: Union[slice, slice]
        The region to propagate.
    intensity_only: bool
        Wether to only asses the intensity or the complex field
    return_propagated: bool
        If true, returns the propagated field
    average_by_power: bool
        Divide metric by the average intensity for every distance
    metric: Callable or string
        The quality metric to be employed. Should have signature function(x, eps) and return a single floating point number.
        Alternatively, one can provide 'TV', 'STD' or 'MIN_STD' as a metric and it will be mapped to the corresponding functions in this module.
    savemem:
        Save memory. Default true

    Returns
    -------

    """
    possible_metrics = {"TV": TV, "STD": std, "MIN_STD": min_std}
    if not isinstance(type(metric), type(callable)):
        try:
            metric = possible_metrics[metric.upper()]
        except KeyError:
            raise KeyError(
                f"Could not map {metric} to a metric. Allowed keywords are: {[k for k in possible_metrics.keys()]}"
            )

    if savemem:
        # SY and sx are extracting after propagation, sy1 and sx1 are extracting before
        sy, sx = slice(None, None), slice(None, None)
        sy1, sx1 = ss
    else:
        sy, sx = ss
        sy1, sx1 = slice(None, None), slice(None, None)

    xp = getArrayModule(object_estimate)
    if isGpuArray(dz):
        dz = dz.get()

    OE_ff = fft2c(object_estimate[...,sy1,sx1])
    if intensity_only:
        op = lambda x: abs(x).real ** 2
    else:
        op = lambda x: x

    scores = []
    OEs = []
    for z in dz:
        OE = op(
            aspw(
                xp.squeeze(OE_ff),
                z=float(z),
                wavelength=float(wavelength),
                L=dx * object_estimate.shape[-1],
                bandlimit=False,
                is_FT=True,
            )[0][sy, sx]
        )  # [...,sy,sx])
        if average_by_power and not intensity_only:
            OE = OE / abs(OE**2).mean()
        elif average_by_power and intensity_only:
            OE = OE / OE.mean()

        score = metric(OE)
        OEs.append(asNumpyArray(OE))
        scores.append(score)
    if return_propagated:
        return np.array(scores), np.array(OEs)
    else:
        return np.array(scores)


def divergence(f):
    """
    Calculate the divergence of a vector field.

    Parameters
    ----------
    f

    Returns
    -------

    """
    xp = getArrayModule(f[0])
    return xp.gradient(f[0], axis=(4, 5))[0] + xp.gradient(f[1], axis=(4, 5))[1]

def divergence_new(f):
    xp = getArrayModule(f[0])
    grad_y = xp.gradient(f[0], axis=-2)
    grad_x = xp.gradient(f[1], axis=-1)
    return grad_y + grad_x


def grad_TV(field, epsilon=1e-2):
    xp = getArrayModule(field)
    gradient = xp.array(xp.gradient(field, axis=(-2,-1)))
    norm = xp.sqrt(xp.sum(gradient, axis=0)**2+epsilon)
    gradient /= norm
    TV_update = divergence_new(gradient)
    return TV_update
