import numpy as np
from typing import List, Union, Tuple

from PtyLab.Operators.Operators import aspw
from PtyLab.utils.gpuUtils import getArrayModule, isGpuArray, asNumpyArray
from PtyLab.utils.utils import fft2c


def std(field, aleph=1e-2):
    """
    Return the standard deviation of a field
    Parameters
    ----------
    field
    aleph

    Returns
    -------

    """
    xp = getArrayModule(field)
    return asNumpyArray(xp.std(field))


def min_std(*args, **kwargs):
    """
    Return minus the standard deviation of a field
    Parameters
    ----------
    field
    aleph

    Returns
    -------

    """
    return -std(*args, **kwargs)


def TV(field, aleph=1e-2):
    """
    Return the Total Variation of a field.

    :param field:
    :return:
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
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Return the value or a metric function over a range of distances. Standard we implement std, minus std and

    :param object_estimate:
    :param dz:
    :param dx:
    :param wavelength:
    :param ss: Subset. Either slice object or list of two ints.
    :param average_by_power: Wether or not to normalize the intensity in the area in which we measure.
    :return:
    """
    possible_metrics = {"TV": TV, "STD": std, "MIN_STD": min_std}
    if not isinstance(type(metric), type(callable)):
        try:
            metric = possible_metrics[metric.upper()]
        except KeyError:
            raise KeyError(
                f"Could not map {metric} to a metric. Allowed keywords are: {[k for k in possible_metrics.keys()]}"
            )
    sy, sx = ss
    xp = getArrayModule(object_estimate)
    if isGpuArray(dz):
        dz = dz.get()
    OE_ff = fft2c(object_estimate)
    if intensity_only:
        op = lambda x: abs(x).real ** 2
    else:
        op = lambda x: x

    scores = []
    OEs = []
    for z in dz:
        OE = op(
            aspw(
                xp.squeeze(OE_ff[..., sy, sx]),
                z=float(z),
                wavelength=float(wavelength),
                L=dx * object_estimate.shape[-1],
                bandlimit=False,
                is_FT=True,
            )[0]
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
