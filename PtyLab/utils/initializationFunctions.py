import logging
import numpy as np
from PtyLab.utils.utils import circ, fft2c, ifft2c
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from skimage.transform import rescale


def initialProbeOrObject(shape, type_of_init, data, logger: logging.Logger = None):
    """
    Initialization objects are created for the reconstruction. Currently
    implemented:
        ones - every element is set to 1 + random noise
        circ - same as 'ones' but with a circular boundary constraint
        upsampled - upsampled low-resolution estimate (used for FPM)

    Random noise is added to the arrays to enforce linear independence required
    for orthogonalization of modes

    :return:
    """
    if type(type_of_init) is np.ndarray:  # it has already been run
        if logger is not None:
            logger.warning(
                "initialObjectOrProbe was called but the object has already "
                "been initialized. Skipping."
            )
        return type_of_init
    supported_shapes = ["circ", 'circ_smooth', "rand", "gaussian", "ones", "upsampled"]
    if type_of_init not in supported_shapes:
        raise NotImplementedError(f'Got {type_of_init} for shape. Supported shapes are: {supported_shapes}')

    if type_of_init == "ones":
        return np.ones(shape) + 0.001 * np.random.rand(*shape)

    if type_of_init in ["circ", 'circ_smooth']:
        try:
            # BUG: This only works for the probe, not for the object
            pupil = circ(data.Xp, data.Yp, data.data.entrancePupilDiameter)
            initial_field = np.ones(shape, dtype=np.complex64) + 0.001 * np.random.rand(*shape)

            if 'smooth' in type_of_init:
                dia_pixel = data.data.entrancePupilDiameter / data.dxo
                pupil = ndimage.gaussian_filter(pupil.astype(np.float64), 0.1 * dia_pixel)

            initial_field *= pupil
            return initial_field

        except AttributeError as e:
            raise AttributeError(
                e, "probe/aperture/entrancePupilDiameter was not defined"
            )
 

    if type_of_init == "upsampled":
        low_res = ifft2c(np.sqrt(np.mean(data.data.ptychogram, 0)))
        pad_size = (int((data.No - data.Np) / 2), int((data.No - data.Np) / 2))
        upsampled = np.pad(
            low_res, pad_size, mode="constant", constant_values=0
        )  # * data.No / data.Np
        return np.ones(shape) * upsampled
