from functools import lru_cache

try:
    import cupy as cp
except ImportError:
    # print("Cupy unavailable")
    import numpy as np

# from PtyLab.Operators.Operators import cache_size
cache_size = 30


@lru_cache(maxsize=cache_size)
def __make_quad_phase(zo, wavelength, Np, dxp, on_gpu):
    """
    Make a quadratic phase profile corresponding to distance zo at wavelength wl. The result is cached and can be
    called again with almost no time lost.
    :param wavelength:  wavelength in meters
    :param zo:
    :param Np:
    :param dxp:
    :param on_gpu:
    :return:
    """
    if on_gpu:
        xp = cp
    else:
        xp = np

    x_p = xp.linspace(-Np / 2, Np / 2, int(Np)) * dxp
    Xp, Yp = xp.meshgrid(x_p, x_p)

    quadraticPhase = xp.exp(1.0j * xp.pi / (wavelength * zo) * (Xp**2 + Yp**2))
    return quadraticPhase
