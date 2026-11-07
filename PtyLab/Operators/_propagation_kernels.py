from functools import lru_cache

try:
    import cupy as cp
except ImportError:
    # print("Cupy unavailable")
    pass

import numpy as np

# from PtyLab.Operators.Operators import cache_size
cache_size = 30


@lru_cache(maxsize=cache_size)
def __make_quad_phase(zo, wavelength, Np, dxp, on_gpu):
    r"""
    Construct and cache a spatial-domain quadratic Fresnel phase factor.

    Evaluates

    $$
    Q(x, y) = \exp\left[\frac{i\pi}{\lambda z}(x^2 + y^2)\right]
    $$

    on a square grid. Used by the Fresnel propagators and the two-step
    polychromatic kernel builder. This function constructs a phase factor
    only; it does not propagate a field or apply amplitude normalization.

    Args:
        zo (float):
            Nonzero propagation distance in meters. For real parameters,
            reversing its sign conjugates the phase factor.

        wavelength (float):
            Positive wavelength in meters.

        Np (int):
            Number of samples along each side of the square grid.
            Use an integer greater than 1.

        dxp (float):
            Positive pixel spacing in meters.

        on_gpu (bool):
            If True, construct the array using CuPy on the GPU.
            Otherwise, construct it using NumPy on the CPU.

    Returns:
        ndarray:
            Complex phase array of shape ``(Np, Np)`` on the requested
            backend. For real, valid parameters its magnitude is one,
            up to floating-point precision.

    Raises:
        ImportError:
            If ``on_gpu`` is True and CuPy could not be imported.

    Notes:
        Coordinates use ``(arange(Np) - Np // 2) * dxp``, matching the
        centered FFT grid. Samples are spaced by ``dxp``, with the origin
        at index ``Np // 2`` for both odd and even grid sizes.

        Up to 30 argument combinations are cached. Cache hits return the
        stored array itself; treat it as read-only. GPU device identity
        is not part of the cache key, so clear the cache when switching
        GPU devices before reusing the same arguments.
    """
    if on_gpu:
        if "cp" not in globals():
            raise ImportError("CuPy is not installed but on_gpu=True was requested.")
        xp = cp
    else:
        xp = np

    x_p = (xp.arange(Np) - Np // 2) * dxp
    Xp, Yp = xp.meshgrid(x_p, x_p)

    quadraticPhase = xp.exp(1.0j * xp.pi / (wavelength * zo) * (Xp**2 + Yp**2))
    return quadraticPhase
