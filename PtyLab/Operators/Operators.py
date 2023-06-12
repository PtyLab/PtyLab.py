import logging

try:  # pre 3.10
    from collections import Callable
except ImportError:
    from collections.abc import Callable

from functools import lru_cache

try:
    import cupy as cp
except ImportError:
    print("cupy not avialable")
import numpy as np

from PtyLab import Params, Reconstruction
from PtyLab.Operators._propagation_kernels import __make_quad_phase
from PtyLab.utils.gpuUtils import getArrayModule, isGpuArray
from PtyLab.utils.utils import circ, fft2c, ifft2c

# how many kernels are kept in memory for every type of propagator? Higher can be faster but comes
# at the expense of (GPU) memory.
cache_size = 5


def propagate_fraunhofer(
    fields, params: Params, reconstruction: Reconstruction, z=None
):
    """
    Propagate using the fraunhofer approximation.

    Parameters
    ----------
    fields: np.ndarray
        Electric field to propagate
    params: Params
        Parameter object. The parameter params.fftshiftSwitch is inspected for the fourier transform
    reconstruction: Reconstruction
        Reconstruction object.
    z: float
        propagation distance. Is ignored in this function.

    Returns
    -------

    A tuple of (reconstruction.esw, Propagated field)

    """
    return reconstruction.esw, fft2c(fields, params.fftshiftSwitch)


def propagate_fraunhofer_inv(
    fields, params: Params, reconstruction: Reconstruction, z=None
):
    """
    Inverse transform. See propagate_frauhofer for the arguments.

    Parameters
    ----------
    fields: np.ndarray
        Electric field to propagate
    params: Params
        Parameter object. The parameter params.fftshiftSwitch is inspected for the fourier transform
    reconstruction: Reconstruction
        Reconstruction object.
    z: float
        propagation distance. Is ignored in this function.

    Returns
    -------
    A tuple of (reconstruction.esw, inverse transformed field)
    """
    return reconstruction.esw, ifft2c(fields, params.fftshiftSwitch)


def propagate_fresnel(fields, params: Params, reconstruction: Reconstruction, z=None):
    # make the quad phase if it's not available yet
    """
    Propagate using the fresnel approximation.

    Parameters
    ----------
    fields: np.ndarray
       Electric field to propagate
    params: Params
       Parameter object. The parameter params.fftshiftSwitch is inspected for the fourier transform
    reconstruction: Reconstruction
       Reconstruction object.
    z: float
       propagation distance in meter

    Returns
    -------

    A tuple of (reconstruction.esw, Propagated field)

    """
    if z is None:
        z = reconstruction.zo
    on_gpu = isGpuArray(fields)
    quadratic_phase = __make_quad_phase(
        z,
        reconstruction.wavelength,
        fields.shape[-1],
        reconstruction.dxp,
        on_gpu=on_gpu,
    )

    eswUpdate = fft2c(fields * quadratic_phase, params.fftshiftSwitch)
    # for legacy reasons, as far as I can see there's no reason to do this
    # esw = reconstruction.esw * quadratic_phase
    return reconstruction.esw, eswUpdate


def propagate_fresnel_inv(
    fields, params: Params, reconstruction: Reconstruction, z=None
):
    """
    Propagate using the inverse fresnel approximation.

    Parameters
    ----------
    fields: np.ndarray
      Electric field to propagate
    params: Params
      Parameter object. The parameter params.fftshiftSwitch is inspected for the fourier transform
    reconstruction: Reconstruction
      Reconstruction object.
    z: float
      propagation distance in meter

    Returns
    -------

    A tuple of (reconstruction.esw, Propagated field)

    """
    # make the quad phase if it's not available yet
    if z is None:
        z = reconstruction.zo
    quadratic_phase = __make_quad_phase(
        z,
        reconstruction.wavelength,
        reconstruction.Np,
        reconstruction.dxp,
        on_gpu=isGpuArray(fields),
    ).conj()

    eswUpdate = ifft2c(fields, params.fftshiftSwitch) * quadratic_phase
    # esw = reconstruction.esw * quadratic_phase
    return reconstruction.esw, eswUpdate


def propagate_ASP(
    fields, params: Params, reconstruction: Reconstruction, inverse=False, z=None,
        fftflag=True,
):
    """
    Propagate using the angular spectrum method


    Parameters
    ----------
    fields: np.ndarray
      Electric field to propagate
    params: Params
      Parameter object. The parameter params.fftshiftSwitch is inspected for the fourier transform
    reconstruction: Reconstruction
      Reconstruction object.
    z: float
      propagation distance in meter
    fftflag: bool
      Specified wether or not to use a centered fft internally. Set to false for debugging but should generally be turned on.

    Returns
    -------
    reconstruction.esw: np.ndarray
        exit surface wave
    result: np.ndarray
        propagated field
    """

    if params.fftshiftSwitch:
        raise ValueError(
            "ASP propagator only works with fftshiftswitch == False")
    if reconstruction.nlambda > 1:
        raise ValueError(
            "For multi-wavelength, set polychromeASP instead of ASP")
    if z is None:
        z = reconstruction.zo
    xp = getArrayModule(fields)
    transfer_function = __make_transferfunction_ASP(
        params.fftshiftSwitch,
        reconstruction.nosm,
        reconstruction.npsm,
        reconstruction.Np,
        z,
        reconstruction.wavelength,
        reconstruction.Lp,
        reconstruction.nlambda,
        isGpuArray(fields),
    )
    if fftflag:
        transfer_function = xp.fft.ifftshift(transfer_function, axes=(-2, -1))
    if inverse:
        transfer_function = transfer_function.conj()
    result = ifft2c(fft2c(fields, fftshiftSwitch=fftflag) *
                    transfer_function, fftshiftSwitch=fftflag)
    return reconstruction.esw, result


def propagate_ASP_inv(*args, **kwargs):
    """
    See propagate_ASP

    Parameters
    ----------
    args
    kwargs

    Returns
    -------

    """
    return propagate_ASP(*args, **kwargs, inverse=True)


def propagate_twoStepPolychrome(
    fields, params: Params, reconstruction: Reconstruction, inverse=False, z=None
):
    """
    Two-step polychrome propagation.

    Parameters
    ----------
    fields: np.ndarray
        Field to propagate
    params: Params
        Parameters
    reconstruction: Reconstruction
    inverse: bool
        Reverse propagation
    z: float
        Propagation distance

    Returns
    -------
    reconstruction.esw, propagated field:
        Exit surface wave and the propagated field

    """
    if z is None:
        z = reconstruction.zo
    transfer_function, quadratic_phase = __make_cache_twoStepPolychrome(
        params.fftshiftSwitch,
        reconstruction.nlambda,
        reconstruction.nosm,
        reconstruction.npsm,
        reconstruction.Np,
        z,
        # this has to be cast to a tuple to
        # make sure it is reused
        tuple(reconstruction.spectralDensity),
        reconstruction.Lp,
        reconstruction.dxp,
        params.gpuSwitch,
    )
    if inverse:
        result = ifft2c(
            fft2c(fields * quadratic_phase.conj()) * transfer_function.conj()
        )
        return reconstruction.esw, result
    else:
        result = ifft2c(fft2c(fields) * transfer_function) * quadratic_phase
        result = fft2c(result, params.fftshiftSwitch)
        return reconstruction.esw, result


def propagate_twoStepPolychrome_inv(
    fields, params: Params, reconstruction: Reconstruction, z=None
):
    """
    See propagate_twoStepPolychrome.

    Parameters
    ----------
    fields
    params
    reconstruction
    z

    Returns
    -------

    """
    F = propagate_twoStepPolychrome(fields, params, reconstruction, inverse=True, z=z)[
        1
    ]
    G = propagate_twoStepPolychrome(
        reconstruction.ESW, params, reconstruction, inverse=True, z=z
    )[1]  # tODO: What is G here? Why are we not returning reconstruction.esw?
    return G, F


def propagate_scaledASP(
    fields, params: Params, reconstruction: Reconstruction, inverse=False, z=None
):
    """
    Propagate using the scaled angular spectrum method.

    Parameters
    ----------
    fields
    params
    reconstruction
    inverse
    z

    Returns
    -------

    """
    if z is None:
        z = reconstruction.zo
    Q1, Q2 = __make_transferfunction_scaledASP(
        params.propagatorType,
        params.fftshiftSwitch,
        reconstruction.nlambda,
        reconstruction.nosm,
        reconstruction.npsm,
        reconstruction.Np,
        z,
        reconstruction.wavelength,
        reconstruction.dxo,
        reconstruction.dxd,
        params.gpuSwitch,
    )
    if inverse:
        Q1, Q2 = Q1.conj(), Q2.conj()
        return reconstruction.esw, ifft2c(fft2c(fields) * Q2) * Q1
    return reconstruction.esw, ifft2c(fft2c(fields * Q1) * Q2)


def propagate_scaledASP_inv(
    fields, params: Params, reconstruction: Reconstruction, z=None
):
    """
    Reverse scaled angular spectrum propagation. See scaledASP for details.

    Parameters
    ----------
    fields: np.ndarray
        Field to propagate
    params: Params
        Parameters
    reconstruction: Reconstruction
    z: float
        Propagation distance

    Returns
    -------
    reconstruction.esw, propagated field:
        Exit surface wave and the propagated field

    """
    return propagate_scaledASP(fields, params, reconstruction, inverse=True, z=z)


def propagate_scaledPolychromeASP(
    fields, params: Params, reconstruction: Reconstruction, inverse=False, z=None
):
    """
    Scaled angular spectrum for multiple wavelengths.

    Parameters
    ----------
    fields: np.ndarray
        Field to propagate
    params: Params
        Parameters
    reconstruction: Reconstruction
    inverse: bool
        Reverse propagation
    z: float
        Propagation distance

    Returns
    -------
    reconstruction.esw, propagated field:
        Exit surface wave and the propagated field

    Returns
    -------

    """
    if z is None:
        z = reconstruction.zo
    Q1, Q2 = __make_transferfunction_scaledPolychromeASP(
        params.fftshiftSwitch,
        reconstruction.nlambda,
        reconstruction.nosm,
        reconstruction.npsm,
        z,
        reconstruction.Np,
        tuple(reconstruction.spectralDensity),
        reconstruction.dxo,
        reconstruction.dxp,
        params.gpuSwitch,
    )
    if inverse:
        Q1, Q2 = Q1.conj(), Q2.conj()
        return reconstruction.esw, ifft2c(fft2c(fields) * Q1) * Q2
    return reconstruction.esw, ifft2c(fft2c(fields * Q1) * Q2)


def propagate_scaledPolychromeASP_inv(
    fields, params: Params, reconstruction: Reconstruction, z=None
):
    """
     Reverse Scaled angular spectrum for multiple wavelengths.

     Parameters
     ----------
     fields: np.ndarray
         Field to propagate
     params: Params
         Parameters
     reconstruction: Reconstruction
     inverse: bool
         Reverse propagation
     z: float
         Propagation distance

     Returns
     -------
     reconstruction.esw, propagated field:
         Exit surface wave and the propagated field

     Returns
     -------

     """
    return propagate_scaledPolychromeASP(
        fields, params, reconstruction, inverse=True, z=z
    )


def propagate_polychromeASP(
    fields, params: Params, reconstruction: Reconstruction, inverse=False, z=None
):
    """
     ASP propagation  for multiple wavelengths.

     Parameters
     ----------
     fields: np.ndarray
         Field to propagate
     params: Params
         Parameters
     reconstruction: Reconstruction
     inverse: bool
         Reverse propagation
     z: float
         Propagation distance

     Returns
     -------
     reconstruction.esw, propagated field:
         Exit surface wave and the propagated field

     Returns
     -------

     """
    if z is None:
        z = reconstruction.zo
    transfer_function = __make_transferfunction_polychrome_ASP(
        params.propagatorType,
        params.fftshiftSwitch,
        reconstruction.nosm,
        reconstruction.npsm,
        reconstruction.Np,
        z,
        reconstruction.wavelength,
        reconstruction.Lp,
        reconstruction.nlambda,
        tuple(reconstruction.spectralDensity),
        params.gpuSwitch,
    )

    if inverse:
        transfer_function = transfer_function.conj()
    result = ifft2c(fft2c(fields) * transfer_function)
    return reconstruction.esw, result


def propagate_identity(
    fields, params: Params, reconstruction: Reconstruction, inverse=False, z=None
):
    """
    Identity propagator (aka does nothing).

    Can probably be used to figure out orientation or to perform some kind of stitching.


    Parameters
    ----------
    fields
    params
    reconstruction
    inverse
    z

    Returns
    -------

    """
    transfer_function = __make_quad_phase(
        1e-3, 532e-9, reconstruction.Np, reconstruction.dxp, isGpuArray(fields)
    )
    transfer_function = transfer_function * 0 + 1
    return reconstruction.esw, fields * transfer_function


def propagate_polychromeASP_inv(fields, params, reconstruction, z=None):
    """
     inverse scaled angular spectrum for multiple wavelengths.

     Parameters
     ----------
     fields: np.ndarray
         Field to propagate
     params: Params
         Parameters
     reconstruction: Reconstruction
     inverse: bool
         Reverse propagation
     z: float
         Propagation distance

     Returns
     -------
     reconstruction.esw, propagated field:
         Exit surface wave and the propagated field

     """
    return propagate_polychromeASP(fields, params, reconstruction, inverse=True, z=z)


def detector2object(fields, params: Params, reconstruction: Reconstruction):
    """
    Implements detector2object.m. Returns a propagated version of the field.

    If field is not given, reconstruction.esw is taken
    :return: esw, updated esw
    """
    if fields is None:
        fields = reconstruction.ESW
    method: Callable[[np.ndarray, Params], Reconstruction] = reverse_lookup_dictionary[
        params.propagatorType.lower()
    ]
    return method(fields, params, reconstruction)


def object2detector(fields, params: Params, reconstruction: Reconstruction):
    """Propagate a field from the object to the detector. Return the new object, do not update in-place."""

    method: Callable[[np.ndarray, Params], Reconstruction] = forward_lookup_dictionary[
        params.propagatorType.lower()
    ]
    if fields is None:
        fields = reconstruction.esw
    return method(fields, params, reconstruction)


def aspw(u, z, wavelength, L, bandlimit=True, is_FT=True):
    """
    Angular spectrum plane wave propagation function.
    following: Matsushima et al., "Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space
    Propagation in Far and Near Fields", Optics Express, 2009


    Parameters
    ----------
    u: np.ndarray
        a 2D field distribution at z = 0 (u is assumed to be square, i.e. N x N)
    z: float
        propagation distance in meter
    wavelength: float
        Wavelength in meter
    L: float
        total size of the field in meter
    bandlimit: bool
        Wether or not to band limit the sample
    is_FT: bool
        If the field has already been fourier transformed.

    Returns
    -------
    U_prop, Q  (field distribution after propagation and the bandlimited transfer function)

    """
    N = u.shape[-1]
    phase_exp = __aspw_transfer_function(
        float(z),
        float(wavelength),
        int(N),
        float(L),
        on_gpu=isGpuArray(u),
        bandlimit=bandlimit,
    )
    if is_FT:
        U = u
    else:
        U = fft2c(u)
    u_prop = ifft2c(U * phase_exp)
    return u_prop, phase_exp


@lru_cache(cache_size)
def __aspw_transfer_function(z, wavelength, N, L, on_gpu=False, bandlimit=True):
    """
    Angular spectrum optical transfer function. You likely don't need to use this directly.

    The result of this call is cached so it can be reused and called as often as you need without having
    to worry about recalculating everything all the time.


    Parameters
    ----------
    z: float
        distance
    wavelength: float
        wavelength in meter
    N: int
        Number of pixels per side
    L: int
        Physical size
    on_gpu: bool
        If true, a cupy array is returned
    bandlimit: bool
        If the transfer function should be band-limited.

    Returns
    -------

    """
    if on_gpu:
        xp = cp
    else:
        xp = np

    a_z = abs(z)
    k = 2 * np.pi / wavelength
    X = xp.arange(-N / 2, N / 2) / L
    Fx, Fy = xp.meshgrid(X, X)
    f_max = L / (wavelength * xp.sqrt(L**2 + 4 * a_z**2))
    # note: see the paper above if you are not sure what this bandlimit has to do here
    # W = rect(Fx/(2*f_max)) .* rect(Fy/(2*f_max));
    W = xp.array(circ(Fx, Fy, 2 * f_max))
    # note: accounts for circular symmetry of transfer function and imposes bandlimit to avoid sampling issues
    exponent = 1 - (Fx * wavelength) ** 2 - (Fy * wavelength) ** 2
    # take out stuff that cannot exist
    mask = exponent > 0
    if not bandlimit:
        mask = 0 * mask + 1
    # put the out of range values to 0 so the square root can be taken
    exponent = xp.clip(exponent, 0, xp.inf)
    H = xp.array(mask * complexexp(k * a_z * xp.sqrt(exponent)))
    if z < 0:
        H = H.conj()
    phase_exp = H * W
    return phase_exp


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


def scaledASP(u, z, wavelength, dx, dq, bandlimit=True, exactSolution=False):
    """
    Angular spectrum propagation with customized grid spacing dq (within Fresnel(or paraxial) approximation)
    :param u: a 2D square input field
    :param z: propagation distance
    :param wavelength: propagation wavelength
    :param dx: grid spacing in original plane (u)
    :param dq: grid spacing in destination plane (Uout)
    :return: propagated field and two quadratic phases

    note: to be analytically correct, add Q3 (see below)
    if only intensities matter, leave it out
    """
    # optical wavenumber
    k = 2 * np.pi / wavelength
    # assume square grid
    N = u.shape[-1]
    # source plane coordinates
    x1 = np.arange(-N // 2, N // 2) * dx
    X1, Y1 = np.meshgrid(x1, x1)
    r1sq = X1**2 + Y1**2
    # spatial frequencies(of source plane)
    f = np.arange(-N // 2, N // 2) / (N * dx)
    FX, FY = np.meshgrid(f, f)
    fsq = FX**2 + FY**2
    # scaling parameter
    m = dq / dx

    # quadratic phase factors
    Q1 = np.exp(1.0j * k / 2 * (1 - m) / z * r1sq)
    Q2 = np.exp(-1.0j * np.pi**2 * 2 * z / m / k * fsq)

    if bandlimit:
        if m != 1:
            r1sq_max = wavelength * z / (2 * dx * (1 - m))
            Wr = np.array(circ(X1, Y1, 2 * r1sq_max))
            Q1 = Q1 * Wr

        fsq_max = m / (2 * z * wavelength * (1 / (N * dx)))
        Wf = np.array(circ(FX, FY, 2 * fsq_max))
        Q2 = Q2 * Wf

    if exactSolution:  # if only intensities matter, leave it out
        # observation plane coordinates
        x2 = np.arange(-N // 2, N // 2) * dq
        X2, Y2 = np.meshgrid(x2, x2)
        r2sq = X2**2 + Y2**2
        Q3 = np.exp(1.0j * k / 2 * (m - 1) / (m * z) * r2sq)
        # compute the propagated field
        Uout = Q3 * ifft2c(Q2 * fft2c(Q1 * u))
        return Uout, Q1, Q2, Q3
    else:  # ignore the phase part in the observation plane
        Uout = ifft2c(Q2 * fft2c(Q1 * u))
        return Uout, Q1, Q2


def scaledASPinv(u, z, wavelength, dx, dq):
    """
    :param u:  a 2D square input field
    :param z:   propagation distance
    :param wavelength: wavelength
    :param dx:  grid spacing in original plane (u)
    :param dq:  grid spacing in destination plane (Uout)
    :return: propagated field

    note: to be analytically correct, add Q3 (see below)
    if only intensities matter, leave it out
    """
    # optical wavenumber
    k = 2 * np.pi / wavelength
    # assume square grid
    N = u.shape[-1]
    # source-plane coordinates
    x1 = np.arange(-N / 2, N / 2) * dx
    Y1, X1 = np.meshgrid(x1, x1)
    r1sq = np.square(X1) + np.square(Y1)
    # spatial frequencies(of source plane)
    f = np.arange(-N / 2, N / 2) / (N * dx)
    FX, FY = np.meshgrid(f, f)
    fsq = FX**2 + FY**2
    # scaling parameter
    m = dq / dx

    # quadratic phase factors
    Q1 = np.exp(1j * k / 2 * (1 - m) / z * r1sq)
    Q2 = np.exp(-1j * 2 * np.pi**2 * z / m / k * fsq)
    Uout = np.conj(Q1) * ifft2c(np.conj(Q2) * fft2c(u))

    # x2 = np.arange(-N / 2, N / 2) * dq
    # X2, Y2 = np.meshgrid(x2,x2)
    # r2sq = X2**2 + Y2**2
    # Q3 = np.exp(1.j * k / 2 * (m - 1) / (m * z) * r2sq)
    # # compute the propagated field
    # Uout = np.conj(Q1) * ifft2c(np.conj(Q2) * fft2c(u*np.conj(Q3)))

    return Uout


def fresnelPropagator(u, z, wavelength, L):
    """
    One-step Fresnel propagation, performing Fresnel-Kirchhoff integral.
    :param u:   field distribution at z = 0(u is assumed to be square, i.e.N x N)
    :param z:   propagation distance
    :param wavelength: wavelength
    :param L: total size[m] of the source plane
    :return: propagated field
    """
    xp = getArrayModule(u)

    k = 2 * np.pi / wavelength
    # source coordinates, assuming square grid
    N = u.shape[-1]
    dx = L / N  # source-plane pixel size
    x = xp.arange(-N // 2, N // 2) * dx
    [Y, X] = xp.meshgrid(x, x)

    # observation coordinates
    dq = wavelength * z / L  # observation-plane pixel size
    q = xp.arange(-N // 2, N // 2) * dq
    [Qy, Qx] = xp.meshgrid(q, q)

    # quadratic phase terms
    Q1 = xp.exp(
        1j * k / (2 * z) * (X**2 + Y**2)
    )  # quadratic phase inside the integral
    Q2 = xp.exp(1j * k / (2 * z) * (Qx**2 + Qy**2))

    # pre-factor
    A = 1 / (1j * wavelength * z)

    # Fresnel-Kirchhoff integral
    u_out = A * Q2 * fft2c(u * Q1)
    return u_out, dq, Q1, Q2


def clear_cache(logger: logging.Logger = None):
    """Clear the cache of all cached functions in this module. Use if GPU memory is not available.

    IF logger is available, print some information about the methods being cleared.

    Returns nothing"""
    list_of_methods = [
        __aspw_transfer_function,
        __make_quad_phase,
        __make_transferfunction_ASP,
        __make_transferfunction_scaledASP,
        __make_cache_twoStepPolychrome,
        __make_transferfunction_polychrome_ASP,
        __make_transferfunction_scaledPolychromeASP,
    ]
    for method in list_of_methods:
        if logger is not None:
            logger.debug(method.cache_info())
            logger.info("clearing cache for %s", method)
        method.cache_clear()


@lru_cache(cache_size)
def __make_transferfunction_ASP(
    fftshiftSwitch, nosm, npsm, Np, zo, wavelength, Lp, nlambda, on_gpu
):
    if fftshiftSwitch:
        raise ValueError(
            "ASP propagatorType works only with fftshiftSwitch = False!")
    if nlambda > 1:
        raise ValueError(
            "For multi-wavelength, polychromeASP needs to be used instead of ASP"
        )

    dummy = np.ones((1, nosm, npsm, 1, Np, Np), dtype="complex64")
    _transferFunction = np.array(
        [
            [
                [
                    [
                        __aspw_transfer_function(zo, wavelength, Np, Lp)
                        for nslice in range(1)
                    ]
                    for npsm in range(npsm)
                ]
                for nosm in range(nosm)
            ]
            for nlambda in range(nlambda)
        ],
        dtype=np.complex64,
    )

    if on_gpu:
        return cp.array(_transferFunction)
    else:
        return _transferFunction


def aspw_cached(u, z, wavelength, L):
    """Cached version of aspw."""
    transferFunction = __aspw_transfer_function(
        z, wavelength, u.shape[-1], L, isGpuArray(u)
    )
    # __make_transferfunction_ASP(False, 1, 1, u.shape[-1],
    #                                               z, wavelength, L, 1, isGpuArray(u))
    # transferFunction = transferFunction[0,0,0,0]
    U = fft2c(u)
    u_prime = ifft2c(U * transferFunction)
    return u_prime


@lru_cache(cache_size)
def __make_transferfunction_polychrome_ASP(
    propagatorType,
    fftshiftSwitch,
    nosm,
    npsm,
    Np,
    zo,
    wavelength,
    Lp,
    nlambda,
    spectralDensity_as_tuple,
    gpuSwitch,
) -> np.ndarray:
    spectralDensity = np.array(spectralDensity_as_tuple)
    if fftshiftSwitch:
        raise ValueError(
            "ASP propagatorType works only with fftshiftSwitch = False!")
    dummy = np.ones((nlambda, nosm, npsm, 1, Np, Np), dtype="complex64")
    transferFunction = np.array(
        [
            [
                [
                    [
                        __aspw_transfer_function(
                            zo, spectralDensity[nlambda], Np, Lp, False,
                        )
                        for nslice in range(1)
                    ]
                    for npsm in range(npsm)
                ]
                for nosm in range(nosm)
            ]
            for nlambda in range(nlambda)
        ]
    )
    if gpuSwitch:
        return cp.array(transferFunction, dtype=cp.complex64)
    else:
        return transferFunction


@lru_cache(cache_size)
def __make_transferfunction_scaledASP(
    propagatorType,
    fftshiftSwitch,
    nlambda,
    nosm,
    npsm,
    Np,
    zo,
    wavelength,
    dxo,
    dxd,
    gpuSwitch,
):
    if fftshiftSwitch:
        raise ValueError(
            "scaledASP propagatorType works only with fftshiftSwitch = False!"
        )
    if nlambda > 1:
        raise ValueError(
            "For multi-wavelength, scaledPolychromeASP needs to be used instead of scaledASP"
        )
    dummy = np.ones((1, nosm, npsm, 1, Np, Np), dtype="complex64")
    _Q1 = np.ones_like(dummy)
    _Q2 = np.ones_like(dummy)
    for nosm in range(nosm):
        for npsm in range(npsm):
            _, _Q1[0, nosm, npsm, 0, ...], _Q2[0, nosm, npsm, 0, ...] = scaledASP(
                dummy[0, nosm, npsm, 0, :, :], zo, wavelength, dxo, dxd
            )

    if gpuSwitch:
        return cp.array(_Q1, dtype=np.complex64), cp.array(_Q2, dtype=np.complex64)
    return _Q1, _Q2


@lru_cache(cache_size)
def __make_transferfunction_scaledPolychromeASP(
    fftshiftSwitch,
    nlambda,
    nosm,
    npsm,
    zo,
    Np,
    spectralDensity_as_tuple,
    dxo,
    dxd,
    on_gpu,
):
    spectralDensity = np.array(spectralDensity_as_tuple)
    if fftshiftSwitch:
        raise ValueError(
            "scaledPolychromeASP propagatorType works only with fftshiftSwitch = False!"
        )
    if on_gpu:
        xp = cp
    else:
        xp = np
    dummy = xp.ones((nlambda, nosm, npsm, 1, Np, Np), dtype="complex64")
    Q1 = xp.ones_like(dummy)
    Q2 = xp.ones_like(dummy)
    for nlambda in range(nlambda):
        Q1_candidate, Q2_candidate = __make_transferfunction_scaledASP(
            None,
            fftshiftSwitch,
            1,
            nosm,
            npsm,
            Np,
            zo,
            spectralDensity[nlambda],
            dxo,
            dxd,
            gpuSwitch=on_gpu,
        )
        Q1[nlambda], Q2[nlambda] = Q1_candidate[0], Q2_candidate[0]
    return Q1, Q2


@lru_cache(cache_size)
def __make_cache_twoStepPolychrome(
    fftshiftSwitch,
    nlambda,
    nosm,
    npsm,
    Np,
    zo,
    spectralDensity_as_tuple,
    Lp,
    dxp,
    on_gpu,
):
    if on_gpu:
        xp = cp
    else:
        xp = np
    spectralDensity = np.array(spectralDensity_as_tuple)
    if fftshiftSwitch:
        raise ValueError(
            "twoStepPolychrome propagatorType works only with fftshiftSwitch = False!"
        )
    transferFunction = xp.array(
        [
            [
                [
                    [
                        __aspw_transfer_function(
                            z=zo *
                            (1 - spectralDensity[0] /
                             spectralDensity[nlambda]),
                            wavelength=spectralDensity[nlambda],
                            N=Np,
                            L=Lp,
                            on_gpu=on_gpu,
                        )
                        for nslice in range(1)
                    ]
                    for npsm in range(npsm)
                ]
                for nosm in range(nosm)
            ]
            for nlambda in range(nlambda)
        ]
    )
    if on_gpu:
        transferFunction = cp.array(transferFunction)
    quadraticPhase = __make_quad_phase(zo, spectralDensity[0], Np, dxp, on_gpu)
    return transferFunction, quadraticPhase


forward_lookup_dictionary = {
    "fraunhofer": propagate_fraunhofer,
    "fresnel": propagate_fresnel,
    "asp": propagate_ASP,
    "polychromeasp": propagate_polychromeASP,
    "scaledasp": propagate_scaledASP,
    "scaledpolychromeasp": propagate_scaledPolychromeASP,
    "twosteppolychrome": propagate_twoStepPolychrome,
    "identity": propagate_identity,
}


reverse_lookup_dictionary = {
    "fraunhofer": propagate_fraunhofer_inv,
    "fresnel": propagate_fresnel_inv,
    "asp": propagate_ASP_inv,
    "polychromeasp": propagate_polychromeASP_inv,
    "scaledasp": propagate_scaledASP_inv,
    "scaledpolychromeasp": propagate_scaledPolychromeASP_inv,
    "twosteppolychrome": propagate_twoStepPolychrome_inv,
    "identity": propagate_identity,
}
