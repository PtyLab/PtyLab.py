from functools import lru_cache

import numpy as np

from PtyLab import Params, Reconstruction
from PtyLab.Operators._propagation_kernels import __make_quad_phase
from PtyLab.Operators.propagator_utils import (complexexp, convolve2d,
                                               gaussian2D, iterate_6d_fields)
from PtyLab.utils.gpuUtils import getArrayModule
from PtyLab.utils.utils import fft2c, ifft2c

try:
    import cupy as cp
except ImportError:
    cp = None

CACHE_SIZE = 5


def propagate_off_axis_sas(
    fields,
    params: Params,
    reconstruction: Reconstruction,
    z: float = None,
):
    """
    Scalable Off-axis Angular Spectrum (SOAS) Propagation method that assumes that the source and
    destination planes are coplanar, but off-axis.

    Parameters
    ----------
    fields: np.ndarray
        Field to propagate
    params: Params
        Instance of the Params class
    reconstruction: Reconstruction
        Instance of the Reconstruction class.
    z: float
        Propagation distance

    Returns
    -------
    reconstruction.esw, propagated field:
        Exit surface wave and the propagated field
    """

    xp = getArrayModule(fields)
    # pad the original field (last 2 dimensions) with zeros to be twice it's size
    # no padding in the first 4 dimensions
    
    # ideally pad factor is 2, however can be modiefied by user.
    # NOTE: Hacky way of getting this, as the Reconstruction class is not flexible enough,
    # to add more attributes (constrained by the design of PtyLab). 
    try:
        pad_factor = reconstruction.pad_factor
    except AttributeError:
        pad_factor = 2
        
    rows, cols = fields.shape[-2:]
    pad_width = (
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (rows // pad_factor, rows // pad_factor),
        (cols // pad_factor, cols // pad_factor),
    )
    fields_padded = xp.pad(fields, pad_width, "constant")

    # reconstruction parameters
    Np = reconstruction.Np
    dxp = reconstruction.dxp
    wavelength = reconstruction.wavelength
    
    # specifying z1 (aspw) and z2 (Fresnel) propagator for relaxing 
    # sampling requirements. Similar issue as the NOTE on pad_factor above.
    z1 = reconstruction.zo if z is None else z
    try:
        z2 = reconstruction.z2
    except AttributeError:
        z2 = z1
        
    # quadratic phase Q2 (currently zo, but this can be z2 and z1 separated)
    quad_phase = __make_quad_phase(
        z2, wavelength, Np * pad_factor, dxp, params.gpuSwitch
    )

    # precompensated transfer function
    H_precomp = __make_transferfunction_off_axis_sas(
        params, reconstruction, pad_factor, params.gpuSwitch, z1, z2
    )

    # field propagation
    psi_precomp = ifft2c(H_precomp * fft2c(fields_padded))
    prop_fields = fft2c(quad_phase * psi_precomp)

    # crop the field by a factor of 2 (as it was originally padded by 2)
    rows_padded, cols_padded = prop_fields.shape[-2:]
    start_h, start_w = (
        (rows_padded - rows) // 2,
        (cols_padded - cols) // 2,
    )
    slicey, slicex = (
        slice(start_h, start_h + rows),
        slice(start_w, start_w + cols),
    )
    prop_fields = prop_fields[..., slicey, slicex]

    return reconstruction.esw, prop_fields


@lru_cache(CACHE_SIZE)
def __make_transferfunction_off_axis_sas(
    params: Params,
    reconstruction: Reconstruction,
    pad_factor: int,
    on_gpu: bool,
    z1: float,
    z2: float
):
    """
    Allows for a 6-dimensional (nlambda, nosm, npsm, nslice, Np, Np) array when computing the transfer function
    for a scalable off-axis angular spectrum propagator.

    Parameters
    ----------
    params: Params
        Instance of the Params class
    reconstruction: Reconstruction
        Instance of the Reconstruction class.
    z: float
        Propagation distance

    Returns
    -------
    np.ndarray or cp.ndarray
        The calculated transfer function with shape (nlambda, nosm, npsm, nslice, Np, Np).
    """

    # convert theta to a tuple of two floats.
    #   - If theta is a single number, return (float(number), 0.0)
    #   - If theta is a tuple of two numbers, convert both to float
    #   - Raise ValueError for invalid theta
    theta = reconstruction.theta
    if isinstance(theta, (int, float)):
        theta = (float(theta), 0.0)
    elif isinstance(theta, tuple) and len(theta) == 2:
        theta = (float(theta[0]), float(theta[1]))
    else:
        raise ValueError("theta must be a scalar or a tuple of two numbers")

    fftshiftSwitch = params.fftshiftSwitch
    Np = reconstruction.Np  # Pixel size along each dimension.
    wavelength = reconstruction.wavelength  # Wavelength used in the scanning probe.
    Lp = reconstruction.Lp  # length of the sample.
    nosm = reconstruction.nosm  # no. of spatial modes for the object.
    npsm = reconstruction.npsm  # no. of spatial modes for the probe.
    nlambda = reconstruction.nlambda  # no. of wavelengths for multi-wavelength.
    nslice = reconstruction.nslice  # no. of slices for multi-slice operation
    xp = cp if on_gpu else np

    # ensuring some checks
    if fftshiftSwitch:
        raise ValueError("ASP propagatorType works only with fftshiftSwitch = False!")

    if nlambda > 1:
        raise ValueError("Currently for multi-wavelength, off-axis SAS does not work")

    if nslice > 1:
        raise ValueError(
            " Currently off-axis SAS not valid for multi-slice ptychography"
        )

    # Array shape (nlambda, nosm, npsm, nslice, Np, Np)
    transfer_function = xp.zeros(
        (nlambda, nosm, npsm, nslice, Np * pad_factor, Np * pad_factor),
        dtype="complex64",
    )
    for inds in iterate_6d_fields(transfer_function):
        transfer_function[*inds] = __off_axis_sas_transfer_function(
            wavelength, Lp, pad_factor * Np, theta, z1, z2, on_gpu
        )

    return transfer_function


@lru_cache(CACHE_SIZE)
def __off_axis_sas_transfer_function(wavelength, Lp, Np, theta, z1, z2, on_gpu):
    """Precompensation transfer function for scalable off-axis transfer function.

    Parameters
    ----------
    wavelength : float
        wavelength
    Lp : float
        Physical size
    Np : float
        _description_
    theta : tuple / scalar
        Theta (angle in degrees) in the x-y plane.
    z1 : float
        propagation distance (ASPW)
    z2 : float
        propagation distance (Fresnel) - relaxing sampling requirements.
    on_gpu : bool
        checks if the array is on GPU or not.

    Returns
    -------
    np.ndarray
        precompensated transfer function `H_precomp`
    """

    # cp/np array
    xp = cp if on_gpu else np

    # Fourier grid
    df = 1 / Lp
    f = xp.arange(-Np / 2, Np / 2) * df
    Fx, Fy = xp.meshgrid(f, f)

    # off-axis sines and tangents (theta in degrees)
    thetax, thetay = theta
    sx = xp.sin(xp.radians(thetax))
    sy = xp.sin(xp.radians(thetay))
    tx = xp.tan(xp.radians(thetax))
    ty = xp.tan(xp.radians(thetay))

    # transfer function
    # eq. 12 includes chi parameter under square root
    chi = (
        1 / wavelength**2
        - (Fx + (sx / wavelength)) ** 2
        - (Fy + (sy / wavelength)) ** 2
    )
    sqrt_chi = np.sqrt(np.maximum(0, chi))

    def _create_bandpass_filter(smooth_filter=True, eps=1e-10):
        """ Creating a bandpass filter """
        
        # for the field in x-direction
        Omegax = z1 * (tx - (Fx + sx / wavelength) / (sqrt_chi + eps))
        Omegax += wavelength * z2 * Fx

        # for the field in y-direction
        Omegay = z1 * (ty - (Fy + sy / wavelength) / (sqrt_chi + eps))
        Omegay += wavelength * z2 * Fy

        # Fourier Bandpass filter (W is a mask below)
        sampling_rate = 2
        W_mask = xp.logical_and(
            df <= xp.abs(1 / (sampling_rate * Omegax + eps)), 
            df <= xp.abs(1 / (sampling_rate * Omegay + eps))
            )

        # smooth the bandpass filter corners with a Gaussian kernel
        if smooth_filter:
            kernel_gauss = gaussian2D(8, 2, on_gpu)
            bandpass_filter = convolve2d(W_mask, kernel_gauss, on_gpu, mode="same")
        else:
            bandpass_filter = W_mask

        return bandpass_filter

    # Pre-compensation transfer function

    # implements the angular spectrum transfer function (see eq. 23, part of the precompensation factor)
    # zo is z1 in the document.
    H_AS = complexexp(2 * np.pi * z1 * sqrt_chi)

    # Fresnel transfer function
    H_Fr = complexexp(
        -np.pi * z2 / wavelength * ((wavelength * Fx) ** 2 + (wavelength * Fy) ** 2)
    )

    # off-axis consideration of the transfer function
    H_offaxis = complexexp(2 * np.pi * z1 * (tx * Fx + ty * Fy))

    # precompensation with bandpass filter
    bandpass_filter = _create_bandpass_filter(smooth_filter=True, eps=1e-10)
    H_precomp = H_AS * xp.conj(H_Fr) * H_offaxis * bandpass_filter

    return H_precomp
