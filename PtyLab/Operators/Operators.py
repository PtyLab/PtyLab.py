import logging

from collections.abc import Callable

from functools import lru_cache

try:
    import cupy as cp
except ImportError:
    pass
    # print("Cupy not available, will not be able to run GPU based computation")
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
    r"""
    Propagate a field using the Fraunhofer approximation.

    $$
    u_{out} = F(u).
    $$

    Here $F$ is the unitary 2D Fourier transform, with ordering controlled
    by ``params.fftshiftSwitch``.

    The propagation is implemented as a centered 2D Fourier transform.
    The Fourier-shift convention is controlled by
    ``params.fftshiftSwitch``.

    Args:
        fields (np.ndarray):
            Electric field to propagate. The Fourier transform is applied
            along the two spatial dimensions.

        params (Params):
            Reconstruction parameters. ``params.fftshiftSwitch`` controls
            the Fourier-shift convention.

        reconstruction (Reconstruction):
            Reconstruction object containing the current exit surface wave.

        z (float, optional):
            Propagation distance. This argument is ignored for Fraunhofer
            propagation.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing ``reconstruction.esw`` and the propagated
            field obtained from the centered 2D Fourier transform of
            ``fields``.

    """
    return reconstruction.esw, fft2c(fields, params.fftshiftSwitch)


def propagate_fraunhofer_inv(
    fields, params: Params, reconstruction: Reconstruction, z=None
):
    """
    Back-propagate a field using the inverse Fraunhofer transform.

    The back-propagation is implemented as a centered 2D inverse Fourier
    transform. The Fourier-shift convention is controlled by
    ``params.fftshiftSwitch``.

    Args:
        fields (np.ndarray):
            Electric field to back-propagate. The inverse Fourier transform
            is applied along the two spatial dimensions.

        params (Params):
            Reconstruction parameters. ``params.fftshiftSwitch`` controls
            the Fourier-shift convention.

        reconstruction (Reconstruction):
            Reconstruction object containing the current exit surface wave.

        z (float, optional):
            Propagation distance. This argument is ignored for inverse
            Fraunhofer propagation.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing ``reconstruction.esw`` and the back-propagated
            field obtained from the centered 2D inverse Fourier transform
            of ``fields``.

    """
    return reconstruction.esw, ifft2c(fields, params.fftshiftSwitch)


def propagate_fresnel(fields, params: Params, reconstruction: Reconstruction, z=None):
    r"""
    Propagate a field using the Fresnel approximation.

    $$
    u_{out} = F(u Q), \qquad Q = \exp[i\pi(x^2+y^2)/(\lambda z)].
    $$

    Here $F$ is the unitary 2D Fourier transform with the configured
    ordering, and $Q$ is the source-plane quadratic phase.

    The input field is multiplied by a quadratic phase factor and then
    transformed using a centered 2D Fourier transform. If ``z`` is not
    provided, ``reconstruction.zo`` is used as the propagation distance.

    Args:
        fields (np.ndarray):
            Complex-valued field to propagate. NumPy and CuPy arrays are
            supported. The last two dimensions represent the spatial field.

        params (Params):
            Reconstruction parameters. ``params.fftshiftSwitch`` controls
            the Fourier-shift convention.

        reconstruction (Reconstruction):
            Reconstruction object providing the wavelength, pixel size, and
            default propagation distance.

        z (float, optional):
            Propagation distance in meters. If None,
            ``reconstruction.zo`` is used.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing the unchanged ``reconstruction.esw`` and the
            Fresnel-propagated field.

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
    Back-propagate a field using the inverse Fresnel approximation.

    The input field is transformed using a centered 2D inverse Fourier
    transform and multiplied by the complex conjugate of the quadratic
    phase factor. If ``z`` is not provided, ``reconstruction.zo`` is used
    as the propagation distance.

    Args:
        fields (np.ndarray):
            Complex-valued field to back-propagate. NumPy and CuPy arrays
            are supported.

        params (Params):
            Reconstruction parameters. ``params.fftshiftSwitch`` controls
            the Fourier-shift convention.

        reconstruction (Reconstruction):
            Reconstruction object providing the wavelength, pixel size,
            field size, and default propagation distance.

        z (float, optional):
            Propagation distance in meters. If None,
            ``reconstruction.zo`` is used.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing the unchanged ``reconstruction.esw`` and the
            back-propagated field.

    """
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
    fields,
    params: Params,
    reconstruction: Reconstruction,
    inverse=False,
    z=None,
    fftflag=True,
):
    r"""
    Propagate a field using the angular spectrum method.

    $$
    u_{out} = F^{-1}[H F(u)].
    $$

    Here $F$ is the unitary 2D Fourier transform and $H$ is the
    band-limited angular-spectrum transfer function in matching ordering.
    This equation describes forward propagation (``inverse=False``).

    The field is transformed to the spatial-frequency domain, multiplied
    by an angular-spectrum transfer function, and transformed back to the
    spatial domain. If ``z`` is not provided, ``reconstruction.zo`` is
    used as the propagation distance.

    This propagator supports single-wavelength reconstruction only and
    requires ``params.fftshiftSwitch`` to be False.

    Args:
        fields (np.ndarray):
            Complex-valued field to propagate. NumPy and CuPy arrays are
            supported.

        params (Params):
            Reconstruction parameters. ``params.fftshiftSwitch`` must be
            False when using this propagator.

        reconstruction (Reconstruction):
            Reconstruction object providing the wavelength, propagation
            grid, field size, and default propagation distance.

        inverse (bool, optional):
            If True, use the complex conjugate of the transfer function
            for back-propagation.
            Defaults to False.

        z (float, optional):
            Propagation distance in meters. If None,
            ``reconstruction.zo`` is used.
            Defaults to None.

        fftflag (bool, optional):
            If True, apply centered Fourier transforms internally and
            shift the transfer function accordingly.
            Defaults to True.

    Returns:
        tuple:
            Tuple containing the unchanged ``reconstruction.esw`` and the
            propagated field.

    Raises:
        ValueError:
            If ``params.fftshiftSwitch`` is True or if more than one
            wavelength is used.

    """

    if params.fftshiftSwitch:
        raise ValueError("ASP propagator only works with fftshiftswitch == False")
    if reconstruction.nlambda > 1:
        raise ValueError("For multi-wavelength, set polychromeASP instead of ASP")
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
    result = ifft2c(
        fft2c(fields, fftshiftSwitch=fftflag) * transfer_function,
        fftshiftSwitch=fftflag,
    )
    return reconstruction.esw, result


def propagate_ASP_inv(*args, **kwargs):
    """
    Back-propagate a field using the angular spectrum method.

    This function forwards all arguments to ``propagate_ASP`` with
    ``inverse=True``.

    Args:
        *args:
            Positional arguments forwarded to ``propagate_ASP``.

        **kwargs:
            Keyword arguments forwarded to ``propagate_ASP``.

    Returns:
        tuple:
            Tuple containing the unchanged ``reconstruction.esw`` and the
            back-propagated field.

    """
    return propagate_ASP(*args, **kwargs, inverse=True)


def propagate_twoStepPolychrome(
    fields, params: Params, reconstruction: Reconstruction, inverse=False, z=None
):
    r"""
    Propagate a multi-wavelength field using the two-step polychrome method.

    $$
    u_{out,\lambda} = F\left[Q\,F^{-1}[H_\lambda F(u_\lambda)]\right].
    $$

    Here $F$ is the centered unitary 2D Fourier transform, $H_\lambda$
    propagates over $z(1-\lambda_0/\lambda)$, and $Q$ is the quadratic
    phase for reference wavelength $\lambda_0$ at distance $z$.
    This equation describes forward propagation (``inverse=False``).

    The propagation combines wavelength-dependent angular-spectrum
    propagation with a quadratic phase correction and a final Fourier
    transform. The first entry of ``reconstruction.spectralDensity`` is
    used as the reference wavelength.

    If ``z`` is not provided, ``reconstruction.zo`` is used as the
    propagation distance. This propagator requires
    ``params.fftshiftSwitch`` to be False.

    Args:
        fields (np.ndarray):
            Complex-valued field to propagate. NumPy and CuPy arrays are
            supported.

        params (Params):
            Reconstruction parameters. ``params.fftshiftSwitch`` must be
            False when using this propagator.

        reconstruction (Reconstruction):
            Reconstruction object providing the spectral density,
            propagation geometry, sampling parameters, and mode
            dimensions.

        inverse (bool, optional):
            If True, apply the conjugated propagation factors for
            back-propagation.
            Defaults to False.

        z (float, optional):
            Propagation distance in meters. If None,
            ``reconstruction.zo`` is used.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing the unchanged ``reconstruction.esw`` and the
            propagated field.

    Raises:
        ValueError:
            If ``params.fftshiftSwitch`` is True.

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
        # device placement follows the data, not the global switch
        isGpuArray(fields),
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
    Back-propagate a field using the two-step polychrome method.

    Both the input field and the detector-plane field stored in
    ``reconstruction.ESW`` are back-propagated. The latter is returned as
    the first element of the result, while the back-propagated input field
    is returned as the second element.

    Args:
        fields (np.ndarray):
            Complex-valued detector-plane field to back-propagate. NumPy
            and CuPy arrays are supported.

        params (Params):
            Reconstruction parameters passed to
            ``propagate_twoStepPolychrome``.

        reconstruction (Reconstruction):
            Reconstruction object containing the detector-plane field
            ``reconstruction.ESW`` and the propagation parameters.

        z (float, optional):
            Propagation distance in meters. If None,
            ``reconstruction.zo`` is used.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing the back-propagated
            ``reconstruction.ESW`` and the back-propagated input field.

    """
    F = propagate_twoStepPolychrome(fields, params, reconstruction, inverse=True, z=z)[
        1
    ]
    G = propagate_twoStepPolychrome(
        reconstruction.ESW, params, reconstruction, inverse=True, z=z
    )[
        1
    ]  # tODO: What is G here? Why are we not returning reconstruction.esw?
    return G, F


def propagate_scaledASP(
    fields, params: Params, reconstruction: Reconstruction, inverse=False, z=None
):
    r"""
    Propagate a field using the scaled angular spectrum method.

    $$
    u_{out} = F^{-1}[Q_2 F(Q_1 u)].
    $$

    Here $F$ is the centered unitary 2D Fourier transform; $Q_1$ and
    $Q_2$ are the source-plane and frequency-domain scaled Fresnel
    factors, including sampling masks. This describes ``inverse=False``.

    The propagation uses two phase factors to account for different
    sampling intervals in the object and detector planes. If ``z`` is not
    provided, ``reconstruction.zo`` is used as the propagation distance.

    This propagator supports single-wavelength reconstruction only and
    requires ``params.fftshiftSwitch`` to be False.

    Args:
        fields (np.ndarray):
            Complex-valued field to propagate. NumPy and CuPy arrays are
            supported.

        params (Params):
            Reconstruction parameters. ``params.fftshiftSwitch`` must be
            False when using this propagator.

        reconstruction (Reconstruction):
            Reconstruction object providing the wavelength, propagation
            distance, mode dimensions, and object- and detector-plane
            sampling intervals.

        inverse (bool, optional):
            If True, use the complex conjugates of the propagation factors
            for back-propagation.
            Defaults to False.

        z (float, optional):
            Propagation distance in meters. If None,
            ``reconstruction.zo`` is used.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing the unchanged ``reconstruction.esw`` and the
            propagated field.

    Raises:
        ValueError:
            If ``params.fftshiftSwitch`` is True or if more than one
            wavelength is used.

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
        # device placement follows the data, not the global switch
        isGpuArray(fields),
    )
    if inverse:
        Q1, Q2 = Q1.conj(), Q2.conj()
        return reconstruction.esw, ifft2c(fft2c(fields) * Q2) * Q1
    return reconstruction.esw, ifft2c(fft2c(fields * Q1) * Q2)


def propagate_scaledASP_inv(
    fields, params: Params, reconstruction: Reconstruction, z=None
):
    """
    Back-propagate a field using the scaled angular spectrum method.

    This function forwards the input arguments to ``propagate_scaledASP``
    with ``inverse=True``.

    Args:
        fields (np.ndarray):
            Complex-valued field to back-propagate. NumPy and CuPy arrays
            are supported.

        params (Params):
            Reconstruction parameters passed to ``propagate_scaledASP``.

        reconstruction (Reconstruction):
            Reconstruction object providing the propagation geometry and
            sampling parameters.

        z (float, optional):
            Propagation distance in meters. If None,
            ``reconstruction.zo`` is used.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing the unchanged ``reconstruction.esw`` and the
            back-propagated field.

    """
    return propagate_scaledASP(fields, params, reconstruction, inverse=True, z=z)


def propagate_scaledPolychromeASP(
    fields, params: Params, reconstruction: Reconstruction, inverse=False, z=None
):
    r"""
    Propagate a multi-wavelength field using the scaled angular spectrum method.

    $$
    u_{out,\lambda} = F^{-1}[Q_{2,\lambda} F(Q_{1,\lambda} u_\lambda)].
    $$

    Here $F$ is the centered unitary 2D Fourier transform; $Q_{1,\lambda}$
    and $Q_{2,\lambda}$ are wavelength-dependent scaled Fresnel factors,
    including sampling masks. This describes ``inverse=False``.

    Separate propagation factors are constructed for each wavelength in
    ``reconstruction.spectralDensity`` to account for wavelength-dependent
    propagation and different sampling intervals in the object and detector
    planes. If ``z`` is not provided, ``reconstruction.zo`` is used as the
    propagation distance.

    This propagator requires ``params.fftshiftSwitch`` to be False.

    Args:
        fields (np.ndarray):
            Complex-valued field to propagate. NumPy and CuPy arrays are
            supported.

        params (Params):
            Reconstruction parameters. ``params.fftshiftSwitch`` must be
            False when using this propagator.

        reconstruction (Reconstruction):
            Reconstruction object providing the spectral density, propagation
            distance, mode dimensions, and object- and detector-plane sampling
            intervals.

        inverse (bool, optional):
            If True, use the complex conjugates of the propagation factors
            for back-propagation.
            Defaults to False.

        z (float, optional):
            Propagation distance in meters. If None,
            ``reconstruction.zo`` is used.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing the unchanged ``reconstruction.esw`` and the
            propagated field.

    Raises:
        ValueError:
            If ``params.fftshiftSwitch`` is True.

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
        reconstruction.dxd,
        # device placement follows the data, not the global switch
        isGpuArray(fields),
    )
    if inverse:
        Q1, Q2 = Q1.conj(), Q2.conj()
        return reconstruction.esw, ifft2c(fft2c(fields) * Q2) * Q1
    return reconstruction.esw, ifft2c(fft2c(fields * Q1) * Q2)


def propagate_scaledPolychromeASP_inv(
    fields, params: Params, reconstruction: Reconstruction, z=None
):
    """
    Back-propagate a multi-wavelength field using the scaled angular
    spectrum method.

    This function forwards the input arguments to
    ``propagate_scaledPolychromeASP`` with ``inverse=True``.

    Args:
        fields (np.ndarray):
            Complex-valued field to back-propagate. NumPy and CuPy arrays
            are supported.

        params (Params):
            Reconstruction parameters passed to
            ``propagate_scaledPolychromeASP``.

        reconstruction (Reconstruction):
            Reconstruction object providing the spectral density,
            propagation geometry, and sampling parameters.

        z (float, optional):
            Propagation distance in meters. If None,
            ``reconstruction.zo`` is used.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing the unchanged ``reconstruction.esw`` and the
            back-propagated field.

    """
    return propagate_scaledPolychromeASP(
        fields, params, reconstruction, inverse=True, z=z
    )


def propagate_polychromeASP(
    fields, params: Params, reconstruction: Reconstruction, inverse=False, z=None
):
    r"""
    Propagate a multi-wavelength field using the angular spectrum method.

    $$
    u_{out,\lambda} = F^{-1}[H_\lambda F(u_\lambda)].
    $$

    Here $F$ is the centered unitary 2D Fourier transform and $H_\lambda$
    is the band-limited angular-spectrum kernel for each wavelength.
    This equation describes forward propagation (``inverse=False``).

    A wavelength-dependent transfer function is constructed for each entry
    in ``reconstruction.spectralDensity``. If ``z`` is not provided,
    ``reconstruction.zo`` is used as the propagation distance.

    This propagator requires ``params.fftshiftSwitch`` to be False.

    Args:
        fields (np.ndarray):
            Complex-valued field to propagate. NumPy and CuPy arrays are
            supported.

        params (Params):
            Reconstruction parameters. ``params.fftshiftSwitch`` must be
            False when using this propagator.

        reconstruction (Reconstruction):
            Reconstruction object providing the spectral density,
            propagation geometry, and mode dimensions.

        inverse (bool, optional):
            If True, use the complex conjugate of the transfer function
            for back-propagation.
            Defaults to False.

        z (float, optional):
            Propagation distance in meters. If None,
            ``reconstruction.zo`` is used.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing the unchanged ``reconstruction.esw`` and the
            propagated field.

    Raises:
        ValueError:
            If ``params.fftshiftSwitch`` is True.

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
        # device placement follows the data, not the global switch
        isGpuArray(fields),
    )

    if inverse:
        transfer_function = transfer_function.conj()
    result = ifft2c(fft2c(fields) * transfer_function)
    return reconstruction.esw, result


def propagate_identity(
    fields, params: Params, reconstruction: Reconstruction, inverse=False, z=None
):
    """
    Return the input field without applying propagation.

    Args:
        fields (np.ndarray):
            Complex-valued field to return unchanged. NumPy and CuPy arrays
            are supported.

        params (Params):
            Reconstruction parameters. This argument is accepted for
            compatibility with the common propagator interface and is not
            used.

        reconstruction (Reconstruction):
            Reconstruction object containing the current exit surface wave.

        inverse (bool, optional):
            Accepted for compatibility with the common propagator interface
            and ignored.
            Defaults to False.

        z (float, optional):
            Accepted for compatibility with the common propagator interface
            and ignored.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing ``reconstruction.esw`` and the unchanged input
            field.

    """
    return reconstruction.esw, fields


def propagate_polychromeASP_inv(fields, params, reconstruction, z=None):
    """
    Back-propagate a multi-wavelength field using the angular spectrum method.

    This function forwards the input arguments to
    ``propagate_polychromeASP`` with ``inverse=True``.

    Args:
        fields (np.ndarray):
            Complex-valued field to back-propagate. NumPy and CuPy arrays
            are supported.

        params (Params):
            Reconstruction parameters passed to
            ``propagate_polychromeASP``.

        reconstruction (Reconstruction):
            Reconstruction object providing the spectral density,
            propagation geometry, and mode dimensions.

        z (float, optional):
            Propagation distance in meters. If None,
            ``reconstruction.zo`` is used.
            Defaults to None.

    Returns:
        tuple:
            Tuple containing the unchanged ``reconstruction.esw`` and the
            back-propagated field.

    """
    return propagate_polychromeASP(fields, params, reconstruction, inverse=True, z=z)


def detector2object(fields, params: Params, reconstruction: Reconstruction):
    """
    Propagate a field from the detector plane to the object plane.

    The inverse propagation method is selected from
    ``params.propagatorType``. If ``fields`` is None, the detector-plane
    field stored in ``reconstruction.ESW`` is used.

    Args:
        fields (np.ndarray or None):
            Detector-plane field to back-propagate. If None,
            ``reconstruction.ESW`` is used.

        params (Params):
            Reconstruction parameters. ``params.propagatorType`` selects
            the inverse propagation method.

        reconstruction (Reconstruction):
            Reconstruction object providing the propagation parameters and
            detector-plane field.

    Returns:
        tuple:
            Result returned by the selected inverse propagator, typically
            containing an object-plane exit surface wave and the
            back-propagated field.

    Raises:
        KeyError:
            If ``params.propagatorType`` does not match a registered
            inverse propagator.

    """
    if fields is None:
        fields = reconstruction.ESW
    method: Callable[[np.ndarray, Params], Reconstruction] = reverse_lookup_dictionary[
        params.propagatorType.lower()
    ]
    return method(fields, params, reconstruction)


def object2detector(fields, params: Params, reconstruction: Reconstruction):
    """
    Propagate a field from the object plane to the detector plane.

    The forward propagation method is selected from
    ``params.propagatorType``. If ``fields`` is None, the exit surface
    wave stored in ``reconstruction.esw`` is used.

    Args:
        fields (np.ndarray or None):
            Object-plane field to propagate. If None,
            ``reconstruction.esw`` is used.

        params (Params):
            Reconstruction parameters. ``params.propagatorType`` selects
            the forward propagation method.

        reconstruction (Reconstruction):
            Reconstruction object providing the propagation parameters and
            current exit surface wave.

    Returns:
        tuple:
            Result returned by the selected forward propagator, typically
            containing ``reconstruction.esw`` and the propagated
            detector-plane field.

    Raises:
        KeyError:
            If ``params.propagatorType`` does not match a registered
            forward propagator.

    """
    method: Callable[[np.ndarray, Params], Reconstruction] = forward_lookup_dictionary[
        params.propagatorType.lower()
    ]
    if fields is None:
        fields = reconstruction.esw
    return method(fields, params, reconstruction)


def aspw(u, z, wavelength, L, bandlimit=True, is_FT=False):
    r"""
    Propagate a field using the angular spectrum plane-wave method.

    $$
    u_{out} = F^{-1}(H U), \qquad
    U = \begin{cases}F(u), & \text{if is\_FT=False},\\
    u, & \text{if is\_FT=True}.\end{cases}
    $$

    Here $F$ is the centered unitary 2D Fourier transform and $H$ is
    the angular-spectrum transfer function returned by this function.

    The propagation uses a band-limited angular-spectrum transfer function
    to reduce sampling artifacts in numerical free-space propagation.[^matsushima2009]
    The input is assumed to be sampled on a square grid.

    [^matsushima2009]: K. Matsushima and T. Shimobaba,
        "Band-Limited Angular Spectrum Method for Numerical Simulation
        of Free-Space Propagation in Far and Near Fields,"
        Opt. Express 17, 19662-19673 (2009).
        <https://doi.org/10.1364/OE.17.019662>

    Args:
        u (np.ndarray):
            Complex-valued field or its Fourier transform. NumPy and CuPy
            arrays are supported. The last two dimensions are assumed to
            describe a square spatial grid.

        z (float):
            Propagation distance in meters. Negative values correspond to
            back-propagation.

        wavelength (float):
            Wavelength in meters.

        L (float):
            Physical side length of the sampled field in meters.

        bandlimit (bool, optional):
            If True, apply a band limit to the angular-spectrum transfer
            function to reduce sampling artifacts.
            Defaults to True.

        is_FT (bool, optional):
            If True, treat ``u`` as an already Fourier-transformed field.
            If False, apply a centered 2D Fourier transform before
            propagation.
            Defaults to False.

    Returns:
        tuple:
            Tuple containing the propagated field and the angular-spectrum
            transfer function.

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
    r"""
    Construct the angular-spectrum transfer function for free-space propagation.

    The transfer function is evaluated as

    $$
    H(f_x, f_y; z) = \exp\left[i k z \sqrt{1-(\lambda f_x)^2-(\lambda f_y)^2}\right]
    $$

    where $k = 2\pi / \lambda$. Negative propagation distances are handled by
    complex conjugation of the forward transfer function. The result is cached
    for repeated use.

    Args:
        z (float):
            Propagation distance in meters.

        wavelength (float):
            Wavelength in meters.

        N (int):
            Number of pixels along each spatial dimension.

        L (float):
            Physical side length of the sampled field in meters.

        on_gpu (bool, optional):
            If True, construct and return the transfer function as a CuPy
            array. Otherwise, return a NumPy array.
            Defaults to False.

        bandlimit (bool, optional):
            If True, apply the sampling-related band-limiting conditions
            used by the angular-spectrum propagation model.
            Defaults to True.

    Returns:
        np.ndarray:
            Complex-valued angular-spectrum transfer function with shape
            ``(N, N)``. A CuPy array is returned when ``on_gpu`` is True.

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
    r"""
    Compute the complex exponential of a phase angle elementwise.

    Uses Euler's identity,

    $$
    e^{i\theta} = \cos(\theta) + i\sin(\theta),
    $$

    to construct a complex phase factor. For real-valued input, the
    result has unit magnitude up to floating-point precision.
    NumPy and CuPy arrays are supported.

    Args:
        angle (float or ndarray):
            Phase angle in radians. Array inputs may have any shape.

    Returns:
        complex or ndarray:
            Complex phase factor with the same shape as the input.
            CuPy inputs produce CuPy outputs; otherwise, NumPy is used.

    Examples:
        >>> import numpy as np
        >>> phase = complexexp(np.array([0.0, np.pi / 2, np.pi]))
        >>> np.allclose(phase, [1.0, 1j, -1.0])
        True
    """
    xp = getArrayModule(angle)
    return xp.cos(angle) + 1j * xp.sin(angle)


def scaledASP(u, z, wavelength, dx, dq, bandlimit=True, exactSolution=False):
    r"""
    Propagate a field with independently specified input and output pixel sizes.

    Uses scaled angular-spectrum propagation within the Fresnel (paraxial)
    approximation. The output retains the input grid shape, with pixel size
    ``dq`` instead of ``dx``. For the scaling ratio $m = dq / dx$, propagation
    is evaluated as

    $$
    U_{\mathrm{out}} = \mathcal{F}^{-1}
        \left[Q_2\,\mathcal{F}\left[Q_1 u\right]\right],
    $$

    where $Q_1$ is a source-plane quadratic phase factor and $Q_2$ is a
    spatial-frequency quadratic phase factor. If ``exactSolution=True``,
    the result is also multiplied by the output-plane phase factor $Q_3$.
    This restores the output phase within the paraxial model; it does not
    switch to nonparaxial propagation. Omitting $Q_3$ leaves the intensity
    unchanged because this factor has unit magnitude for real parameters.

    Args:
        u (np.ndarray):
            Complex-valued input field on a square grid of shape ``(N, N)``.

        z (float):
            Nonzero propagation distance in meters.

        wavelength (float):
            Positive wavelength in meters.

        dx (float):
            Positive source-plane pixel size in meters.

        dq (float):
            Positive destination-plane pixel size in meters.

        bandlimit (bool, optional):
            Apply circular sampling masks to ``Q1`` and ``Q2`` to limit
            sampling artifacts. The mask on ``Q1`` is skipped when
            ``dq == dx``. Defaults to True.

        exactSolution (bool, optional):
            Include the output-plane quadratic phase factor ``Q3`` and
            return it as a fourth tuple element. Defaults to False.

    Returns:
        tuple:
            ``(Uout, Q1, Q2)`` when ``exactSolution=False``, or
            ``(Uout, Q1, Q2, Q3)`` when ``exactSolution=True``.
            ``Uout`` is the propagated complex field sampled at pixel size
            ``dq``. The phase factors have shape ``(N, N)``; ``Q1`` and
            ``Q2`` include the sampling masks when band limiting is enabled.

    Examples:
        >>> u = np.ones((64, 64), dtype=complex)
        >>> u_out, q1, q2 = scaledASP(
        ...     u, z=0.01, wavelength=500e-9, dx=5e-6, dq=4e-6
        ... )
        >>> u_out.shape
        (64, 64)
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
    r"""
    Back-propagate a field using conjugated scaled Fresnel phase factors.

    Reverses the unmasked scaled propagation performed by ``scaledASP``
    with ``exactSolution=False``. The backward operation is

    $$
    u_{\mathrm{back}} =
    Q_1^*\,\mathcal{F}^{-1}
    \left[Q_2^*\,\mathcal{F}[u]\right],
    $$

    where $Q_1$ and $Q_2$ are constructed from the original forward
    propagation geometry.

    Pass the same propagation distance and pixel sizes as in the forward
    call. Do not negate ``z`` or interchange ``dx`` and ``dq``.

    Args:
        u (np.ndarray):
            Complex-valued destination-plane field sampled at pixel size
            ``dq``. Use a square array of shape ``(N, N)`` with even ``N``.

        z (float):
            Nonzero propagation distance used in the forward operation,
            in meters.

        wavelength (float):
            Positive wavelength in meters.

        dx (float):
            Positive source-plane pixel size from the forward operation,
            in meters. This is the pixel size of the returned field.

        dq (float):
            Positive destination-plane pixel size from the forward
            operation, in meters. This is the pixel size of ``u``.

    Returns:
        np.ndarray:
            Back-propagated complex field with the same shape as ``u``,
            sampled at pixel size ``dx``.

    Notes:
        This function uses the Fresnel (paraxial) approximation and does
        not apply band-limit masks. It is the inverse of the unmasked
        forward operator for even-sized grids, up to numerical precision.
        Information discarded by forward band limiting is not recovered.

        If the forward calculation used ``exactSolution=True``, multiply
        its output by ``Q3.conj()`` before passing it to this function.

        Odd-sized grids currently use coordinates inconsistent with
        ``scaledASP`` and should be avoided.

    Examples:
        >>> u = np.ones((32, 32), dtype=complex)
        >>> v, _, _ = scaledASP(
        ...     u, 0.01, 500e-9, 5e-6, 4e-6,
        ...     bandlimit=False, exactSolution=False
        ... )
        >>> recovered = scaledASPinv(v, 0.01, 500e-9, 5e-6, 4e-6)
        >>> np.allclose(recovered, u)
        True
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
    r"""
    Propagate a field using a single-FFT Fresnel approximation.

    Multiplies the source field by a quadratic phase factor ``Q1``,
    applies a centered orthonormal Fourier transform, and multiplies
    the result by the destination-plane phase factor ``Q2`` and
    ``1 / (1j * wavelength * z)``.

    The source and destination pixel sizes are related by

    $$
    dx = \frac{L}{N}, \qquad dq = \frac{\lambda z}{L}.
    $$

    Unlike ``scaledASP``, this method does not allow the destination
    pixel size to be chosen independently.

    Args:
        u (ndarray):
            Source-plane field on a square grid of shape ``(N, N)``.
            NumPy and CuPy arrays are supported. Use even ``N`` because
            the current coordinate construction is inconsistent with
            centered FFT bins for odd-sized grids.

        z (float):
            Nonzero propagation distance in meters. Positive values
            give a positive destination pixel spacing.

        wavelength (float):
            Positive wavelength in meters.

        L (float):
            Positive physical side length of the source grid in meters.
            The source pixel size is ``L / N``.

    Returns:
        tuple:
            ``(u_out, dq, Q1, Q2)``, containing the propagated field,
            destination pixel spacing in meters, source-plane quadratic
            phase factor, and destination-plane quadratic phase factor.
            All returned arrays have shape ``(N, N)`` and use the same
            NumPy or CuPy backend as the input.

    Notes:
        This standalone helper is currently unused by the reconstruction
        workflow. The repository contains an unused import and a skipped
        test; reconstruction uses ``propagate_fresnel`` and
        ``propagate_fresnel_inv`` instead. External usage has not been
        assessed.

        This implementation uses the paraxial approximation and does
        not apply explicit band limiting or zero-padding.

        Absolute physical amplitude normalization is incomplete:
        the orthonormal FFT requires an additional factor ``N * dx**2``
        to represent the discretized Fresnel integral. The global
        propagation phase ``exp(1j * k * z)`` is also omitted.

        Zero-distance propagation is not supported.

    Examples:
        >>> u = np.ones((64, 64), dtype=complex)
        >>> u_out, dq, q1, q2 = fresnelPropagator(
        ...     u, z=0.01, wavelength=500e-9, L=320e-6
        ... )
        >>> u_out.shape
        (64, 64)
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
    Q1 = xp.exp(1j * k / (2 * z) * (X**2 + Y**2))  # quadratic phase inside the integral
    Q2 = xp.exp(1j * k / (2 * z) * (Qx**2 + Qy**2))

    # pre-factor
    A = 1 / (1j * wavelength * z)

    # Fresnel-Kirchhoff integral
    u_out = A * Q2 * fft2c(u * Q1)
    return u_out, dq, Q1, Q2


def clear_cache(logger: logging.Logger = None):
    """
    Clear cached propagation kernels and quadratic phase factors.

    Clears the seven registered propagation-helper caches, including
    the shared quadratic-phase cache imported from
    ``PtyLab.Operators._propagation_kernels``. Both CPU and GPU cache
    entries are removed, and cache hit/miss statistics are reset.

    Subsequent propagation calls recompute the required arrays.

    Args:
        logger (logging.Logger, optional):
            Logger used to report cache statistics at DEBUG level and
            cache-clearing messages at INFO level. If None, no messages
            are emitted by this function. Defaults to None.

    Notes:
        This is a manual maintenance utility; no internal calls were
        found in the current reconstruction workflow.

        Clearing the caches removes their references to stored arrays.
        Arrays referenced elsewhere remain allocated. This function
        does not explicitly release memory retained by CuPy's memory
        pool, so reported GPU memory usage may not immediately decrease.

        Avoid clearing caches inside reconstruction loops, since this
        forces repeated computation of propagation factors.

    Examples:
        >>> from PtyLab.Operators.Operators import clear_cache
        >>> clear_cache()
    """
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
    """
    Build and cache a mode-shaped angular-spectrum transfer function.

    Constructs a band-limited two-dimensional kernel through
    ``__aspw_transfer_function`` and repeats it across the object and
    probe modes. Used internally by ``propagate_ASP`` for forward and
    backward propagation.

    The kernel is constructed on the CPU, converted to complex64, and
    optionally transferred to the GPU. Its spatial-frequency ordering
    is centered; the calling propagator handles any required shifting.

    Args:
        fftshiftSwitch (bool):
            Propagation configuration flag. Must be False.

        nosm (int):
            Number of object modes.

        npsm (int):
            Number of probe modes.

        Np (int):
            Number of pixels along each side of the square propagation
            grid. Use an even value because the underlying kernel's
            frequency coordinates are inconsistent with FFT bins for
            odd-sized grids.

        zo (float):
            Propagation distance in meters.

        wavelength (float):
            Positive illumination wavelength in meters.

        Lp (float):
            Positive physical side length of the propagation grid in
            meters.

        nlambda (int):
            Number of wavelengths. This helper is intended for
            single-wavelength propagation; pass 1.

        on_gpu (bool):
            If True, return a CuPy array. Otherwise, return a NumPy
            array.

    Returns:
        ndarray:
            Complex64 transfer function with shape
            ``(nlambda, nosm, npsm, 1, Np, Np)``. The same spatial
            kernel is repeated across all object and probe modes.

    Raises:
        ValueError:
            If ``fftshiftSwitch`` is True or ``nlambda`` is greater
            than 1.

    Notes:
        Results are cached using all function arguments as the key.
        Treat the returned array as read-only: in-place modifications
        would affect subsequent calls that retrieve the same entry.

        The underlying kernel uses its default band limiting.
        Backward propagation is implemented by the caller through
        complex conjugation; filtered-out information is not recovered.
    """
    if fftshiftSwitch:
        raise ValueError("ASP propagatorType works only with fftshiftSwitch = False!")
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
    """
    Propagate a spatial-domain field using a cached angular-spectrum kernel.

    Applies a centered Fourier transform to the input, multiplies it by
    the band-limited transfer function from ``__aspw_transfer_function``,
    and transforms the result back to the spatial domain. NumPy and CuPy
    arrays are supported.

    Only the transfer function is cached. The Fourier transforms and
    propagated field are computed on every call. ``aspw`` also uses the
    cached kernel; this helper provides a field-only return value with
    band limiting always enabled.

    Args:
        u (ndarray):
            Spatial-domain input field with square final two dimensions
            ``(N, N)``. Any leading dimensions are treated as batches.

        z (float):
            Propagation distance in meters. Negative values use the
            complex conjugate of the positive-distance kernel.

        wavelength (float):
            Positive wavelength in meters.

        L (float):
            Positive physical side length of the spatial grid in meters.
            The input and output pixel sizes are both ``L / N``.

    Returns:
        ndarray:
            Propagated complex field with the same shape and NumPy or
            CuPy backend as the input. The transfer function is not
            returned.

    Notes:
        For ordinary scalar geometry parameters, this is equivalent to
        taking the first return value of
        ``aspw(u, z, wavelength, L, bandlimit=True, is_FT=False)``.
        The input must not already be Fourier-transformed.

        Band limiting removes spatial frequencies, so backward
        propagation does not generally recover the original field.
        Use even-sized grids because the underlying kernel currently
        uses frequency coordinates inconsistent with FFT bins for odd N.

    Examples:
        >>> u = np.ones((64, 64), dtype=complex)
        >>> u_out = aspw_cached(u, z=0.01, wavelength=500e-9, L=320e-6)
        >>> u_out.shape
        (64, 64)
    """
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
    """
    Build and cache angular-spectrum kernels for multiple wavelengths.

    Constructs a band-limited kernel for each wavelength in
    ``spectralDensity_as_tuple`` using ``__aspw_transfer_function``.
    Each kernel is repeated across the object and probe modes.

    Used internally by ``propagate_polychromeASP``. Kernels are
    constructed on the CPU and optionally transferred to the GPU.

    Args:
        propagatorType (str):
            Propagator identifier. Currently unused in the calculation,
            but included in the cache key.

        fftshiftSwitch (bool):
            Propagation configuration flag. Must be False.

        nosm (int):
            Number of object modes.

        npsm (int):
            Number of probe modes.

        Np (int):
            Number of pixels along each side of the square propagation
            grid. Use even values because the underlying kernel's
            frequency coordinates are inconsistent with FFT bins for
            odd-sized grids.

        zo (float):
            Propagation distance in meters, shared by all wavelengths.

        wavelength (float):
            Reference wavelength. Currently unused in the calculation,
            but included in the cache key. Actual propagation wavelengths
            come from ``spectralDensity_as_tuple``.

        Lp (float):
            Positive physical side length of the propagation grid in
            meters, shared by all wavelength channels.

        nlambda (int):
            Number of wavelength channels.

        spectralDensity_as_tuple (tuple[float, ...]):
            Wavelengths in meters, ordered to match the field's wavelength
            axis. Supply one positive wavelength per channel.
            These entries are wavelengths, not spectral intensity weights.
            A tuple is required for compatibility with the cache.

        gpuSwitch (bool):
            If True, return a CuPy array with dtype complex64.
            Otherwise, return a NumPy array retaining the kernel dtype.

    Returns:
        ndarray:
            Centered angular-spectrum transfer functions with shape
            ``(nlambda, nosm, npsm, 1, Np, Np)``. Kernels differ between
            wavelengths but are identical across object and probe modes
            within each wavelength channel.

    Raises:
        ValueError:
            If ``fftshiftSwitch`` is True.

    Notes:
        Results are cached using all arguments as the key. Treat the
        returned array as read-only to avoid modifying cached results.

        The underlying kernels use their default band limiting.
        This helper neither propagates fields nor combines wavelength
        channels; it only constructs their transfer functions.
    """
    spectralDensity = np.array(spectralDensity_as_tuple)
    if fftshiftSwitch:
        raise ValueError("ASP propagatorType works only with fftshiftSwitch = False!")
    dummy = np.ones((nlambda, nosm, npsm, 1, Np, Np), dtype="complex64")
    transferFunction = np.array(
        [
            [
                [
                    [
                        __aspw_transfer_function(
                            zo,
                            spectralDensity[nlambda],
                            Np,
                            Lp,
                            False,
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
    """
    Build and cache phase factors for single-wavelength scaled propagation.

    Obtains the source-plane factor ``Q1`` and spatial-frequency factor
    ``Q2`` by calling ``scaledASP`` on a dummy field. The propagated
    dummy field is discarded.

    Used internally by ``propagate_scaledASP`` and by the multi-wavelength
    scaled-kernel builder. The factors use the Fresnel (paraxial)
    approximation, with band limiting enabled and the output-plane
    quadratic phase ``Q3`` omitted.

    Args:
        propagatorType (str or None):
            Propagator identifier. Currently unused in the calculation,
            but included in the cache key.

        fftshiftSwitch (bool):
            Propagation configuration flag. Must be False.

        nlambda (int):
            Number of wavelengths. Pass 1; multiple wavelengths are
            handled by the separate polychromatic helper.

        nosm (int):
            Number of object modes.

        npsm (int):
            Number of probe modes.

        Np (int):
            Number of pixels along each side of the square propagation
            grid. Use even values because the underlying scaled
            propagator has an odd-grid coordinate inconsistency.

        zo (float):
            Nonzero propagation distance in meters.

        wavelength (float):
            Positive illumination wavelength in meters.

        dxo (float):
            Positive source-plane pixel size in meters.

        dxd (float):
            Positive destination-plane pixel size in meters.

        gpuSwitch (bool):
            If True, return CuPy arrays. Otherwise, return NumPy arrays.
            The factors are constructed on the CPU in either case.

    Returns:
        tuple:
            ``(Q1, Q2)``, two complex64 arrays with shape
            ``(1, nosm, npsm, 1, Np, Np)``. Q1 acts in the source plane;
            Q2 acts in spatial-frequency space. Both include the
            applicable sampling masks from ``scaledASP``.

    Raises:
        ValueError:
            If ``fftshiftSwitch`` is True or ``nlambda`` is greater
            than 1.

    Notes:
        Results are cached using all arguments as the key. Treat the
        returned arrays as read-only to avoid changing cached results.

        The current nested loops overwrite the probe-mode count with
        a loop index. For multiple object modes, some mode factors can
        remain incorrectly initialized to ones. This limitation must
        be resolved before relying on those configurations.
    """
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
    """
    Build and cache scaled-propagation phase factors for multiple wavelengths.

    Calls ``__make_transferfunction_scaledASP`` separately for each
    wavelength and assembles the resulting factors along the wavelength
    axis. Used internally by ``propagate_scaledPolychromeASP``.

    All channels share the propagation distance and source/destination
    pixel sizes. The factors use the Fresnel (paraxial) approximation,
    with band limiting enabled and the output-plane phase Q3 omitted.

    Args:
        fftshiftSwitch (bool):
            Propagation configuration flag. Must be False.

        nlambda (int):
            Number of wavelength channels.

        nosm (int):
            Number of object modes.

        npsm (int):
            Number of probe modes.

        zo (float):
            Nonzero propagation distance in meters.

        Np (int):
            Number of pixels along each side of the square propagation
            grid. Use even values because the underlying scaled
            propagator has an odd-grid coordinate inconsistency.

        spectralDensity_as_tuple (tuple[float, ...]):
            Positive wavelengths in meters, ordered to match the field's
            wavelength axis. Supply one wavelength per channel.
            These values are wavelengths, not spectral intensity weights.
            A tuple is required for compatibility with the cache.

        dxo (float):
            Positive source-plane pixel size in meters, shared by all
            wavelength channels.

        dxd (float):
            Positive destination-plane pixel size in meters, shared by
            all wavelength channels.

        on_gpu (bool):
            If True, return CuPy arrays. Otherwise, return NumPy arrays.
            The underlying phase factors are calculated on the CPU
            before any GPU transfer.

    Returns:
        tuple:
            ``(Q1, Q2)``, two complex64 arrays with shape
            ``(nlambda, nosm, npsm, 1, Np, Np)``. Q1 acts in the source
            plane and Q2 in spatial-frequency space. Both include the
            applicable sampling masks.

    Raises:
        ValueError:
            If ``fftshiftSwitch`` is True.

    Notes:
        Results are cached using all arguments as the key. Treat the
        returned arrays as read-only to avoid modifying cached results.
        The underlying single-wavelength factors are cached separately.

        This helper inherits the mode-loop bug in
        ``__make_transferfunction_scaledASP``: for multiple object modes,
        some factors can remain incorrectly initialized to ones.

        This function constructs factors only. It does not propagate
        fields, combine wavelength channels, or apply spectral weights.
    """
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
                            z=zo * (1 - spectralDensity[0] / spectralDensity[nlambda]),
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
