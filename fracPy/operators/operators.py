import numpy as np
from fracPy.utils.utils import circ, fft2c, ifft2c
from fracPy.utils.gpuUtils import getArrayModule


def aspw(u, z, wavelength, L):
    """
    Angular spectrum plane wave propagation function.
    following: Matsushima et al., "Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space
    Propagation in Far and Near Fields", Optics Express, 2009
    :param u: a 2D field distribution at z = 0 (u is assumed to be square, i.e. N x N)
    :param z: propagation distance
    :param wavelength: propagation wavelength in meter
    :param L: total size of the field in meter
    :return: field distribution after propagation and the bandlimited transfer function
    """
    xp = getArrayModule(u)
    k = 2*np.pi/wavelength
    N = u.shape[0]
    X = np.arange(-N/2, N/2)/L
    Fx, Fy = np.meshgrid(X, X)
    f_max = L/(wavelength*np.sqrt(L**2+4*z**2))
    # note: see the paper above if you are not sure what this bandlimit has to do here
    # W = rect(Fx/(2*f_max)) .* rect(Fy/(2*f_max));
    W = xp.array(circ(Fx, Fy, 2*f_max))
    # note: accounts for circular symmetry of transfer function and imposes bandlimit to avoid sampling issues
    H = xp.array(np.exp(1.j * k * z * np.sqrt(1 - (Fx*wavelength)**2 - (Fy*wavelength)**2)))
    U = fft2c(u)
    u = ifft2c(U * H * W)
    return u, H*W

def scaledASP(u, z, wavelength, dx, dq):
    """
    Angular spectrum propagation with customized grid spacing dq
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
    N = u.shape[0]
    # source plane coordinates
    x1 = np.arange(-N / 2, N / 2) * dx
    X1, Y1 = np.meshgrid(x1, x1)
    r1sq = X1**2+Y1**2
    # spatial frequencies(of source plane)
    f = np.arange(-N / 2, N / 2)/ (N * dx)
    FX, FY = np.meshgrid(f, f)
    fsq = FX**2 + FY**2
    # scaling parameter
    m = dq / dx

    # quadratic phase factors todo: bandlimit implementation?
    Q1 = np.exp(1.j * k / 2 * (1 - m) / z * r1sq)
    Q2 = np.exp(-1.j * np.pi**2 * 2 * z / m / k * fsq)
    Uout = ifft2c(Q2 * fft2c(Q1 * u))

    # note: to be analytically correct, add Q3 (see below)
    # if only intensities matter, leave it out
    # x2 = np.arange(-N / 2, N / 2) * dq
    # X2, Y2 = np.meshgrid(x2,x2)
    # r2sq = X2**2 + Y2**2
    # Q3 = np.exp(1.j * k / 2 * (m - 1) / (m * z) * r2sq)
    # # compute the propagated field
    # Uout = Q3 * ifft2c(Q2 * fft2c(Q1 * u))

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
    fsq = FX ** 2 + FY ** 2
    # scaling parameter
    m = dq / dx

    # quadratic phase factors
    Q1 = np.exp(1j * k / 2 * (1 - m) / z * r1sq)
    Q2 = np.exp(-1j * 2 * np.pi ** 2 * z / m / k * fsq)
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
    :param u:   field distribution at z = 0(u is assumed to be square, i.e.N x N)
    :param z:   propagation distance
    :param wavelength: wavelength
    :param L: total size[m] of
    :return: propagated field
    """
    k = 2 * np.pi /wavelength
    # source coordinates, this assumes that the field is NxN pixels
    N = u.shape[-1]
    dx = L / N
    x = np.arange(-N / 2, N / 2) * dx
    [Y, X] = np.meshgrid(x,x)

    # target coordinates
    dq = wavelength *z / L
    q = np.arange(-N / 2, N / 2) * dq
    [Qy, Qx] = np.meshgrid(q, q);

    # phase terms
    Qin = np.exp(1j * k / (2 * z) * (np.square(X) + np.square(Y)))
    r = fft2c(Qin * u)
    return r, dq, q, Qx, Qy

