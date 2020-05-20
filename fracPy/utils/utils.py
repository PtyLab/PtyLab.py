import pickle
import numpy as np
from scipy import linalg
import scipy.stats as st



def fft2c(array):
    """
    Unitary 2D forward Fourier transform
    :param array:
    :return:
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array),norm='ortho'))



def ifft2c(array):
    """
    Unitary 2D inverse Fourier transform
    :param array:
    :return:
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array),norm='ortho'))


# def load(filename):
#     with open(filename, 'rb') as f:
#         return pickle.load(f)

def circ(x,y,D):
    """
    generate a circle on a 2D grid
    :param x: 2D array
    :param y: 2D array
    :param D: diameter 
    :return: a 2D array
    """
    circle = (x**2+y**2)<(D/2)**2
    return circle

def orthogonalizeModes(p):
    """
    Imposes orthogonality through singular value decomposition
    :return:
    """
    # orthogonolize modes only for npsm and nosm which are lcoated and indices 1, 2
    U, s, V = linalg.svd(p.reshape(p.shape[0], p.shape[1]*p.shape[2]), full_matrices=False )
    p = np.dot(np.diag(s), V).reshape(p.shape[0], p.shape[1], p.shape[2])
    normalizedEigenvalues = s**2/np.sum(s**2)
    return p, normalizedEigenvalues, U

def aspw(u,z,wavelength,L):
    """
    Angular spectrum plane wave propagation function.
    following: Matsushima et al., "Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space
    Propagation in Far and Near Fields", Optics Express, 2009
    :param u: a 2D field distribution at z = 0 (u is assumed to be square, i.e. N x N)
    :param z: propagation distance
    :param wavelength: propagation wavelength in meter
    :param L: total size of the field in meter
    :return: field distribution after propagation
    """
    k = 2*np.pi/wavelength
    N = u.shape[0]
    X = np.arange(-N/2, N/2)/L
    Fx, Fy = np.meshgrid(X, X)
    f_max = L/(wavelength*np.sqrt(L**2+4*z**2))
    # note: see the paper above if you are not sure what this bandlimit has to do here
    # W = rect(Fx/(2*f_max)) .* rect(Fy/(2*f_max));
    W = circ(Fx, Fy, 2*f_max)
    # note: accounts for circular symmetry of transfer function and imposes bandlimit to avoid sampling issues
    H = np.exp(1.j * k * z * np.sqrt(1 - (Fx*wavelength)**2 - (Fy*wavelength)**2))
    U = fft2c(u)
    u = ifft2c(U * H * W)
    return u, H

def scaledASP(u,z,wavelength,dx,dq):
    """

    :param u:
    :param z:
    :param wavelength:
    :param dx:grid spacing in original plane (u)
    :param dq: grid spacing in destination plane (Uout)
    :return:
    """
    k = 2 * np.pi / wavelength
    N = u.shape[0]

    # source plane coordinates
    x1 = np.arange(-N / 2, N / 2) * dx
    X1, Y1 = np.meshgrid(x1,x1)
    r1sq = X1**2+Y1**2
    # spatial frequencies(of source plane)
    f = np.arange(-N / 2, N / 2)/ (N * dx)
    FX, FY = np.meshgrid(f,f)
    fsq = FX**2 + FY**2
    # scaling parameter
    m = dq / dx

    # quadratic phase factors
    Q1 = np.exp(1. * k / 2 * (1 - m) / z * r1sq)
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