import numpy as np
from .fft import fft2c, ifft2c


def fresnelPropagator(u, z, wavelength, L):
    k = 2 * np.pi /wavelength
    #source coordinates, this assumes that the field is NxN pixels
    N = u.shape[-1]
    dx = L / N
    x = np.arange(-N / 2, N / 2) * dx
    [Y, X] = np.meshgrid(x,x)

    #target coordinates
    dq = wavelength *z / L
    q = np.arange(-N / 2, N / 2) * dq
    [Qy, Qx] = np.meshgrid(q,q);

    #phase terms
    Qin = np.exp(1j * k / (2 * z) * (np.square(X) + np.square(Y)))
    #Qout = exp(1i * k / (2 * z) * (Qx. ^ 2 + Qy. ^ 2));
    #r = Qout. * fft2c(Qin. * u)
    r = fft2c(Qin * u)
    return r, dq, q, Qx, Qy

def angularSpectrumPropagator():
    raise NotImplementedError


def ft2(f, delta):
    g = fft2c(f) * delta**2
    return g


def ift2(g, delta):
    N = g.shape[-1]
    f = ifft2c(g) * (N * delta)**2
    return f

def generateFresnelIR():
    raise NotImplementedError

def inverseTwoStepPropagator():
    raise NotImplementedError

def scaledASP():
    raise NotImplementedError

def scaledSPinv():
    raise NotImplementedError

def two_step_prop():
    raise NotImplementedError