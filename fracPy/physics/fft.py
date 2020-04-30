import numpy as np


def fft2c(g):
    """
    performs 2 - dimensional Fourier transformation
    fft2c is normalized(i.e.norm(g) = norm(G) ), i.e.it preserves the L2 - norm
    if g is two - dimensional, fft2c(g) yields the 2D DFT of g
    if g is multi - dimensional, fft2c(g) yields the 2D DFT of g for each slice along the third dimension
    (important for correct normalization % under partially coherent conditions)
    """

    G = np.fft.fftshift(np.fft.fft2(g, norm="ortho"))

    return G


def ifft2c(G):
    """
    performs 2 - dimensional inverse Fourier transformation
    fft2c is normalized(i.e.norm(g) = norm(G) ), i.e.it preserves the L2 - norm
    if G is two - dimensional, fft2c(G) yields the 2D iDFT of G
    if G is multi - dimensional, fft2c(G) yields the 2D iDFT of G for each slice along the third dimension
    (important for correct normalization % under partially coherent conditions)
    """

    g = np.fft.ifft2(np.fft.ifftshift(G), norm="ortho")

    return g
