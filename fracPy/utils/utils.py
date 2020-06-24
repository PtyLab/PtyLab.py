import pickle
import numpy as np
from scipy import linalg
import scipy.stats as st

from fracPy.utils.gpuUtils import getArrayModule


def fft2c(array):
    """
    performs 2 - dimensional unitary Fourier transformation, where energy is reserved abs(g)**2==abs(fft2c(g))**2
    if g is two - dimensional, fft2c(g) yields the 2D DFT of g
    if g is multi - dimensional, fft2c(g) yields the 2D DFT of g along the last two axes
    :param array:
    :return:
    """
    xp = getArrayModule(array)
    return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(array), norm='ortho'))



def ifft2c(array):
    """
    performs 2 - dimensional inverse Fourier transformation, where energy is reserved abs(G)**2==abs(fft2c(g))**2
    if G is two - dimensional, fft2c(G) yields the 2D iDFT of G
    if G is multi - dimensional, fft2c(G) yields the 2D iDFT of G along the last two axes
    :param array:
    :return:
    """
    xp = getArrayModule(array)
    return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(array), norm='ortho'))


def circ(x,y,D):
    """
    generate a circle on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param y: 2D y coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
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
    xp = getArrayModule(p)
    # TODO: check, most likely this is faster to perform on the CPU rather than GPU
    # if hasattr(p, 'device'):
    #     p = p.get()
    # U, s, V = np.linalg.svd(p.reshape(p.shape[0], p.shape[1]*p.shape[2]), full_matrices=False )
    # p = np.dot(np.diag(s), V).reshape(p.shape[0], p.shape[1], p.shape[2])
    # normalizedEigenvalues = s**2/xp.sum(s**2)
    
    U, s, V = xp.linalg.svd(p.reshape(p.shape[0], p.shape[1]*p.shape[2]), full_matrices=False )
    p = xp.dot(xp.diag(s), V).reshape(p.shape[0], p.shape[1], p.shape[2])
    normalizedEigenvalues = s**2/xp.sum(s**2)
    return xp.array(p), normalizedEigenvalues, U

