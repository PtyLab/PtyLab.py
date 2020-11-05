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

def rect(x):
    """
    """
    x = abs(x)
    y = (x<1/2)
    return y

def posit(x):
    """
    returns 0 when x negative
    """
    r = (x+abs(x))/2
    # r[r<0]=0 #todo check which way is faster
    return r

def fraccircshift(A, shiftsize):
    """
    fraccircshift expands numpy.roll to fractional shifts values, using linear interpolation.
    :param A: ndarray
    :param shiftsize: shift size in each dimension of A, len(shiftsize)==A.ndim.
    """
    integer = np.floor(shiftsize).astype(int)  # integer portions of shiftsize
    fraction = shiftsize-integer
    dim = len(shiftsize)
    # the dimensions are treated one after another
    for n in np.arange(dim):
        intn = integer[n]
        fran = fraction[n]
        shift1 = intn
        shift2 = intn+1
        # linear interpolation
        A = (1-fran)*np.roll(A, shift1, axis=n)+fran*np.roll(A, shift2, axis=n)
    return A

def cart2pol(x,y):
    """
    Transform Cartesian to polar coordinates
    :param x:
    :param y:
    :return:
    """
    th = np.arctan2(y,x)
    r = np.hypot(x,y)
    return th, r

def gaussian2D(n,std):
    # create the grid of (x,y) values
    n = (n-1)//2
    x, y = np.meshgrid(np.arange(-n, n+1), np.arange(-n, n+1))
    # analytic function
    h = np.exp(- (x**2 + y**2)/(2 * std**2))
    # truncate very small values to zero
    mask = h < np.finfo(float).eps*np.max(h)
    h *= (1-mask)
    # normalize filter to unit L1 energy
    sumh = np.sum(h)
    if sumh != 0:
        h = h/sumh
    return h


def orthogonalizeModes(p):
    """
    Imposes orthogonality through singular value decomposition
    :return:
    """
    # orthogonolize modes only for npsm and nosm which are lcoated and indices 1, 2
    xp = getArrayModule(p)
    try:
        U, s, V = xp.linalg.svd(p.reshape(p.shape[0], p.shape[1]*p.shape[2]), full_matrices=False )
        p = xp.dot(xp.diag(s), V).reshape(p.shape[0], p.shape[1], p.shape[2])
        normalizedEigenvalues = s**2/xp.sum(s**2)
    except Exception as e:
        print('Warning: performing SVD on CPU rather than GPU due to error', e)
        #print('Exception: ', e)
        # TODO: check, most likely this is faster to perform on the CPU rather than GPU
        if hasattr(p, 'device'):
            p = p.get()
        U, s, V = np.linalg.svd(p.reshape(p.shape[0], p.shape[1]*p.shape[2]), full_matrices=False )
        p = np.dot(np.diag(s), V).reshape(p.shape[0], p.shape[1], p.shape[2])
        normalizedEigenvalues = s**2/xp.sum(s**2)

    return xp.asarray(p), normalizedEigenvalues, U


