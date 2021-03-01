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
    generate a binary array containing a circle on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param y: 2D y coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param D: diameter 
    :return: a binary 2D array
    """
    circle = (x**2+y**2)<(D/2)**2
    return circle

def rect(arr, threshold = 0.5):
    """
    generate a binary array containing a rectangle on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param threshold: threshold value to binarilize the input array, default value 0.5
    :return: a binary array
    """
    arr = abs(arr)
    return arr<threshold

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


def zernikeAberrations(Xp,Yp,D,z_coeff):
    """
    Compute the first 19 Zernike aberrations based on Zernike polynomials
    Based on https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices
    
    Xp,Yp - meshgrid coordinates
    D - radius within which to generate the zernike aberrations
    z_coeff - 19 element long list containing coefficients.
    
    minimal example:
    
        import matplotlib.pyplot as plt
        import numpy as np
        
        # create the circular dimensions which will define the size
        # of a unit circle used for zernike aberration calculations
        Xp,Yp = np.mgrid[-128:128, -128:128]
        D = 128
        
        # Get defocus aberration (4th index)
        z_coeff = np.zeros(19)
        z_coeff[4] = 3
        Z = zernikeAberrations(Xp,Yp,D,z_coeff)
        
        # plot the polynoial
        plt.figure(1)
        plt.imshow(np.angle(Z))
        plt.show()
    """
    
    aperture = circ(Xp,Yp,D)
    angle = np.double(np.arctan2(Yp, Xp)) * aperture
    p = np.double(np.hypot(Xp, Yp)) * aperture
    p = p /np.max(p)
    
    Z = dict()
    Z[0] = z_coeff[0] # pistom
    Z[1] = z_coeff[1] * 4**(1/2.)*p*np.cos(angle); # tip
    Z[2] = z_coeff[2] * 4**(1/2.)*p* np.sin(angle); # tilt
    Z[3] = z_coeff[3] * 3**(1/2.)*(2* p**2 - 1); # defocus
    Z[4] = z_coeff[4] * 6**(1/2.)*(p**2)*np.sin(2*angle); # astigmatism
    Z[5] = z_coeff[5] * 6**(1/2.)*(p**2)*np.cos(2*angle); # astigmatism
    Z[6] = z_coeff[6] * 8**(1/2.)*(3* p**3 - 2*p)*np.sin(angle); # coma
    Z[7] = z_coeff[7] * 8**(1/2.)*(3* p**3 - 2*p)*np.cos(angle); # coma
    Z[8] = z_coeff[8] * 8**(1/2.)*(p**3)*np.sin(3*angle); # trefoil
    Z[9] = z_coeff[9] * 8**(1/2.)*(p**3)*np.cos(3*angle); # trefoil
    Z[10] = z_coeff[10] * 5**(1/2.)*(6*p**4 - 6*p**2 + 1); # spherical
    Z[11] = z_coeff[11] * 10**(1/2.)*(4*p**4 - 3*p**2)*np.cos(2.*angle); # 2nd astigmatism
    Z[12] = z_coeff[12] * 10**(1/2.)*(4*p**4 - 3*p**2)*np.sin(2.*angle); # 2nd astigmatism
    Z[13] = z_coeff[13] * 10**(1/2.)*(p**4)*np.cos(4.*angle); 
    Z[14] = z_coeff[14] * 10**(1/2.)*(p**4)*np.sin(4.*angle); 
    Z[15] = z_coeff[15] * 12**(1/2.)*(10*p**5 - 12*p**3 + 3*p)*np.cos(angle);
    Z[16] = z_coeff[16] * 12**(1/2)*(10*p**5 - 12*p**3 + 3*p)*np.sin(angle);
    Z[17] = z_coeff[17] * 12**(1/2)*(5*p**5 - 4*p**3)*np.cos(3*angle);
    Z[18] = z_coeff[18] * 12**(1/2)*(5*p**5 - 4*p**3)*np.sin(3*angle);
    
    return aperture * np.exp(1j*np.sum(list(Z.values())))
