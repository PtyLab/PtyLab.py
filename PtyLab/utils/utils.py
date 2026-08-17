import logging

import numpy as np
from scipy import linalg
import scipy.stats as st

from PtyLab.utils.gpuUtils import asNumpyArray, getArrayModule

logger = logging.getLogger(__name__)


def fft2c(field, fftshiftSwitch=False, *args, **kwargs):
    """
    performs 2 - dimensional unitary Fourier transformation, where energy is preserved sum( abs(g)**2 ) == sum( abs(fft2c(g))**2 )
    if g is two - dimensional, fft2c(g) yields the 2D DFT of g
    if g is multi - dimensional, fft2c(g) yields the 2D DFT of g along the last two axes
    :param array:
    :return:
    """
    xp = getArrayModule(field)

    if fftshiftSwitch:
        return xp.fft.fft2(field, norm="ortho")
    else:
        axes = (-2, -1)
        return xp.fft.fftshift(
            xp.fft.fft2(xp.fft.ifftshift(field, axes=axes), norm="ortho"), axes=axes
        )


def ifft2c(field, fftshiftSwitch=False):
    """
    performs 2 - dimensional inverse Fourier transformation, where energy is preserved sum( abs(G)**2 ) == sum( abs(fft2c(g))**2 ) 
    if G is two - dimensional, fft2c(G) yields the 2D iDFT of G
    if G is multi - dimensional, fft2c(G) yields the 2D iDFT of G along the last two axes
    :param array:
    :return:
    """
    xp = getArrayModule(field)

    if fftshiftSwitch:
        return xp.fft.ifft2(field, norm="ortho")
    else:
        axes = (-2, -1)
        return xp.fft.fftshift(
            xp.fft.ifft2(xp.fft.ifftshift(field, axes=axes), norm="ortho"), axes=axes
        )


def circ(x, y, D):
    """
    generate a binary array containing a circle on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param y: 2D y coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param D: diameter
    :return: a binary 2D array
    """
    circle = (x**2 + y**2) < (D / 2) ** 2
    return circle


def rect(arr, threshold=0.5):
    """
    generate a binary array containing a rectangle on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param threshold: threshold value to binarilize the input array, default value 0.5
    :return: a binary array
    """
    arr = abs(arr)
    return arr < threshold


def posit(x):
    """
    returns 0 when x negative
    """
    r = (x + abs(x)) / 2
    # r[r<0]=0 #todo check which way is faster
    return r


def fraccircshift(A, shiftsize):
    """
    fraccircshift expands numpy.roll to fractional shifts values, using linear interpolation.
    :param A: ndarray
    :param shiftsize: shift size in each dimension of A, len(shiftsize)==A.ndim.
    """
    integer = np.floor(shiftsize).astype(int)  # integer portions of shiftsize
    fraction = shiftsize - integer
    dim = len(shiftsize)
    # the dimensions are treated one after another
    for n in np.arange(dim):
        intn = integer[n]
        fran = fraction[n]
        shift1 = intn
        shift2 = intn + 1
        # linear interpolation
        A = (1 - fran) * np.roll(A, shift1, axis=n) + fran * np.roll(A, shift2, axis=n)
    return A


def cart2pol(x, y):
    """
    Transform Cartesian to polar coordinates
    :param x:
    :param y:
    :return:
    """
    th = np.arctan2(y, x)
    r = np.hypot(x, y)
    return th, r


def gaussian2D(n, std):
    # create the grid of (x,y) values
    n = (n - 1) // 2
    x, y = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    # analytic function
    h = np.exp(-(x**2 + y**2) / (2 * std**2))
    # truncate very small values to zero
    mask = h < np.finfo(float).eps * np.max(h)
    h *= 1 - mask
    # normalize filter to unit L1 energy
    sumh = np.sum(h)
    if sumh != 0:
        h = h / sumh
    return h


def _snapShotModes(p, xp):
    """
    Orthogonalize the modes of ``p`` through the Gram matrix, using ``xp``
    (numpy or cupy) for every step.

    The two matmuls -- forming ``G = p2D p2D^H`` and applying the result back to
    the modes -- run under ``xp``, so they stay wherever the data already lives.
    The eigendecomposition of ``G`` always runs on the host: ``G`` is only
    (nmodes x nmodes), a few hundred bytes, so there is no parallelism for a GPU
    to exploit, and cuSOLVER is measurably slow on matrices that small -- it
    costs 0.17-0.86 ms against 0.03-0.17 ms for both matmuls combined, i.e. the
    decomposition would be the whole cost of this function. Keeping it on the
    host is 2-5x faster overall and, as a side effect, means orthogonalization
    no longer depends on cuSOLVER at all: cupyx.cusolver is not importable in
    every CuPy/CUDA installation, which used to send this function down its
    fallback path for entire reconstructions.

    :param p: modes, shaped (nmodes, Ny, Nx)
    :param xp: numpy or cupy
    :return: (orthogonalized modes, normalized eigenvalues, mode mixing matrix)
    """
    p2D = p.reshape(p.shape[0], p.shape[1] * p.shape[2])
    G = xp.dot(p2D.conj(), xp.transpose(p2D))

    # G is Hermitian, so eigh rather than eig: real eigenvalues, orthonormal
    # eigenvectors, and no complex residue to discard. eigh returns them
    # ascending, and the callers want the dominant mode first.
    w, V = np.linalg.eigh(asNumpyArray(G))
    order = np.argsort(w)[::-1]

    # G is also positive semi-definite, so w is non-negative in exact
    # arithmetic; clip guards a vanishing mode that rounding pushed below zero,
    # which would otherwise turn the sqrt into a nan.
    s = np.sqrt(np.clip(w[order].real, 0.0, None))
    normalizedEigenvalues = s**2 / np.sum(s**2)

    # The modes are V^T p2D -- plain transpose, not conjugate: G above is
    # conj(p2D p2D^H), so eigh hands back the conjugated eigenvectors and the
    # conjugation cancels. Writing the product directly avoids dividing the
    # projection by s only to multiply it back in, which is what the regularized
    # inverse of diag(s) used to do -- and which lost precision on near-zero
    # modes.
    V = xp.asarray(np.ascontiguousarray(V[:, order].T))
    modes = xp.dot(V, p2D).reshape(p.shape[0], p.shape[1], p.shape[2])

    return modes, normalizedEigenvalues, V


def orthogonalizeModes(p, method=None):
    """
    Imposes orthogonality through singular value decomposition
    :return:
    """
    # orthogonolize modes only for npsm and nosm which are lcoated and indices 1, 2
    xp = getArrayModule(p)

    if method == "snapShots":
        try:
            p, normalizedEigenvalues, V = _snapShotModes(p, xp)
        except Exception as e:
            # cuSOLVER can fail for reasons that have nothing to do with this
            # data -- a CUDA install that cannot load libcusolver, or a memory
            # pool that has left it no room to allocate its workspace. Fall back
            # to the host rather than losing the reconstruction, but say loudly
            # what went wrong: this costs a device round trip every call, so it
            # is not something to run for a whole reconstruction unnoticed.
            logger.warning(
                "Orthogonalizing modes on the CPU rather than the GPU: %s: %s",
                type(e).__name__,
                e,
            )
            p, normalizedEigenvalues, V = _snapShotModes(asNumpyArray(p), np)
        return xp.asarray(p), normalizedEigenvalues, V

    else:
        U, s, V = xp.linalg.svd(
            p.reshape(p.shape[0], p.shape[1] * p.shape[2]), full_matrices=False
        )
        p = xp.dot(xp.diag(s), V).reshape(p.shape[0], p.shape[1], p.shape[2])
        normalizedEigenvalues = s**2 / xp.sum(s**2)

        return xp.asarray(p), normalizedEigenvalues, U.T.conj()


def zernikeAberrations(Xp, Yp, D, z_coeff):
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

    aperture = circ(Xp, Yp, D)
    angle = np.double(np.arctan2(Yp, Xp)) * aperture
    p = np.double(np.hypot(Xp, Yp)) * aperture
    p = p / np.max(p)

    Z = dict()
    Z[0] = z_coeff[0]  # pistom
    Z[1] = z_coeff[1] * 4 ** (1 / 2.0) * p * np.cos(angle)
    # tip
    Z[2] = z_coeff[2] * 4 ** (1 / 2.0) * p * np.sin(angle)
    # tilt
    Z[3] = z_coeff[3] * 3 ** (1 / 2.0) * (2 * p**2 - 1)
    # defocus
    Z[4] = z_coeff[4] * 6 ** (1 / 2.0) * (p**2) * np.sin(2 * angle)
    # astigmatism
    Z[5] = z_coeff[5] * 6 ** (1 / 2.0) * (p**2) * np.cos(2 * angle)
    # astigmatism
    Z[6] = z_coeff[6] * 8 ** (1 / 2.0) * (3 * p**3 - 2 * p) * np.sin(angle)
    # coma
    Z[7] = z_coeff[7] * 8 ** (1 / 2.0) * (3 * p**3 - 2 * p) * np.cos(angle)
    # coma
    Z[8] = z_coeff[8] * 8 ** (1 / 2.0) * (p**3) * np.sin(3 * angle)
    # trefoil
    Z[9] = z_coeff[9] * 8 ** (1 / 2.0) * (p**3) * np.cos(3 * angle)
    # trefoil
    Z[10] = z_coeff[10] * 5 ** (1 / 2.0) * (6 * p**4 - 6 * p**2 + 1)
    # spherical
    Z[11] = (
        z_coeff[11] * 10 ** (1 / 2.0) * (4 * p**4 - 3 * p**2) * np.cos(2.0 * angle)
    )
    # 2nd astigmatism
    Z[12] = (
        z_coeff[12] * 10 ** (1 / 2.0) * (4 * p**4 - 3 * p**2) * np.sin(2.0 * angle)
    )
    # 2nd astigmatism
    Z[13] = z_coeff[13] * 10 ** (1 / 2.0) * (p**4) * np.cos(4.0 * angle)
    Z[14] = z_coeff[14] * 10 ** (1 / 2.0) * (p**4) * np.sin(4.0 * angle)
    Z[15] = (
        z_coeff[15]
        * 12 ** (1 / 2.0)
        * (10 * p**5 - 12 * p**3 + 3 * p)
        * np.cos(angle)
    )
    Z[16] = (
        z_coeff[16]
        * 12 ** (1 / 2)
        * (10 * p**5 - 12 * p**3 + 3 * p)
        * np.sin(angle)
    )
    Z[17] = z_coeff[17] * 12 ** (1 / 2) * (5 * p**5 - 4 * p**3) * np.cos(3 * angle)
    Z[18] = z_coeff[18] * 12 ** (1 / 2) * (5 * p**5 - 4 * p**3) * np.sin(3 * angle)

    return aperture * np.exp(1j * np.sum(list(Z.values())))


def p2bin(im, binningFactor):
    """
    perform binning at a factor of power of 2, return binned image and the indices for before and after binning.
    :Params im: input image for binning
    :Params binningFactor: must be power of 2 in the current implementation
    :return:
    """
    M, N = im.shape
    if np.mod(binningFactor, 2) != 0 and binningFactor != 1:
        raise ValueError("binning factor needs to be a power of 2")
    if np.mod(M, binningFactor) != 0 or np.mod(N, binningFactor) != 0:
        raise ValueError(
            "#rows and #columns of reference need to be divided by binningFactor!"
        )

    if binningFactor != 1:
        for k in range(1, int(np.log2(binningFactor))):
            im_binned = bin2(im)
        im_binned_ind = range(im_binned.size)
        im_ind = np.arange(M * N).reshape(M // binningFactor, N, binningFactor)
        im_ind = np.stack(im_ind, axis=1).reshape(M, N)
    else:
        im_binned = im
        im_binned_ind = range(im_binned.size)
        im_ind = im_binned_ind
    return im_binned, im_ind, im_binned_ind


def bin2(X):
    """
    perform 2-by-2 binning.
    :Params X: input 2D image for binning
    return: Y: output 2D image after 2-by-2 binning
    """
    # simple 2-fold binning
    m, n = X.shape
    Y = np.sum(X.reshape(2, m // 2, 2, n // 2), axis=(0, 2))
    return Y
