"""=============================================================================
Randomized SVD. See Halko, Martinsson, Tropp's 2011 SIAM paper:

This file has been adopted to fit in PtyLab by making it GPU-aware by Dirk Boonzajer

"Finding structure with randomness: Probabilistic algorithms for constructing
approximate matrix decompositions"
============================================================================="""

import numpy as np
from PtyLab.utils.gpuUtils import getArrayModule, isGpuArray

# ------------------------------------------------------------------------------

def rsvd(A, rank, n_oversamples=None, n_subspace_iters=None,
         return_range=False):
    """Randomized SVD (p. 227 of Halko et al).

    :param A:                (m x n) matrix.
    :param rank:             Desired rank approximation.
    :param n_oversamples:    Oversampling parameter for Gaussian random samples.
    :param n_subspace_iters: Number of power iterations.
    :param return_range:     If `True`, return basis for approximate range of A.
    :return:                 U, S, and Vt as in truncated SVD.
    """
    xp = getArrayModule(A)
    if n_oversamples is None:
        # This is the default used in the paper.
        n_samples = 2 * rank
    else:
        n_samples = rank + n_oversamples

    # Stage A.
    Q = find_range(A, n_samples, n_subspace_iters)

    # Stage B.
    B = Q.T.conj() @ A
    U_tilde, S, Vt = xp.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde

    # Truncate.
    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]

    # This is useful for computing the actual error of our approximation.
    if return_range:
        return U, S, Vt, Q
    return U, S, Vt

# ------------------------------------------------------------------------------

def find_range(A, n_samples, n_subspace_iters=None):
    """Algorithm 4.1: Randomized range finder (p. 240 of Halko et al).

    Given a matrix A and a number of samples, computes an orthonormal matrix
    that approximates the range of A.

    :param A:                (m x n) matrix.
    :param n_samples:        Number of Gaussian random samples.
    :param n_subspace_iters: Number of subspace iterations.
    :return:                 Orthonormal basis for approximate range of A.
    """
    xp = getArrayModule(A)
    m, n = A.shape
    O = 1j*xp.random.normal(0,1.0, size=(n, n_samples))
    O +=xp.random.normal(0,1.0, size=(n, n_samples))
    O = O.astype(xp.complex64)
    #O = xp.random.randn(n, n_samples) + 1j * xp.random.randn(n, n_samples)
    Y = A @ O

    if n_subspace_iters:
        return subspace_iter(A, Y, n_subspace_iters)
    else:
        return ortho_basis(Y)

# ------------------------------------------------------------------------------

def subspace_iter(A, Y0, n_iters):
    """Algorithm 4.4: Randomized subspace iteration (p. 244 of Halko et al).

    Uses a numerically stable subspace iteration algorithm to down-weight
    smaller singular values.

    :param A:       (m x n) matrix.
    :param Y0:      Initial approximate range of A.
    :param n_iters: Number of subspace iterations.
    :return:        Orthonormalized approximate range of A after power
                    iterations.
    """
    Q = ortho_basis(Y0)
    for _ in range(n_iters):
        Z = ortho_basis(A.T.conj() @ Q)
        Q = ortho_basis(A @ Z)
    return Q

# ------------------------------------------------------------------------------

def ortho_basis(M):
    """Computes an orthonormal basis for a matrix.

    :param M: (m x n) matrix.
    :return:  An orthonormal basis for M.
    """
    xp = getArrayModule(M)
    Q, _ = xp.linalg.qr(M)
    return Q
