"""Unit tests for the OPR truncated-SVD replacement.

These run on the CPU so CI covers them, even though the OPR engine itself is
currently GPU-only.
"""

import numpy as np
import pytest

from PtyLab.Engines.OPR import OPR

try:
    import cupy as cp

    HAS_GPU = cp.cuda.is_available()
except Exception:
    cp = None
    HAS_GPU = False


def _low_rank(m, n, rank, seed=0, noise=1e-3):
    """A tall (m, n) matrix with a clear rank-`rank` structure plus noise."""
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(m, rank)) + 1j * rng.normal(size=(m, rank))
    right = rng.normal(size=(rank, n)) + 1j * rng.normal(size=(rank, n))
    # decaying weights so the singular values are well separated
    weights = np.diag(np.linspace(1.0, 0.2, rank))
    signal = left @ weights @ right
    perturbation = noise * (rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n)))
    return (signal + perturbation).astype(np.complex64)


@pytest.mark.parametrize("rank", [1, 3, 4])
def test_gram_tsvd_matches_full_svd_truncation(rank):
    """gram_tsvd must reproduce the rank-k truncation that OPR relies on."""
    A = _low_rank(2048, 40, rank=rank)

    U, s, Vh = OPR.gram_tsvd(A, rank)
    got = U @ (s[:, None] * Vh)

    Uf, sf, Vhf = np.linalg.svd(A, full_matrices=False)
    sf = sf.copy()
    sf[rank:] = 0
    expected = Uf @ (sf[:, None] * Vhf)

    # The truncation is gauge-invariant even though the individual singular
    # vectors are not, so this can be compared directly.
    err = np.linalg.norm(got - expected) / np.linalg.norm(expected)
    assert err < 1e-4, f"rank-{rank} truncation differs by {err:.2e}"


def test_gram_tsvd_singular_values_match():
    A = _low_rank(2048, 40, rank=4)
    _U, s, _Vh = OPR.gram_tsvd(A, 4)
    expected = np.linalg.svd(A, compute_uv=False)[:4]
    np.testing.assert_allclose(s, expected, rtol=1e-4)


def test_gram_tsvd_returns_orthonormal_left_vectors():
    A = _low_rank(2048, 40, rank=4)
    U, _s, _Vh = OPR.gram_tsvd(A, 4)
    gram = U.conj().T @ U
    np.testing.assert_allclose(gram, np.eye(4), rtol=1e-3, atol=1e-3)


def test_gram_tsvd_clamps_rank_to_available_columns():
    """Asking for more components than columns must not blow up."""
    A = _low_rank(512, 6, rank=3)
    U, s, Vh = OPR.gram_tsvd(A, 10)
    assert U.shape == (512, 6)
    assert s.shape == (6,)
    assert Vh.shape == (6, 6)


def test_gram_tsvd_handles_rank_deficient_input():
    """A genuinely rank-deficient matrix has zero singular values; the
    division by s must not produce NaNs."""
    rng = np.random.default_rng(3)
    left = rng.normal(size=(256, 2)) + 1j * rng.normal(size=(256, 2))
    right = rng.normal(size=(2, 8)) + 1j * rng.normal(size=(2, 8))
    A = (left @ right).astype(np.complex64)  # exactly rank 2

    U, s, Vh = OPR.gram_tsvd(A, 5)
    assert np.all(np.isfinite(U)), "gram_tsvd produced non-finite left vectors"
    assert np.all(np.isfinite(s))
    reconstructed = U @ (s[:, None] * Vh)
    err = np.linalg.norm(reconstructed - A) / np.linalg.norm(A)
    assert err < 1e-4


@pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU available")
def test_gram_tsvd_gpu_matches_cpu():
    A = _low_rank(2048, 40, rank=4)
    U_c, s_c, Vh_c = OPR.gram_tsvd(A, 4)
    U_g, s_g, Vh_g = OPR.gram_tsvd(cp.asarray(A), 4)

    cpu = U_c @ (s_c[:, None] * Vh_c)
    gpu = cp.asnumpy(U_g @ (s_g[:, None] * Vh_g))
    err = np.linalg.norm(gpu - cpu) / np.linalg.norm(cpu)
    assert err < 1e-4, f"GPU gram_tsvd differs from CPU by {err:.2e}"
