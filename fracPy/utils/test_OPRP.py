import unittest

from numpy.testing import assert_almost_equal

from fracPy.utils.OPRP import OPRP_storage
import numpy as np


def test_push():
    N = 100
    npix = 512
    probes = np.random.rand(N//10, npix, npix)
    probes = np.repeat(probes, axis=0, repeats=10)
    storage = OPRP_storage(N, probes[0], False,
                           5)
    for i, p in enumerate(probes):
        storage.push(p, i)

    # perform TSVD
    storage.N_probes = N
    storage.tsvd()

    # get one element out
    p1 = storage.get(0)
    assert_almost_equal(p1, probes[0])

    # get a new element out after smaller TSVD
    storage.N_probes = N//10
    storage.tsvd()
    for i, p in enumerate(probes):
        p1 = storage.get(i)
        assert_almost_equal(p1, probes[i])
