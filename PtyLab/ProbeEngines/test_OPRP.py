from numpy.testing import assert_almost_equal

from PtyLab.ProbeEngines.OPRP import OPRP_storage
import numpy as np


def test_push():
    N = 100
    npix = 128
    probes = np.random.rand(N // 10, 4, npix, npix)
    probes = np.repeat(probes, axis=0, repeats=10)
    storage = OPRP_storage(5)

    # push one and check that it's the same
    for tsvd in [False, True]:
        storage.push(probes[0], 0, N)
        if tsvd:
            storage.tsvd()
        # try to get all of them
        p1 = storage.get(0)
        assert_almost_equal(p1, probes[0], decimal=5)
        p1 = storage.get(1)  # should fail as we didn't specify it yet
        assert_almost_equal(p1, probes[0], decimal=5)
        storage.clear()

    for i, p in enumerate(probes):
        storage.push(p, i, N)

    # perform TSVD
    storage.N_probes = N
    storage.tsvd()

    # get one element out
    p1 = storage.get(0)
    assert_almost_equal(p1, probes[0], decimal=5)

    # get a new element out after smaller TSVD
    storage.N_probes = N // 10
    storage.tsvd()
    for i, p in enumerate(probes):
        p1 = storage.get(i)
        try:
            assert_almost_equal(p1, probes[i], decimal=5)

        except AssertionError:
            print(f"Failed for i={i}")
            print(p1.shape, probes[i].shape)
            raise


def test_center_probe():
    N = 100
    npix = 128
    probes = np.random.rand(N // 10, 4, npix, npix)
    probes[:, :, : npix // 3, :] = 0
    probes = np.repeat(probes, axis=0, repeats=10)
    storage = OPRP_storage(5)
    storage.push(probes[0], 0, N)
    for i, p in enumerate(probes):
        p_inout, _ = storage.uncenter_probe(storage.center_probe(p, i)[0], i)
        assert_almost_equal(p, p_inout)
        probes[i], shift = storage.center_probe(p, i)
        print("first round: ", shift)

        probes[i], shift = storage.center_probe(p, i)
        print("second round", shift)
