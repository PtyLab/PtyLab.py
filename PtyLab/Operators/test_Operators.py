from unittest import TestCase

from numpy.testing import assert_allclose

from PtyLab import Operators, easyInitialize
import numpy as np
from numpy.testing import assert_allclose
import time


def test_caching_aspw():
    try:
        import cupy as cp

        xp = cp
    except ImportError:
        xp = np
    E = xp.random.rand(10, 1, 3, 512, 512)
    z = 1e-3
    wl = 512e-9
    pixel_pitch = 10e-6
    L = pixel_pitch * E.shape[-1]

    from cupyx import time as timer

    # run this to warm up the GPU
    timer.repeat(Operators.Operators.aspw, (E, z, wl, L), {}, n_repeat=200, n_warmup=50)

    t0 = time.time()
    for i in range(100):
        E_prop = Operators.Operators.aspw_cached(E, z, wl, L)
    if xp is not np:
        E_prop = xp.asnumpy(E_prop)
    t1 = time.time()
    t_cached = t1 - t0
    for i in range(100):
        E_prop2 = Operators.Operators.aspw(E, z, wl, L)[0]
    if xp is not np:
        E_prop2 = E_prop2.get()
    t2 = time.time()
    t_noncached = t2 - t1
    print(f"\n\nNon-cached took: {t_noncached}", f"Cached took {t_cached}s")
    # E_prop = np.squeeze(E_prop)

    assert_allclose(E_prop, E_prop2)


def test_object2detector():
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )
    params.gpuSwitch = True
    reconstruction._move_data_to_gpu()
    _doit(reconstruction, params)

    params.gpuSwitch = False
    reconstruction._move_data_to_cpu()
    _doit(reconstruction, params)


def _doit(reconstruction, params):
    for operator_name in Operators.Operators.forward_lookup_dictionary:
        params.propagatorType = operator_name
        reconstruction.esw = reconstruction.probe
        print("\n")
        import time

        for i in range(3):
            t0 = time.time()
            Operators.Operators.object2detector(
                reconstruction.esw, params, reconstruction
            )
            # out = operator(reconstruction.probe, params, reconstruction)
            t1 = time.time()
            print(operator_name, i, 1e3 * (t1 - t0), "ms")


def test__propagate_fresnel():
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )

    reconstruction.initializeObjectProbe()
    reconstruction.esw = 2
    for operator in [
        Operators.Operators.propagate_fresnel,
        Operators.Operators.propagate_ASP,
        Operators.Operators.propagate_scaledASP,
        Operators.Operators.propagate_twoStepPolychrome,
        Operators.Operators.propagate_scaledPolychromeASP,
    ]:
        params.gpuSwitch = True
        reconstruction._move_data_to_gpu()

        import time

        for i in range(3):
            t0 = time.time()
            out = operator(reconstruction.probe, params, reconstruction)
            t1 = time.time()
            print(i, 1e3 * (t1 - t0), "ms")

        params.gpuSwitch = False
        reconstruction._move_data_to_cpu()

        import time

        for i in range(10):
            t0 = time.time()
            out = operator(reconstruction.probe, params, reconstruction)
            t1 = time.time()
            print(i, 1e3 * (t1 - t0), "ms")


def test_aspw_cached():
    assert False


class TestASP(TestCase):
    def test_propagate_asp(self):
        experimentalData, reconstruction, params, monitor, engine = easyInitialize(
            "example:simulation_cpm"
        )
        reconstruction.esw = None
        a = reconstruction.probe
        P1 = Operators.Operators.propagate_ASP(a,params, reconstruction,z=1e-3, fftflag=False)[1]
        P2 = Operators.Operators.propagate_ASP(a, params, reconstruction, z=1e-3, fftflag=True)[1]
        assert_allclose(P1, P2)
