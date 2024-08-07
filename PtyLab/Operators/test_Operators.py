import time
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose

from PtyLab import easyInitialize
from PtyLab.Operators.Operators import (
    aspw,
    aspw_cached,
    forward_lookup_dictionary,
    object2detector,
    propagate_ASP,
    propagate_fresnel,
    propagate_off_axis_sas,
    propagate_scaledASP,
    propagate_scaledPolychromeASP,
    propagate_twoStepPolychrome,
)


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
    timer.repeat(aspw, (E, z, wl, L), {}, n_repeat=200, n_warmup=50)

    t0 = time.time()
    for i in range(100):
        E_prop = aspw_cached(E, z, wl, L)
    if xp is not np:
        E_prop = xp.asnumpy(E_prop)
    t1 = time.time()
    t_cached = t1 - t0
    for i in range(100):
        E_prop2 = aspw(E, z, wl, L)[0]
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
    for operator_name in forward_lookup_dictionary:
        params.propagatorType = operator_name
        reconstruction.esw = reconstruction.probe
        print("\n")
        import time

        for i in range(3):
            t0 = time.time()
            object2detector(reconstruction.esw, params, reconstruction)
            # out = operator(reconstruction.probe, params, reconstruction)
            t1 = time.time()
            print(operator_name, i, 1e3 * (t1 - t0), "ms")


def test_propagate_fresnel(test_device: str = "CPU", nruns: int = 10):
    """Checks if Fresnel based propagators are bug-free.

    Parameters
    ----------
    test_device : str, optional
        Specify hardware either as "CPU" or "GPU", by default "CPU"
    nruns : int, optional
        No. of runs for each propagator
    """
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )

    reconstruction.initializeObjectProbe()
    reconstruction.esw = 2
    for operator in [
        propagate_fresnel,
        propagate_ASP,
        propagate_scaledASP,
        propagate_twoStepPolychrome,
        propagate_scaledPolychromeASP,
    ]:
        if test_device == "GPU":
            if params.gpuSwitch:
                reconstruction._move_data_to_gpu()

                for i in range(nruns):
                    t0 = time.time()
                    _ = operator(reconstruction.probe, params, reconstruction)
                    t1 = time.time()
                    print(i, 1e3 * (t1 - t0), "ms")
            else:
                print("No GPU hardware found, please set test_device = 'CPU' ")

        elif test_device == "CPU":
            params.gpuSwitch = False
            reconstruction._move_data_to_cpu()

            print("\n---------------")
            print(f"{operator.__name__}\n")
            for i in range(nruns):
                t0 = time.time()
                _ = operator(reconstruction.probe, params, reconstruction)
                t1 = time.time()
                print(f"Run {i}: {1e3 * (t1 - t0):.3f} ms")
        else:
            raise SyntaxError("Set test_device = 'CPU' or 'GPU' ")


def test_aspw_cached():
    assert False


class TestASP(TestCase):
    def test_propagate_asp(self):
        experimentalData, reconstruction, params, monitor, engine = easyInitialize(
            "example:simulation_cpm"
        )
        reconstruction.esw = None
        a = reconstruction.probe
        P1 = propagate_ASP(a, params, reconstruction, z=1e-3, fftflag=False)[1]
        P2 = propagate_ASP(a, params, reconstruction, z=1e-3, fftflag=True)[1]
        assert_allclose(P1, P2)


def test_off_axis_sas(test_device: str = "CPU", nruns: int = 10):
    """Checks if Fresnel based propagators are bug-free.

    Parameters
    ----------
    test_device : str, optional
        Specify hardware either as "CPU" or "GPU", by default "CPU"
    nruns : int, optional
        No. of runs for each propagator
    """

    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )

    reconstruction.initializeObjectProbe()
    reconstruction.esw = 2
    reconstruction.theta = (40, 0)

    if test_device == "GPU":
        if params.gpuSwitch:
            reconstruction._move_data_to_gpu()

            for i in range(nruns):
                t0 = time.time()
                _ = propagate_off_axis_sas(reconstruction.probe, params, reconstruction)
                t1 = time.time()
                print(i, 1e3 * (t1 - t0), "ms")
        else:
            print("\nNo GPU hardware found, please set test_device = 'CPU' ")

    elif test_device == "CPU":
        params.gpuSwitch = False
        reconstruction._move_data_to_cpu()

        print("\n---------------")
        print(f"{propagate_off_axis_sas.__name__}\n")
        for i in range(nruns):
            t0 = time.time()
            _ = propagate_off_axis_sas(reconstruction.probe, params, reconstruction)
            t1 = time.time()
            print(f"Run {i}: {1e3 * (t1 - t0):.3f} ms")
    else:
        raise SyntaxError("\nSet test_device = 'CPU' or 'GPU' ")


test_off_axis_sas(test_device="CPU")
