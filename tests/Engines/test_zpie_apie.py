"""Smoke tests for the calibration engines zPIE and aPIE.

They only check that the engines run end to end on the CPU and that aPIE's
resampling is exact where it should be, not that the calibration converges.
"""

import matplotlib
import numpy as np
import pytest

import PtyLab
from PtyLab import Engines

matplotlib.use("Agg")  # aPIE draws its angle history with pyplot


def setup(engine, **attributes):
    experimentalData, reconstruction, params, monitor, _ = PtyLab.easyInitialize(
        "example:simulation_cpm", operationMode="CPM", dummyMonitor=True
    )
    params.gpuSwitch = False
    for name, value in attributes.items():
        setattr(reconstruction, name, value)
    return engine(reconstruction, experimentalData, params, monitor)


def test_zpie_runs_and_updates_z():
    engine = setup(Engines.zPIE)
    engine.numIterations = 3
    z0 = engine.reconstruction.zo
    engine.reconstruct()

    z = engine.reconstruction.zHistory
    assert len(z) == 3
    assert np.all(np.isfinite(z))
    assert z[0] == z0  # no z update in the first iteration
    assert len(engine.reconstruction.merit) == len(engine.reconstruction.dz) == 11


def test_apie_resampling_is_identity_at_zero_angle():
    engine = setup(Engines.aPIE, theta=0.0)
    ptychogram, W = engine.resampleDetector(0.0)
    # at theta = 0, T_inv is the identity, so every pixel samples itself
    np.testing.assert_allclose(ptychogram, engine.ptychogramUntransformed, rtol=1e-6)
    np.testing.assert_allclose(W, 1.0)


def test_apie_runs_and_records_theta():
    engine = setup(Engines.aPIE, theta=45.0)
    engine.numIterations = 2
    engine.reconstruct()

    assert engine.reconstruction.thetaHistory.shape == (2,)
    assert np.isfinite(float(engine.reconstruction.theta))
    assert np.all(np.isfinite(engine.reconstruction.object))


def test_apie_doReconstruction_is_deprecated():
    engine = setup(Engines.aPIE, theta=45.0)
    engine.numIterations = 1
    with pytest.warns(DeprecationWarning, match="reconstruct"):
        engine.doReconstruction()


def test_apie_scan_returns_best_candidate():
    engine = setup(Engines.aPIE, theta=45.0)
    engine.thetaSearch = "scan"
    engine.numIterations = 2
    engine.thetaScanRange = 2.0
    engine.thetaScanPoints = 5
    engine.thetaScanRefinements = 1
    best, error = engine.reconstruct()

    # 5 points 1 deg apart, then 5 points 0.5 deg apart around the best one,
    # of which 2 (best at the edge) or 3 were already scanned
    assert len(engine.thetaScanValues) in (7, 8)
    assert error == min(engine.thetaScanErrors)
    assert best == engine.thetaScanValues[engine.thetaScanErrors.index(error)]
    assert engine.reconstruction.theta == best
    assert np.all(np.isfinite(engine.reconstruction.object))
