from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

import PtyLab
from PtyLab import Engines
from PtyLab.Reconstruction.Reconstruction import Reconstruction


Z0 = 50e-3
SCAN_RANGE = 2e-3
SCAN_POINTS = 3
N_ITERATIONS = 5
RNG_SEED = 0


def configure_reconstruction(reconstruction):
    reconstruction.npsm = 4
    reconstruction.nosm = 1
    reconstruction.nlambda = 1
    reconstruction.nslice = 1

    reconstruction.initialProbe = "circ"
    reconstruction.initialObject = "ones"

    reconstruction.initializeObjectProbe()


def configure_params(params):
    params.positionOrder = "random"
    params.propagatorType = "Fraunhofer"

    params.gpuSwitch = False
    params.fftshiftSwitch = False

    params.intensityConstraint = "standard"

    params.positionCorrectionSwitch = False
    params.modulusEnforcedProbeSwitch = False
    params.probePowerCorrectionSwitch = True

    params.probeSmoothenessSwitch = True
    params.probeSmoothnessAleph = 1e-2
    params.probeSmoothenessWidth = 10

    params.comStabilizationSwitch = 10

    # Keep normal mPIE behavior
    params.orthogonalizationSwitch = True
    params.orthogonalizationFrequency = 10

    params.absorbingProbeBoundary = False

    params.objectContrastSwitch = False
    params.absObjectSwitch = False
    params.backgroundModeSwitch = False

    params.couplingSwitch = True
    params.couplingAleph = 1

    params.TV_autofocus = False
    params.l2reg = False


def create_fresh_reference(
    source_data,
    source_reconstruction,
    source_params,
    z,
):
    """
    Independently create one fresh reconstruction at candidate z.
    This deliberately does NOT use purityPIE's helper.
    """

    candidate_data = deepcopy(source_data)
    candidate_params = deepcopy(source_params)

    candidate_data.zo = z

    candidate_reconstruction = Reconstruction(
        candidate_data,
        candidate_params,
    )

    candidate_reconstruction.npsm = source_reconstruction.npsm
    candidate_reconstruction.nosm = source_reconstruction.nosm
    candidate_reconstruction.nlambda = source_reconstruction.nlambda
    candidate_reconstruction.nslice = source_reconstruction.nslice

    candidate_reconstruction.initialProbe = (
        source_reconstruction.initialProbe
    )
    candidate_reconstruction.initialObject = (
        source_reconstruction.initialObject
    )

    if hasattr(source_reconstruction, "initialProbe_filename"):
        candidate_reconstruction.initialProbe_filename = (
            source_reconstruction.initialProbe_filename
        )

    if hasattr(source_reconstruction, "initialObject_filename"):
        candidate_reconstruction.initialObject_filename = (
            source_reconstruction.initialObject_filename
        )

    candidate_reconstruction.initializeObjectProbe()

    return (
        candidate_data,
        candidate_reconstruction,
        candidate_params,
    )


@pytest.fixture
def scan_setup(generate_simu_hdf5):
    """Create a CPU reconstruction and restore global RNG state after the test."""
    random_state = np.random.get_state()
    try:
        np.random.seed(RNG_SEED)
        data, reconstruction, params, monitor, _ = PtyLab.easyInitialize(
            generate_simu_hdf5,
            operationMode="CPM",
            dummyMonitor=True,
        )
        reconstruction.zo = Z0
        configure_reconstruction(reconstruction)
        configure_params(params)
        params.purityZScanRange = SCAN_RANGE
        params.purityZScanPoints = SCAN_POINTS
        params.purityZScanPlot = False
        yield data, reconstruction, params, monitor
    finally:
        np.random.set_state(random_state)


def test_fresh_curve_matches_new_purity_scan(scan_setup):
    """The scan must match independent mPIE runs and select their best distance."""
    data, reconstruction, params, monitor = scan_setup
    z_values = np.linspace(Z0 - SCAN_RANGE, Z0 + SCAN_RANGE, SCAN_POINTS)

    # Seed once per complete scan; candidates consume the RNG sequence naturally.
    np.random.seed(RNG_SEED)
    fresh_purities = []
    for z in z_values:
        fresh_data, fresh_reconstruction, fresh_params = create_fresh_reference(
            data, reconstruction, params, z
        )
        fresh_engine = Engines.mPIE(
            fresh_reconstruction, fresh_data, fresh_params, monitor
        )
        fresh_engine.numIterations = N_ITERATIONS
        fresh_engine.reconstruct()
        fresh_engine.orthogonalization()
        fresh_purities.append(float(np.asarray(fresh_reconstruction.purityProbe).squeeze()))

    fresh_purities = np.asarray(fresh_purities)

    # Use exactly the same random sequence for the purityPIE scan.
    np.random.seed(RNG_SEED)
    purity_engine = Engines.purityPIE(reconstruction, data, params, monitor)
    purity_engine.numIterations = N_ITERATIONS
    best_z, best_purity = purity_engine.reconstruct()

    assert_allclose(purity_engine.purityZValues, z_values)
    assert_allclose(
        purity_engine.purityZMetrics, fresh_purities, rtol=1e-6, atol=1e-7
    )
    assert best_z == pytest.approx(z_values[np.argmax(fresh_purities)])
    assert best_purity == pytest.approx(np.max(fresh_purities), rel=1e-6, abs=1e-7)
