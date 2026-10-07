import numpy as np
import pytest

from types import SimpleNamespace

from PtyLab.Engines.purityPIE import purityPIE


# ============================================================================
# Helpers
# ============================================================================


def make_candidate(
    error=0.1,
    dxp=1.0,
):
    """
    Create a minimal candidate reconstruction for purityPIE scan tests.
    """

    return SimpleNamespace(
        object=np.ones(
            (1, 1, 8, 8),
            dtype=complex,
        ),
        probe=np.ones(
            (1, 1, 8, 8),
            dtype=complex,
        ),
        error=np.asarray([error]),
        dxp=dxp,
    )


@pytest.fixture
def engine():
    """
    Create a minimal purityPIE instance without running the full
    BaseEngine / Reconstruction initialization.

    The expensive mPIE candidate reconstructions are mocked in individual
    tests through `_evaluatePurityAtZ()` or
    `_evaluatePurityAtWavelength()`.
    """

    eng = object.__new__(purityPIE)

    # ------------------------------------------------------------------
    # Minimal reconstruction
    # ------------------------------------------------------------------

    eng.reconstruction = SimpleNamespace(
        npsm=2,
        zo=47e-3,
        wavelength=13.0e-9,
        object=np.zeros(
            (1, 1, 8, 8),
            dtype=complex,
        ),
        probe=np.zeros(
            (1, 1, 8, 8),
            dtype=complex,
        ),
        error=np.asarray([]),
        purityProbe=None,
        dxp=1.0,
    )

    # ------------------------------------------------------------------
    # Minimal parameter set
    # ------------------------------------------------------------------

    eng.params = SimpleNamespace(

        # Fixed z scan
        purityZScanRange=3e-3,
        purityZScanPoints=31,

        # Adaptive z scan
        purityZInitialStep=200e-6,
        purityZStepGrowth=1.5,
        purityZStepShrink=0.5,
        purityZMinStep=10e-6,
        purityZPurityTolerance=1e-12,

        # Fixed wavelength scan
        purityWavelengthScanRange=1.0e-9,
        purityWavelengthScanPoints=21,

        # Adaptive wavelength scan
        purityWavelengthInitialStep=0.1e-9,
        purityWavelengthStepGrowth=1.5,
        purityWavelengthStepShrink=0.5,
        purityWavelengthMinStep=0.01e-9,
        purityWavelengthPurityTolerance=1e-12,

        # Shared adaptive limit
        purityMaxEvaluations=30,

        # reconstruct()
        purityCalibrationTarget="z",
        purityAdaptive=False,
        purityScanPlot=False,
        gpuFlag=0,
    )

    # Number of mPIE iterations used for candidate reconstructions.
    # Required only for progress / monitor bookkeeping in scan methods.
    eng.numIterations = 1

    # Disable graphical monitor updates.
    eng.showReconstruction = (
        lambda *args, **kwargs: None
    )

    return eng


# ============================================================================
# Fixed-grid axial calibration
# ============================================================================


def test_fixed_z_scan_finds_parabolic_peak(
    engine,
):

    true_z = 50e-3

    engine.reconstruction.zo = 50e-3

    engine.params.purityZScanRange = 1e-3
    engine.params.purityZScanPoints = 21

    def fake_evaluate(z):

        purity = (
            1.0
            - ((z - true_z) / 1e-3) ** 2
        )

        candidate = make_candidate(
            error=1.0 - purity,
        )

        return purity, candidate

    engine._evaluatePurityAtZ = (
        fake_evaluate
    )

    best_z, best_purity = (
        engine.purityZScan()
    )

    assert np.isclose(
        best_z,
        true_z,
    )

    assert np.isclose(
        best_purity,
        1.0,
    )

    assert len(
        engine.purityZValues
    ) == engine.params.purityZScanPoints

    assert len(
        engine.purityZMetrics
    ) == engine.params.purityZScanPoints

    assert len(
        engine.purityZErrors
    ) == engine.params.purityZScanPoints

    assert np.all(
        np.diff(engine.purityZValues) > 0
    )

    assert np.isclose(
        engine.reconstruction.zo,
        best_z,
    )

    assert np.isclose(
        engine.reconstruction.purityProbe,
        best_purity,
    )


# ============================================================================
# Adaptive axial calibration
# ============================================================================


def test_adaptive_z_scan_finds_parabolic_peak(
    engine,
):

    true_z = 50e-3

    engine.reconstruction.zo = 47e-3

    def fake_evaluate(z):

        purity = (
            1.0
            - ((z - true_z) / 1e-3) ** 2
        )

        candidate = make_candidate(
            error=1.0 - purity,
        )

        return purity, candidate

    engine._evaluatePurityAtZ = (
        fake_evaluate
    )

    best_z, best_purity = (
        engine.adaptivePurityZScan()
    )

    error_um = (
        abs(best_z - true_z)
        * 1e6
    )

    assert error_um <= 20

    assert np.isclose(
        best_purity,
        1.0,
        atol=1e-3,
    )

    assert len(
        engine.purityZValues
    ) <= engine.params.purityMaxEvaluations

    assert len(
        engine.purityZValues
    ) == len(
        engine.purityZMetrics
    )

    assert len(
        engine.purityZValues
    ) == len(
        engine.purityZErrors
    )

    assert np.all(
        np.diff(engine.purityZValues) > 0
    )

    assert np.isclose(
        engine.reconstruction.zo,
        best_z,
    )

    assert np.isclose(
        engine.reconstruction.purityProbe,
        best_purity,
    )


def test_adaptive_z_scan_with_noise(
    engine,
):

    true_z = 50e-3

    rng = np.random.default_rng(0)

    engine.reconstruction.zo = 47e-3

    engine.params.purityZMinStep = 20e-6
    engine.params.purityZPurityTolerance = (
        5e-4
    )

    def fake_evaluate(z):

        clean = (
            1.0
            - ((z - true_z) / 1e-3) ** 2
        )

        noise = rng.normal(
            loc=0.0,
            scale=0.002,
        )

        purity = clean + noise

        return (
            purity,
            make_candidate(
                error=1.0 - purity,
            ),
        )

    engine._evaluatePurityAtZ = (
        fake_evaluate
    )

    best_z, best_purity = (
        engine.adaptivePurityZScan()
    )

    error_um = (
        abs(best_z - true_z)
        * 1e6
    )

    assert error_um <= 100

    assert len(
        engine.purityZValues
    ) <= engine.params.purityMaxEvaluations


@pytest.mark.parametrize(
    "seed",
    range(20),
)
def test_adaptive_z_scan_with_noise_multiple_seeds(
    engine,
    seed,
):

    true_z = 50e-3

    rng = np.random.default_rng(seed)

    engine.reconstruction.zo = 47e-3

    engine.params.purityZMinStep = 20e-6
    engine.params.purityZPurityTolerance = (
        5e-4
    )

    def fake_evaluate(z):

        clean = (
            1.0
            - ((z - true_z) / 1e-3) ** 2
        )

        noise = rng.normal(
            loc=0.0,
            scale=0.002,
        )

        purity = clean + noise

        return (
            purity,
            make_candidate(
                error=1.0 - purity,
            ),
        )

    engine._evaluatePurityAtZ = (
        fake_evaluate
    )

    best_z, _ = (
        engine.adaptivePurityZScan()
    )

    error_um = (
        abs(best_z - true_z)
        * 1e6
    )

    assert error_um <= 100

    assert len(
        engine.purityZValues
    ) <= engine.params.purityMaxEvaluations


def test_adaptive_z_scan_respects_max_evaluations(
    engine,
):

    true_z = 50e-3

    engine.reconstruction.zo = 47e-3

    engine.params.purityMaxEvaluations = 10

    calls = []

    def fake_evaluate(z):

        calls.append(float(z))

        purity = (
            1.0
            - ((z - true_z) / 1e-3) ** 2
        )

        return (
            purity,
            make_candidate(
                error=1.0 - purity,
            ),
        )

    engine._evaluatePurityAtZ = (
        fake_evaluate
    )

    engine.adaptivePurityZScan()

    assert len(calls) <= 10

    assert len(
        engine.purityZValues
    ) <= 10


def test_adaptive_z_scan_does_not_repeat_evaluations(
    engine,
):

    true_z = 50e-3

    engine.reconstruction.zo = 47e-3

    calls = []

    def fake_evaluate(z):

        calls.append(float(z))

        purity = (
            1.0
            - ((z - true_z) / 1e-3) ** 2
        )

        return (
            purity,
            make_candidate(
                error=1.0 - purity,
            ),
        )

    engine._evaluatePurityAtZ = (
        fake_evaluate
    )

    engine.adaptivePurityZScan()

    assert len(calls) == len(
        set(calls)
    )


def test_adaptive_z_scan_when_initial_guess_is_near_peak(
    engine,
):

    true_z = 50e-3

    engine.reconstruction.zo = (
        49.95e-3
    )

    engine.params.purityZInitialStep = (
        100e-6
    )

    def fake_evaluate(z):

        purity = (
            1.0
            - ((z - true_z) / 1e-3) ** 2
        )

        return (
            purity,
            make_candidate(
                error=1.0 - purity,
            ),
        )

    engine._evaluatePurityAtZ = (
        fake_evaluate
    )

    best_z, _ = (
        engine.adaptivePurityZScan()
    )

    error_um = (
        abs(best_z - true_z)
        * 1e6
    )

    assert error_um <= 20


# ============================================================================
# Fixed-grid wavelength calibration
# ============================================================================


def test_fixed_wavelength_scan_finds_peak(
    engine,
):

    true_wavelength = 13.5e-9

    engine.reconstruction.wavelength = (
        13.5e-9
    )

    engine.params.purityWavelengthScanRange = (
        0.5e-9
    )

    engine.params.purityWavelengthScanPoints = (
        21
    )

    def fake_evaluate(wavelength):

        purity = (
            1.0
            - (
                (
                    wavelength
                    - true_wavelength
                )
                / 0.5e-9
            ) ** 2
        )

        candidate = make_candidate(
            error=1.0 - purity,
            dxp=2.0,
        )

        return purity, candidate

    engine._evaluatePurityAtWavelength = (
        fake_evaluate
    )

    (
        best_wavelength,
        best_purity,
    ) = engine.purityWavelengthScan()

    assert np.isclose(
        best_wavelength,
        true_wavelength,
    )

    assert np.isclose(
        best_purity,
        1.0,
    )

    assert len(
        engine.purityWavelengthValues
    ) == (
        engine.params
        .purityWavelengthScanPoints
    )

    assert len(
        engine.purityWavelengthMetrics
    ) == (
        engine.params
        .purityWavelengthScanPoints
    )

    assert len(
        engine.purityWavelengthErrors
    ) == (
        engine.params
        .purityWavelengthScanPoints
    )

    assert np.all(
        np.diff(
            engine.purityWavelengthValues
        ) > 0
    )

    assert np.isclose(
        engine.reconstruction.wavelength,
        best_wavelength,
    )

    assert np.isclose(
        engine.reconstruction.purityProbe,
        best_purity,
    )


# ============================================================================
# Adaptive wavelength calibration
# ============================================================================


def test_adaptive_wavelength_scan_finds_peak(
    engine,
):

    true_wavelength = 13.5e-9

    engine.reconstruction.wavelength = (
        12.5e-9
    )

    def fake_evaluate(wavelength):

        purity = (
            1.0
            - (
                (
                    wavelength
                    - true_wavelength
                )
                / 0.5e-9
            ) ** 2
        )

        candidate = make_candidate(
            error=1.0 - purity,
            dxp=2.0,
        )

        return purity, candidate

    engine._evaluatePurityAtWavelength = (
        fake_evaluate
    )

    (
        best_wavelength,
        best_purity,
    ) = (
        engine
        .adaptivePurityWavelengthScan()
    )

    error_pm = (
        abs(
            best_wavelength
            - true_wavelength
        )
        * 1e12
    )

    assert error_pm <= 20

    assert np.isclose(
        best_purity,
        1.0,
        atol=1e-3,
    )

    assert len(
        engine.purityWavelengthValues
    ) <= engine.params.purityMaxEvaluations

    assert len(
        engine.purityWavelengthValues
    ) == len(
        engine.purityWavelengthMetrics
    )

    assert len(
        engine.purityWavelengthValues
    ) == len(
        engine.purityWavelengthErrors
    )

    assert np.all(
        np.diff(
            engine.purityWavelengthValues
        ) > 0
    )

    assert np.isclose(
        engine.reconstruction.wavelength,
        best_wavelength,
    )

    assert np.isclose(
        engine.reconstruction.purityProbe,
        best_purity,
    )


def test_adaptive_wavelength_scan_respects_max_evaluations(
    engine,
):

    true_wavelength = 13.5e-9

    engine.reconstruction.wavelength = (
        12.5e-9
    )

    engine.params.purityMaxEvaluations = 10

    calls = []

    def fake_evaluate(wavelength):

        calls.append(
            float(wavelength)
        )

        purity = (
            1.0
            - (
                (
                    wavelength
                    - true_wavelength
                )
                / 0.5e-9
            ) ** 2
        )

        return (
            purity,
            make_candidate(
                error=1.0 - purity,
                dxp=2.0,
            ),
        )

    engine._evaluatePurityAtWavelength = (
        fake_evaluate
    )

    engine.adaptivePurityWavelengthScan()

    assert len(calls) <= 10

    assert len(
        engine.purityWavelengthValues
    ) <= 10


def test_adaptive_wavelength_scan_does_not_repeat_evaluations(
    engine,
):

    true_wavelength = 13.5e-9

    engine.reconstruction.wavelength = (
        12.5e-9
    )

    calls = []

    def fake_evaluate(wavelength):

        calls.append(
            float(wavelength)
        )

        purity = (
            1.0
            - (
                (
                    wavelength
                    - true_wavelength
                )
                / 0.5e-9
            ) ** 2
        )

        return (
            purity,
            make_candidate(
                error=1.0 - purity,
                dxp=2.0,
            ),
        )

    engine._evaluatePurityAtWavelength = (
        fake_evaluate
    )

    engine.adaptivePurityWavelengthScan()

    assert len(calls) == len(
        set(calls)
    )


# ============================================================================
# Probe-mode requirements
# ============================================================================


@pytest.mark.parametrize(
    "method_name",
    [
        "purityZScan",
        "adaptivePurityZScan",
        "purityWavelengthScan",
        "adaptivePurityWavelengthScan",
    ],
)
def test_purity_scan_requires_two_probe_modes(
    engine,
    method_name,
):

    engine.reconstruction.npsm = 1

    method = getattr(
        engine,
        method_name,
    )

    with pytest.raises(
        ValueError,
        match="at least two probe modes",
    ):
        method()


# ============================================================================
# Convenience wrapper tests
# ============================================================================


def test_evaluate_purity_at_z_delegates_to_generic_helper(
    engine,
):

    calls = []

    expected = (
        0.9,
        make_candidate(),
    )

    def fake_generic(
        target,
        value,
    ):
        calls.append(
            (target, value)
        )
        return expected

    engine._evaluatePurityAtParameter = (
        fake_generic
    )

    result = engine._evaluatePurityAtZ(
        50e-3
    )

    assert result is expected

    assert calls == [
        ("z", 50e-3)
    ]


def test_evaluate_purity_at_wavelength_delegates_to_generic_helper(
    engine,
):

    calls = []

    expected = (
        0.9,
        make_candidate(),
    )

    def fake_generic(
        target,
        value,
    ):
        calls.append(
            (target, value)
        )
        return expected

    engine._evaluatePurityAtParameter = (
        fake_generic
    )

    result = (
        engine
        ._evaluatePurityAtWavelength(
            13.5e-9
        )
    )

    assert result is expected

    assert calls == [
        ("wavelength", 13.5e-9)
    ]


# ============================================================================
# reconstruct() dispatch
# ============================================================================


@pytest.mark.parametrize(
    (
        "target",
        "adaptive",
        "expected_method",
    ),
    [
        (
            "z",
            False,
            "purityZScan",
        ),
        (
            "z",
            True,
            "adaptivePurityZScan",
        ),
        (
            "wavelength",
            False,
            "purityWavelengthScan",
        ),
        (
            "wavelength",
            True,
            "adaptivePurityWavelengthScan",
        ),
    ],
)
def test_reconstruct_dispatches_to_correct_scan(
    engine,
    target,
    adaptive,
    expected_method,
):

    engine.params.purityCalibrationTarget = (
        target
    )

    engine.params.purityAdaptive = (
        adaptive
    )

    engine.params.purityScanPlot = False

    # Avoid BaseEngine initialization work.
    engine.changeExperimentalData = (
        lambda data: None
    )

    engine.changeOptimizable = (
        lambda reconstruction: None
    )

    engine._prepareReconstruction = (
        lambda: None
    )

    calls = []

    def make_fake_scan(name):

        def fake_scan():

            calls.append(name)

            return (
                1.23,
                0.95,
            )

        return fake_scan

    engine.purityZScan = (
        make_fake_scan(
            "purityZScan"
        )
    )

    engine.adaptivePurityZScan = (
        make_fake_scan(
            "adaptivePurityZScan"
        )
    )

    engine.purityWavelengthScan = (
        make_fake_scan(
            "purityWavelengthScan"
        )
    )

    engine.adaptivePurityWavelengthScan = (
        make_fake_scan(
            "adaptivePurityWavelengthScan"
        )
    )

    best_value, best_purity = (
        engine.reconstruct()
    )

    assert calls == [
        expected_method
    ]

    assert best_value == 1.23

    assert best_purity == 0.95


def test_reconstruct_rejects_unknown_target(
    engine,
):

    engine.params.purityCalibrationTarget = (
        "invalid"
    )

    engine.params.purityAdaptive = False
    engine.params.purityScanPlot = False

    engine.changeExperimentalData = (
        lambda data: None
    )

    engine.changeOptimizable = (
        lambda reconstruction: None
    )

    engine._prepareReconstruction = (
        lambda: None
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported purity calibration target"
        ),
    ):
        engine.reconstruct()


# ============================================================================
# reconstruct() plotting
# ============================================================================


def test_reconstruct_calls_plot_when_enabled(
    engine,
):

    engine.params.purityCalibrationTarget = (
        "z"
    )

    engine.params.purityAdaptive = False
    engine.params.purityScanPlot = True

    engine.changeExperimentalData = (
        lambda data: None
    )

    engine.changeOptimizable = (
        lambda reconstruction: None
    )

    engine._prepareReconstruction = (
        lambda: None
    )

    engine.purityZScan = (
        lambda: (
            50e-3,
            0.95,
        )
    )

    plot_calls = []

    engine.plotPurityScan = (
        lambda: plot_calls.append(
            True
        )
    )

    engine.reconstruct()

    assert plot_calls == [True]