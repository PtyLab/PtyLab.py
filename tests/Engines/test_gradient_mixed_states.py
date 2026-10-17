"""Mixed object and probe states in the optional Torch engine."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PtyLab import DummyMonitor, ExperimentalData, Params, Reconstruction  # noqa: E402
from PtyLab.Engines.GradientEngine import (  # noqa: E402
    GradientEngine,
    PhaseRampRemoval,
    ProbeCentering,
)
from PtyLab.Operators.Operators import object2detector  # noqa: E402
from PtyLab.utils.utils import fft2c  # noqa: E402

COUNTS = [(1, 3), (2, 1), (2, 3)]  # (nosm, npsm)


def mixed_engine(nosm, npsm, propagator="Fraunhofer", seed=3):
    """8x8 frames, 9 positions, data from random object and probe states."""
    data = ExperimentalData(operationMode="CPM")
    data.wavelength = 632.8e-9
    data.zo = 0.05
    data.dxd = 75e-6
    nd = 8
    dx = data.wavelength * data.zo / (nd * data.dxd)
    grid = np.array([-2, 0, 2])
    data.encoder = np.stack(np.meshgrid(grid, grid), -1).reshape(-1, 2) * dx
    data.ptychogram = np.ones((len(data.encoder), nd, nd), dtype=np.float32)
    data.entrancePupilDiameter = 4 * dx
    data.spectralDensity = None
    data.theta = None
    data._setData()
    params = Params()
    params.gpuSwitch = False
    params.positionOrder = "sequential"
    params.propagatorType = propagator
    reconstruction = Reconstruction(data, params)
    reconstruction.nlambda = reconstruction.nslice = 1
    reconstruction.nosm, reconstruction.npsm = nosm, npsm
    reconstruction.initialObject = "ones"
    reconstruction.initialProbe = "circ"
    reconstruction.initializeObjectProbe()

    rng = np.random.default_rng(seed)
    No = reconstruction.No
    objects = (0.5 + rng.random((nosm, No, No))) * np.exp(
        0.3j * rng.standard_normal((nosm, No, No))
    )
    yy, xx = np.mgrid[-4:4, -4:4]
    gaussian = np.exp(-(xx**2 + yy**2) / 6)
    # Probe states with decreasing power and different phase structure.
    probes = np.stack(
        [
            gaussian * (0.6**m) * np.exp(1j * (0.2 * xx + 0.4 * m * yy))
            for m in range(npsm)
        ]
    )
    # Data: incoherent sum over every (a, m) pair, computed with NumPy.
    for index, (row, col) in enumerate(reconstruction.positions):
        patch = objects[:, row : row + nd, col : col + nd]
        data.ptychogram[index] = sum(
            abs(fft2c(patch[a] * probes[m])) ** 2
            for a in range(nosm)
            for m in range(npsm)
        )
    data._setData()
    # Start: perturbed truth, so every mode is distinct and non-zero.
    reconstruction.object = (
        (objects * 0.9 + 0.05 * rng.standard_normal(objects.shape))
        .reshape(reconstruction.object.shape)
        .astype(np.complex64)
    )
    reconstruction.probe = (
        (probes * 0.85 + 0.02 * rng.standard_normal(probes.shape))
        .reshape(reconstruction.probe.shape)
        .astype(np.complex64)
    )
    engine = GradientEngine(reconstruction, data, params, DummyMonitor())
    engine.device = "cpu"
    return engine


def _all_intensities(engine):
    with torch.no_grad():
        return engine.forward(torch.arange(len(engine.positions))).clone()


@pytest.mark.parametrize("nosm, npsm", COUNTS)
@pytest.mark.parametrize("propagator", ["Fraunhofer", "Fresnel"])
def test_mixed_prediction_matches_a_numpy_loop_over_state_pairs(nosm, npsm, propagator):
    engine = mixed_engine(nosm, npsm, propagator)
    engine.prepareReconstruction()
    r = engine.reconstruction
    indices = torch.arange(len(engine.positions))
    fields = engine.model.detector_fields(indices)
    assert fields.shape == (len(indices), 1, nosm, npsm, 1, r.Np, r.Np)

    r.esw = None  # PtyLab's propagators return it alongside the field
    objects = r.object.reshape(nosm, r.No, r.No)
    probes = r.probe.reshape(npsm, r.Np, r.Np)
    expected = np.zeros((len(indices), r.Np, r.Np))
    for index, (row, col) in enumerate(engine.positions):
        for a in range(nosm):
            for m in range(npsm):
                exit_wave = objects[a, row : row + r.Np, col : col + r.Np] * probes[m]
                _, detector = object2detector(exit_wave, engine.params, r)
                expected[index] += abs(np.squeeze(detector)) ** 2
    np.testing.assert_allclose(
        engine.forward(indices).detach().numpy(),
        expected,
        rtol=1e-5,
        atol=1e-7 * expected.max(),
    )


@pytest.mark.parametrize("nosm, npsm", COUNTS)
def test_gradients_reach_every_state_and_pass_gradcheck(nosm, npsm):
    from torch.func import functional_call

    engine = mixed_engine(nosm, npsm)
    engine.prepareReconstruction()
    model = engine.model
    indices = torch.tensor([0, 4, 8])
    obj = model.object.detach().to(torch.complex128).requires_grad_()
    field = model.probe.field.detach().to(torch.complex128).requires_grad_()

    def intensity(o, f):
        return functional_call(model, {"object": o, "probe.field": f}, (indices,))

    assert torch.autograd.gradcheck(
        intensity, (obj, field), eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True
    )
    intensity(obj, field).sum().backward()
    # Every object state is lit by every probe state, so no state has a zero
    # gradient where the scan illuminates it.
    for gradient, axis in ((obj.grad, 1), (field.grad, 2)):
        per_state = gradient.abs().sum(dim=(-2, -1)).flatten()
        assert len(per_state) == gradient.shape[axis]
        assert (per_state > 0).all()


def test_an_empty_extra_probe_mode_reduces_to_one_mode():
    single = mixed_engine(1, 1)
    double = mixed_engine(1, 2)
    probe = double.reconstruction.probe.copy()
    probe[:, :, 0] = single.reconstruction.probe[:, :, 0]
    probe[:, :, 1] = 0
    double.reconstruction.probe = probe
    double.reconstruction.object = single.reconstruction.object.copy()
    double.experimentalData = single.experimentalData
    for candidate in (single, double):
        candidate.prepareReconstruction()
        indices = torch.arange(len(candidate.positions))
        candidate.optimizer.zero_grad(set_to_none=True)
        candidate.computeLoss(
            candidate.forward(indices), candidate.measuredIntensities
        ).backward()
    torch.testing.assert_close(_all_intensities(double), _all_intensities(single))
    torch.testing.assert_close(double.model.object.grad, single.model.object.grad)
    torch.testing.assert_close(
        double.model.probe.field.grad[:, :, :1], single.model.probe.field.grad
    )


def test_a_unitary_rotation_of_probe_modes_keeps_the_intensity():
    engine = mixed_engine(2, 3)
    engine.prepareReconstruction()
    before = _all_intensities(engine)
    # Random unitary 3x3 from the QR decomposition of a complex Gaussian.
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    unitary = torch.as_tensor(q, dtype=torch.complex64)
    with torch.no_grad():
        field = engine.model.probe.field
        # Q_n = sum_m P_m U_mn over the probe-state axis 2.
        field.copy_(torch.einsum("lamsyx,mn->lansyx", field, unitary))
    after = _all_intensities(engine)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6 * before.max())


@pytest.mark.parametrize("batch_size", [2, 4])  # partial final batch of 1
def test_mixed_batches_match_the_full_scan(batch_size):
    full, batched = mixed_engine(2, 3), mixed_engine(2, 3)
    batched.batchSize = batch_size
    for candidate in (full, batched):
        candidate.prepareReconstruction()
        candidate.setPositionOrder = lambda: None
        candidate.positionIndices = np.arange(len(candidate.positions))
    assert batched.runIteration(0) == pytest.approx(full.runIteration(0), rel=2e-6)
    for name in ("object", "probe.field"):
        actual = dict(batched.model.named_parameters())[name]
        expected = dict(full.model.named_parameters())[name]
        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=2e-7)


def test_mixed_preconditioners_sum_over_the_other_tensor_states():
    engine = mixed_engine(2, 3)
    engine.prepareReconstruction()
    r = engine.reconstruction
    indices = torch.tensor([0, 3, 5])
    curvature = engine.model.gradient_preconditioners(indices)
    objects = r.object.reshape(2, r.No, r.No)
    probes = r.probe.reshape(3, r.Np, r.Np)
    illumination = np.zeros((r.No, r.No))
    coverage = np.zeros((r.Np, r.Np))
    for row, col in r.positions[[0, 3, 5]]:
        illumination[row : row + r.Np, col : col + r.Np] += (abs(probes) ** 2).sum(0)
        coverage += (abs(objects[:, row : row + r.Np, col : col + r.Np]) ** 2).sum(0)
    coverage *= engine.model.probe.scale.item() ** 2
    assert curvature["object"].shape == engine.model.object.shape
    assert curvature["probe.field"].shape == engine.model.probe.field.shape
    for a in range(2):
        np.testing.assert_allclose(
            curvature["object"][0, a, 0, 0].numpy(), illumination, rtol=1e-5
        )
    for m in range(3):
        np.testing.assert_allclose(
            curvature["probe.field"][0, 0, m, 0].numpy(), coverage, rtol=1e-5
        )


@pytest.mark.parametrize("optimizer", ["adam", "preconditioned"])
def test_mixed_reconstruction_reduces_the_loss_and_keeps_the_shapes(optimizer):
    engine = mixed_engine(2, 3)
    engine.optimizerType = optimizer
    if optimizer == "preconditioned":
        engine.parameters = {"object": {"lr": 0.3}, "probe.field": {"lr": 0.3}}
    engine.numIterations = 30
    shapes = engine.reconstruction.object.shape, engine.reconstruction.probe.shape
    engine.reconstruct()
    r = engine.reconstruction
    assert (r.object.shape, r.probe.shape) == shapes
    assert np.isfinite(r.error).all()
    assert r.error[-1] < 0.5 * r.error[0]


def test_mode_powers_are_the_physical_power_of_each_mode():
    engine = mixed_engine(1, 3)
    engine.prepareReconstruction()
    probes = engine.reconstruction.probe.reshape(3, 8, 8)
    expected = (abs(probes) ** 2).sum(axis=(-2, -1))
    powers = engine.model.probe.mode_powers()
    assert powers.shape == (1, 3)
    np.testing.assert_allclose(powers[0].numpy(), expected, rtol=1e-5)


def test_probe_centering_with_modes_keeps_far_field_predictions():
    engine = mixed_engine(2, 3)
    probe = np.zeros_like(engine.reconstruction.probe)
    # Compact modes, so a whole-pixel roll stays inside the window.
    probe[..., 2:6, 2:6] = engine.reconstruction.probe[..., 2:6, 2:6]
    engine.reconstruction.probe = np.roll(probe, (2, -1), axis=(-2, -1))
    engine.prepareReconstruction()
    before = _all_intensities(engine)
    ProbeCentering(threshold=1)(engine, 0)
    row, col = ProbeCentering.centre_of_mass(engine.model.probe.field)
    assert max(abs(row), abs(col)) < 1
    torch.testing.assert_close(
        _all_intensities(engine), before, rtol=1e-5, atol=1e-6 * before.max()
    )


@pytest.mark.parametrize("propagator", ["Fraunhofer", "Fresnel"])
def test_phase_ramp_removal_with_modes_keeps_predictions(propagator):
    engine = mixed_engine(2, 3, propagator)
    r = engine.reconstruction
    rows, cols = np.mgrid[: r.No, : r.No]
    r.object = (r.object * np.exp(1j * (0.3 * rows - 0.2 * cols) + 0.7j)).astype(
        np.complex64
    )
    engine.prepareReconstruction()
    before = _all_intensities(engine)
    PhaseRampRemoval()(engine, 0)
    with torch.no_grad():
        k_row, k_col, _ = PhaseRampRemoval().estimate(engine)
    assert abs(k_row) < 0.05 and abs(k_col) < 0.05
    torch.testing.assert_close(
        _all_intensities(engine), before, rtol=1e-4, atol=1e-6 * before.max()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_mixed_objective_and_gradients_match_cpu():
    cpu, cuda = mixed_engine(2, 3), mixed_engine(2, 3)
    cuda.device = "cuda"
    for candidate in (cpu, cuda):
        candidate.prepareReconstruction()
        candidate.optimizer.zero_grad(set_to_none=True)
        loss = candidate.computeLoss(
            candidate.forward(torch.arange(9, device=candidate.device)),
            candidate.measuredIntensities,
        )
        loss.backward()
        candidate.loss = loss.item()
    assert cuda.loss == pytest.approx(cpu.loss, rel=2e-5)
    for name in ("object", "probe.field"):
        torch.testing.assert_close(
            dict(cuda.model.named_parameters())[name].grad.cpu(),
            dict(cpu.model.named_parameters())[name].grad,
            rtol=3e-5,
            atol=2e-7,
        )
