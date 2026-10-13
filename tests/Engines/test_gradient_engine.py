"""Numerical and API checks for the optional Torch engine."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PtyLab import DummyMonitor, ExperimentalData, Params, Reconstruction
from PtyLab.Engines.GradientEngine import GradientEngine
from PtyLab.utils.utils import fft2c


@pytest.fixture
def engine():
    data = ExperimentalData(operationMode="CPM")
    data.wavelength = 632.8e-9
    data.zo = 0.05
    data.dxd = 75e-6
    nd = 8
    dx = data.wavelength * data.zo / (nd * data.dxd)
    data.encoder = np.array([[-2, -2], [-2, 2], [2, -2], [2, 2]]) * dx
    data.ptychogram = np.ones((4, nd, nd), dtype=np.float32)
    data.entrancePupilDiameter = 4 * dx
    data.spectralDensity = None
    data.theta = None
    data._setData()
    params = Params()
    params.gpuSwitch = False
    params.positionOrder = "sequential"
    reconstruction = Reconstruction(data, params)
    reconstruction.npsm = reconstruction.nosm = reconstruction.nlambda = (
        reconstruction.nslice
    ) = 1
    reconstruction.initialObject = "ones"
    reconstruction.initialProbe = "circ"
    reconstruction.initializeObjectProbe()
    rng = np.random.default_rng(42)
    true_object = (0.5 + rng.random((reconstruction.No, reconstruction.No))) * np.exp(
        0.2j * rng.standard_normal((reconstruction.No, reconstruction.No))
    )
    yy, xx = np.mgrid[-4:4, -4:4]
    true_probe = np.exp(-(xx**2 + yy**2) / 12) * np.exp(0.1j * xx)
    for index, (row, col) in enumerate(reconstruction.positions):
        data.ptychogram[index] = (
            abs(fft2c(true_object[row : row + nd, col : col + nd] * true_probe)) ** 2
        )
    data._setData()
    reconstruction.probe[...] = true_probe * 0.85
    return GradientEngine(reconstruction, data, params, DummyMonitor())


def test_reconstruction_reduces_loss_and_preserves_numpy_api(engine, tmp_path):
    engine.batchSize = 3
    engine.numIterations = 35
    before_object = engine.reconstruction.object.copy()
    before_probe = engine.reconstruction.probe.copy()
    engine.reconstruct()
    r = engine.reconstruction
    assert len(r.error) == 35
    assert r.error[-1] < 0.3 * r.error[0]
    for initial, result, tensor, physical in (
        (before_object, r.object, engine.model.object, engine.model.object),
        (
            before_probe,
            r.probe,
            engine.model.probe.field,
            engine.model.probe.physical_field(),
        ),
    ):
        assert isinstance(result, np.ndarray)
        assert result.shape == initial.shape
        assert not np.allclose(result, initial)
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
        np.testing.assert_array_equal(result, physical.detach().cpu().numpy())
        assert not np.shares_memory(result, physical.detach().cpu().numpy())
    r.saveResults(tmp_path / "autodiff.hdf5")
    assert (tmp_path / "autodiff.hdf5").exists()


@pytest.mark.parametrize(
    "name,value",
    [
        ("propagator", "polychromeASP"),
        ("comStabilizationSwitch", True),
        ("fftshiftSwitch", True),
        ("positionCorrectionSwitch", True),
    ],
)
def test_unsupported_settings_fail_explicitly(engine, name, value):
    setattr(engine.params, name, value)
    with pytest.raises(NotImplementedError):
        engine.reconstruct()


def test_raw_negative_measurements_are_accepted(engine):
    engine.device = "cpu"
    engine.experimentalData.ptychogram[0, 0, 0] = -0.5
    engine.prepareReconstruction()
    assert engine.measuredIntensities.min().item() == pytest.approx(-0.5)


def test_subclass_can_extend_model_optimizer_and_constraints(engine):
    from PtyLab.Engines.GradientEngine.models import PtychographyModel, SharedProbe

    class GainModel(PtychographyModel):
        def __init__(self, object_field, probe, positions):
            super().__init__(object_field, probe, positions)
            self.gain = torch.nn.Parameter(torch.tensor(0.8))

        def forward(self, indices):
            return self.gain * super().forward(indices)

    class GainEngine(GradientEngine):
        def createOptimizer(self):
            return torch.optim.SGD(self.parameterGroups())

        def regularizationLoss(self):
            return 0.01 * (self.model.gain - 1).square()

        def afterStep(self, iteration):
            assert not torch.is_grad_enabled()
            self.model.gain.clamp_(min=0.1)
            self.steps.append(iteration)

    extended = GainEngine(
        engine.reconstruction, engine.experimentalData, engine.params, engine.monitor
    )
    r = extended.reconstruction
    extended.model = GainModel(r.object, SharedProbe(r.probe), r.positions)
    extended.parameters = {
        "object": {"lr": 0.003},
        "probe.field": {"lr": 0.001},
        "gain": {"lr": 0.002},
    }
    extended.steps = []
    extended.numIterations = 3
    extended.reconstruct()
    assert isinstance(extended.optimizer, torch.optim.SGD)
    assert extended.steps == [0, 1, 2]
    assert extended.model.gain.item() != pytest.approx(0.8)
    assert torch.isfinite(extended.model.gain.grad)
    assert np.isfinite(extended.reconstruction.error).all()


@pytest.mark.parametrize("propagator", ["Fraunhofer", "Fresnel", "ASP", "scaledASP"])
@pytest.mark.parametrize("size", [7, 8])
def test_propagators_match_ptylab_and_pass_gradcheck(engine, propagator, size):
    from types import SimpleNamespace

    from PtyLab.Operators.Operators import object2detector

    rng = np.random.default_rng(17)
    obj = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    probe = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    dx = 5e-6
    engine.reconstruction = SimpleNamespace(
        Np=size,
        dxp=dx,
        dxo=dx,
        dxd=dx if propagator == "ASP" else 1.3 * dx,
        zo=1e-4,
        wavelength=632.8e-9,
        Lp=size * dx,
        nlambda=1,
        nosm=1,
        npsm=1,
        esw=obj * probe,
    )
    engine.params.propagator = propagator
    engine.device = "cpu"
    engine.preparePropagation()
    obj_t = torch.tensor(obj, requires_grad=True)
    probe_t = torch.tensor(probe, requires_grad=True)
    actual = engine.propagate(obj_t * probe_t)
    _, expected = object2detector(obj * probe, engine.params, engine.reconstruction)
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), np.squeeze(expected), rtol=2e-6, atol=2e-6
    )
    assert torch.autograd.gradcheck(
        lambda o, p: engine.propagate(o * p).abs().square(),
        (obj_t, probe_t),
        fast_mode=True,
    )


def test_asp_rejects_different_pixel_spacings(engine):
    engine.params.propagator = "ASP"
    with pytest.raises(ValueError, match="pixel spacings"):
        engine.reconstruct()


def test_component_regularizer_matches_subclass_hook(engine):
    from copy import deepcopy
    from functools import partial

    from PtyLab.Engines.GradientEngine.regularizers import object_smoothness

    class RegularizedEngine(GradientEngine):
        def regularizationLoss(self):
            return object_smoothness(self.model, weight=0.02)

    initial = deepcopy(engine.reconstruction)
    reference = RegularizedEngine(
        initial, engine.experimentalData, initial.params, DummyMonitor()
    )
    engine.regularizers = [partial(object_smoothness, weight=0.02)]
    engine.numIterations = reference.numIterations = 4
    # CUDA scatter-adds in the patch-gather backward are not bitwise
    # reproducible; CPU keeps this equivalence check exact.
    engine.device = reference.device = "cpu"
    engine.reconstruct()
    reference.reconstruct()
    np.testing.assert_allclose(
        engine.reconstruction.error, reference.reconstruction.error
    )
    np.testing.assert_allclose(
        engine.reconstruction.object, reference.reconstruction.object
    )
    np.testing.assert_allclose(
        engine.reconstruction.probe, reference.reconstruction.probe
    )


# Loss additivity over batches is checked per loss in test_gradient_losses.py.
@pytest.mark.parametrize("batch_size", [3, 8])  # a partial final batch; one batch
@pytest.mark.parametrize("propagator", GradientEngine.supportedPropagators)
def test_batches_match_single_frame_step(engine, batch_size, propagator):
    from copy import deepcopy
    from unittest.mock import Mock

    engine.params.propagator = propagator
    if propagator == "ASP":
        engine.reconstruction.dxp = engine.reconstruction.dxd
    initial = deepcopy(engine.reconstruction)
    reference = GradientEngine(
        initial, engine.experimentalData, initial.params, DummyMonitor()
    )
    engine.batchSize = batch_size
    reference.batchSize = 1
    for candidate in (engine, reference):
        candidate.regularizers = [
            lambda model: (
                0.002
                * (
                    model.object.abs().square().mean()
                    + model.probe.field.abs().square().mean()
                )
            )
        ]
        candidate.prepareReconstruction()

        # Nonsequential ordering also verifies that measurements track each patch.
        def order(candidate=candidate):
            candidate.positionIndices = np.array([2, 0, 3, 1])

        candidate.setPositionOrder = order
        candidate.regularizationLoss = Mock(wraps=candidate.regularizationLoss)
        candidate.optimizer.step = Mock(wraps=candidate.optimizer.step)

    expected_intensity = reference.centeredIntensity(
        reference.forward(torch.tensor([1], device=reference.device))[0]
    )
    expected_loss = reference.runIteration(0)
    actual_loss = engine.runIteration(0)
    assert actual_loss == pytest.approx(expected_loss, rel=2e-6, abs=1e-7)
    engine.regularizationLoss.assert_called_once()
    engine.optimizer.step.assert_called_once()
    for actual, expected in (
        (engine.model.object, reference.model.object),
        (engine.model.probe.field, reference.model.probe.field),
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=2e-7)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)
    np.testing.assert_allclose(
        engine.reconstruction.Iestimated, expected_intensity, rtol=3e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        engine.reconstruction.Imeasured,
        engine.experimentalData.ptychogram[1],
        rtol=2e-6,
    )


def test_custom_model_uses_the_same_forward_for_full_and_partial_batches(engine):
    from PtyLab.Engines.GradientEngine.models import PtychographyModel, SharedProbe

    class CountingModel(PtychographyModel):
        def __init__(self, object_field, probe, positions):
            super().__init__(object_field, probe, positions)
            self.batch_lengths = []

        def forward(self, indices):
            self.batch_lengths.append(len(indices))
            return super().forward(indices)

    r = engine.reconstruction
    engine.model = CountingModel(r.object, SharedProbe(r.probe), r.positions)
    engine.batchSize = 3
    engine.numIterations = 1
    engine.reconstruct()
    assert engine.model.batch_lengths == [3, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_batches_match_cpu(engine):
    from copy import deepcopy

    initial = deepcopy(engine.reconstruction)
    reference = GradientEngine(
        initial, engine.experimentalData, initial.params, DummyMonitor()
    )
    reference.device = "cpu"
    engine.device = "cuda"
    engine.batchSize = 3
    for candidate in (engine, reference):
        candidate.prepareReconstruction()
    assert engine.runIteration(0) == pytest.approx(reference.runIteration(0), rel=2e-5)
    for actual, expected in (
        (engine.model.object, reference.model.object),
        (engine.model.probe.field, reference.model.probe.field),
    ):
        torch.testing.assert_close(
            actual.grad.cpu(), expected.grad, rtol=3e-5, atol=2e-7
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-5, atol=2e-6)


def test_replacement_data_uses_its_scan_positions(engine):
    from copy import deepcopy

    initial = deepcopy(engine.reconstruction)
    reference = GradientEngine(
        initial, engine.experimentalData, initial.params, DummyMonitor()
    )
    replacement = deepcopy(engine.experimentalData)
    order = [2, 0, 3, 1]
    expected_positions = initial.positions[order].copy()
    replacement.encoder = replacement.encoder[order].copy()
    replacement.ptychogram = replacement.ptychogram[order].copy()
    replacement._setData()
    engine.batchSize = reference.batchSize = 3
    engine.numIterations = reference.numIterations = 1
    reference.reconstruct()
    engine.reconstruct(replacement)

    np.testing.assert_array_equal(engine.positions, expected_positions)
    np.testing.assert_allclose(engine.reconstruction.error, initial.error, rtol=2e-6)
    np.testing.assert_allclose(engine.reconstruction.object, initial.object, atol=2e-6)
    np.testing.assert_allclose(engine.reconstruction.probe, initial.probe, atol=2e-6)
    assert not np.shares_memory(
        engine.reconstruction.encoder_corrected, replacement.encoder
    )


@pytest.mark.parametrize("field", ["wavelength", "dxd", "zo", "operationMode"])
def test_replacement_data_rejects_changed_geometry_without_mutation(engine, field):
    from copy import deepcopy

    original = engine.experimentalData
    replacement = deepcopy(original)
    setattr(
        replacement,
        field,
        "FPM" if field == "operationMode" else getattr(original, field) * 2,
    )
    with pytest.raises(ValueError, match="new Reconstruction"):
        engine.reconstruct(replacement)
    assert engine.experimentalData is original
    assert engine.reconstruction.data is original


def test_reconstruct_without_replacement_preserves_corrected_positions(engine):
    corrected = engine.reconstruction.encoder_corrected[::-1].copy()
    engine.reconstruction.encoder_corrected = corrected.copy()
    engine.numIterations = 1
    engine.reconstruct()
    np.testing.assert_array_equal(engine.reconstruction.encoder_corrected, corrected)


def test_component_model_preserves_six_axis_fields_and_returns_intensity(engine):
    from PtyLab.Engines.GradientEngine.models import PtychographyModel, SharedProbe

    engine.device = "cpu"
    engine.prepareReconstruction()

    assert isinstance(engine.model, PtychographyModel)
    assert isinstance(engine.model.probe, SharedProbe)
    indices = torch.tensor([0, 2], dtype=torch.long)
    fields = engine.model.detector_fields(indices)
    intensity = engine.model(indices)

    assert fields.shape == (2, 1, 1, 1, 1, 8, 8)
    assert intensity.shape == (2, 8, 8)
    # Fraunhofer intensities are in unshifted FFT order, fields stay centered.
    intensity = torch.fft.fftshift(intensity, dim=(-2, -1))
    torch.testing.assert_close(intensity, fields.abs().square().sum(dim=(1, 2, 3, 4)))

    expected = []
    obj = engine.reconstruction.object.reshape(engine.reconstruction.No, -1)
    probe = engine.reconstruction.probe.reshape(8, 8)
    for index in indices.tolist():
        row, col = engine.positions[index]
        expected.append(abs(fft2c(obj[row : row + 8, col : col + 8] * probe)) ** 2)
    np.testing.assert_allclose(
        intensity.detach().numpy(), expected, rtol=1e-5, atol=1e-7
    )


def test_default_device_is_cuda_when_available_else_cpu(engine):
    assert engine.device == ("cuda" if torch.cuda.is_available() else "cpu")


def test_default_prediction_objective_and_update_match_field_reference(engine):
    # Adam acts on the scale-free probe P / s; prediction and objective are
    # those of the physical probe P.
    engine.device = "cpu"
    engine.batchSize = 3
    engine.prepareReconstruction()
    frame_power = engine.experimentalData.ptychogram.sum() / 4
    scale = 2.0 ** np.round(np.log2(np.sqrt(frame_power / 8**2)))
    assert engine.model.probe.scale.item() == scale

    object_reference = torch.nn.Parameter(
        torch.tensor(engine.reconstruction.object, dtype=torch.complex64)
    )
    probe_reference = torch.nn.Parameter(
        torch.tensor(engine.reconstruction.probe, dtype=torch.complex64) / scale
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [object_reference], "lr": engine.learningRateObject},
            {"params": [probe_reference], "lr": engine.learningRateProbe},
        ],
        foreach=False,
    )
    optimizer.zero_grad(set_to_none=True)
    reference_loss = torch.zeros(())
    measured = torch.tensor(engine.experimentalData.ptychogram).sqrt()
    total_power = measured.square().sum()
    for index, (row, col) in enumerate(engine.positions):
        patch = object_reference[..., row : row + 8, col : col + 8]
        detector = GradientEngine.fft2c(patch * scale * probe_reference)
        detector = detector.reshape(8, 8)
        loss = (detector.abs() - measured[index]).square().sum() / total_power
        loss.backward()
        reference_loss += loss.detach()
    optimizer.step()

    actual_loss = engine.runIteration(0)
    assert actual_loss == pytest.approx(reference_loss.item(), rel=2e-6, abs=1e-7)
    torch.testing.assert_close(
        engine.model.object.grad, object_reference.grad, rtol=3e-5, atol=2e-7
    )
    torch.testing.assert_close(
        engine.model.probe.field.grad, probe_reference.grad, rtol=3e-5, atol=2e-7
    )
    torch.testing.assert_close(
        engine.model.object, object_reference, rtol=3e-5, atol=2e-6
    )
    torch.testing.assert_close(
        engine.model.probe.field, probe_reference, rtol=3e-5, atol=2e-6
    )
    np.testing.assert_array_equal(
        engine.snapshot()["probe.field"],
        scale * engine.model.probe.field.detach().numpy(),
    )


@pytest.mark.parametrize("estimated_name", ["object", "probe.field"])
def test_omitted_parameter_stays_fixed_during_reconstruction(engine, estimated_name):
    engine.device = "cpu"
    engine.parameters = {estimated_name: {"lr": 0.02}}
    engine.numIterations = 2
    before = {
        "object": engine.reconstruction.object.copy(),
        "probe.field": engine.reconstruction.probe.copy(),
    }
    engine.reconstruct()
    after = engine.snapshot()

    fixed_name = "probe.field" if estimated_name == "object" else "object"
    np.testing.assert_array_equal(after[fixed_name], before[fixed_name])
    assert not np.allclose(after[estimated_name], before[estimated_name])


def test_snapshot_restart_and_reset_have_explicit_ownership(engine):
    engine.device = "cpu"
    engine.prepareReconstruction()
    first_optimizer = engine.optimizer
    retained_object = engine.model.object.detach().clone()

    detached = engine.snapshot()
    detached["object"][...] = 0
    assert torch.count_nonzero(engine.model.object).item() > 0

    engine.reconstruction.object[...] = 2 + 3j
    engine.prepareReconstruction()
    assert engine.optimizer is not first_optimizer
    torch.testing.assert_close(engine.model.object, retained_object)
    np.testing.assert_array_equal(
        engine.reconstruction.object, retained_object.detach().numpy()
    )

    engine.reconstruction.object[...] = 2 + 3j
    object_identity = id(engine.model.object)
    engine.reset()
    assert id(engine.model.object) == object_identity
    torch.testing.assert_close(
        engine.model.object, torch.full_like(engine.model.object, 2 + 3j)
    )


def test_exception_path_synchronizes_component_state(engine):
    class InterruptedEngine(GradientEngine):
        def afterStep(self, iteration):
            self.model.object.add_(1)
            raise RuntimeError("intentional interruption")

    interrupted = InterruptedEngine(
        engine.reconstruction, engine.experimentalData, engine.params, DummyMonitor()
    )
    interrupted.device = "cpu"
    interrupted.numIterations = 1
    with pytest.raises(RuntimeError, match="intentional interruption"):
        interrupted.reconstruct()
    np.testing.assert_array_equal(
        interrupted.reconstruction.object,
        interrupted.model.object.detach().numpy(),
    )
    assert not np.shares_memory(
        interrupted.reconstruction.object,
        interrupted.model.object.detach().numpy(),
    )


def test_duplicate_parameter_ownership_is_rejected(engine):
    engine.device = "cpu"
    engine.reset()
    engine.model.object_alias = engine.model.object
    engine.parameters = {
        "object": {"lr": 0.03},
        "object_alias": {"lr": 0.03},
    }
    with pytest.raises(ValueError, match="same tensor"):
        engine.prepareReconstruction()


def test_explicit_cpu_device_moves_components_and_batches(engine):
    engine.device = "cpu"
    engine.prepareReconstruction()
    assert {parameter.device.type for parameter in engine.model.parameters()} == {"cpu"}
    assert engine.measuredIntensities.device.type == "cpu"
    assert all(
        kernel.device.type == "cpu" for kernel in engine.propagation.propagationKernels
    )


def test_update_per_batch_with_one_batch_matches_full_scan_step(engine):
    from copy import deepcopy

    initial = deepcopy(engine.reconstruction)
    reference = GradientEngine(
        initial, engine.experimentalData, initial.params, DummyMonitor()
    )
    engine.updatePerBatch = True
    for candidate in (engine, reference):
        candidate.device = "cpu"
        candidate.batchSize = engine.experimentalData.numFrames
        candidate.numIterations = 3
        candidate.reconstruct()
    np.testing.assert_array_equal(
        engine.reconstruction.error, reference.reconstruction.error
    )
    np.testing.assert_array_equal(
        engine.reconstruction.object, reference.reconstruction.object
    )


@pytest.mark.parametrize("batch_size, steps", [(1, 4), (3, 2), (4, 1)])
def test_update_per_batch_steps_once_per_batch(engine, batch_size, steps):
    from functools import partial

    from PtyLab.Engines.GradientEngine.regularizers import object_smoothness

    engine.device = "cpu"
    engine.batchSize = batch_size
    engine.updatePerBatch = True
    engine.regularizers = [partial(object_smoothness, weight=0.02)]
    after_step = []
    engine.afterStep = after_step.append
    engine.prepareReconstruction()
    original_step = engine.optimizer.step
    counted = []

    def step(*args, **kwargs):
        counted.append(1)
        return original_step(*args, **kwargs)

    engine.optimizer.step = step
    loss = engine.runIteration(0)
    assert len(counted) == len(after_step) == steps
    assert np.isfinite(loss)


def test_gradient_preconditioners_match_numpy_illumination(engine):
    engine.device = "cpu"
    engine.prepareReconstruction()
    r = engine.reconstruction
    indices = torch.tensor([0, 2, 3])
    curvature = engine.model.gradient_preconditioners(indices)

    probe = np.squeeze(r.probe)
    obj = np.squeeze(r.object)
    illumination = np.zeros((r.No, r.No))
    coverage = np.zeros((r.Np, r.Np))
    for row, col in r.positions[[0, 2, 3]]:
        illumination[row : row + r.Np, col : col + r.Np] += abs(probe) ** 2
        coverage += abs(obj[row : row + r.Np, col : col + r.Np]) ** 2
    # The probe map is the curvature of the stored, scale-free parameter.
    coverage *= engine.model.probe.scale.item() ** 2
    assert curvature["object"].shape == engine.model.object.shape
    assert curvature["probe.field"].shape == engine.model.probe.field.shape
    np.testing.assert_allclose(
        curvature["object"].squeeze().numpy(), illumination, rtol=1e-5
    )
    np.testing.assert_allclose(
        curvature["probe.field"].squeeze().numpy(), coverage, rtol=1e-5
    )


def test_preconditioned_step_divides_gradient_by_regularized_curvature(engine):
    engine.device = "cpu"
    engine.optimizerType = "preconditioned"
    engine.preconditionerAlpha = 0.25
    engine.parameters = {"object": {"lr": 0.3}, "probe.field": {"lr": 0.2}}
    engine.prepareReconstruction()
    assert isinstance(engine.optimizer, torch.optim.SGD)
    before = {
        name: parameter.detach().clone()
        for name, parameter in engine.model.named_parameters()
    }

    indices = torch.arange(engine.experimentalData.numFrames)
    engine.optimizer.zero_grad(set_to_none=True)
    loss = engine.computeLoss(
        engine.forward(indices), engine.measuredIntensities[indices]
    )
    loss.backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in engine.model.named_parameters()
    }
    curvature = engine.model.gradient_preconditioners(indices)

    engine.setPositionOrder = lambda: None
    engine.positionIndices = np.arange(engine.experimentalData.numFrames)
    engine.runIteration(0)
    for name, rate in (("object", 0.3), ("probe.field", 0.2)):
        values = curvature[name]
        denominator = (0.75 * values + 0.25 * values.max()) / engine.totalPower
        expected = before[name] - rate * gradients[name] / denominator
        torch.testing.assert_close(
            dict(engine.model.named_parameters())[name].detach(),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )


def test_probe_scale_follows_the_data_and_keeps_the_physical_probe(engine):
    engine.device = "cpu"
    probe = engine.reconstruction.probe.copy()
    engine.prepareReconstruction()
    scale = engine.model.probe.scale.item()
    assert scale == 2.0 ** round(np.log2(scale))  # a power of two
    np.testing.assert_array_equal(engine.snapshot()["probe.field"], probe)

    engine.experimentalData.ptychogram = 16 * engine.experimentalData.ptychogram
    engine.prepareReconstruction()
    assert engine.model.probe.scale.item() == 4 * scale
    np.testing.assert_array_equal(engine.snapshot()["probe.field"], probe)

    engine.reconstruction.probe = 2 * probe
    engine.reset()
    assert engine.model.probe.scale.item() == 4 * scale
    np.testing.assert_array_equal(engine.snapshot()["probe.field"], 2 * probe)


def _data_from_current_model(engine, background):
    """Replace the data by the engine's current prediction plus ``background``."""
    engine.device = "cpu"
    engine.prepareReconstruction()
    with torch.no_grad():
        clean = engine.centeredIntensity(
            engine.forward(torch.arange(len(engine.positions)))
        )
    engine.experimentalData.ptychogram = (clean + background).astype(np.float32)


def test_background_is_added_in_centered_order_and_can_be_removed(engine):
    engine.device = "cpu"
    size = engine.reconstruction.Np
    image = np.arange(size * size, dtype=np.float32).reshape(size, size) / size**2
    engine.prepareReconstruction()
    indices = torch.arange(len(engine.positions))
    without = engine.centeredIntensity(engine.forward(indices))

    engine.background = image
    engine.prepareReconstruction()
    with_background = engine.centeredIntensity(engine.forward(indices))
    np.testing.assert_allclose(with_background, without + image, rtol=1e-6, atol=1e-6)
    assert not engine.model.background.requires_grad

    engine.background = None
    engine.prepareReconstruction()
    assert engine.model.background is None


@pytest.mark.parametrize("truth, start", [(0.3, 0.0), (0.0, 0.05)])
def test_selected_background_is_estimated_and_stays_nonnegative(engine, truth, start):
    _data_from_current_model(engine, truth)
    engine.background = start
    engine.parameters = {"background": {"lr": 0.05}}
    engine.numIterations = 200
    engine.reconstruct()
    learned = engine.snapshot()["background"]
    assert learned >= 0
    assert learned == pytest.approx(truth, abs=0.02)
