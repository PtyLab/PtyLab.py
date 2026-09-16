"""Numerical and API checks for the optional Torch engine."""

import os
import subprocess
import sys

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


@pytest.mark.parametrize("size", [7, 8])
def test_forward_matches_numpy_and_has_complex_gradients(size):
    rng = np.random.default_rng(3)
    obj = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    probe = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    obj_t = torch.tensor(obj, requires_grad=True)
    probe_t = torch.tensor(probe, requires_grad=True)
    actual = GradientEngine.fft2c(obj_t * probe_t)
    np.testing.assert_allclose(actual.detach().numpy(), fft2c(obj * probe), atol=1e-12)
    assert torch.autograd.gradcheck(
        lambda o, p: GradientEngine.fft2c(o * p).abs().square().sum(),
        (obj_t, probe_t),
    )


@pytest.mark.parametrize("batch_size", [1, 3])
def test_reconstruction_reduces_loss_and_preserves_numpy_api(
    engine, tmp_path, batch_size
):
    engine.batchSize = batch_size
    engine.numIterations = 35
    before_object = engine.reconstruction.object.copy()
    before_probe = engine.reconstruction.probe.copy()
    engine.reconstruct()
    r = engine.reconstruction
    assert len(r.error) == 35
    assert r.error[-1] < 0.3 * r.error[0]
    for initial, result, tensor in (
        (before_object, r.object, engine.objectTensor),
        (before_probe, r.probe, engine.probeTensor),
    ):
        assert isinstance(result, np.ndarray)
        assert result.shape == initial.shape
        assert not np.allclose(result, initial)
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
        np.testing.assert_array_equal(result, tensor.detach().numpy())
        assert not np.shares_memory(result, tensor.detach().numpy())
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


def test_invalid_intensities_fail(engine):
    engine.experimentalData.ptychogram[0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        engine.reconstruct()


def test_base_package_imports_without_torch():
    code = """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("Torch intentionally unavailable", name="torch")
sys.meta_path.insert(0, NoTorch())
import PtyLab
assert "torch" not in sys.modules
try:
    from PtyLab.Engines.GradientEngine import GradientEngine
except ImportError as exc:
    assert "pip install torch" in str(exc), str(exc)
else:
    raise AssertionError("Expected installation guidance")
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )


def test_subclass_can_extend_model_optimizer_and_constraints(engine):
    class GainEngine(GradientEngine):
        def initializeParameters(self):
            super().initializeParameters()
            self.gain = torch.nn.Parameter(torch.tensor(0.8, device=self.device))
            self.steps = []

        def parameterGroups(self):
            return super().parameterGroups() + [{"params": [self.gain], "lr": 0.02}]

        def createOptimizer(self):
            return torch.optim.SGD(self.parameterGroups())

        def forward(self, positionIndex):
            return self.gain * super().forward(positionIndex)

        def computeLoss(self, detectorWave, measuredAmplitude):
            return (
                detectorWave.abs().square() - measuredAmplitude.square()
            ).square().sum() / self.totalPower

        def regularizationLoss(self):
            return 0.01 * (self.gain - 1).square()

        def afterStep(self, iteration):
            assert not torch.is_grad_enabled()
            self.gain.clamp_(min=0.1)
            self.steps.append(iteration)

    extended = GainEngine(
        engine.reconstruction, engine.experimentalData, engine.params, engine.monitor
    )
    extended.numIterations = 3
    extended.reconstruct()
    assert isinstance(extended.optimizer, torch.optim.SGD)
    assert extended.steps == [0, 1, 2]
    assert extended.gain.item() != pytest.approx(0.8)
    assert torch.isfinite(extended.gain.grad)
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
    engine.preparePropagation()
    obj_t = torch.tensor(obj, requires_grad=True)
    probe_t = torch.tensor(probe, requires_grad=True)
    actual = engine.propagate(obj_t, probe_t)
    _, expected = object2detector(obj * probe, engine.params, engine.reconstruction)
    np.testing.assert_allclose(
        actual.detach().numpy(), np.squeeze(expected), rtol=2e-6, atol=2e-6
    )
    assert torch.autograd.gradcheck(
        lambda o, p: engine.propagate(o, p).abs().square(),
        (obj_t, probe_t),
        fast_mode=True,
    )


@pytest.mark.parametrize("propagator", ["Fresnel", "ASP", "scaledASP"])
def test_reconstruction_with_other_propagators(engine, propagator):
    from PtyLab.Operators.Operators import object2detector

    r = engine.reconstruction
    engine.params.propagator = propagator
    if propagator == "ASP":
        r.dxp = r.dxd
    # Generate matching data with the existing NumPy operator, independently of Torch.
    for index, (row, col) in enumerate(r.positions):
        r.esw = 0.7 * r.object[..., row : row + r.Np, col : col + r.Np] * r.probe
        _, detector = object2detector(r.esw, engine.params, r)
        engine.experimentalData.ptychogram[index] = np.squeeze(abs(detector) ** 2)
    engine.experimentalData._setData()
    engine.numIterations = 15
    engine.reconstruct()
    assert np.isfinite(r.error).all()
    assert r.error[-1] < r.error[0]
    assert torch.isfinite(engine.objectTensor.grad).all()
    assert torch.isfinite(engine.probeTensor.grad).all()


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
            return object_smoothness(self.objectTensor, self.probeTensor, weight=0.02)

    initial = deepcopy(engine.reconstruction)
    reference = RegularizedEngine(
        initial, engine.experimentalData, initial.params, DummyMonitor()
    )
    engine.regularizers = [partial(object_smoothness, weight=0.02)]
    engine.numIterations = reference.numIterations = 4
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


def test_assigned_model_and_loss_receive_gradients(engine):
    from PtyLab.Engines.GradientEngine.losses import intensity_loss
    from PtyLab.Engines.GradientEngine.models import SingleSliceModel

    class ScaledFieldModel(SingleSliceModel):
        calls = 0

        def __call__(self, obj, probe, position, propagate):
            self.calls += 1
            return 0.9 * super().__call__(obj, probe, position, propagate)

    engine.model = ScaledFieldModel()
    engine.lossFunction = intensity_loss
    engine.numIterations = 3
    engine.reconstruct()
    assert engine.model.calls == 3 * engine.experimentalData.numFrames
    assert engine.objectTensor.grad.abs().sum() > 0
    assert engine.probeTensor.grad.abs().sum() > 0
    assert np.isfinite(engine.reconstruction.error).all()


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
@pytest.mark.parametrize("propagator", GradientEngine.supportedPropagators)
@pytest.mark.parametrize(
    "loss_name", ["amplitude_loss", "intensity_loss", "poisson_loss"]
)
def test_batches_match_single_frame_step(engine, batch_size, propagator, loss_name):
    from copy import deepcopy
    from unittest.mock import Mock

    from PtyLab.Engines.GradientEngine import losses

    engine.params.propagator = propagator
    if propagator == "ASP":
        engine.reconstruction.dxp = engine.reconstruction.dxd
    initial = deepcopy(engine.reconstruction)
    reference = GradientEngine(
        initial, engine.experimentalData, initial.params, DummyMonitor()
    )
    engine.batchSize = batch_size
    for candidate in (engine, reference):
        candidate.lossFunction = getattr(losses, loss_name)
        candidate.regularizers = [
            lambda obj, probe: (
                0.002 * (obj.abs().square().mean() + probe.abs().square().mean())
            )
        ]
        candidate.prepareReconstruction()

        # Nonsequential ordering also verifies that measurements track each patch.
        def order(candidate=candidate):
            candidate.positionIndices = np.array([2, 0, 3, 1])

        candidate.setPositionOrder = order
        candidate.regularizationLoss = Mock(wraps=candidate.regularizationLoss)
        candidate.optimizer.step = Mock(wraps=candidate.optimizer.step)

    expected_wave = reference.forward(1).detach().abs().square().numpy()
    expected_loss = reference.runIteration(0)
    actual_loss = engine.runIteration(0)
    assert actual_loss == pytest.approx(expected_loss, rel=2e-6, abs=1e-7)
    engine.regularizationLoss.assert_called_once()
    engine.optimizer.step.assert_called_once()
    for actual, expected in (
        (engine.objectTensor, reference.objectTensor),
        (engine.probeTensor, reference.probeTensor),
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=2e-7)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)
    np.testing.assert_allclose(
        engine.reconstruction.Iestimated, expected_wave, rtol=3e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        engine.reconstruction.Imeasured,
        engine.experimentalData.ptychogram[1],
        rtol=2e-6,
    )


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, "2", None, True, np.bool_(True)])
def test_invalid_batch_size(engine, batch_size):
    engine.batchSize = batch_size
    with pytest.raises(ValueError, match="batchSize must be a positive integer"):
        engine.prepareReconstruction()


def test_numpy_integer_batch_size(engine):
    engine.batchSize = np.int64(3)
    engine.numIterations = 1
    engine.reconstruct()


@pytest.mark.parametrize("customization", ["model", "forward"])
def test_custom_forward_requires_explicit_batch_hook(engine, customization):
    from PtyLab.Engines.GradientEngine.models import SingleSliceModel

    if customization == "model":

        class CustomModel(SingleSliceModel):
            def __call__(self, *args):
                return 0.9 * super().__call__(*args)

        engine.model = CustomModel()
    else:
        original_forward = engine.forward
        engine.forward = lambda index: 0.9 * original_forward(index)
    engine.batchSize = 2
    with pytest.raises(NotImplementedError, match="Override forwardBatch"):
        engine.prepareReconstruction()
    engine.batchSize = 1
    engine.numIterations = 1
    engine.reconstruct()


def test_custom_batch_hook_is_used(engine):
    class GainEngine(GradientEngine):
        def initializeParameters(self):
            super().initializeParameters()
            self.gain = torch.nn.Parameter(torch.tensor(0.8, device=self.device))
            self.batch_lengths = []

        def parameterGroups(self):
            return super().parameterGroups() + [{"params": [self.gain], "lr": 0.02}]

        def forward(self, index):
            return self.gain * super().forward(index)

        def forwardBatch(self, indices):
            self.batch_lengths.append(len(indices))
            return self.gain * super().forwardBatch(indices)

    custom = GainEngine(
        engine.reconstruction, engine.experimentalData, engine.params, DummyMonitor()
    )
    custom.batchSize = 3
    custom.numIterations = 1
    custom.reconstruct()
    assert custom.batch_lengths == [3, 1]
    assert torch.isfinite(custom.gain.grad)
    assert custom.gain.item() != pytest.approx(0.8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_batches_match_cpu(engine):
    from copy import deepcopy

    initial = deepcopy(engine.reconstruction)
    reference = GradientEngine(
        initial, engine.experimentalData, initial.params, DummyMonitor()
    )
    engine.device = "cuda"
    engine.batchSize = 3
    for candidate in (engine, reference):
        candidate.prepareReconstruction()
    assert engine.runIteration(0) == pytest.approx(reference.runIteration(0), rel=2e-5)
    for actual, expected in (
        (engine.objectTensor, reference.objectTensor),
        (engine.probeTensor, reference.probeTensor),
    ):
        torch.testing.assert_close(
            actual.grad.cpu(), expected.grad, rtol=3e-5, atol=2e-7
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-5, atol=2e-6)


def test_poisson_loss_matches_likelihood_and_has_finite_gradients():
    from PtyLab.Engines.GradientEngine.losses import poisson_loss

    wave = torch.tensor([1 + 2j, 2 - 1j], dtype=torch.complex128, requires_grad=True)
    amplitude = torch.tensor([2.0, 3.0], dtype=torch.float64)
    power = amplitude.square().sum()
    counts = amplitude.square()
    expected = (
        -torch.distributions.Poisson(wave.abs().square()).log_prob(counts)
        - torch.lgamma(counts + 1)
    ).sum() / power
    torch.testing.assert_close(poisson_loss(wave, amplitude, power), expected)
    assert torch.autograd.gradcheck(
        lambda w: poisson_loss(w, amplitude, power), (wave,)
    )

    zero_wave = torch.zeros(2, dtype=torch.complex128, requires_grad=True)
    loss = poisson_loss(zero_wave, torch.tensor([0.0, 1.0]), power)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(zero_wave.grad).all()


@pytest.mark.parametrize("batch_size", [1, 3])
def test_replacement_data_uses_its_scan_positions(engine, batch_size):
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
    engine.batchSize = reference.batchSize = batch_size
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
