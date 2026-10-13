"""Memory-bounded batching of the optional Torch engine."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_gradient_engine import engine  # noqa: E402,F401  (shared fixture)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _one_iteration(engine):
    """Objective and the gradients its step is computed from.

    Post-step parameters are not compared: Adam's first step
    lr * g / (|g| + eps) flips sign with rounding noise in near-zero gradients,
    which CUDA's atomic accumulation produces nondeterministically.
    """
    engine.prepareReconstruction()
    loss = engine.runIteration(0)
    return loss, {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in engine.model.named_parameters()
    }


@pytest.mark.parametrize("device", DEVICES)
def test_host_resident_data_match_device_resident_data(engine, device):
    engine.device = device
    engine.batchSize = 3
    host = copy.deepcopy(engine)
    host.dataDevice = "cpu"
    expected_loss, expected = _one_iteration(engine)
    loss, actual = _one_iteration(host)
    assert host.measuredIntensities.device.type == "cpu"
    if device == "cuda":
        assert host.measuredIntensities.is_pinned()
    assert loss == pytest.approx(expected_loss, rel=1e-6)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_host_resident_step_does_not_depend_on_batch_size(engine, device):
    engine.device = device
    engine.dataDevice = "cpu"
    batched = copy.deepcopy(engine)
    batched.batchSize = 3  # batches of 3 and 1
    expected_loss, expected = _one_iteration(engine)
    loss, actual = _one_iteration(batched)
    assert loss == pytest.approx(expected_loss, rel=2e-6)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=3e-5, atol=2e-7)


def _larger_engine(frames=16, size=128):
    """Random CPM data large enough for per-frame memory to be measurable."""
    from PtyLab import DummyMonitor, ExperimentalData, Params, Reconstruction
    from PtyLab.Engines.GradientEngine import GradientEngine

    data = ExperimentalData(operationMode="CPM")
    data.wavelength, data.zo, data.dxd = 632.8e-9, 0.05, 75e-6
    dx = data.wavelength * data.zo / (size * data.dxd)
    rng = np.random.default_rng(0)
    data.encoder = rng.uniform(-20, 20, (frames, 2)) * dx
    data.ptychogram = rng.random((frames, size, size)).astype(np.float32)
    data.entrancePupilDiameter = size / 3 * dx
    data.spectralDensity = data.theta = None
    data._setData()
    params = Params()
    params.gpuSwitch = False
    reconstruction = Reconstruction(data, params)
    reconstruction.npsm = reconstruction.nosm = 1
    reconstruction.nlambda = reconstruction.nslice = 1
    reconstruction.initialObject, reconstruction.initialProbe = "ones", "circ"
    reconstruction.initializeObjectProbe()
    return GradientEngine(reconstruction, data, params, DummyMonitor())


def _free_cuda_memory():
    """The budget counts all allocated CUDA memory, so drop other tests' tensors."""
    import gc

    gc.collect()
    torch.cuda.empty_cache()


@needs_cuda
def test_memory_budget_bounds_the_peak_and_picks_the_batch():
    from PtyLab.Engines.GradientEngine.batching import iteration_peak

    _free_cuda_memory()
    engine = _larger_engine()
    engine.device = "cuda"
    engine.prepareReconstruction()
    assert engine.suggestBatchSize(1e12)["batch size"] == 16
    # A budget halfway between the peaks of batches of 6 and 7 frames.
    peaks = {size: iteration_peak(engine, size) for size in (6, 7)}
    assert peaks[7] > peaks[6]
    budget = 0.5 * (peaks[6] + peaks[7])
    engine.memoryBudget = budget
    engine.prepareReconstruction()
    assert engine.effectiveBatchSize == 6
    # The budget bounds the iterations; the measurement in preparation briefly
    # holds a copy of the parameters on top.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    errors = [engine.runIteration(iteration) for iteration in range(3)]
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() <= budget
    assert errors[-1] < errors[0]

    del engine
    _free_cuda_memory()
    fresh = _larger_engine()
    fresh.device = "cuda"
    fresh.memoryBudget = budget
    fresh.numIterations = 2
    fresh.reconstruct()  # the budget is measured at the first preparation
    assert fresh.effectiveBatchSize == 6
    with pytest.raises(MemoryError):
        fresh.suggestBatchSize(1)


@pytest.mark.parametrize("device", DEVICES)
def test_compiled_loss_matches_eager(engine, device):
    # Objective and gradients are compared for identical inputs. Multi-step
    # Adam trajectories are not: Adam's first step lr * g / (|g| + eps) flips
    # sign with rounding noise in near-zero gradients.
    from functools import partial

    from PtyLab.Engines.GradientEngine.losses import poisson_loss

    engine.device = device
    compiled = copy.deepcopy(engine)
    compiled.compile = True
    indices = torch.arange(4, device=device)
    for loss in (engine.lossFunction, partial(poisson_loss, read_out_sigma=0.5)):
        results = []
        for run in (engine, compiled):
            run.lossFunction = loss
            run.prepareReconstruction()
            value = run.computeLoss(run.forward(indices), run.measuredBatch(indices))
            value.backward()
            results.append(
                (value.detach(), [p.grad.clone() for p in run.model.parameters()])
            )
        (eager_value, eager_grads), (value, grads) = results
        torch.testing.assert_close(value, eager_value, rtol=1e-6, atol=0)
        for grad, eager_grad in zip(grads, eager_grads):
            torch.testing.assert_close(grad, eager_grad, rtol=1e-5, atol=1e-7)
    assert len(compiled._compiledLosses) == 2

    compiled.batchSize = 3  # a partial final batch: a second shape
    compiled.numIterations = 5
    compiled.reconstruct()
    assert compiled.reconstruction.error[-1] < compiled.reconstruction.error[0]
