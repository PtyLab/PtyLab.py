"""Orthogonal probe relaxation (per-frame probe stack) in the optional Torch engine."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PtyLab import DummyMonitor, ExperimentalData, Params, Reconstruction  # noqa: E402
from PtyLab.Engines.GradientEngine import (  # noqa: E402
    AbsorbingProbeBoundary,
    GradientEngine,
    OPRProbe,
    OPRProjection,
    PhaseRampRemoval,
    ProbeCentering,
    ProbeModeOrthogonalization,
)
from PtyLab.Operators.Operators import object2detector  # noqa: E402
from PtyLab.utils.utils import fft2c  # noqa: E402

FRAMES = 9


def true_stack(npsm, seed=5):
    """Frame probes of rank 2 per mode: P_jm = (1 + a_j) B_m + b_j C_m.

    a_j is a flux change, b_j the weight of a second pattern C_m (a tilt), so
    every mode's stack has exactly two components. Shape (J, npsm, 8, 8).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-4:4, -4:4]
    gaussian = np.exp(-(xx**2 + yy**2) / 6)
    B = np.stack([gaussian * 0.6**m * np.exp(0.4j * m * yy) for m in range(npsm)])
    C = B * (xx / 3)
    a = 0.3 * rng.standard_normal(FRAMES)
    b = 0.4 * (rng.standard_normal(FRAMES) + 1j * rng.standard_normal(FRAMES))
    return (1 + a)[:, None, None, None] * B + b[:, None, None, None] * C


def opr_engine(npsm=1, nosm=1, propagator="Fraunhofer", seed=3):
    """8x8 frames, 9 positions, data from a frame-varying probe stack."""
    data = ExperimentalData(operationMode="CPM")
    data.wavelength = 632.8e-9
    data.zo = 0.05
    data.dxd = 75e-6
    nd = 8
    dx = data.wavelength * data.zo / (nd * data.dxd)
    grid = np.array([-2, 0, 2])
    data.encoder = np.stack(np.meshgrid(grid, grid), -1).reshape(-1, 2) * dx
    data.ptychogram = np.ones((FRAMES, nd, nd), dtype=np.float32)
    data.entrancePupilDiameter = 4 * dx
    data.spectralDensity = None
    data.theta = None
    data._setData()
    params = Params()
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
    stack = true_stack(npsm)
    # Data: frame j sees its own probes P_jm, incoherent over (a, m).
    for j, (row, col) in enumerate(reconstruction.positions):
        patch = objects[:, row : row + nd, col : col + nd]
        data.ptychogram[j] = sum(
            abs(fft2c(patch[a] * stack[j, m])) ** 2
            for a in range(nosm)
            for m in range(npsm)
        )
    data._setData()
    reconstruction.object = (
        (objects * 0.9 + 0.05 * rng.standard_normal(objects.shape))
        .reshape(reconstruction.object.shape)
        .astype(np.complex64)
    )
    # Shared start: the frame-mean probe, perturbed.
    mean = stack.mean(0) * 0.9 + 0.02 * rng.standard_normal(stack.shape[1:])
    reconstruction.probe = mean.reshape(reconstruction.probe.shape).astype(np.complex64)
    engine = GradientEngine(reconstruction, data, params, DummyMonitor())
    engine.device = "cpu"
    return engine


def set_stack(engine, stack):
    """Copy a physical (J, npsm, Np, Np) stack into the engine's OPRProbe."""
    shape = engine.model.probe.field.shape
    engine.model.probe.reset_from_array(stack.reshape(shape).astype(np.complex64))


def all_intensities(engine):
    with torch.no_grad():
        return engine.forward(torch.arange(FRAMES)).clone()


def prepared(npsm=1, nosm=1, propagator="Fraunhofer", stack=True):
    """Prepared engine with an OPRProbe set to the true stack."""
    engine = opr_engine(npsm, nosm, propagator)
    engine.relaxProbe()
    engine.prepareReconstruction()
    if stack:
        set_stack(engine, true_stack(npsm))
    return engine


def low_rank_numpy(rows, rank):
    """Best rank-K approximation of the rows (J, N) by NumPy's SVD."""
    V, s, Uh = np.linalg.svd(rows, full_matrices=False)
    return (V[:, :rank] * s[:rank]) @ Uh[:rank]


@pytest.mark.parametrize("npsm, nosm", [(1, 1), (3, 1), (2, 2)])
@pytest.mark.parametrize("propagator", ["Fraunhofer", "Fresnel"])
def test_opr_prediction_matches_a_numpy_loop_with_one_probe_per_frame(
    npsm, nosm, propagator
):
    engine = prepared(npsm, nosm, propagator)
    r = engine.reconstruction
    stack = true_stack(npsm)
    r.esw = None
    objects = r.object.reshape(nosm, r.No, r.No)
    expected = np.zeros((FRAMES, r.Np, r.Np))
    for j, (row, col) in enumerate(engine.positions):
        for a in range(nosm):
            for m in range(npsm):
                exit_wave = objects[a, row : row + r.Np, col : col + r.Np] * stack[j, m]
                _, detector = object2detector(exit_wave, engine.params, r)
                expected[j] += abs(np.squeeze(detector)) ** 2
    np.testing.assert_allclose(
        all_intensities(engine).numpy(), expected, rtol=1e-5, atol=1e-7 * expected.max()
    )


def test_an_equal_stack_predicts_like_the_shared_probe_and_sums_its_gradients():
    shared, relaxed = opr_engine(3), opr_engine(3)
    relaxed.relaxProbe()
    for engine in (shared, relaxed):
        engine.prepareReconstruction()
        engine.optimizer.zero_grad(set_to_none=True)
        engine.computeLoss(
            engine.forward(torch.arange(FRAMES)), engine.measuredIntensities
        ).backward()
    torch.testing.assert_close(all_intensities(relaxed), all_intensities(shared))
    torch.testing.assert_close(relaxed.model.object.grad, shared.model.object.grad)
    # The shared probe's gradient is the sum of the frame probes' gradients.
    torch.testing.assert_close(
        relaxed.model.probe.field.grad.sum(0),
        shared.model.probe.field.grad,
        rtol=2e-5,
        atol=1e-7,
    )


def test_opr_gradients_reach_only_the_frames_of_the_batch_and_pass_gradcheck():
    from torch.func import functional_call

    engine = prepared(2)
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
    per_frame = field.grad.abs().sum(dim=(1, 2, 3, 4, 5, 6))
    assert (per_frame[indices] > 0).all()
    others = torch.ones(FRAMES, dtype=torch.bool)
    others[indices] = False
    assert (per_frame[others] == 0).all()


@pytest.mark.parametrize("batch_size", [2, 4])  # partial final batch of 1
def test_opr_batches_match_the_full_scan(batch_size):
    full, batched = prepared(2), prepared(2)
    batched.effectiveBatchSize = batch_size
    for engine in (full, batched):
        engine.setPositionOrder = lambda: None
        engine.positionIndices = np.arange(FRAMES)
    assert batched.runIteration(0) == pytest.approx(full.runIteration(0), rel=2e-6)
    for name in ("object", "probe.field"):
        actual = dict(batched.model.named_parameters())[name]
        expected = dict(full.model.named_parameters())[name]
        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=2e-7)


@pytest.mark.parametrize("relax", [0.0, 0.3])
def test_projection_matches_numpy_svd_truncation_with_blending(relax):
    engine = prepared(2, stack=False)
    rng = np.random.default_rng(1)
    stack = rng.standard_normal((FRAMES, 2, 8, 8)) + 1j * rng.standard_normal(
        (FRAMES, 2, 8, 8)
    )
    set_stack(engine, stack)
    physical = engine.model.probe.physical_field().detach().numpy()[:, 0, 0, :, 0]
    OPRProjection(rank=2, relax=relax)(engine, 0)
    actual = engine.model.probe.physical_field().detach().numpy()[:, 0, 0, :, 0]
    for m in range(2):
        rows = physical[:, m].reshape(FRAMES, -1).astype(np.complex128)
        expected = relax * rows + (1 - relax) * low_rank_numpy(rows, 2)
        np.testing.assert_allclose(
            actual[:, m].reshape(FRAMES, -1), expected, rtol=1e-4, atol=1e-5
        )


def test_projection_matches_the_classic_opr_gram_tsvd():
    from PtyLab.Engines.OPR import OPR

    engine = prepared(1, stack=False)
    rng = np.random.default_rng(2)
    set_stack(engine, rng.standard_normal((FRAMES, 1, 8, 8)) + 0j)
    rows = engine.model.probe.physical_field().detach().numpy()[:, 0, 0, 0, 0]
    rows = rows.reshape(FRAMES, -1).astype(np.complex128)
    # Classic OPR: columns are frames, A (N, J); update = U (s Vh).
    U, s, Vh = OPR.gram_tsvd(rows.T, 3)
    expected = (U @ (s[:, None] * Vh)).T
    OPRProjection(rank=3)(engine, 0)
    actual = engine.model.probe.physical_field().detach().numpy()[:, 0, 0, 0, 0]
    np.testing.assert_allclose(
        actual.reshape(FRAMES, -1), expected, rtol=1e-4, atol=1e-5
    )


def test_projection_keeps_a_stack_already_of_that_rank_and_needs_an_opr_probe():
    engine = prepared(2)  # the true stack has rank 2 per mode
    before = engine.model.probe.field.detach().clone()
    OPRProjection(rank=2)(engine, 0)
    torch.testing.assert_close(engine.model.probe.field, before, rtol=1e-4, atol=1e-5)
    OPRProjection(rank=FRAMES)(engine, 0)  # rank >= frames: nothing to do

    shared = opr_engine(1)
    shared.prepareReconstruction()
    with pytest.raises(TypeError, match="relaxProbe"):
        OPRProjection()(shared, 0)
    with pytest.raises(ValueError, match="rank"):
        OPRProjection(rank=0)


def test_mode_orthogonalization_acts_on_every_frame_and_keeps_its_intensity():
    engine = prepared(3)
    before = all_intensities(engine)
    ProbeModeOrthogonalization()(engine, 0)
    torch.testing.assert_close(
        all_intensities(engine), before, rtol=1e-5, atol=1e-6 * before.max()
    )
    field = engine.model.probe.physical_field().detach()
    for j in range(FRAMES):
        modes = field[j, 0, 0, :, 0].flatten(-2)  # (npsm, N)
        gram = modes.conj() @ modes.T
        off_diagonal = gram - torch.diag(torch.diagonal(gram))
        assert off_diagonal.abs().max() < 1e-5 * gram.abs().max()
        powers = torch.diagonal(gram).real
        assert (powers[:-1] >= powers[1:]).all()


def test_opr_preconditioner_gives_each_frame_its_own_patch():
    engine = prepared(2, nosm=2)
    r = engine.reconstruction
    indices = torch.tensor([0, 3, 5])
    curvature = engine.model.gradient_preconditioners(indices)
    probe_map = curvature["probe.field"]
    assert probe_map.shape == engine.model.probe.field.shape
    objects = r.object.reshape(2, r.No, r.No)
    scale = engine.model.probe.scale.item()
    for j in range(FRAMES):
        if j in (0, 3, 5):
            row, col = r.positions[j]
            expected = (abs(objects[:, row : row + 8, col : col + 8]) ** 2).sum(0)
            expected = expected * scale**2
        else:
            expected = np.zeros((8, 8))
        for m in range(2):
            np.testing.assert_allclose(
                probe_map[j, 0, 0, m, 0].numpy(), expected, rtol=1e-5
            )


def test_centering_phase_ramp_and_boundary_work_on_a_stack():
    engine = prepared(2)
    before = all_intensities(engine)
    # Shift every frame probe by one pixel: centering rolls the whole stack back.
    with torch.no_grad():
        engine.model.probe.field.copy_(torch.roll(engine.model.probe.field, 3, -1))
    shifted = all_intensities(engine)
    ProbeCentering(threshold=1)(engine, 0)
    PhaseRampRemoval()(engine, 0)
    # Fraunhofer: whole-pixel rolls and phase gauges leave intensities unchanged.
    torch.testing.assert_close(
        all_intensities(engine), shifted, rtol=1e-4, atol=1e-6 * before.max()
    )

    old = engine.model.probe.field.detach().clone()
    boundary = AbsorbingProbeBoundary()
    boundary(engine, 0)
    window = boundary.window(8, old.device, old.real.dtype)
    torch.testing.assert_close(engine.model.probe.field, old * (0.95 + 0.05 * window))


def test_export_snapshot_and_reset_round_trip():
    engine = opr_engine(2)
    engine.relaxProbe()
    engine.constraints = [OPRProjection(rank=2)]
    engine.numIterations = 3
    engine.reconstruct()
    r = engine.reconstruction
    model_stack = engine.model.probe.physical_field().detach().numpy()
    # Classic OPR layout: (nlambda, 1, npsm, 1, Np, Np, frames).
    assert r.probe_stack.shape == (1, 1, 2, 1, 8, 8, FRAMES)
    np.testing.assert_allclose(np.moveaxis(r.probe_stack, -1, 0), model_stack)
    assert r.probe.shape == (1, 1, 2, 1, 8, 8)
    state = engine.snapshot(stack=False)
    assert "probe.stack" not in state
    np.testing.assert_allclose(
        state["probe.field"], engine.model.probe.shared_estimate().numpy()
    )

    # reset() re-imports the exported stack after the model was changed.
    with torch.no_grad():
        engine.model.probe.field.zero_()
    engine.reset()
    np.testing.assert_allclose(
        engine.model.probe.physical_field().detach().numpy(), model_stack, rtol=1e-6
    )


def test_shared_probe_is_exported_only_when_the_monitor_draws(monkeypatch):
    # The shared estimate is a J x J eigendecomposition, far costlier than an
    # iteration at large J; mid-run, only the monitor's draws need it.
    engine = opr_engine()
    engine.relaxProbe()
    engine.monitor.figureUpdateFrequency = 3
    engine.numIterations = 7  # draws at 0, 3 and 6 (the last)
    estimate = OPRProbe.shared_estimate
    calls = []

    def counted(probe):
        calls.append(1)
        return estimate(probe)

    monkeypatch.setattr(OPRProbe, "shared_estimate", counted)
    seen = {}

    def show(loop, force=False):
        r, model = engine.reconstruction, engine.model
        seen[loop] = {
            "calls": len(calls),
            "force": force,
            "probe fresh": np.array_equal(r.probe, estimate(model.probe).numpy()),
            "object fresh": np.array_equal(r.object, model.object.detach().numpy()),
        }

    engine.showReconstruction = show
    engine.reconstruct()

    draws = {0, 3, 6}
    for loop, state in seen.items():
        assert state["object fresh"]  # the object is still exported every iteration
        assert state["probe fresh"] == (loop in draws)
        new_calls = state["calls"] - seen[loop - 1]["calls"] if loop else 1
        assert new_calls == (loop in draws)
    assert seen[6]["force"]
    # The end of the run exports the probe and the stack as before.
    np.testing.assert_array_equal(
        engine.reconstruction.probe, estimate(engine.model.probe).numpy()
    )


def test_reset_after_a_run_imports_an_edited_probe_to_every_frame():
    # A run leaves reconstruction.probe_stack behind; an edited probe must
    # still win over that stale export.
    engine = opr_engine()
    engine.relaxProbe()
    engine.numIterations = 2
    engine.reconstruct()
    r = engine.reconstruction
    for edit in ("assigned", "in place"):
        if edit == "assigned":
            r.probe = np.full_like(r.probe, 0.25 + 0.5j)
        else:
            r.probe *= 2
        expected = r.probe.copy()
        engine.reset()
        stack = engine.model.probe.physical_field().detach().numpy()
        for frame in stack:
            np.testing.assert_allclose(frame, expected, rtol=1e-6)
        engine.reconstruct()  # exports a fresh stack and probe


def test_data_with_another_frame_count_are_rejected():
    from copy import deepcopy

    engine = opr_engine()
    engine.relaxProbe()
    engine.numIterations = 1
    engine.reconstruct()
    data = deepcopy(engine.experimentalData)
    data.encoder = data.encoder[:-1]  # 8 frames for a stack of 9
    data.ptychogram = data.ptychogram[:-1]
    data._setData()
    with pytest.raises(ValueError, match="9 frames but the data have 8"):
        engine.reconstruct(data)


def test_shared_estimate_is_the_probe_of_an_equal_stack():
    engine = opr_engine(2)
    engine.relaxProbe()
    engine.prepareReconstruction()
    np.testing.assert_allclose(
        engine.model.probe.shared_estimate().numpy(),
        engine.reconstruction.probe,
        rtol=1e-5,
        atol=1e-6 * abs(engine.reconstruction.probe).max(),
    )
    powers = engine.model.probe.frame_powers()
    assert powers.shape == (FRAMES, 1, 2)
    torch.testing.assert_close(engine.model.probe.mode_powers(), powers.mean(0))


def test_relax_probe_keeps_scale_and_parameter_name_and_is_idempotent():
    engine = opr_engine(1)
    engine.prepareReconstruction()
    scale = engine.model.probe.scale.item()
    engine.relaxProbe()
    probe = engine.model.probe
    assert isinstance(probe, OPRProbe)
    assert probe.scale.item() == scale
    assert probe.field.shape == (FRAMES, 1, 1, 1, 1, 8, 8)
    assert "probe.field" in dict(engine.model.named_parameters())
    engine.relaxProbe()
    assert engine.model.probe is probe


def test_opr_fits_frame_changes_that_a_shared_probe_cannot():
    shared, relaxed = opr_engine(1), opr_engine(1)
    relaxed.relaxProbe()
    relaxed.constraints = [OPRProjection(rank=2)]
    for engine in (shared, relaxed):
        engine.numIterations = 150
        engine.reconstruct()
    assert relaxed.reconstruction.error[-1] < 0.2 * shared.reconstruction.error[-1]
    # The projected stack has rank <= 2: the third singular value is ~0.
    rows = relaxed.model.probe.field.detach()[:, 0, 0, 0, 0].flatten(-2)
    s = torch.linalg.svdvals(rows)
    assert s[2] < 1e-4 * s[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_opr_objective_gradients_and_projection_match_cpu():
    cpu, cuda = opr_engine(2), opr_engine(2)
    cuda.device = "cuda"
    for engine in (cpu, cuda):
        engine.relaxProbe()
        engine.prepareReconstruction()
        set_stack(engine, true_stack(2) * 1.1)
        engine.optimizer.zero_grad(set_to_none=True)
        loss = engine.computeLoss(
            engine.forward(torch.arange(FRAMES, device=engine.device)),
            engine.measuredIntensities,
        )
        loss.backward()
        engine.loss = loss.item()
        OPRProjection(rank=1, relax=0.05)(engine, 0)
    assert cuda.loss == pytest.approx(cpu.loss, rel=2e-5)
    for name in ("object", "probe.field"):
        torch.testing.assert_close(
            dict(cuda.model.named_parameters())[name].grad.cpu(),
            dict(cpu.model.named_parameters())[name].grad,
            rtol=3e-5,
            atol=2e-7,
        )
    torch.testing.assert_close(
        cuda.model.probe.field.detach().cpu(),
        cpu.model.probe.field.detach(),
        rtol=1e-4,
        atol=1e-5,
    )
