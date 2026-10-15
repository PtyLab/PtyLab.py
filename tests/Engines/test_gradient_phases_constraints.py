"""Constraints and run phases of the optional Torch engine."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_gradient_engine import engine  # noqa: E402,F401  (shared fixture)

from PtyLab.Engines.GradientEngine import (  # noqa: E402
    Constraint,
    ObjectAmplitudeClamp,
    Phase,
    PhaseRampRemoval,
    ProbeCentering,
    blind_recipe,
)


def _all_intensities(engine):
    with torch.no_grad():
        return engine.forward(torch.arange(len(engine.positions))).clone()


def test_schedule_selects_start_step_and_end():
    constraint = Constraint(start=2, step=3, end=10)
    assert [i for i in range(15) if constraint.active(i)] == [2, 5, 8]
    assert [i for i in range(4) if Constraint().active(i)] == [0, 1, 2, 3]


def test_engine_calls_constraints_after_each_step_with_the_iteration(engine):
    calls = []

    def record(_, iteration):
        assert not torch.is_grad_enabled()
        calls.append(iteration)

    engine.device = "cpu"
    engine.constraints = [record, ProbeCentering(start=1, step=2)]
    engine.numIterations = 3
    engine.reconstruct()
    assert calls == [0, 1, 2]


def test_probe_centering_rolls_probe_object_and_adam_state(engine):
    # A probe that fits inside the window after a shift, so the roll is exact
    # and far-field predictions do not change.
    engine.device = "cpu"
    probe = np.zeros_like(engine.reconstruction.probe)
    probe[..., 2:6, 2:6] = engine.reconstruction.probe[..., 2:6, 2:6]
    engine.reconstruction.probe = np.roll(probe, (2, -1), axis=(-2, -1))
    engine.reset()
    engine.prepareReconstruction()
    engine.runIteration(0)  # creates Adam state
    with torch.no_grad():
        engine.model.probe.field.copy_(
            torch.as_tensor(np.roll(probe, (2, -1), axis=(-2, -1)))
            / engine.model.probe.scale
        )
    before = _all_intensities(engine)
    fields = {
        name: (p.detach().clone(), engine.optimizer.state[p]["exp_avg"].clone())
        for name, p in engine.model.named_parameters()
    }

    ProbeCentering(threshold=1)(engine, 0)

    row, col = ProbeCentering.centre_of_mass(engine.model.probe.field)
    assert max(abs(row), abs(col)) < 1
    for name, p in engine.model.named_parameters():
        value, moment = fields[name]
        torch.testing.assert_close(
            p.detach(), torch.roll(value, (-2, 1), dims=(-2, -1))
        )
        torch.testing.assert_close(
            engine.optimizer.state[p]["exp_avg"],
            torch.roll(moment, (-2, 1), dims=(-2, -1)),
        )
    torch.testing.assert_close(_all_intensities(engine), before, rtol=1e-5, atol=1e-7)


def test_probe_centering_below_threshold_does_nothing(engine):
    engine.device = "cpu"
    engine.prepareReconstruction()
    before = engine.model.probe.field.detach().clone()
    ProbeCentering(threshold=2)(engine, 0)
    torch.testing.assert_close(
        engine.model.probe.field.detach(), before, rtol=0, atol=0
    )


@pytest.mark.parametrize("relax", [0.0, 0.5])
def test_object_amplitude_clamp_keeps_phase(engine, relax):
    engine.device = "cpu"
    obj = engine.reconstruction.object
    obj[...] = np.linspace(0.2, 2.0, obj.size).reshape(obj.shape) * np.exp(0.7j)
    engine.reset()
    before = engine.model.object.detach().clone()
    ObjectAmplitudeClamp(1.0, relax=relax)(engine, 0)
    after = engine.model.object.detach()
    expected = relax * before.abs() + (1 - relax) * before.abs().clamp(max=1.0)
    torch.testing.assert_close(after.abs(), expected)
    torch.testing.assert_close(torch.angle(after), torch.angle(before))


@pytest.mark.parametrize("propagator", ["Fraunhofer", "Fresnel"])
def test_phase_ramp_removal_keeps_predictions_and_removes_the_ramp(engine, propagator):
    engine.device = "cpu"
    engine.params.propagatorType = propagator
    r = engine.reconstruction
    rows, cols = np.mgrid[: r.No, : r.No]
    rng = np.random.default_rng(3)
    texture = np.exp(0.2j * rng.standard_normal((r.No, r.No)))
    r.object[...] = texture * np.exp(1j * (0.4 * rows - 0.25 * cols + 1.1))
    engine.prepareReconstruction()
    before = _all_intensities(engine)

    removal = PhaseRampRemoval()
    removal(engine, 0)

    torch.testing.assert_close(_all_intensities(engine), before, rtol=1e-4, atol=1e-6)
    k_row, k_col, phi = removal.estimate(engine)
    assert abs(k_row) < 1e-5 and abs(k_col) < 1e-5 and abs(phi) < 1e-5


def test_phase_ramp_removal_rotates_adam_first_moment(engine):
    engine.device = "cpu"
    engine.prepareReconstruction()
    engine.runIteration(0)
    removal = PhaseRampRemoval(ramp=False)
    _, _, phi = removal.estimate(engine)
    obj = engine.model.object
    first = engine.optimizer.state[obj]["exp_avg"].clone()
    second = engine.optimizer.state[obj]["exp_avg_sq"].clone()
    removal(engine, 0)
    factor = complex(np.exp(-1j * phi))
    torch.testing.assert_close(engine.optimizer.state[obj]["exp_avg"], first * factor)
    mean = 0.5 * (second.real + second.imag)
    torch.testing.assert_close(
        engine.optimizer.state[obj]["exp_avg_sq"], torch.complex(mean, mean)
    )


def test_run_matches_consecutive_reconstructions_and_restores_settings(engine):
    engine.device = "cpu"
    manual = copy.deepcopy(engine)
    phases = [
        Phase(2, {"object": {"lr": 0.02}}, name="object only"),
        Phase(3, {"probe.field": {"lr": 0.05}}, optimizerType="preconditioned"),
    ]
    history = engine.run(phases)

    manual.parameters = {"object": {"lr": 0.02}}
    manual.numIterations = 2
    manual.reconstruct()
    errors = list(manual.reconstruction.error)
    manual.parameters = {"probe.field": {"lr": 0.05}}
    manual.optimizerType = "preconditioned"
    manual.numIterations = 3
    manual.reconstruct()
    errors += manual.reconstruction.error

    assert [item["name"] for item in history] == ["object only", ""]
    assert [len(item["error"]) for item in history] == [2, 3]
    np.testing.assert_array_equal(engine.reconstruction.error, errors)
    np.testing.assert_array_equal(
        engine.reconstruction.object, manual.reconstruction.object
    )
    np.testing.assert_array_equal(
        engine.reconstruction.probe, manual.reconstruction.probe
    )
    assert engine.parameters is None
    assert engine.optimizerType == "adam"
    assert engine.numIterations == 100


def test_blind_recipe_reduces_the_loss(engine):
    engine.device = "cpu"
    engine.run(blind_recipe(30, constraints=[ProbeCentering()]))
    error = engine.reconstruction.error
    assert len(error) == 30 and error[-1] < 0.5 * error[0]


def test_blind_recipe_passes_readout_sigma_to_both_phases(engine):
    from PtyLab.Engines.GradientEngine.losses import amplitude_loss, poisson_loss

    seen = []

    def recording(loss):
        """Record the sigma the loss receives, then evaluate it."""

        def wrapped(predicted, measured, total_power, read_out_sigma=0.0):
            seen.append(read_out_sigma)
            return loss(predicted, measured, total_power, read_out_sigma=read_out_sigma)

        return wrapped

    engine.device = "cpu"
    size = engine.reconstruction.Np
    sigma = np.full((size, size), 0.5, dtype=np.float32)

    # Either way the plain amplitude start gets 0 and the Adam phase the map:
    # first from engine.readoutSigma, then from the recipe's own read_out_sigma.
    for engine_sigma, recipe_sigma in ((sigma, None), (None, sigma)):
        engine.readoutSigma = engine_sigma
        seen.clear()
        phases = blind_recipe(
            5, loss=recording(poisson_loss), read_out_sigma=recipe_sigma
        )
        assert phases[0].loss is amplitude_loss
        phases[0].loss = recording(phases[0].loss)
        engine.run(phases)
        assert np.isfinite(engine.reconstruction.error).all()
        assert seen[0] == 0.0
        torch.testing.assert_close(seen[-1], torch.as_tensor(sigma))
        assert engine.readoutSigma is engine_sigma  # restored after run()
