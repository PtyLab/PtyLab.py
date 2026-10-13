"""Numerical contracts for intensity-based differentiable objectives."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PtyLab.Engines.GradientEngine.losses import (
    amplitude_loss,
    mixed_poisson_gaussian_loss,
    poisson_loss,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_amplitude_stabilization_value_and_dark_pixel_gradients(device):
    intensity = torch.tensor(
        [0.0, 1e-14, 0.5, 2.0], device=device, dtype=torch.float64, requires_grad=True
    )
    measured = torch.tensor([1.0, 0.0, 0.4, 1.5], device=device, dtype=torch.float64)
    power = measured.sum()
    loss = amplitude_loss(intensity, measured, power)
    expected = (
        np.sum(
            (
                np.sqrt(intensity.detach().cpu().numpy() + 1e-12)
                - np.sqrt(measured.cpu().numpy())
            )
            ** 2
        )
        / power.item()
    )
    assert loss.item() == pytest.approx(expected, rel=1e-12)
    loss.backward()
    assert torch.isfinite(intensity.grad).all()

    field = torch.tensor(
        [0j, 1e-7 + 2e-7j, 0.5 + 0.2j, 1j],
        device=device,
        dtype=torch.complex128,
        requires_grad=True,
    )
    amplitude_loss(field.abs().square(), measured, power).backward()
    assert torch.isfinite(field.grad).all()
    assert field.grad[0] == 0
    assert field.grad[1:].abs().min() > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "loss", [amplitude_loss, mixed_poisson_gaussian_loss, poisson_loss]
)
def test_loss_gradcheck_and_partial_batch_additivity(device, loss):
    intensity = torch.tensor(
        [[0.2, 0.7], [1.1, 2.0], [0.5, 0.3]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    measured = torch.tensor(
        [[0.1, 0.9], [0.0, 1.6], [0.4, 0.2]], device=device, dtype=torch.float64
    )
    power = measured.sum()
    assert torch.autograd.gradcheck(
        lambda x: loss(x, measured, power), (intensity,), eps=1e-6, atol=1e-5, rtol=1e-3
    )
    full = loss(intensity, measured, power)
    batches = loss(intensity[:2], measured[:2], power) + loss(
        intensity[2:], measured[2:], power
    )
    torch.testing.assert_close(full, batches)
    torch.testing.assert_close(
        torch.autograd.grad(full, intensity)[0],
        torch.autograd.grad(batches, intensity)[0],
    )


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_noise_matches_numpy_and_has_finite_dark_pixel_gradient(device):
    intensity = torch.tensor(
        [0.0, 0.2, 2.0], device=device, dtype=torch.float64, requires_grad=True
    )
    measured = torch.tensor([0.0, 0.5, 1.8], device=device, dtype=torch.float64)
    gain, sigma = 1.7, 0.3
    variance = gain * intensity.detach().cpu().numpy() + sigma**2 + 1e-8
    residual = intensity.detach().cpu().numpy() - measured.cpu().numpy()
    expected = (
        0.5 * np.sum(residual**2 / variance + np.log(variance)) / measured.sum().item()
    )
    loss = mixed_poisson_gaussian_loss(intensity, measured, measured.sum(), gain, sigma)
    assert loss.item() == pytest.approx(expected, rel=1e-12)
    loss.backward()
    assert torch.isfinite(intensity.grad).all()


def test_poisson_loss_matches_likelihood_and_has_finite_gradients():
    intensity = torch.tensor([5.0, 5.0], dtype=torch.float64, requires_grad=True)
    measured = torch.tensor([4.0, 9.0], dtype=torch.float64)
    power = measured.sum()
    expected = (
        -torch.distributions.Poisson(intensity).log_prob(measured)
        - torch.lgamma(measured + 1)
    ).sum() / power
    torch.testing.assert_close(poisson_loss(intensity, measured, power), expected)

    zero_intensity = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    loss = poisson_loss(zero_intensity, torch.tensor([0.0, 1.0]), power)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(zero_intensity.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_poisson_with_readout_matches_formula_on_raw_negative_data(device):
    sigma = 1.5
    intensity = torch.tensor(
        [0.05, 0.3, 4.0], device=device, dtype=torch.float64, requires_grad=True
    )
    measured = torch.tensor([-1.2, 0.4, 3.1], device=device, dtype=torch.float64)
    power = torch.tensor(5.0, dtype=torch.float64, device=device)
    shifted = intensity.detach().cpu().numpy() + sigma**2
    expected = (
        np.sum(shifted - (measured.cpu().numpy() + sigma**2) * np.log(shifted)) / 5.0
    )

    loss = poisson_loss(intensity, measured, power, read_out_sigma=sigma)

    assert loss.item() == pytest.approx(expected, rel=1e-12)
    loss.backward()
    expected_grad = (1 - (measured.cpu().numpy() + sigma**2) / shifted) / 5.0
    np.testing.assert_allclose(intensity.grad.cpu().numpy(), expected_grad, rtol=1e-12)


@pytest.mark.parametrize(
    "gain, sigma", [(1.0, 0.0), (2.5, 0.0), (1.0, 1.0), (2.5, 1.5)]
)
def test_poisson_is_stationary_at_the_mean_of_unclamped_data(gain, sigma):
    # In detector units the optimum is I = mean(y) for every gain, including
    # the negative raw values that clamping would have raised.
    measured = gain * torch.tensor([-0.9, -0.2, 0.0, 0.4, 1.7], dtype=torch.float64)
    intensity = torch.full_like(measured, measured.mean().item(), requires_grad=True)
    loss = poisson_loss(
        intensity, measured, torch.tensor(1.0), gain=gain, read_out_sigma=sigma
    )
    (gradient,) = torch.autograd.grad(loss, intensity)
    assert gradient.sum().item() == pytest.approx(0.0, abs=1e-12)


def test_generalized_anscombe_shifts_both_sides_before_clamping():
    sigma = 1.0
    intensity = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64, requires_grad=True)
    measured = torch.tensor([-0.8, 0.3, 2.4], dtype=torch.float64)
    offset = 0.375 + sigma**2
    expected = np.sum(
        (
            np.sqrt(intensity.detach().numpy() + offset)
            - np.sqrt(measured.numpy() + offset)
        )
        ** 2
    )

    loss = amplitude_loss(
        intensity,
        measured,
        torch.tensor(1.0),
        anscombe_offset_switch=True,
        read_out_sigma=sigma,
    )

    assert loss.item() == pytest.approx(expected, rel=1e-12)
    loss.backward()
    assert torch.isfinite(intensity.grad).all()


def test_amplitude_clamps_negative_raw_measurements():
    intensity = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    loss = amplitude_loss(
        intensity, torch.tensor([-0.5, 1.0], dtype=torch.float64), torch.tensor(1.0)
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(intensity.grad).all()
