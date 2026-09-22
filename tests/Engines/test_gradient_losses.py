"""Numerical contracts for intensity-based differentiable objectives."""

import numpy as np
import pytest
from scipy.special import log_ndtr

torch = pytest.importorskip("torch")

from PtyLab.Engines.GradientEngine.losses import (
    amplitude_loss,
    anscombe_loss,
    censored_mixed_poisson_gaussian_loss,
    mixed_poisson_gaussian_loss,
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
@pytest.mark.parametrize("loss", [amplitude_loss, mixed_poisson_gaussian_loss])
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
        0.5
        * np.sum(residual**2 / variance + np.log(variance))
        / measured.sum().item()
    )
    loss = mixed_poisson_gaussian_loss(intensity, measured, measured.sum(), gain, sigma)
    assert loss.item() == pytest.approx(expected, rel=1e-12)
    loss.backward()
    assert torch.isfinite(intensity.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_anscombe_loss_matches_transform_and_has_finite_gradients(device):
    intensity = torch.tensor(
        [0.0, 0.2, 2.0], device=device, dtype=torch.float64, requires_grad=True
    )
    measured = torch.tensor([0.0, 0.5, 1.8], device=device, dtype=torch.float64)
    offset = 0.375
    expected = (
        np.sum(
            (
                np.sqrt(intensity.detach().cpu().numpy() + offset)
                - np.sqrt(measured.cpu().numpy() + offset)
            )
            ** 2
        )
        / measured.sum().item()
    )

    loss = anscombe_loss(intensity, measured, measured.sum(), offset)

    assert loss.item() == pytest.approx(expected, rel=1e-12)
    loss.backward()
    assert torch.isfinite(intensity.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_censored_mixed_noise_uses_floor_probability_for_clipped_pixels(device):
    intensity = torch.tensor(
        [0.1, 0.5, 2.0], device=device, dtype=torch.float64, requires_grad=True
    )
    measured = torch.tensor([0.0, 0.3, 1.7], device=device, dtype=torch.float64)
    gain, sigma, floor = 1.2, 0.25, 0.0
    variance = gain * intensity.detach().cpu().numpy() + sigma**2 + 1e-8
    uncensored = 0.5 * (
        (measured.cpu().numpy() - intensity.detach().cpu().numpy()) ** 2 / variance
        + np.log(variance)
    )
    censored = -log_ndtr((floor - intensity.detach().cpu().numpy()) / np.sqrt(variance))
    expected = np.where(measured.cpu().numpy() <= floor, censored, uncensored)

    loss = censored_mixed_poisson_gaussian_loss(
        intensity,
        measured,
        measured.sum(),
        gain=gain,
        read_out_sigma=sigma,
        lower_bound=floor,
    )

    assert loss.item() == pytest.approx(expected.sum() / measured.sum().item())
    loss.backward()
    assert torch.isfinite(intensity.grad).all()


@pytest.mark.parametrize(
    "options",
    [
        {"gain": -1},
        {"gain": float("nan")},
        {"read_out_sigma": -1},
        {"eps": 0},
        {"eps": float("inf")},
    ],
)
def test_mixed_noise_rejects_invalid_noise_configuration(options):
    with pytest.raises(ValueError):
        mixed_poisson_gaussian_loss(
            torch.ones(2), torch.ones(2), torch.tensor(2.0), **options
        )


@pytest.mark.parametrize("eps", [0, -1e-12, float("nan"), float("inf")])
def test_amplitude_rejects_invalid_stabilization(eps):
    with pytest.raises(ValueError):
        amplitude_loss(torch.zeros(2), torch.ones(2), torch.tensor(2.0), eps=eps)


@pytest.mark.parametrize("offset", [0, -1e-12, float("nan"), float("inf")])
def test_anscombe_rejects_invalid_offset(offset):
    with pytest.raises(ValueError):
        anscombe_loss(
            torch.zeros(2),
            torch.ones(2),
            torch.tensor(2.0),
            anscombe_offset=offset,
        )


@pytest.mark.parametrize(
    "options",
    [
        {"gain": -1},
        {"read_out_sigma": -1},
        {"eps": 0},
        {"lower_bound": float("nan")},
    ],
)
def test_censored_mixed_noise_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        censored_mixed_poisson_gaussian_loss(
            torch.ones(2), torch.ones(2), torch.tensor(2.0), **options
        )
