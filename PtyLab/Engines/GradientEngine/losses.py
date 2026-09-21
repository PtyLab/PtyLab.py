"""Differentiable objectives for predicted and measured detector intensity.

Each objective returns one real scalar for a frame or batch. Losses sum over pixels
and frames; GradientEngine applies one optimizer step per full scan.
"""

from math import isfinite

import torch


def intensity_loss(
    predicted_intensity,
    measured_intensity,
    total_power,
):
    """Squared intensity residual normalized by squared total scan intensity.

    L = (1 / P^2) * sum_i (|u_i|^2 - a_i^2)^2

    Proportional to a Gaussian negative log-likelihood with constant variance,
    for example spatially uniform read-out noise. High photon counts alone do
    not imply constant variance: Poisson variance still grows with intensity.
    """
    residual = predicted_intensity - measured_intensity
    return residual.square().sum() / total_power.square()


def poisson_loss(
    predicted_intensity,
    measured_intensity,
    total_power,
    eps=1e-10,
):
    """NLL of the Poisson distribution.

    L = (1 / P) * sum_i [I_i - y_i * log(max(I_i, epsilon))]

    Poisson photon-count objective, omitting terms independent of the prediction
    and clipping the logarithm near zero. Use photon-count units for this model.
    """
    return (
        predicted_intensity
        - measured_intensity * predicted_intensity.clamp_min(eps).log()
    ).sum() / total_power


def amplitude_loss(
    predicted_intensity,
    measured_intensity,
    total_power,
    eps=1e-12,
):
    """Stabilized squared amplitude residual, normalized by full-scan power.

    L = sum_i (sqrt(max(I_i, 0) + eps) - sqrt(y_i))^2 / P

    I is predicted intensity, y is measured intensity, and P is total measured
    scan intensity. With positive scalar eps, ordinary Torch autodiff remains
    finite at I=0. Stabilization changes the objective near zero; even I=y=0
    contributes eps/P. eps has intensity units and defaults to 1e-12.

    Square-root residuals approximate variance stabilization of Poisson counts;
    this is not an exact Poisson likelihood or the full Anscombe transform.
    Inputs must be nonnegative real intensities in float32 or float64, with P>0.
    """
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite scalar.")
    predicted_amplitude = torch.sqrt(predicted_intensity.clamp_min(0.0) + eps)
    residual = predicted_amplitude - measured_intensity.sqrt()
    return residual.square().sum() / total_power


def mixed_poisson_gaussian_loss(
    predicted_intensity,
    measured_intensity,
    total_power,
    gain=1.0,
    read_out_sigma=0.0,
    eps=1e-8,
):
    """Gaussian approximation to a mixed Poisson-Gaussian noise objective.

    L = (1 / (2 * P)) * sum_i [ (I_i - y_i)^2 / var_i + log(var_i) ]

    where var_i = g * max(I_i, 0) + sigma_readout^2 + epsilon

    This is a Gaussian NLL with prediction-dependent variance, not the exact
    Poisson-Gaussian convolution likelihood; accuracy can be poor at low counts.
    If y = g*N + readout and N is Poisson, I = g*E[N] and var(y) = g*I + sigma^2.
    Thus gain is detector units per photon, read_out_sigma is in detector units,
    and eps is in squared detector units. All three are fixed scalar settings:
    gain and read_out_sigma must be nonnegative; eps must be strictly positive.
    The omitted additive constant means this objective can be negative.
    """
    if not isfinite(gain) or gain < 0:
        raise ValueError("gain must be a nonnegative finite scalar.")
    if not isfinite(read_out_sigma) or read_out_sigma < 0:
        raise ValueError("read_out_sigma must be a nonnegative finite scalar.")
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite scalar.")

    variance = gain * predicted_intensity.clamp_min(0.0) + (read_out_sigma**2) + eps
    residual_sq = (predicted_intensity - measured_intensity).square()

    nll = 0.5 * ((residual_sq / variance) + variance.log())
    return nll.sum() / total_power
