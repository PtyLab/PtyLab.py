"""Differentiable objectives for predicted and measured detector intensity.

Each objective returns one real scalar for a frame or batch. Losses sum over pixels
and frames; GradientEngine applies one optimizer step per full scan.

Unit convention: predicted_intensity is expected in detector units throughout
this module, i.e. predicted_intensity = gain * E[photon count]. For a unit-gain
detector (gain=1.0, the default everywhere below), detector units and photon
counts coincide, so this reduces to the usual photon-count convention. Any
gain != 1.0 must be applied consistently across whichever loss is selected, so
that predicted_intensity always means the same physical quantity regardless of
which objective GradientEngine is configured to use.
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
    gain=1.0,
    eps=1e-10,
):
    """NLL of the Poisson distribution.

    L = (1 / P) * sum_i [I_i - (y_i / g) * log(max(I_i, epsilon))]

    Poisson photon-count objective, omitting terms independent of the prediction
    and clipping the logarithm near zero. predicted_intensity is in detector
    units (I = g * E[N]); gain reconciles this with measured_intensity, which
    is assumed to already be in the same detector units as predicted_intensity.
    For a unit-gain detector (default), this is the ordinary photon-count
    Poisson NLL with I and y both equal to photon counts.
    """
    if not isfinite(gain) or gain <= 0:
        raise ValueError("gain must be a positive finite scalar.")
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite scalar.")
    return (
        predicted_intensity
        - (measured_intensity / gain) * predicted_intensity.clamp_min(eps).log()
    ).sum() / total_power


def amplitude_loss(
    predicted_intensity,
    measured_intensity,
    total_power,
    anscombe_offset=0.375,
):
    """Anscombe-stabilized squared amplitude residual, normalized by full-scan power.

    L = sum_i (sqrt(max(I_i, 0) + c) - sqrt(max(y_i, 0) + c))^2 / P,  c = anscombe_offset

    Approximates the Poisson NLL by working in amplitude space, where Poisson
    noise is roughly constant-variance. c = 3/8 (Anscombe, 1948) sharpens this
    approximation at low counts and must be applied to both sides equally, and
    must stay strictly positive to keep the gradient finite at I = 0.
    """
    if not isfinite(anscombe_offset) or anscombe_offset <= 0:
        raise ValueError("anscombe_offset must be a positive finite scalar.")
    predicted_amplitude = torch.sqrt(
        predicted_intensity.clamp_min(0.0) + anscombe_offset
    )
    measured_amplitude = torch.sqrt(measured_intensity.clamp_min(0.0) + anscombe_offset)
    residual = predicted_amplitude - measured_amplitude
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
