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

Raw data: measured_intensity may be negative (offset-subtracted frames with
readout noise). A detector background belongs in predicted_intensity, which
GradientEngine's model provides via ``engine.background``, not subtracted from
the data. :func:`poisson_loss` and :func:`anscombe_loss` with
``read_out_sigma`` model such data without clipping; :func:`amplitude_loss`
clamps negative measurements to zero before its square root.
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
    read_out_sigma=0.0,
    eps=1e-10,
):
    """Poisson NLL in photon counts, optionally with Gaussian readout noise.

    L = (1 / P) * sum_i [m_i - (z_i + s^2) * log(max(m_i, epsilon))],
    m_i = I_i / g + s^2,   z_i = y_i / g,   s = sigma / g

    I and y are in detector units (I = g * E[N]), so I / g and y / g are photon
    counts and the minimum lies at I = y for every gain. With
    ``read_out_sigma = 0`` this is the ordinary Poisson NLL, omitting terms
    independent of the prediction and clipping the logarithm near zero.

    With readout noise y = g * N + eta, eta ~ N(0, sigma^2), the shifted count
    z + s^2 has the mean and variance of Poisson(I / g + s^2): Snyder's CCD
    approximation. y is deliberately not clamped: raw negative values are valid
    data, and clamping z + s^2 at zero biases dark pixels upward. The gradient
    1 - (z + s^2) / m is bounded by the floor m >= s^2, unlike the plain Poisson
    gradient 1 - z / m at a dark prediction.
    """
    if not isfinite(gain) or gain <= 0:
        raise ValueError("gain must be a positive finite scalar.")
    if not isfinite(read_out_sigma) or read_out_sigma < 0:
        raise ValueError("read_out_sigma must be a nonnegative finite scalar.")
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite scalar.")
    shift = (read_out_sigma / gain) ** 2
    counts = predicted_intensity / gain + shift
    return (
        counts - (measured_intensity / gain + shift) * counts.clamp_min(eps).log()
    ).sum() / total_power


def amplitude_loss(
    predicted_intensity,
    measured_intensity,
    total_power,
    eps=1e-12,
):
    """Stabilized squared amplitude residual, normalized by full-scan power.

    L = sum_i (sqrt(max(I_i, 0) + eps) - sqrt(y_i))^2 / P

    This is the conventional ptychographic amplitude objective. The small
    positive epsilon keeps autodiff finite at a dark predicted pixel without
    changing the measured amplitude transform. Negative measurements (raw data
    with readout noise) are clamped to zero, which biases dark pixels upward;
    use :func:`anscombe_loss` with ``read_out_sigma`` for such data.
    """
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite scalar.")
    predicted_amplitude = torch.sqrt(predicted_intensity.clamp_min(0.0) + eps)
    residual = predicted_amplitude - measured_intensity.clamp_min(0.0).sqrt()
    return residual.square().sum() / total_power


def anscombe_loss(
    predicted_intensity,
    measured_intensity,
    total_power,
    anscombe_offset=0.375,
    read_out_sigma=0.0,
):
    """Squared residual after the (generalized) Anscombe amplitude transform.

    L = sum_i (sqrt(max(I_i, 0) + c) - sqrt(max(y_i + c, 0)))^2 / P,
    c = anscombe_offset + read_out_sigma^2

    With ``read_out_sigma = 0`` this is the Anscombe 3/8 transform for Poisson
    counts. With Gaussian readout noise of standard deviation sigma (in photon
    counts, unit gain) it is the generalized Anscombe transform, which
    stabilizes the variance of Poisson plus readout data. The shift keeps
    ``y + c`` positive for almost every raw, unclipped pixel, so negative
    measurements are modelled rather than clipped. Keeping this objective
    separate from :func:`amplitude_loss` makes the approximation explicit.
    """
    if not isfinite(anscombe_offset) or anscombe_offset <= 0:
        raise ValueError("anscombe_offset must be a positive finite scalar.")
    if not isfinite(read_out_sigma) or read_out_sigma < 0:
        raise ValueError("read_out_sigma must be a nonnegative finite scalar.")
    offset = anscombe_offset + read_out_sigma**2
    predicted_amplitude = torch.sqrt(predicted_intensity.clamp_min(0.0) + offset)
    measured_amplitude = torch.sqrt((measured_intensity + offset).clamp_min(0.0))
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

