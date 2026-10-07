r"""
Differentiable objectives for predicted and measured detector intensity.

Each objective returns one real scalar for a frame or batch. Losses sum over
pixels and frames, and `GradientEngine` applies one optimizer step per full
scan. Below, $I_i$ is the predicted intensity, $y_i$ the measured intensity
and $P$ the total measured power of the scan.

Unit convention: `predicted_intensity` is expected in detector units throughout
this module,

$$
I = g\,\mathrm{E}[N],
$$

where $N$ is the photon count and $g$ the detector gain. For a unit-gain
detector ($g = 1$, the default everywhere below), detector units and photon
counts coincide. Any $g \neq 1$ must be applied consistently across whichever
loss is selected, so that `predicted_intensity` means the same physical
quantity regardless of the objective `GradientEngine` is configured to use.

Raw data: `measured_intensity` may be negative (offset-subtracted frames with
readout noise). A detector background belongs in the prediction, which the
model provides through `engine.background`, and is not subtracted from the
data. `poisson_loss` with `read_out_sigma` and `amplitude_loss` with
`anscombe_offset_switch=True` and `read_out_sigma` model such data without
clipping. The plain `amplitude_loss` clamps negative measurements to zero
before its square root.
"""

from math import isfinite

import torch


def intensity_loss(
    predicted_intensity,
    measured_intensity,
    total_power,
):
    r"""
    Squared intensity residual normalized by the squared total scan power.

    $$
    \mathcal{L} = \frac{1}{P^2}\sum_i \left(I_i - y_i\right)^2
    $$

    The objective is proportional to a Gaussian negative log-likelihood with
    constant variance, for example spatially uniform readout noise. High
    photon counts alone do not imply constant variance: Poisson variance still
    grows with intensity.

    Args:
        predicted_intensity (torch.Tensor):
            Predicted detector intensity $I$.

        measured_intensity (torch.Tensor):
            Measured detector intensity $y$, with the same shape.

        total_power (torch.Tensor):
            Total measured power $P$ of the full scan.

    Returns:
        torch.Tensor:
            Real scalar loss.
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
    r"""
    Poisson negative log-likelihood, optionally with Gaussian readout noise.

    $$
    \mathcal{L} = \frac{1}{P}\sum_i \left[m_i - \left(z_i + s^2\right) \log\max\left(m_i, \epsilon\right)\right]
    $$

    with

    $$
    m_i = \frac{I_i}{g} + s^2,\qquad z_i = \frac{y_i}{g},\qquad s = \frac{\sigma}{g}.
    $$

    $I$ and $y$ are in detector units ($I = g\,\mathrm{E}[N]$), so $I/g$ and
    $y/g$ are photon counts and the minimum lies at $I = y$ for every gain.
    With $\sigma = 0$ this is the ordinary Poisson negative log-likelihood,
    without terms independent of the prediction and with the logarithm clipped
    near zero.

    With readout noise $y = gN + \eta$, $\eta \sim \mathcal{N}(0, \sigma^2)$,
    the shifted count $z + s^2$ has the mean and variance of
    $\mathrm{Poisson}(I/g + s^2)$, the CCD approximation of Snyder et al.

    Args:
        predicted_intensity (torch.Tensor):
            Predicted detector intensity $I$ in detector units.

        measured_intensity (torch.Tensor):
            Measured detector intensity $y$ in detector units, with the same
            shape. Raw negative values are allowed.

        total_power (torch.Tensor):
            Total measured power $P$ of the full scan.

        gain (float, optional):
            Detector gain $g$ in detector units per photon. Defaults to 1.0.

        read_out_sigma (float, optional):
            Readout noise standard deviation $\sigma$ in detector units.
            Defaults to 0.0.

        eps (float, optional):
            Floor $\epsilon$ of the logarithm's argument. Defaults to 1e-10.

    Returns:
        torch.Tensor:
            Real scalar loss.

    Raises:
        ValueError:
            If `gain` or `eps` is not positive and finite, or `read_out_sigma`
            is not nonnegative and finite.

    Notes:
        $y$ is deliberately not clamped: raw negative values are valid data,
        and clamping $z + s^2$ at zero biases dark pixels upward. The gradient
        $1 - (z + s^2)/m$ is bounded by the floor $m \geq s^2$, unlike the
        plain Poisson gradient $1 - z/m$ at a dark prediction.
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
    anscombe_offset_switch=False,
    read_out_sigma=0.0,
    eps=1e-12,
):
    r"""
    Squared amplitude residual, optionally after the generalized Anscombe shift.

    By default this is the conventional ptychographic amplitude objective,

    $$
    \mathcal{L} = \frac{1}{P}\sum_i \left(\sqrt{\max(I_i, 0) + \epsilon} - \sqrt{\max(y_i, 0)}\right)^2.
    $$

    The small $\epsilon$ keeps autodiff finite at a dark predicted pixel
    without changing the measured amplitude. Negative measurements (raw data
    with readout noise) are clamped to zero, which biases dark pixels upward.

    With `anscombe_offset_switch=True`, both sides are shifted by $c$ before
    the square root:

    $$
    \mathcal{L} = \frac{1}{P}\sum_i \left(\sqrt{\max(I_i, 0) + c} - \sqrt{\max(y_i + c, 0)}\right)^2,\qquad c = \frac{3}{8} + \sigma^2.
    $$

    With $\sigma = 0$ this is the Anscombe transform for Poisson counts. With
    Gaussian readout noise of standard deviation $\sigma$ (in photon counts,
    unit gain) it is the generalized Anscombe transform, which stabilizes the
    variance of Poisson plus readout data. The shift keeps $y + c$ positive
    for almost every raw, unclipped pixel, so negative measurements are
    modelled rather than clipped.

    Args:
        predicted_intensity (torch.Tensor):
            Predicted detector intensity $I$ in photon counts.

        measured_intensity (torch.Tensor):
            Measured detector intensity $y$ in photon counts, with the same
            shape.

        total_power (torch.Tensor):
            Total measured power $P$ of the full scan.

        anscombe_offset_switch (bool, optional):
            If True, apply the fixed $3/8$ Anscombe offset plus $\sigma^2$ to
            both sides. Defaults to False.

        read_out_sigma (float, optional):
            Readout noise standard deviation $\sigma$ in photon counts. Only
            used with `anscombe_offset_switch=True`. Defaults to 0.0.

        eps (float, optional):
            Stabilizer $\epsilon$ of the predicted amplitude without the shift.
            Defaults to 1e-12.

    Returns:
        torch.Tensor:
            Real scalar loss.

    Raises:
        ValueError:
            If `read_out_sigma` is not nonnegative and finite, `eps` is not
            positive and finite, or `read_out_sigma` is nonzero without
            `anscombe_offset_switch=True`.
    """
    if not isfinite(read_out_sigma) or read_out_sigma < 0:
        raise ValueError("read_out_sigma must be a nonnegative finite scalar.")
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite scalar.")
    if anscombe_offset_switch:
        offset = 0.375 + read_out_sigma**2
        predicted_amplitude = torch.sqrt(predicted_intensity.clamp_min(0.0) + offset)
        measured_amplitude = torch.sqrt((measured_intensity + offset).clamp_min(0.0))
    else:
        if read_out_sigma != 0:
            raise ValueError("read_out_sigma requires anscombe_offset_switch=True.")
        predicted_amplitude = torch.sqrt(predicted_intensity.clamp_min(0.0) + eps)
        measured_amplitude = measured_intensity.clamp_min(0.0).sqrt()
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
    r"""
    Gaussian approximation to a mixed Poisson-Gaussian noise objective.

    $$
    \mathcal{L} = \frac{1}{2P}\sum_i \left[\frac{(I_i - y_i)^2}{v_i} + \log v_i\right],\qquad v_i = g\max(I_i, 0) + \sigma^2 + \epsilon.
    $$

    If $y = gN + \eta$ with $N$ Poisson and
    $\eta \sim \mathcal{N}(0, \sigma^2)$, then $I = g\,\mathrm{E}[N]$ and
    $\mathrm{Var}(y) = gI + \sigma^2$. This is a Gaussian negative log-likelihood with
    prediction-dependent variance, not the exact Poisson-Gaussian convolution
    likelihood, and its accuracy can be poor at low counts.

    Args:
        predicted_intensity (torch.Tensor):
            Predicted detector intensity $I$ in detector units.

        measured_intensity (torch.Tensor):
            Measured detector intensity $y$ in detector units, with the same
            shape.

        total_power (torch.Tensor):
            Total measured power $P$ of the full scan.

        gain (float, optional):
            Detector gain $g$ in detector units per photon. Defaults to 1.0.

        read_out_sigma (float, optional):
            Readout noise standard deviation $\sigma$ in detector units.
            Defaults to 0.0.

        eps (float, optional):
            Variance floor $\epsilon$ in squared detector units. Defaults to
            1e-8.

    Returns:
        torch.Tensor:
            Real scalar loss. The omitted additive constant means it can be
            negative.

    Raises:
        ValueError:
            If `gain` or `read_out_sigma` is not nonnegative and finite, or
            `eps` is not positive and finite.
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
