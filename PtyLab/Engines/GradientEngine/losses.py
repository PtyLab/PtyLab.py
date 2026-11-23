r"""
Losses comparing the predicted intensity $I$ with the measured intensity $y$.

Each loss returns a real scalar, summed over all pixels and frames and
normalized by the total measured power $P$ of the scan. Notation:

- $\lVert\cdot\rVert$: Euclidean norm over all pixels and frames.
- $\sqrt{\cdot}$, $\log$, products and quotients act per pixel.
- $g$: detector gain, the signal one photon produces in detector units
  (camera counts, ADU). A pixel that receives $\bar N$ photons on average
  has the predicted intensity $I = g\bar N$. For a photon-counting
  detector, $g = 1$ and $I$ is in photons.
- $\sigma$: readout noise standard deviation (`read_out_sigma`), a scalar or
  a tensor that broadcasts against the intensities, e.g. a per-pixel map
  `(N, N)` in centered order.

$y$ may be negative (offset-subtracted raw data); only the plain
`amplitude_loss` clamps it to zero. See
[Detector background](../differentiable/overview.md#detector-background)
for where background and dark frames belong.
"""

from math import isfinite

import torch


def _check_read_out_sigma(read_out_sigma):
    """Raise if a scalar `read_out_sigma` is negative, NaN or infinite.

    Tensors pass through because `GradientEngine.readoutSigma` validates them
    once per run, which saves a device synchronization per batch.
    """
    if torch.is_tensor(read_out_sigma):
        return
    if not isfinite(read_out_sigma) or read_out_sigma < 0:
        raise ValueError("read_out_sigma must be a nonnegative finite scalar.")


def intensity_loss(
    predicted_intensity,
    measured_intensity,
    total_power,
):
    r"""
    Squared intensity error, for Gaussian noise with constant variance.

    $$
    \mathcal{L} = \frac{\lVert I - y \rVert^2}{P^2}
    $$

    Args:
        predicted_intensity (torch.Tensor): Predicted intensity $I$.
        measured_intensity (torch.Tensor): Measured intensity $y$, same shape.
        total_power (torch.Tensor): Total measured power $P$ of the scan.

    Returns:
        torch.Tensor: Real scalar loss.

    Examples:
        ```python
        from PtyLab.Engines.GradientEngine import intensity_loss

        engine.lossFunction = intensity_loss
        ```
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
    \mathcal{L} = \frac{1}{P}\sum \left(\tilde I - \tilde y \log \tilde I\right),
    \qquad \tilde I = \frac{I}{g} + s^2,\quad \tilde y = \frac{y}{g} + s^2,\quad s = \frac{\sigma}{g}.
    $$

    With $\sigma = 0$ this is the plain Poisson likelihood. With $\sigma > 0$,
    $\tilde y$ has the mean and variance of $\mathrm{Poisson}(\tilde I)$
    (Snyder et al.), so the loss models raw negative $y$ directly. The minimum
    is at $I = y$ for any gain.

    Args:
        predicted_intensity (torch.Tensor): Predicted intensity $I$, detector units.
        measured_intensity (torch.Tensor): Measured intensity $y$, detector units, same shape.
        total_power (torch.Tensor): Total measured power $P$ of the scan.
        gain (float, optional): Detector gain $g$.
        read_out_sigma (float or torch.Tensor, optional): Readout noise
            $\sigma$, detector units.
        eps (float, optional): Lower bound on $\tilde I$ inside the log.

    Returns:
        torch.Tensor: Real scalar loss (can be negative).

    Raises:
        ValueError: If `gain` or `eps` is outside $(0, \infty)$, or a scalar
            `read_out_sigma` is outside $[0, \infty)$.

    Examples:
        ```python
        from functools import partial
        from PtyLab.Engines.GradientEngine import poisson_loss

        engine.lossFunction = partial(poisson_loss, gain=1.0, read_out_sigma=2.0)
        ```
    """
    if not isfinite(gain) or gain <= 0:
        raise ValueError("gain must be a positive finite scalar.")
    _check_read_out_sigma(read_out_sigma)
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite scalar.")
    # s^2 per pixel: a scalar, or a map that broadcasts against the batch.
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
    gain=1.0,
    read_out_sigma=0.0,
    eps=1e-12,
):
    r"""
    Squared amplitude error, optionally with the generalized Anscombe shift.

    $$
    \mathcal{L} = \frac{\lVert \sqrt{I} - \sqrt{y} \rVert^2}{P}
    $$

    This is the default ptychographic objective. Negative $I$ and $y$ are
    clamped to zero, which biases dark pixels upward.

    With `anscombe_offset_switch=True`, both sides are shifted first:

    $$
    \mathcal{L} = \frac{\lVert \sqrt{I + c} - \sqrt{y + c} \rVert^2}{P},
    \qquad c = \tfrac{3}{8}\,g + \frac{\sigma^2}{g}.
    $$

    $\sqrt{y + c}$ is the generalized Anscombe transform (Mäkitalo and Foi,
    2013) up to the constant factor $2/\sqrt{g}$. It gives Poisson plus
    readout noise the same variance at every intensity, and the shift lifts
    almost all raw negative $y$ above zero, so they enter the loss as
    measured. For $g = 1$ and $\sigma = 0$ it is the classic Anscombe
    transform with $c = 3/8$.

    Args:
        predicted_intensity (torch.Tensor): Predicted intensity $I$, detector units.
        measured_intensity (torch.Tensor): Measured intensity $y$, detector units, same shape.
        total_power (torch.Tensor): Total measured power $P$ of the scan.
        anscombe_offset_switch (bool, optional): Apply the shift $c$.
        gain (float, optional): Detector gain $g$. Only used with the shift.
        read_out_sigma (float or torch.Tensor, optional): Readout noise
            $\sigma$, detector units. Only used with the shift.
        eps (float, optional): Added to $I$ under the square root when the
            shift is off, to keep gradients finite at $I = 0$.

    Returns:
        torch.Tensor: Real scalar loss.

    Raises:
        ValueError: If `gain` or `eps` is outside $(0, \infty)$, a scalar
            `read_out_sigma` is outside $[0, \infty)$, or `gain` or
            `read_out_sigma` is set while `anscombe_offset_switch=False`.

    Examples:
        ```python
        from functools import partial
        from PtyLab.Engines.GradientEngine import amplitude_loss

        engine.lossFunction = amplitude_loss  # the default
        engine.lossFunction = partial(
            amplitude_loss, anscombe_offset_switch=True, gain=2.0, read_out_sigma=1.5
        )
        ```
    """
    if not isfinite(gain) or gain <= 0:
        raise ValueError("gain must be a positive finite scalar.")
    _check_read_out_sigma(read_out_sigma)
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite scalar.")
    if anscombe_offset_switch:
        # c = 3g/8 + σ²/g, per pixel when σ is a map.
        offset = 0.375 * gain + read_out_sigma**2 / gain
        predicted_amplitude = torch.sqrt(predicted_intensity.clamp_min(0.0) + offset)
        measured_amplitude = torch.sqrt((measured_intensity + offset).clamp_min(0.0))
    else:
        if gain != 1:
            raise ValueError("gain requires anscombe_offset_switch=True.")
        # A map is refused whatever its values: checking them would need a sync.
        if torch.is_tensor(read_out_sigma) or read_out_sigma != 0:
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
    Gaussian approximation of Poisson plus readout noise.

    $$
    \mathcal{L} = \frac{1}{2P}\left(\left\lVert \frac{I - y}{\sqrt{v}} \right\rVert^2 + \sum \log v\right),
    \qquad v = gI + \sigma^2.
    $$

    $v$ is the variance of $y$ for Poisson photons plus Gaussian readout
    noise. The loss approximates the exact Poisson-Gaussian likelihood and
    loses accuracy at low counts.

    Args:
        predicted_intensity (torch.Tensor): Predicted intensity $I$, detector units.
        measured_intensity (torch.Tensor): Measured intensity $y$, detector units, same shape.
        total_power (torch.Tensor): Total measured power $P$ of the scan.
        gain (float, optional): Detector gain $g$.
        read_out_sigma (float or torch.Tensor, optional): Readout noise
            $\sigma$, detector units.
        eps (float, optional): Added to $v$ to keep it positive.

    Returns:
        torch.Tensor: Real scalar loss (can be negative).

    Raises:
        ValueError: If `gain` or a scalar `read_out_sigma` is outside
            $[0, \infty)$, or `eps` is outside $(0, \infty)$.

    Examples:
        ```python
        from functools import partial
        from PtyLab.Engines.GradientEngine import mixed_poisson_gaussian_loss

        engine.lossFunction = partial(
            mixed_poisson_gaussian_loss, gain=1.0, read_out_sigma=2.0
        )
        ```
    """
    if not isfinite(gain) or gain < 0:
        raise ValueError("gain must be a nonnegative finite scalar.")
    _check_read_out_sigma(read_out_sigma)
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite scalar.")

    # v = g I + σ² + ε, with σ² per pixel when σ is a map.
    variance = gain * predicted_intensity.clamp_min(0.0) + (read_out_sigma**2) + eps
    residual_sq = (predicted_intensity - measured_intensity).square()

    nll = 0.5 * ((residual_sq / variance) + variance.log())
    return nll.sum() / total_power
