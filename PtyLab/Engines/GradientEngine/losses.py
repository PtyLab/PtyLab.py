"""Differentiable data losses returning one real scalar per frame or batch.

All losses accept (complex detector field, measured amplitude, total scan power).
Losses sum over pixels and frames; the engine takes one optimizer step per scan. Different losses
have different scales; learning rates may need adjusting when switching losses.
"""


def amplitude_loss(detector_wave, measured_amplitude, total_power):
    """Squared amplitude residual, normalized by total measured scan intensity.

    L = (1 / P) * sum_i (|u_i| - a_i)^2

    Here u_i = detector_wave_i, a_i = measured_amplitude_i, and
    P = total_power (total measured scan intensity).
    The sum runs over all pixels and frames in the input.
    """
    residual = detector_wave.abs() - measured_amplitude
    return residual.square().sum() / total_power


def intensity_loss(detector_wave, measured_amplitude, total_power):
    """Squared intensity residual, normalized by squared total scan intensity.

    L = (1 / P^2) * sum_i (|u_i|^2 - a_i^2)^2

    Here u_i = detector_wave_i, a_i = measured_amplitude_i, and
    P = total_power (total measured scan intensity).
    The sum runs over all pixels and frames in the input.
    """
    residual = detector_wave.abs().square() - measured_amplitude.square()
    return residual.square().sum() / total_power.square()


def poisson_loss(detector_wave, measured_amplitude, total_power):
    """Poisson negative log-likelihood normalized by total scan intensity.

    L = (1 / P) * sum_i [I_i - y_i * log(max(I_i, epsilon))]

    Here I_i = |detector_wave_i|^2, y_i = measured_amplitude_i^2,
    P = total_power (total measured scan intensity), and epsilon = 1e-8.
    The sum runs over all pixels and frames in the input; log is natural.
    Omits the measurement-only log-factorial term. Predicted intensity is
    floored at epsilon inside the logarithm to handle zero detector fields.
    """
    intensity = detector_wave.abs().square()
    measured_intensity = measured_amplitude.square()
    return (
        intensity - measured_intensity * intensity.clamp_min(1e-8).log()
    ).sum() / total_power
