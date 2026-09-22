import numpy as np

from demos.gradient_engine.noise_model_comparison import (
    _flat_initial_object,
    _losses_for_noise,
    _reconstruction_ssim,
    winner_rows,
)


def test_reconstruction_ssim_is_one_for_a_global_phase_shift():
    rows, columns = np.indices((16, 16))
    truth = (0.2 + rows / 20) * np.exp(1j * columns / 7)
    estimate = truth * np.exp(1j * 0.73)

    aligned, amplitude_ssim, phase_ssim = _reconstruction_ssim(
        estimate, truth, np.s_[:, :]
    )

    np.testing.assert_allclose(aligned, truth)
    np.testing.assert_allclose(amplitude_ssim, 1.0)
    np.testing.assert_allclose(phase_ssim, 1.0)


def test_reconstruction_ssim_detects_changed_structure():
    rows, columns = np.indices((16, 16))
    truth = (0.2 + rows / 20) * np.exp(1j * columns / 7)
    estimate = np.flip(truth, axis=(0, 1)).copy()

    _, amplitude_ssim, phase_ssim = _reconstruction_ssim(estimate, truth, np.s_[:, :])

    assert amplitude_ssim < 0.9
    assert phase_ssim < 0.9


def test_winner_rows_selects_highest_reconstruction_ssim():
    comparison = {
        "scenarios": {"Pure Poisson": {}},
        "photon_counts": (100.0,),
        "summary": [
            {
                "measurement noise": "Pure Poisson",
                "mean photons / frame": 100.0,
                "loss": "Amplitude",
                "amplitude SSIM": 0.8,
                "phase SSIM": 0.7,
            },
            {
                "measurement noise": "Pure Poisson",
                "mean photons / frame": 100.0,
                "loss": "Poisson",
                "amplitude SSIM": 0.9,
                "phase SSIM": 0.6,
            },
        ],
    }

    assert winner_rows(comparison) == [
        {
            "measurement noise": "Pure Poisson",
            "mean photons / frame": 100.0,
            "best amplitude SSIM": "Poisson",
            "best phase SSIM": "Amplitude",
        }
    ]


def test_flat_initialization_does_not_depend_on_object_values():
    first_truth = np.arange(64, dtype=np.float32).reshape(8, 8)
    second_truth = np.flip(first_truth).copy()

    first_initial = _flat_initial_object(first_truth)
    second_initial = _flat_initial_object(second_truth)

    np.testing.assert_array_equal(first_initial, np.ones((8, 8)))
    np.testing.assert_array_equal(first_initial, second_initial)
    assert first_initial.dtype == np.complex64


def test_censored_loss_is_only_used_for_clipped_readout_measurements():
    pure_poisson = _losses_for_noise(0.0)
    clipped_readout = _losses_for_noise(0.25)

    assert "Amplitude" in pure_poisson
    assert "Anscombe" in pure_poisson
    assert "Mixed P-G" not in pure_poisson
    assert "Censored P-G" not in pure_poisson
    assert "Mixed P-G" in clipped_readout
    assert "Censored P-G" in clipped_readout
