import numpy as np
import pytest

def adaptive_search(
    objective,
    z0,
    initial_step,
    growth=1.5,
    shrink=0.5,
    min_step=20e-6,
    tolerance=1e-8,
    max_evaluations=30,
):
    """
    Standalone version of the adaptive z-search logic.

    Parameters
    ----------
    objective : callable
        Function returning the quantity to maximize.
    z0 : float
        Initial z position.
    initial_step : float
        Initial axial step.
    growth : float
        Step expansion factor.
    shrink : float
        Step refinement factor.
    min_step : float
        Minimum allowed step.
    tolerance : float
        Minimum significant improvement.
    max_evaluations : int
        Maximum number of objective evaluations.

    Returns
    -------
    best_z : float
        Best position found.
    best_value : float
        Best objective value found.
    z_values : ndarray
        Evaluated z positions.
    values : ndarray
        Corresponding objective values.
    """

    step = initial_step

    evaluated = {}

    best_z = None
    best_value = -np.inf

    z_history = []
    value_history = []

    def evaluate(z):
        nonlocal best_z, best_value

        key = float(z)

        if key in evaluated:
            return evaluated[key]

        value = float(objective(z))

        evaluated[key] = value
        z_history.append(z)
        value_history.append(value)

        if value > best_value:
            best_value = value
            best_z = z

        return value

    # --------------------------------------------------------------
    # Initial three-point sampling
    # --------------------------------------------------------------

    p0 = evaluate(z0)

    z_left = z0 - step
    z_right = z0 + step

    p_left = evaluate(z_left)
    p_right = evaluate(z_right)

    # --------------------------------------------------------------
    # Determine uphill direction
    # --------------------------------------------------------------

    if p_left > p0 and p_left >= p_right:
        direction = -1

    elif p_right > p0 and p_right > p_left:
        direction = +1

    else:
        direction = 0

    # --------------------------------------------------------------
    # Expansion phase
    # --------------------------------------------------------------

    if direction != 0:

        current_z = z0 + direction * step
        current_value = evaluate(current_z)

        while len(evaluated) < max_evaluations:

            step *= growth

            next_z = current_z + direction * step
            next_value = evaluate(next_z)

            improvement = next_value - current_value

            if improvement > tolerance:
                current_z = next_z
                current_value = next_value
                continue

            break

    # --------------------------------------------------------------
    # Refinement phase
    # --------------------------------------------------------------

    step *= shrink

    while (
        step >= min_step
        and len(evaluated) < max_evaluations
    ):

        left_z = best_z - step
        right_z = best_z + step

        left_value = evaluate(left_z)

        if len(evaluated) >= max_evaluations:
            break

        right_value = evaluate(right_z)

        if (
            left_value <= best_value + tolerance
            and right_value <= best_value + tolerance
        ):
            step *= shrink

    order = np.argsort(z_history)

    return (
        best_z,
        best_value,
        np.asarray(z_history)[order],
        np.asarray(value_history)[order],
    )

def test_adaptive_search_finds_parabolic_peak():

    true_z = 50e-3

    def purity_function(z):
        """
        Synthetic purity curve with a maximum at 50 mm.
        """
        return 1.0 - ((z - true_z) / 1e-3) ** 2

    best_z, best_purity, z_values, purity_values = adaptive_search(
        objective=purity_function,
        z0=47e-3,
        initial_step=200e-6,
        growth=1.5,
        shrink=0.5,
        min_step=10e-6,
        tolerance=1e-12,
        max_evaluations=30,
    )

    print()
    print("=" * 70)
    print("Adaptive search test")
    print("=" * 70)

    for z, purity in zip(z_values, purity_values):
        print(
            f"z = {z * 1e3:8.4f} mm | "
            f"purity = {purity:.8f}"
        )

    print("-" * 70)

    print(
        f"true z       = {true_z * 1e3:.6f} mm"
    )

    print(
        f"best z       = {best_z * 1e3:.6f} mm"
    )

    print(
        f"error         = "
        f"{abs(best_z - true_z) * 1e6:.3f} um"
    )

    print(
        f"best purity   = {best_purity:.8f}"
    )

    print(
        f"evaluations   = {len(z_values)}"
    )

    print("=" * 70)

    assert abs(best_z - true_z) <= 20e-6

    assert np.isclose(
        best_purity,
        1.0,
        atol=1e-3,
    )

    assert len(z_values) <= 30

def test_adaptive_search_with_noise():

    true_z = 50e-3

    rng = np.random.default_rng(0)

    def purity_function(z):
        """
        Synthetic purity curve with small reconstruction-like noise.
        """
        clean = 1.0 - ((z - true_z) / 1e-3) ** 2

        noise = rng.normal(
            loc=0.0,
            scale=0.002,
        )

        return clean + noise

    best_z, best_purity, z_values, purity_values = adaptive_search(
        objective=purity_function,
        z0=47e-3,
        initial_step=200e-6,
        growth=1.5,
        shrink=0.5,
        min_step=20e-6,

        # Larger than the typical point-to-point numerical noise
        tolerance=5e-4,

        max_evaluations=30,
    )

    error_um = abs(best_z - true_z) * 1e6

    print()
    print("=" * 70)
    print("Adaptive search with noise")
    print("=" * 70)

    for z, purity in zip(z_values, purity_values):
        print(
            f"z = {z * 1e3:8.4f} mm | "
            f"purity = {purity:.8f}"
        )

    print("-" * 70)

    print(
        f"true z       = {true_z * 1e3:.6f} mm"
    )

    print(
        f"best z       = {best_z * 1e3:.6f} mm"
    )

    print(
        f"error         = {error_um:.3f} um"
    )

    print(
        f"best purity   = {best_purity:.8f}"
    )

    print(
        f"evaluations   = {len(z_values)}"
    )

    print("=" * 70)

    # The noisy search does not need to hit the exact analytical maximum,
    # but it should remain close to it.
    assert error_um <= 100

    assert len(z_values) <= 30

@pytest.mark.parametrize("seed", range(20))
def test_adaptive_search_with_noise_multiple_seeds(seed):

    true_z = 50e-3

    rng = np.random.default_rng(seed)

    def purity_function(z):
        clean = 1.0 - ((z - true_z) / 1e-3) ** 2

        noise = rng.normal(
            loc=0.0,
            scale=0.002,
        )

        return clean + noise

    best_z, best_purity, z_values, purity_values = adaptive_search(
        objective=purity_function,
        z0=47e-3,
        initial_step=200e-6,
        growth=1.5,
        shrink=0.5,
        min_step=20e-6,
        tolerance=5e-4,
        max_evaluations=30,
    )

    error_um = abs(best_z - true_z) * 1e6

    assert error_um <= 100

    assert len(z_values) <= 30


def test_adaptive_search_avoids_local_bump():

    true_z = 50e-3

    def purity_function(z):
        """
        Synthetic purity curve with:
        - global maximum near 50 mm
        - smaller local bump near 48.5 mm
        """

        # Main global peak
        main_peak = 1.0 - ((z - true_z) / 1e-3) ** 2

        # Local bump centered at 48.5 mm
        local_bump = 0.35 * np.exp(
            -((z - 48.5e-3) / 0.20e-3) ** 2
        )

        return main_peak + local_bump

    best_z, best_purity, z_values, purity_values = adaptive_search(
        objective=purity_function,
        z0=47e-3,
        initial_step=200e-6,
        growth=1.5,
        shrink=0.5,
        min_step=20e-6,
        tolerance=5e-4,
        max_evaluations=30,
    )

    error_um = abs(best_z - true_z) * 1e6

    print()
    print("=" * 70)
    print("Adaptive search with local bump")
    print("=" * 70)

    for z, purity in zip(z_values, purity_values):
        print(
            f"z = {z * 1e3:8.4f} mm | "
            f"purity = {purity:.8f}"
        )

    print("-" * 70)

    print(
        f"true z       = {true_z * 1e3:.6f} mm"
    )

    print(
        f"best z       = {best_z * 1e3:.6f} mm"
    )

    print(
        f"error         = {error_um:.3f} um"
    )

    print(
        f"best purity   = {best_purity:.8f}"
    )

    print(
        f"evaluations   = {len(z_values)}"
    )

    print("=" * 70)

    assert error_um <= 100
    assert len(z_values) <= 30

if __name__ == "__main__":
    test_adaptive_search_finds_parabolic_peak()
    test_adaptive_search_with_noise()
    test_adaptive_search_avoids_local_bump()