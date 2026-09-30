import pytest

from PtyLab import Engines
from PtyLab.Engines.mPIE import mPIE, pcPIE


@pytest.fixture
def engine():
    """Create a wrapper instance without initializing reconstruction data."""
    return object.__new__(pcPIE)


def test_pcpie_is_exported():
    """Engines.pcPIE should remain available for backward compatibility."""
    assert hasattr(Engines, "pcPIE")
    assert Engines.pcPIE is pcPIE


def test_pcpie_inherits_mpie():
    """pcPIE should reuse the mPIE implementation."""
    assert issubclass(pcPIE, mPIE)


@pytest.mark.parametrize(
    "legacy_name,current_name,initial_value,updated_value",
    [
        ("betaM", "feedbackM", 0.3, 0.5),
        ("stepM", "frictionM", 0.7, 0.8),
    ],
    ids=["betaM-feedbackM", "stepM-frictionM"],
)
def test_legacy_momentum_parameter_aliases(
    engine, legacy_name, current_name, initial_value, updated_value
):
    """Legacy pcPIE momentum names should map to the mPIE parameters."""
    setattr(engine, current_name, initial_value)

    # Old pcPIE names should read from the new mPIE attributes
    assert getattr(engine, legacy_name) == initial_value

    # Setting the old names should update the new attributes
    setattr(engine, legacy_name, updated_value)
    assert getattr(engine, current_name) == updated_value
