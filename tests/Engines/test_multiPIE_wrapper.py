import pytest

from PtyLab.Engines.mPIE import mPIE, multiPIE


def test_multiPIE_is_exposed_through_engines():
    from PtyLab import Engines

    assert Engines.multiPIE is multiPIE
    
def test_multiPIE_is_mPIE_subclass():
    assert issubclass(multiPIE, mPIE)


def test_multiPIE_deprecated_parameter_aliases():
    engine = object.__new__(multiPIE)

    engine.feedbackM = 0.3
    engine.frictionM = 0.7

    assert engine.betaM == 0.3
    assert engine.stepM == 0.7

    engine.betaM = 0.4
    engine.stepM = 0.8

    assert engine.feedbackM == 0.4
    assert engine.frictionM == 0.8


def test_multiPIE_emits_deprecation_warning(monkeypatch):
    class DummyParams:
        momentumAcceleration = False

    dummy_params = DummyParams()

    def fake_mpie_init(
        self,
        reconstruction,
        experimentalData,
        params,
        monitor,
    ):
        self.reconstruction = reconstruction
        self.experimentalData = experimentalData
        self.params = params
        self.monitor = monitor

        self.feedbackM = 0.3
        self.frictionM = 0.7

    monkeypatch.setattr(
        mPIE,
        "__init__",
        fake_mpie_init,
    )

    with pytest.warns(
        DeprecationWarning,
        match="multiPIE.*deprecated",
    ):
        engine = multiPIE(
            reconstruction=object(),
            experimentalData=object(),
            params=dummy_params,
            monitor=object(),
        )

    assert engine.params.momentumAcceleration is True
    assert engine.name == "multiPIE"
    assert engine.logger.name == "multiPIE"


def test_multiPIE_preserves_momentum_behavior(monkeypatch):
    class DummyParams:
        momentumAcceleration = False

    dummy_params = DummyParams()

    received = {}

    def fake_mpie_init(
        self,
        reconstruction,
        experimentalData,
        params,
        monitor,
    ):
        received["momentumAcceleration"] = (
            params.momentumAcceleration
        )

        self.params = params
        self.feedbackM = 0.3
        self.frictionM = 0.7

    monkeypatch.setattr(
        mPIE,
        "__init__",
        fake_mpie_init,
    )

    with pytest.warns(DeprecationWarning):
        multiPIE(
            reconstruction=object(),
            experimentalData=object(),
            params=dummy_params,
            monitor=object(),
        )

    assert received["momentumAcceleration"] is True