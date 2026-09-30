import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from PtyLab.Engines.ePIE import ePIE
from PtyLab.Engines.mPIE import mPIE


base_engine_module = importlib.import_module("PtyLab.Engines.BaseEngine")


class DummyEPIE(ePIE):
    """Minimal ePIE subclass for testing TV dispatch."""

    def objectPatchUpdate(self, objectPatch, DELTA):
        self.object_update_called = True
        return objectPatch + 10


class DummyMPIE(mPIE):
    """Minimal mPIE subclass for testing TV dispatch."""

    def objectPatchUpdate(self, objectPatch, DELTA):
        self.object_update_called = True
        return objectPatch + 20


@pytest.mark.parametrize(
    "engine_class,object_update_offset",
    [(DummyEPIE, 10), (DummyMPIE, 20)],
    ids=["ePIE", "mPIE"],
)
def test_tv_update_uses_engine_specific_object_update(
    monkeypatch, engine_class, object_update_offset
):
    # Only initialize the attributes needed by the shared TV update.
    engine = object.__new__(engine_class)
    engine.betaObject = 0.25
    engine.params = SimpleNamespace(objectTVregStepSize=0.4)
    engine.object_update_called = False

    object_patch = np.ones((1, 1, 1, 1, 2, 2), dtype=np.complex64)
    delta = np.zeros_like(object_patch)

    monkeypatch.setattr(
        base_engine_module,
        "grad_TV",
        lambda obj, epsilon=1e-2: 2 * np.ones_like(obj),
    )

    result = engine.objectPatchUpdate_TV(object_patch, delta)

    # TV contribution: step size * betaObject * gradient = 0.4 * 0.25 * 2.
    expected = object_patch + object_update_offset + 0.2
    assert engine.object_update_called
    np.testing.assert_allclose(result, expected)
