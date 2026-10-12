"""DummyMonitor must accept every hook the engines call.

An engine only reaches some hooks under specific params (e.g.
`update_focusing_metric` needs `params.TV_autofocus`), so a missing hook can go
unnoticed until a user turns that option on.
"""

import re
from pathlib import Path

import pytest

import PtyLab
from PtyLab import Engines
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import DummyMonitor
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction

ENGINES_DIR = Path(PtyLab.__file__).parent / "Engines"


def engine_monitor_hooks():
    """Names of every `self.monitor.<name>(...)` call in the engine sources."""
    pattern = re.compile(r"self\.monitor\.([A-Za-z_]\w*)\s*\(")
    return sorted(
        {
            name
            for f in ENGINES_DIR.rglob("*.py")
            for name in pattern.findall(f.read_text())
        }
    )


@pytest.mark.parametrize("hook", engine_monitor_hooks())
def test_dummy_monitor_has_every_engine_hook(hook):
    assert callable(getattr(DummyMonitor(), hook, None))


@pytest.mark.parametrize("engine_name", ["ePIE", "mPIE", "qNewton", "mqNewton"])
def test_engine_runs_with_dummy_monitor_and_autofocus(generate_simu_hdf5, engine_name):
    data = ExperimentalData(str(generate_simu_hdf5), operationMode="CPM")
    params = Params()
    params.gpuSwitch = False
    # autofocus calls monitor.update_focusing_metric, which DummyMonitor once lacked
    params.TV_autofocus = True
    params.TV_autofocus_run_every = 1
    reconstruction = Reconstruction(data, params)
    reconstruction.initializeObjectProbe()

    engine = getattr(Engines, engine_name)(reconstruction, data, params, DummyMonitor())
    engine.numIterations = 2
    engine.reconstruct()

    assert len(reconstruction.error) == 2
