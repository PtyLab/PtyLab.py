"""The parameter-history window shows only the quantities an engine changes."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from PtyLab import Engines
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.frame import MonitorFrame
from PtyLab.Monitor.Monitor import Monitor, tracked_values
from PtyLab.Monitor.Plots import ParameterHistoryPlot
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction


def test_tracked_values_in_display_units_and_without_undefined_purity():
    frame = MonitorFrame(
        error=np.array([]),
        object=None,
        probe=None,
        zo=0.05,
        beam_width=(2e-4, 1e-4),  # (y, x); not plotted
        area_overlap=0.8,  # not plotted
        linear_overlap=0.6,
        npsm=3,
        purity_probe=0.8,
        purity_object=1.0,  # single object mode: undefined, must be left out
    )

    assert tracked_values(frame) == {
        "zo [mm]": {"zo": 50.0},
        "purity [%]": {"probe": 80.0},
    }


def test_constant_quantities_open_no_window():
    plot = ParameterHistoryPlot()
    for it in range(3):
        plot.record(it, {"zo [mm]": {"zo": 50.0}})
    plot.draw()

    assert plot.active_panels() == []
    assert plot.figure is None


def test_a_changed_quantity_gets_a_panel():
    plot = ParameterHistoryPlot()
    for it, zo in enumerate([50.0, 50.0, 49.9]):
        plot.record(it, {"zo [mm]": {"zo": zo}, "theta [deg]": {"theta": 45.0}})
    plot.draw()

    assert plot.active_panels() == ["zo [mm]"]
    assert [ax.get_title() for ax in plot.figure.axes] == ["zo [mm]"]


def test_positions_kept_only_once_moved_and_copied():
    plot = ParameterHistoryPlot()
    measured = np.zeros((4, 2))
    corrected = measured.copy()

    plot.record_positions(measured, corrected)
    assert plot.positions is None

    corrected[0] = 1e-6
    plot.record_positions(measured, corrected)
    corrected[1] = 5e-6  # the engine edits in place; the stored copy must not follow
    assert plot.positions[1][1, 0] == 0


@pytest.mark.usefixtures("generate_simu_hdf5")
@pytest.mark.parametrize("autofocus", [False, True])
def test_zo_panel_appears_only_when_autofocus_moves_zo(autofocus):
    data = ExperimentalData("example_data/simu.hdf5", operationMode="CPM")
    params = Params()
    params.gpuSwitch = False
    params.TV_autofocus = autofocus
    params.TV_autofocus_run_every = 1
    reconstruction = Reconstruction(data, params)
    reconstruction.initializeObjectProbe()
    monitor = Monitor()
    monitor.figureUpdateFrequency = 1

    engine = Engines.mPIE(reconstruction, data, params, monitor)
    engine.numIterations = 3
    engine.reconstruct()

    panels = monitor.parameterHistoryMonitor.active_panels()
    assert ("zo [mm]" in panels) == autofocus
