"""MonitorFrame contents and how monitors consume it."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from numpy.testing import assert_allclose

from PtyLab import Engines
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.frame import MonitorFrame
from PtyLab.Monitor.Monitor import AbstractMonitor, DummyMonitor, Monitor
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.utils import fft2c


def _record(name):
    """A monitor hook that stores its arguments under `name`."""

    def hook(self, *args, **kwargs):
        self.calls.setdefault(name, []).append((args, kwargs))

    return hook


class RecordingMonitor(AbstractMonitor):
    """Uses the default `update`, and records what reaches each hook."""

    figureUpdateFrequency = 1
    verboseLevel = "high"
    objectZoom = 1
    probeZoom = 1

    def __init__(self):
        self.calls = {}

    updateObjectProbeErrorMonitor = _record("updateObjectProbeErrorMonitor")
    writeEngineName = _record("writeEngineName")
    update_encoder = _record("update_encoder")
    updateBeamWidth = _record("updateBeamWidth")
    updateDiffractionDataMonitor = _record("updateDiffractionDataMonitor")
    update_overlap = _record("update_overlap")


def six_d(rng, n):
    """Random complex array in the 6D layout (nlambda, nosm, npsm, nslice, Ny, Nx)."""
    return rng.random((1, 1, 1, 1, n, n)) + 1j * rng.random((1, 1, 1, 1, n, n))


def test_views_crop_to_the_roi_and_drop_singleton_axes():
    rng = np.random.default_rng(0)
    obj, probe = six_d(rng, 16), six_d(rng, 8)
    frame = MonitorFrame(
        error=np.array([1.0]),
        object=obj,
        probe=probe,
        object_roi=(slice(2, 10), slice(4, 12)),
        probe_roi=(slice(1, 5), slice(1, 5)),
    )

    assert_allclose(frame.object_view(), obj[0, 0, 0, 0, 2:10, 4:12])
    assert_allclose(frame.probe_view(), probe[0, 0, 0, 0, 1:5, 1:5])


def test_fpm_object_view_shows_the_transformed_spectrum():
    rng = np.random.default_rng(1)
    spectrum = six_d(rng, 16)
    frame = MonitorFrame(
        error=np.array([]), object=spectrum, probe=six_d(rng, 8), operation_mode="FPM"
    )

    assert_allclose(frame.object_view(), np.squeeze(fft2c(spectrum)))


def test_default_update_forwards_to_hooks_and_skips_empty_fields():
    rng = np.random.default_rng(2)
    monitor = RecordingMonitor()
    frame = MonitorFrame(
        error=np.array([3.0, 2.0]),
        object=six_d(rng, 8),
        probe=six_d(rng, 8),
        zo=0.05,
        engine_name="mPIE",
        beam_width=(1e-4, 2e-4),
    )

    monitor.update(frame)

    ((_, kwargs),) = monitor.calls["updateObjectProbeErrorMonitor"]
    assert_allclose(kwargs["object_estimate"], frame.object_view())
    assert kwargs["zo"] == 0.05
    assert monitor.calls["writeEngineName"] == [(("mPIE",), {})]
    assert monitor.calls["updateBeamWidth"] == [((1e-4, 2e-4), {})]
    # no encoder, intensities or overlaps in the frame, so those hooks stay quiet
    assert set(monitor.calls) == {
        "updateObjectProbeErrorMonitor",
        "writeEngineName",
        "updateBeamWidth",
    }


def test_dummy_monitor_never_reads_the_arrays():
    # object=None would fail in object_view, so this passes only if it is skipped
    DummyMonitor().update(MonitorFrame(error=np.array([]), object=None, probe=None))


def run_mpie(monitor, iterations=2):
    data = ExperimentalData("example_data/simu.hdf5", operationMode="CPM")
    params = Params()
    params.gpuSwitch = False
    reconstruction = Reconstruction(data, params)
    reconstruction.initializeObjectProbe()
    engine = Engines.mPIE(reconstruction, data, params, monitor)
    engine.numIterations = iterations
    engine.reconstruct()
    return engine, reconstruction


@pytest.mark.usefixtures("generate_simu_hdf5")
def test_engine_sends_full_frames_to_a_hook_based_monitor():
    monitor = RecordingMonitor()
    engine, reconstruction = run_mpie(monitor)

    # one call for the initial guess plus one per iteration
    assert len(monitor.calls["updateObjectProbeErrorMonitor"]) == 3
    # the initial-guess frame has no engine extras, every iteration frame does
    for hook in ("writeEngineName", "update_encoder", "updateBeamWidth"):
        assert len(monitor.calls[hook]) == 2, hook
    # verboseLevel "high" adds the diffraction data and the overlaps
    assert len(monitor.calls["updateDiffractionDataMonitor"]) == 2
    assert len(monitor.calls["update_overlap"]) == 2

    # the object reaching the hook is the engine's ROI crop, as before the frame existed
    (_, kwargs) = monitor.calls["updateObjectProbeErrorMonitor"][-1]
    roi = engine.objectROI
    assert_allclose(
        kwargs["object_estimate"],
        np.squeeze(reconstruction.object[..., roi[0], roi[1]]),
    )


@pytest.mark.usefixtures("generate_simu_hdf5")
def test_matplotlib_monitor_draws_every_iteration():
    monitor = Monitor()
    monitor.figureUpdateFrequency = 1
    monitor.verboseLevel = "high"
    run_mpie(monitor, iterations=3)

    error_axis = monitor.defaultMonitor.ax_error_metric
    assert error_axis.get_title() == "Error metric (it 3)"
    assert monitor.diffractionDataMonitor.im_Iestimated is not None


@pytest.mark.usefixtures("generate_simu_hdf5")
def test_save_figures_writes_one_png_per_existing_window(tmp_path):
    monitor = Monitor()
    monitor.verboseLevel = "high"
    run_mpie(monitor)

    saved = monitor.saveFigures(tmp_path / "out" / "run")

    # no history window: plain mPIE changes none of zo, theta, purity, positions
    assert [p.name for p in saved] == ["run_reconstruction.png", "run_diffraction.png"]
    assert all(p.stat().st_size > 0 for p in saved)
    assert DummyMonitor().saveFigures(tmp_path / "dummy") == []


@pytest.mark.usefixtures("generate_simu_hdf5")
def test_save_results_with_snapshot_writes_hdf5_and_figures(tmp_path):
    monitor = Monitor()
    _, reconstruction = run_mpie(monitor)

    reconstruction.saveResults(str(tmp_path / "run.hdf5"), snapshot=monitor)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "run.hdf5",
        "run_reconstruction.png",
    ]
