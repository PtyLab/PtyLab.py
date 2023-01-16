from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.Reconstruction.CalibrationFPM import IlluminationCalibration
from PtyLab.Monitor.Monitor import Monitor, DummyMonitor
from PtyLab.Params.Params import Params
from PtyLab import Engines
from pathlib import Path
from typing import Tuple


def easyInitialize(
    filename: Path,
    engine: Engines.BaseEngine = Engines.ePIE,
    operationMode="CPM",
    dummyMonitor=False,
) -> Tuple[ExperimentalData, Reconstruction, Params, Monitor, Engines.BaseEngine]:
    """Do a 'standard' initialization, and return the items you need with some sensible defaults."""
    if operationMode == "CPM":
        return _easyInitializeCPM(filename, engine, operationMode, dummyMonitor)
    if operationMode == "FPM":
        return _easyInitializeFPM(filename, engine, operationMode, dummyMonitor)
    else:
        raise NotImplementedError()


def _easyInitializeCPM(filename, engine_function, operationMode, dummy_monitor=False):
    experimentalData = ExperimentalData(filename, operationMode)
    params = Params()
    if dummy_monitor:
        monitor = DummyMonitor()
    else:
        monitor = Monitor()
    reconstruction = Reconstruction(experimentalData, params)

    reconstruction.initializeObjectProbe()

    engine = engine_function(reconstruction, experimentalData, params, monitor)
    return experimentalData, reconstruction, params, monitor, engine


def _easyInitializeFPM(filename, engine_function, operationMode, dummy_monitor=False):
    experimentalData = ExperimentalData(filename, operationMode)
    if dummy_monitor:
        monitor = DummyMonitor()
    else:
        monitor = Monitor()

    params = Params()
    reconstruction = Reconstruction(experimentalData, params)
    reconstruction.initializeObjectProbe()
    calib = IlluminationCalibration(reconstruction, experimentalData)

    engine = engine_function(reconstruction, experimentalData, params, monitor)
    params.positionOrder = "NA"
    params.probeBoundary = True
    return experimentalData, reconstruction, params, monitor, engine, calib
