from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Reconstruction.Reconstruction import Reconstruction
from fracPy.Reconstruction.CalibrationFPM import IlluminationCalibration
from fracPy.Monitor.Monitor import Monitor, DummyMonitor
from fracPy.Params.Params import Params
from fracPy import Engines
from pathlib import Path
from typing import Tuple

def easyInitialize(filename: Path, engine: Engines.BaseEngine=Engines.ePIE, operationMode='CPM') ->\
        Tuple[ExperimentalData, Reconstruction, Params, Monitor, Engines.BaseEngine]:
    """ Do a 'standard' initialization, and return the items you need with some sensible defaults. """
    if operationMode == 'CPM':
        return _easyInitializeCPM(filename, engine, operationMode)
    if operationMode == 'FPM':
        return _easyInitializeFPM(filename, engine, operationMode)
    else:
        raise NotImplementedError()



def _easyInitializeCPM(filename, engine_function, operationMode):
    experimentalData = ExperimentalData(filename, operationMode)
    params = Params()
    #monitor = Monitor()
    monitor = DummyMonitor()
    reconstruction = Reconstruction(experimentalData, params)
    reconstruction.initializeObjectProbe()

    engine = engine_function(reconstruction, experimentalData, params, monitor)
    return experimentalData, reconstruction, params, monitor, engine


def _easyInitializeFPM(filename, engine_function, operationMode):
    experimentalData = ExperimentalData(filename, operationMode)
    monitor = Monitor()
    params = Params()
    reconstruction = Reconstruction(experimentalData, params)
    reconstruction.initializeObjectProbe()
    calib = IlluminationCalibration(reconstruction, experimentalData)

    engine = engine_function(reconstruction, experimentalData, params, monitor)
    params.positionOrder = 'NA'
    params.probeBoundary = True
    return experimentalData, reconstruction, params, monitor, engine, calib


