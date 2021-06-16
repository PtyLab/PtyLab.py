from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizables.Reconstruction import Reconstruction
from fracPy.Optimizables.CalibrationFPM import IlluminationCalibration 
from fracPy.Monitors.Monitor import Monitor
from fracPy.Params.Params import Params
from fracPy import Engines
from pathlib import Path
from typing import Tuple

def easyInitialize(filename: Path, engine: Engines.BaseEngine=Engines.ePIE, operationMode='CPM') ->\
        Tuple[Reconstruction, ExperimentalData, Params, Monitor, Engines.BaseEngine]:
    """ Do a 'standard' initialization, and return the items you need with some sensible defaults. """
    if operationMode == 'CPM':
        return _easyInitializeCPM(filename, engine, operationMode)
    if operationMode == 'FPM':
        return _easyInitializeFPM(filename, engine, operationMode)
    else:
        raise NotImplementedError()



def _easyInitializeCPM(filename, engine_function, operationMode):
    experimentalData = ExperimentalData(filename)
    experimentalData.operationMode = operationMode
    monitor = Monitor()
    optimizable = Reconstruction(experimentalData)
    optimizable.prepare_reconstruction()
    params = Params()
    engine = engine_function(optimizable, experimentalData, params, monitor)
    return optimizable, experimentalData, params, monitor, engine


def _easyInitializeFPM(filename, engine_function, operationMode):
    experimentalData = ExperimentalData(filename)
    experimentalData.operationMode = operationMode
    monitor = Monitor()
    optimizable = Reconstruction(experimentalData)
    optimizable.prepare_reconstruction()
    calib = IlluminationCalibration(optimizable, experimentalData)
    params = Params()
    engine = engine_function(optimizable, experimentalData, params, monitor)
    params.positionOrder = 'NA'
    params.probeBoundary = True
    return optimizable, experimentalData, params, monitor, engine, calib


