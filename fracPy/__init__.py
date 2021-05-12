from fracPy.FixedData.DefaultExperimentalData import ExperimentalData
from fracPy.Optimizables.Optimizable import Optimizable
from fracPy.Monitors.Monitor import Monitor
from fracPy.Params.ReconstructionParameters import Reconstruction_parameters
from fracPy import Engines

from pathlib import Path
from typing import Tuple

def easy_initialize(filename: Path, engine: Engines.BaseReconstructor=Engines.ePIE_reconstructor.ePIE, operationMode='CPM') ->\
        Tuple[Optimizable, ExperimentalData, Reconstruction_parameters, Monitor, Engines.BaseReconstructor]:
    """ Do a 'standard' initialization, and return the items you need with some sensible defaults. """
    if operationMode == 'CPM':
        return _easy_initialize_CPM(filename, engine, operationMode)
    else:
        raise NotImplementedError()



def _easy_initialize_CPM(filename, engine_function, operationMode):
    experimentalData = ExperimentalData(filename)
    experimentalData.operationMode = operationMode
    monitor = Monitor()
    optimizable = Optimizable(experimentalData)
    params = Reconstruction_parameters()
    engine = engine_function(optimizable, experimentalData, params, monitor)
    return optimizable, experimentalData, params, monitor, engine


