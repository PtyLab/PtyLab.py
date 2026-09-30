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
    '''
    Initialize the main PtyLab components for CPM or FPM reconstruction.

    Args:
        filename (Path):
            Path to the experimental data file.

        engine (Engines.BaseEngine, optional):
            EReconstruction engine class to instantiate.
            Defaults to ``Engines.ePIE``.

        operationMode (str, optional):
            Operation mode, either ``"CPM"`` or ``"FPM"``.
            Defaults to ``"CPM"``.

        dummyMonitor (bool, optional):
            If True, use a dummy monitor without graphical output.
            Defaults to False.

    Returns:
        tuple:
            Initialized PtyLab objects required for the reconstruction.  
            For FPM, additionally returns an ``IlluminationCalibration`` object.

    Raises:
        NotImplementedError:
            If ``operationMode`` is neither ``"CPM"`` nor ``"FPM"``.
    '''
    if operationMode == "CPM":
        return _easyInitializeCPM(filename, engine, operationMode, dummyMonitor)
    if operationMode == "FPM":
        return _easyInitializeFPM(filename, engine, operationMode, dummyMonitor)
    else:
        raise NotImplementedError()


def _easyInitializeCPM(filename, engine_function, operationMode, dummy_monitor=False):
    '''
    Initialize the main PtyLab components for a conventional ptychography reconstruction.

    Args:
        filename (str or Path):
            Path to the experimental data file.

        engine_function (type[Engines.BaseEngine]):
            Reconstruction engine class to instantiate, for example
            ``Engines.ePIE`` or ``Engines.mPIE``.

        operationMode (str):
            Ptychographic operation mode passed to ``ExperimentalData``.
            For this helper, this is expected to be ``"CPM"``.

        dummy_monitor (bool, optional):
            If True, use a ``DummyMonitor`` without graphical output.
            Otherwise, initialize the standard graphical ``Monitor``.
            Defaults to False.

    Returns:
        tuple:
            A tuple containing:

            - ``ExperimentalData``: loaded diffraction data and acquisition geometry.
            - ``Reconstruction``: initialized reconstruction state.
            - ``Params``: reconstruction parameters.
            - ``Monitor`` or ``DummyMonitor``: reconstruction monitor.
            - ``BaseEngine``: initialized reconstruction engine.

    Notes:
        The object and probe are initialized by calling
        ``reconstruction.initializeObjectProbe()`` before the engine is created.
    '''
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
