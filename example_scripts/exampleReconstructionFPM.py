import matplotlib
matplotlib.use('tkagg')
from fracPy.io import getExampleDataFolder


#matplotlib.use('qt5agg')
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.Optimizable.CalibrationFPM import IlluminationCalibration
from fracPy.engines import ePIE, mPIE, qNewton, zPIE, e3PIE, pcPIE
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.Params.Params import Params
from fracPy.monitors.Monitor import Monitor as Monitor
import logging
logging.basicConfig(level=logging.INFO)
from fracPy.utils.utils import ifft2c
from fracPy.utils.visualisation import hsvplot
from matplotlib import pyplot as plt
import numpy as np

""" 
FPM data reconstructor 
change data visualization and initialization options manually for now
"""

# %% Load in the data
exampleData = ExperimentalData()
exampleData.operationMode = 'FPM'

fileName = 'lung_441leds_fpm.hdf5'  # experimental data
# fileName = 'HeLa_49leds_fpm.hdf5'  # experimental data
filePath = getExampleDataFolder() / fileName
exampleData.loadData(filePath)

# decide whether the positions will be recomputed each time they are called or whether they will be fixed
# without the switch, positions are computed from the encoder values
# with the switch calling exampleData.positions will return positions0
exampleData.fixedPositions = False

# %% FPM position calibration
# NOTE: exampleData.fixedPositions = True must be used!
calib = IlluminationCalibration(exampleData)
# calib.plot=True
calib.fitMode = 'Translation'
calibrated_positions, _, _ = calib.runCalibration()
# update the fixed non-dynamically computed positions with the new ones
exampleData.positions0 = calibrated_positions

# %% Prepare everything for the reconstruction
# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
optimizable = Optimizable(exampleData)
optimizable.initialProbe = 'circ'
optimizable.initialObject = 'upsampled'
optimizable.prepare_reconstruction()

# Set monitor properties
monitor = Monitor()
monitor.figureUpdateFrequency = 1
monitor.objectPlot = 'abs'  # complex abs angle
monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure
monitor.objectPlotZoom = .01  # control object plot FoV
monitor.probePlotZoom = .01  # control probe plot FoV

params = Params()
# params.gpuSwitch = True
params.positionOrder = 'NA'
params.probePowerCorrectionSwitch = False
params.comStabilizationSwitch = False
params.probeBoundary = True
# params.absorbingProbeBoundary = True


# Run the reconstructor
# engine = ePIE.ePIE(optimizable, exampleData, monitor)
engine = qNewton.qNewton(optimizable, exampleData, params, monitor)
# engine = mPIE.mPIE(optimizable, exampleData, monitor)
engine.numIterations = 50
engine.betaProbe = 1
engine.betaObject = 1

# now, run the reconstruction
engine.doReconstruction()