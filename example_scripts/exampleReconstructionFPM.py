import matplotlib
# matplotlib.use('tkagg')
from fracPy.io import getExampleDataFolder


matplotlib.use('qt5agg')
from fracPy.FixedData.DefaultExperimentalData import ExperimentalData
from fracPy.Optimizables.Optimizable import Optimizable
from fracPy.Optimizables.CalibrationFPM import IlluminationCalibration
from fracPy.Engines import ePIE_reconstructor, mPIE_reconstructor, qNewton_reconstructor, mqNewton_reconstructor, zPIE_reconstructor, e3PIE_reconstructor, pcPIE_reconstructor
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.Params.ReconstructionParameters import Reconstruction_parameters
from fracPy.Monitors.Monitor import Monitor as Monitor
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

fileName = 'lung_fpm.hdf5'  # experimental data
# fileName = 'lung_fpm_small.hdf5'  # experimental data
filePath = getExampleDataFolder() / fileName
exampleData.loadData(filePath)

# %% Prepare everything for the reconstruction
# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
optimizable = Optimizable(exampleData)
optimizable.initialProbe = 'circ'
optimizable.initialObject = 'upsampled'

# %% FPM position calibration
calib = IlluminationCalibration(optimizable, exampleData)
calib.plot = True
calib.fit_mode ='Translation'
calibratedPositions, NA, calibMatrix = calib.runCalibration()
calib.updatePositions()

# %% Prepare reconstruction post-calibration
optimizable.prepare_reconstruction()

# %% Set monitor properties
monitor = Monitor()
monitor.figureUpdateFrequency = 1
monitor.objectPlot = 'abs'  # complex abs angle
monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure
monitor.objectPlotZoom = .01  # control object plot FoVW
monitor.probePlotZoom = .01  # control probe plot FoV

# %% Set param
params = Reconstruction_parameters()
params.gpuSwitch = True
params.positionOrder = 'NA'
params.probePowerCorrectionSwitch = False
params.comStabilizationSwitch = False
params.probeBoundary = True
# params.absorbingProbeBoundary = True
params.adaptiveDenoisingSwitch = True

#%% Run the reconstructors
# engine = qNewton.qNewton(optimizable, exampleData, params, monitor)
# engine.numIterations = 50
# engine.betaProbe = 1
# engine.betaObject = 1
# engine.doReconstruction()

# Run momentum accelerated reconstructor
engine = mqNewton_reconstructor.mqNewton(optimizable, exampleData, params, monitor)
engine.numIterations = 50
engine.betaProbe = 1
engine.betaObject = 1
engine.beta1 = 0.5
engine.beta2 = 0.5
engine.betaProbe_m = 0.25
engine.betaObject_m = 0.25
engine.momentum_method = 'NADAM'
engine.doReconstruction()