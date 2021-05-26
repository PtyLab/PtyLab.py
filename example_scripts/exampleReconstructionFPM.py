""" 
FPM data reconstructor 
change data visualization and initialization options manually for now
"""
import matplotlib
try:matplotlib.use('tkagg')
except:pass
import fracPy
from fracPy.io import getExampleDataFolder
from fracPy import Engines
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np

""" 
ptycho data reconstructor 
change data visualization and initialization options manually for now
"""

fileName = 'lung_fpm.hdf5'  # simu.hdf5 or Lenspaper.hdf5
filePath = getExampleDataFolder() / fileName
optimizable, exampleData, params, monitor, engine, calib = fracPy.easy_initialize(filePath, operationMode = 'FPM')

# %% Prepare everything for the reconstruction
# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
optimizable.initialProbe = 'circ'
optimizable.initialObject = 'upsampled'

# %% FPM position calibration
calib.plot = False
calib.fit_mode ='Translation'
calibratedPositions, NA, calibMatrix = calib.runCalibration()
calib.updatePositions()

# %% Prepare reconstruction post-calibration
optimizable.prepare_reconstruction()

# %% Set monitor properties
monitor.figureUpdateFrequency = 1
monitor.objectPlot = 'abs'  # complex abs angle
monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure
monitor.objectPlotZoom = .01  # control object plot FoVW
monitor.probePlotZoom = .01  # control probe plot FoV

# %% Set param
params.gpuSwitch = True
params.positionOrder = 'NA'
params.probePowerCorrectionSwitch = False
params.comStabilizationSwitch = False
params.probeBoundary = True
params.adaptiveDenoisingSwitch = True

#%% Run the reconstructors
# Run momentum accelerated reconstructor
engine = Engines.mqNewton(optimizable, exampleData, params, monitor)
engine.numIterations = 50
engine.betaProbe = 1
engine.betaObject = 1
engine.beta1 = 0.5
engine.beta2 = 0.5
engine.betaProbe_m = 0.25
engine.betaObject_m = 0.25
engine.momentum_method = 'NADAM'
engine.doReconstruction()