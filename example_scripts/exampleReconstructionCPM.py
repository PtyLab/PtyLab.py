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
ptycho data reconstructor 
change data visualization and initialization options manually for now
"""

exampleData = ExperimentalData()

import os
fileName = 'Lenspaper.hdf5'  # simu.hdf5 or Lenspaper.hdf5
filePath = getExampleDataFolder() / fileName

exampleData.loadData(filePath)

exampleData.operationMode = 'CPM'
exampleData.showPtychogram()

# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
optimizable = Optimizable(exampleData)
optimizable.npsm = 1 # Number of probe modes to reconstruct
optimizable.nosm = 1 # Number of object modes to reconstruct
optimizable.nlambda = 1 # len(exampleData.spectralDensity) # Number of wavelength
optimizable.nslice = 1 # Number of object slice
exampleData.dz = 1e-4  # slice
# exampleData.dxp = exampleData.dxd


optimizable.initialProbe = 'circ'
exampleData.entrancePupilDiameter = exampleData.Np / 3 * exampleData.dxp  # initial estimate of beam
optimizable.initialObject = 'ones'
# initialize probe and object and related params
optimizable.prepare_reconstruction()

# customize initial probe quadratic phase
optimizable.probe = optimizable.probe*np.exp(1.j*2*np.pi/exampleData.wavelength *
                                             (exampleData.Xp**2+exampleData.Yp**2)/(2*6e-3))

# this will copy any attributes from experimental data that we might care to optimize
# # Set monitor properties
monitor = Monitor()
monitor.figureUpdateFrequency = 1
monitor.objectPlot = 'complex'  # complex abs angle
monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure
monitor.objectPlotZoom = 1.5   # control object plot FoV
monitor.probePlotZoom = 0.5   # control probe plot FoV

# exampleData.zo = exampleData.zo
# exampleData.dxp = exampleData.dxp/1
# Run the reconstruction

params = Params()
## main parameters
params.positionOrder = 'random'  # 'sequential' or 'random'
params.propagator = 'Fresnel'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP


## switches
params.gpuSwitch = True
params.probePowerCorrectionSwitch = True
params.modulusEnforcedProbeSwitch = False
params.comStabilizationSwitch = True
params.orthogonalizationSwitch = False
params.orthogonalizationFrequency = 10
params.fftshiftSwitch = False
params.intensityConstraint = 'standard'  # standard fluctuation exponential poission
params.absorbingProbeBoundary = False
params.objectContrastSwitch = False
params.absObjectSwitch = False
params.backgroundModeSwitch = False
params.couplingSwitch = True
params.couplingAleph = 1
params.positionCorrectionSwitch = False

## choose engine
# ePIE
# engine_ePIE = ePIE.ePIE(optimizable, exampleData, params,monitor)
# engine_ePIE.numIterations = 50
# engine_ePIE.betaProbe = 0.25
# engine_ePIE.betaObject = 0.25
# engine_ePIE.doReconstruction()

# mPIE
engine_mPIE = mPIE.mPIE(optimizable, exampleData, params, monitor)
engine_mPIE.numIterations = 100
engine_mPIE.betaProbe = 0.25
engine_mPIE.betaObject = 0.25
engine_mPIE.doReconstruction()

# zPIE
engine_zPIE = zPIE.zPIE(optimizable, exampleData, params,monitor)
engine_zPIE.numIterations = 50
engine_zPIE.betaProbe = 0.25
engine_zPIE.betaObject = 0.25
engine_zPIE.zPIEgradientStepSize = 200  # gradient step size for axial position correction (typical range [1, 100])
engine_zPIE.doReconstruction()

# e3PIE
# engine_e3PIE = e3PIE.e3PIE(optimizable, exampleData, params,monitor)
# engine_e3PIE.numIteration = 50
# engine_e3PIE.doReconstruction

# pcPIE
# engine_pcPIE = pcPIE.pcPIE(optimizable, exampleData, params,monitor)
# engine_pcPIE.numIteration = 50
# engine_pcPIE.doReconstruction()

# now save the data
# optimizable.saveResults('reconstruction.hdf5')
