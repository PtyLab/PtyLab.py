import matplotlib
matplotlib.use('tkagg')
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

fileName = 'Lenspaper.hdf5'  # simu.hdf5 or Lenspaper.hdf5
filePath = getExampleDataFolder() / fileName

optimizable, exampleData, params, monitor, ePIE_engine = fracPy.easyInitialize(filePath)

## altternative
# exampleData = ExperimentalData()
# exampleData.loadData(filePath)
# exampleData.operationMode = 'CPM'
exampleData.showPtychogram()
exampleData.zo = exampleData.zo*0.9

# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
optimizable.npsm = 1 # Number of probe modes to reconstruct
optimizable.nosm = 1 # Number of object modes to reconstruct
optimizable.nlambda = 1 # len(exampleData.spectralDensity) # Number of wavelength
optimizable.nslice = 1 # Number of object slice
# reconstruction.dxp = reconstruction.dxd


optimizable.initialProbe = 'circ'
exampleData.entrancePupilDiameter = optimizable.Np / 3 * optimizable.dxp  # initial estimate of beam
optimizable.initialObject = 'ones'
# initialize probe and object and related Params
optimizable.prepare_reconstruction()

# customize initial probe quadratic phase
optimizable.probe = optimizable.probe*np.exp(1.j*2*np.pi/optimizable.wavelength *
                                             (optimizable.Xp**2+optimizable.Yp**2)/(2*6e-3))

# this will copy any attributes from experimental data that we might care to optimize
# # Set monitor properties
# monitor = Monitor()
monitor.figureUpdateFrequency = 1
monitor.objectPlot = 'complex'  # complex abs angle
monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure
monitor.objectZoom = 1.5   # control object plot FoV
monitor.probeZoom = 0.5   # control probe plot FoV

# Run the reconstruction

# Params = Params()
## main parameters
params.positionOrder = 'random'  # 'sequential' or 'random'
params.propagatorType = 'Fresnel'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP


## how do we want to reconstruct?
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


engine = Engines.mqNewton(optimizable, exampleData, params, monitor)
engine.numIterations = 5
engine.betaProbe = 1
engine.betaObject = 1
engine.beta1 = 0.5
engine.beta2 = 0.5
engine.betaProbe_m = 1
engine.betaObject_m = 1
engine.momentum_method = 'NADAM'
engine.reconstruct()

## choose engine
# ePIE
# engine_ePIE = ePIE.ePIE(reconstruction, exampleData, Params,monitor)
# engine_ePIE.numIterations = 50
# engine_ePIE.betaProbe = 0.25
# engine_ePIE.betaObject = 0.25
# engine_ePIE.reconstruct()

# mPIE
engine_mPIE = Engines.mPIE(optimizable, exampleData, params, monitor)
engine_mPIE.numIterations = 5
engine_mPIE.betaProbe = 0.25
engine_mPIE.betaObject = 0.25
engine_mPIE.reconstruct()

# zPIE
engine_zPIE = Engines.zPIE(optimizable, exampleData, params, monitor)
engine_zPIE.numIterations = 5
engine_zPIE.betaProbe = 0.35
engine_zPIE.betaObject = 0.35
engine_zPIE.zPIEgradientStepSize = 1000  # gradient step size for axial position correction (typical range [1, 100])
engine_zPIE.reconstruct()

# do another round of mPIE
engine_mPIE.reconstruct()

# e3PIE
# engine_e3PIE = e3PIE.e3PIE(reconstruction, exampleData, Params,monitor)
# engine_e3PIE.numIteration = 50
# engine_e3PIE.reconstruct()

# pcPIE
# engine_pcPIE = pcPIE.pcPIE(reconstruction, exampleData, Params,monitor)
# engine_pcPIE.numIteration = 50
# engine_pcPIE.reconstruct()

# now save the data
# reconstruction.saveResults('reconstruction.hdf5')
