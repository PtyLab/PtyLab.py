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

experimentalData, reconstruction, params, monitor, ePIE_engine = fracPy.easyInitialize(filePath, operationMode='CPM')

## altternative
# experimentalData = ExperimentalData()
# experimentalData.loadData(filePath)
# experimentalData.operationMode = 'CPM'
experimentalData.showPtychogram()
# experimentalData.zo = experimentalData.zo * 0.9

# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
reconstruction.npsm = 1 # Number of probe modes to reconstruct
reconstruction.nosm = 1 # Number of object modes to reconstruct
reconstruction.nlambda = 1 # len(experimentalData.spectralDensity) # Number of wavelength
reconstruction.nslice = 1 # Number of object slice
# reconstruction.dxp = reconstruction.dxd


reconstruction.initialProbe = 'circ'
# experimentalData.entrancePupilDiameter = reconstruction.Np / 3 * reconstruction.dxp  # initial estimate of beam
reconstruction.initialObject = 'ones'
# initialize probe and object and related Params
reconstruction.initializeObjectProbe()

# customize initial probe quadratic phase
reconstruction.probe = reconstruction.probe * np.exp(1.j * 2 * np.pi / reconstruction.wavelength *
                                                     (reconstruction.Xp ** 2 + reconstruction.Yp ** 2) / (2 * 6e-3))

# this will copy any attributes from experimental data that we might care to optimize
# # Set monitor properties
# monitor = Monitor()
monitor.figureUpdateFrequency = 1
monitor.objectPlot = 'complex'  # complex abs angle
monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure
monitor.objectZoom = 2   # control object plot FoV
monitor.probeZoom = 2   # control probe plot FoV

# Run the reconstruction

# Params = Params()
## main parameters
params.positionOrder = 'random'  # 'sequential' or 'random'
params.propagatorType = 'Fresnel'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP


## how do we want to reconstruct?
params.gpuSwitch = False
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

## choose mqNewton engine
mqNewton = Engines.mqNewton(reconstruction, experimentalData, params, monitor)
mqNewton.numIterations = 5
mqNewton.betaProbe = 1
mqNewton.betaObject = 1
mqNewton.beta1 = 0.5
mqNewton.beta2 = 0.5
mqNewton.betaProbe_m = 1
mqNewton.betaObject_m = 1
mqNewton.momentum_method = 'NADAM'
# mqNewton.reconstruct()

#
# ## choose ePIE engine
ePIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
ePIE.numIterations = 5
ePIE.betaProbe = 0.25
ePIE.betaObject = 0.25
ePIE.reconstruct()
#
# ## choose mPIE engine
mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
mPIE.numIterations = 100
mPIE.betaProbe = 0.25
mPIE.betaObject = 0.25
mPIE.reconstruct()
#
# ## choose zPIE engine
# zPIE = Engines.zPIE(reconstruction, experimentalData, params, monitor)
# zPIE.numIterations = 5
# zPIE.betaProbe = 0.35
# zPIE.betaObject = 0.35
# zPIE.zPIEgradientStepSize = 1000  # gradient step size for axial position correction (typical range [1, 100])
# zPIE.reconstruct()
#
# # do another round of mPIE
# mPIE.reconstruct()
#
#
# ## switch to pcPIE
# pcPIE = Engines.pcPIE(reconstruction, experimentalData, params, monitor)
# pcPIE.numIterations = 5
# pcPIE.reconstruct()

# now save the data
# reconstruction.saveResults('reconstruction.hdf5')
