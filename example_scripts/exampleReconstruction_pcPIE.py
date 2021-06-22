import matplotlib
matplotlib.use('tkagg')
import fracPy
from fracPy.io import getExampleDataFolder
from fracPy import Params
from fracPy import Monitor
from fracPy import Reconstruction
from fracPy import ExperimentalData
from fracPy import Engines
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np


""" 
ptycho data reconstructor 
change data visualization and initialization options manually for now
"""


fileName = 'simu.hdf5'  # simu.hdf5 or Lenspaper.hdf5
filePath = getExampleDataFolder() / fileName

exampleData = ExperimentalData(filePath, operationMode='CPM')
#initialize again reconstruction and engine with the wrong encoder data
reconstruction = Reconstruction(exampleData)

# perturb encoder positions
maxPosError = 10
encoder0 = exampleData.encoder.copy()
exampleData.encoder = encoder0 + maxPosError * reconstruction.dxo * (2 * np.random.rand(encoder0.shape[0], encoder0.shape[1]) - 1)
import matplotlib.pyplot as plt
figure, ax = plt.subplots(1, 1, num=112, squeeze=True, clear=True, figsize=(5, 5))
ax.set_title('Original and perturbed scan grid positions')
ax.set_xlabel('(um)')
ax.set_ylabel('(um)')
line1, = plt.plot(encoder0[:, 1] * 1e6,
         encoder0[:, 0] * 1e6, 'bo', label='correct')
line2, = plt.plot(exampleData.encoder[:, 1] * 1e6,
         exampleData.encoder[:, 0] * 1e6, 'yo', label='false')
plt.legend(handles=[line1, line2])
plt.tight_layout()
plt.show(block=False)

exampleData.showPtychogram()

# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
reconstruction.npsm = 1 # Number of probe modes to reconstruct
reconstruction.nosm = 1 # Number of object modes to reconstruct
reconstruction.nlambda = 1 # len(exampleData.spectralDensity) # Number of wavelength
reconstruction.nslice = 1 # Number of object slice
# reconstruction.dxp = reconstruction.dxd


reconstruction.initialProbe = 'circ'
exampleData.entrancePupilDiameter = reconstruction.Np / 3 * reconstruction.dxp  # initial estimate of beam
reconstruction.initialObject = 'ones'
# initialize probe and object and related params
reconstruction.initializeObjectProbe()

# customize initial probe quadratic phase
reconstruction.probe = reconstruction.probe*np.exp(1.j*2*np.pi/reconstruction.wavelength *
                                             (reconstruction.Xp**2+reconstruction.Yp**2)/(2*6e-3))

# this will copy any attributes from experimental data that we might care to optimize
# # Set monitor properties
monitor = Monitor()
monitor.figureUpdateFrequency = 1
monitor.objectPlot = 'complex'  # complex abs angle
monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure
monitor.objectPlotZoom = 1.5   # control object plot FoV
monitor.probePlotZoom = 0.5   # control probe plot FoV

# Run the reconstruction

params = Params()
## main parameters
params.positionOrder = 'random'  # 'sequential' or 'random'
params.propagator = 'Fresnel'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP


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

ePIE_engine = Engines.ePIE(reconstruction, exampleData, params, monitor)
## choose engine
# ePIE
engine_ePIE = Engines.ePIE(reconstruction, exampleData, params,monitor)
engine_ePIE.numIterations = 50
engine_ePIE.betaProbe = 0.25
engine_ePIE.betaObject = 0.25
engine_ePIE.reconstruct()

#reset object and probe to initial guesses
reconstruction.initialProbe = 'circ'
exampleData.entrancePupilDiameter = reconstruction.Np / 3 * reconstruction.dxp  # initial estimate of beam
reconstruction.initialObject = 'ones'
# initialize probe and object and related params
reconstruction.initializeObjectProbe()

# customize initial probe quadratic phase
reconstruction.probe = reconstruction.probe*np.exp(1.j*2*np.pi/reconstruction.wavelength *
                                             (reconstruction.Xp**2+reconstruction.Yp**2)/(2*6e-3))
reconstruction.error = []
params.positionCorrectionSwitch = True

engine_pcPIE = Engines.pcPIE(reconstruction, exampleData, params,monitor)
engine_pcPIE.numIterations = 70
engine_pcPIE.reconstruct()

#compare original encoder and result from pcPIE
figure, ax = plt.subplots(1, 1, num=113, squeeze=True, clear=True, figsize=(5, 5))
ax.set_title('Original and corrected after perturbation scan grid positions')
ax.set_xlabel('(um)')
ax.set_ylabel('(um)')
line1, = plt.plot(encoder0[:, 1] * 1e6,
         encoder0[:, 0] * 1e6, 'bo', label='correct')
line2, = plt.plot((reconstruction.positions[:, 1] - reconstruction.No // 2 + reconstruction.Np // 2) * reconstruction.dxo * 1e6,
                  (reconstruction.positions[:, 0] - reconstruction.No // 2 + reconstruction.Np // 2) * reconstruction.dxo * 1e6, 'yo', label='estimated')
plt.legend(handles=[line1, line2])
plt.tight_layout()
plt.show(block=False)