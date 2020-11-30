import matplotlib
matplotlib.use('tkagg')
from fracPy.io import getExampleDataFolder


#matplotlib.use('qt5agg')
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines import ePIE, mPIE, qNewton, zPIE, e3PIE
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.monitors.Monitor import Monitor as Monitor
import logging
logging.basicConfig(level=logging.INFO)
from fracPy.utils.utils import ifft2c
from fracPy.utils.visualisation import hsvplot
from matplotlib import pyplot as plt
import numpy as np



exampleData = ExperimentalData()

import os
fileName = 'WFS_fundamental.hdf5'  #  simuRecent  Lenspaper WFS_1_bin4 WFS_fundamental
filePath = getExampleDataFolder() / fileName

exampleData.loadData(filePath)

exampleData.operationMode = 'CPM'
exampleData.No = 2**11
# M = (1+np.sqrt(1-4*exampleData.dxo/exampleData.dxd)/2*exampleData.dxo/exampleData.dxd)
# exampleData.zo = exampleData.zo/M
# exampleData.dxd = exampleData.dxd/M
# absorbedPhase = np.exp(1.j*np.pi/exampleData.wavelength *
#                                              (exampleData.Xp**2+exampleData.Yp**2)/(exampleData.zo))
# absorbedPhase2 = np.exp(1.j*np.pi/exampleData.wavelength *
#                                              (exampleData.Xp**2+exampleData.Yp**2)/(exampleData.zo))

# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
optimizable = Optimizable(exampleData)
optimizable.npsm = 1 # Number of probe modes to reconstruct
optimizable.nosm = 1 # Number of object modes to reconstruct
# exampleData.spectralDensity = 0.8*exampleData.spectralDensity
exampleData.wavelength = np.min(exampleData.wavelength)
optimizable.nlambda = len(exampleData.spectralDensity) # Number of wavelength
optimizable.nslice = 1 # Number of object slice
exampleData.dz = 1e-4  # slice
exampleData.dxp = exampleData.dxd/4
# exampleData.No = 2**11+2**10
# exampleData.zo = exampleData.zo


optimizable.initialProbe = 'circ'
exampleData.entrancePupilDiameter = 700e-6  # exampleData.Np / 3 * exampleData.dxp  # initial estimate of beam size
optimizable.initialObject = 'ones'
# initialize probe and object and related params
optimizable.prepare_reconstruction()

# customize initial probe quadratic phase
# optimizable.probe = optimizable.probe*np.exp(1.j*2*np.pi/exampleData.wavelength *
#                                              (exampleData.Xp**2+exampleData.Yp**2)/(2*6e-3))

# optimizable.object = optimizable.object*np.exp(1.j*2*np.pi/exampleData.wavelength *
#                                              (exampleData.Xo**2+exampleData.Yo**2)/(2*1e+0))
# hsvplot(np.squeeze(optimizable.object[0, 0, 0, 0, :, :]), pixelSize=exampleData.dxp, axisUnit='mm')
# plt.show(block=False)

from fracPy.utils.utils import rect, fft2c, ifft2c
from fracPy.utils.scanGrids import GenerateRasterGrid
pinholeDiameter = 730e-6
fullPeriod = 6 * 13.5e-6
apertureSize = 4 * 13.5e-6
WFS = 0 * exampleData.Xp
n = int(pinholeDiameter // fullPeriod)
R, C = GenerateRasterGrid(n, np.round(fullPeriod / exampleData.dxp))
print('WFS size: %d um' % (2 * max(max(np.abs(C)), max(np.abs(R))) * exampleData.dxp * 1e6))
print('WFS size: %d um' % ((max(max(R) - min(R), max(C) - min(C)) * exampleData.dxp + apertureSize) * 1e6))
R = R + exampleData.Np // 2
C = C + exampleData.Np // 2

np.random.seed(1)
R_offset = np.random.randint(1, 3, len(R))
np.random.seed(2)
C_offset = np.random.randint(1, 3, len(C))
R = R + R_offset - 2
C = C + C_offset - 2

for k in np.arange(len(R)):
    WFS[R[k], C[k]] = 1

subaperture = rect(exampleData.Xp / apertureSize) * rect(exampleData.Yp / apertureSize)
WFS = np.abs(ifft2c(fft2c(WFS) * fft2c(subaperture)))  # convolution of the subaperture with the scan grid
WFS = WFS / np.max(WFS)

optimizable.probe[..., :, :] = WFS.astype('complex64')
hsvplot(np.squeeze(optimizable.probe[0, 0, 0, 0, :, :]), pixelSize=exampleData.dxp, axisUnit='mm')
plt.show(block=False)
#
# this will copy any attributes from experimental data that we might care to optimize
# Set monitor properties
monitor = Monitor()
monitor.figureUpdateFrequency = 1
monitor.objectPlot = 'complex'  # complex abs angle
monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure
monitor.probePlotZoom = 0.5  # control probe plot FoV
monitor.objectPlotZoom = 0.5  # control object plot FoV


# Run the reconstruction
## choose engine
# ePIE
# engine = ePIE.ePIE_GPU(optimizable, exampleData, monitor)
# engine = ePIE.ePIE(optimizable, exampleData, monitor)
# mPIE
engine = mPIE.mPIE_GPU(optimizable, exampleData, monitor)
# engine = mPIE.mPIE(optimizable, exampleData, monitor)

## main parameters
engine.numIterations = 10
engine.positionOrder = 'random'  # 'sequential' or 'random'
engine.propagator = 'scaledPolychromeASP'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP
engine.betaProbe = 0.05
engine.betaObject = 0.75

## engine specific parameters:
engine.zPIEgradientStepSize = 100  # gradient step size for axial position correction (typical range [1, 100])

## switches
engine.probePowerCorrectionSwitch = False
engine.modulusEnforcedProbeSwitch = False
engine.comStabilizationSwitch = False
engine.orthogonalizationSwitch = False
engine.orthogonalizationFrequency = 10
engine.fftshiftSwitch = False
engine.intensityConstraint = 'standard'  # standard fluctuation exponential poission
engine.absorbingProbeBoundary = False
engine.objectContrastSwitch = False
engine.absObjectSwitch = False
engine.backgroundModeSwitch = False
engine.couplingSwitch = True
engine.couplingAleph = 1

engine.doReconstruction()


# now save the data
# optimizable.saveResults('reconstruction.hdf5')


