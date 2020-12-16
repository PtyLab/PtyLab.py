import matplotlib
matplotlib.use('tkagg')
from fracPy.io import getExampleDataFolder


#matplotlib.use('qt5agg')
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines import ePIE, mPIE, qNewton, zPIE, e3PIE, multiPIE
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
fileName = 'WFS_8.hdf5'  #  simu  Lenspaper WFS_1_bin4 WFS_fundamental    data_637nm_662nm WFS_fundamental_20201207
filePath = getExampleDataFolder() / fileName

exampleData.loadData(filePath)
exampleData.showPtychogram()

exampleData.operationMode = 'CPM'
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
# exampleData.spectralDensity = [662e-9, 637e-9]
# exampleData.spectralDensity = [29.9e-9, 32.11e-9, 34.71e-9, 37.74e-9, 41.35e-9]
# exampleData.spectralDensity = 870e-9/np.linspace(29,19,6)
# exampleData.spectralDensity = 800*1e-9/np.linspace(15, 31, 9)
# exampleData.spectralDensity = [exampleData.wavelength]
exampleData.wavelength = np.min(exampleData.spectralDensity)
optimizable.nlambda = len(exampleData.spectralDensity) # Number of wavelength
optimizable.nslice = 1 # Number of object slice
exampleData.dz = 1e-4  # slice
# binningFactor = 4
exampleData.dxp = exampleData.dxd/8
exampleData.No = 2**11
# exampleData.zo = 0.20
# exampleData.zo = 230e-3


optimizable.initialProbe = 'circ'
exampleData.entrancePupilDiameter = exampleData.Np / 3 * exampleData.dxp  # initial estimate of beam size
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

pinholeDiameter = 250e-6
fullPeriod = 12e-6
apertureSize = 6e-6

# pinholeDiameter = 730e-6
# fullPeriod = 6 * 13.5e-6
# apertureSize = 3 * 13.5e-6
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
monitor.probePlotZoom = 0.8  # control probe plot FoV
monitor.objectPlotZoom =0.7   # control object plot FoV
monitor.objectPlotContrast = 1
monitor.probePlotContrast = 1


# Run the reconstruction
## choose engine
# ePIE
# engine = ePIE.ePIE_GPU(optimizable, exampleData, monitor)
# engine = ePIE.ePIE(optimizable, exampleData, monitor)
# mPIE
# engine = mPIE.mPIE_GPU(optimizable, exampleData, monitor)
# engine = mPIE.mPIE(optimizable, exampleData, monitor)
# multiPIE
engine = multiPIE.multiPIE_GPU(optimizable, exampleData, monitor)
# engine = multiPIE.multiPIE(optimizable, exampleData, monitor)

## main parameters
engine.numIterations = 1000
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
engine.absorbingProbeBoundary = True
engine.objectContrastSwitch = False
engine.absObjectSwitch = False
engine.backgroundModeSwitch = False
engine.couplingSwitch = False
engine.couplingAleph = 1

engine.doReconstruction()


# now save the data
# optimizable.saveResults('reconstruction.hdf5')


