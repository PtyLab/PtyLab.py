import matplotlib
matplotlib.use('tkagg')
from fracPy.io import getExampleDataFolder
from fracPy.utils.scanGrids import GenerateConcentricGrid, GenerateRasterGrid
from fracPy.utils.utils import rect, fft2c, ifft2c
from scipy.ndimage import rotate

#matplotlib.use('qt5agg')
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines import multiPIE
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
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
filePath = 'WFS_fundamental.hdf5'#r"D:\Du\Workshop\fracpy\example_data" # D:\ptyLab\example_data D:\Du\Workshop\fracpy\example_data
filePath = getExampleDataFolder() / filePath

exampleData.loadData(filePath)  # simuRecent  Lenspaper

exampleData.operationMode = 'CPM'
exampleData.No = 2**11
# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
optimizable = Optimizable(exampleData)
optimizable.npsm = 1 # Number of probe modes to reconstruct
optimizable.nosm = 1 # Number of object modes to reconstruct
optimizable.nlambda = 1 # Number of wavelength
optimizable.nslice = 1 # Number of object slice
exampleData.dxp = exampleData.dxd/2
exampleData.spectralDensity = [exampleData.wavelength]


optimizable.initialProbe = 'circ'
exampleData.entrancePupilDiameter = exampleData.Np / 3 * exampleData.dxp  # initial estimate of beam
optimizable.initialObject = 'ones'
# initialize probe and object and related params
optimizable.prepare_reconstruction()

# customize initial probe quadratic phase
fullPeriod = 6*13.5e-6
apertureSize = 4*13.5e-6
WFS = 0*exampleData.Xp
n = 9
R, C = GenerateRasterGrid(n, round(fullPeriod/exampleData.dxp))
print('WFS size: %d um'%( 2 * max(max(abs(C)), max(abs(R)))*exampleData.dxp * 1e6))
print('WFS size: %d um'%((max( max(R) - min(R), max(C) - min(C)) *exampleData.dxp + apertureSize) * 1e6))
R = R+exampleData.Np//2
C = C+exampleData.Np//2
 
# rng(0,'v5uniform')
# for k = 1:len(R):
#     R[k] = R[k] + randi(2) - 2
#     C[k] = C[k] + randi(2) - 2
#
for k in np.arange(len(R)):
    WFS[R[k], C[k]] = 1

subaperture = rect(exampleData.Xp / apertureSize) * rect(exampleData.Yp / apertureSize)
WFS = abs(ifft2c(fft2c(WFS) * fft2c(subaperture)))
WFS = WFS/np.max(WFS)

optimizable.probe[..., :, :] = rotate(WFS.conj(), np.arctan(11/256), reshape=False, mode='nearest')
optimizable.object[..., :, :] = np.exp(-(exampleData.Xo**2 + exampleData.Yo**2)/(2*(3e-3/2.355)**2))\
                                * np.exp(1.j * np.arctan2(exampleData.Yo, exampleData.Xo))

# # initial guess for HHG
# i = 0
# for harmonicNumber in np.arange(19, 33, 2):
#     optimizable.object[i, ..., :, :] = np.exp(-(exampleData.Xo**2 + exampleData.Yo**2)/(2*(3e-3/2.355)**2)) \
#                                        * np.exp(1.j * np.arctan2(exampleData.Yo, exampleData.Xo )* harmonicNumber)
#     i = i+1

optimizable.object = optimizable.object / np.linalg.norm(optimizable.object, axis=(-1, -2)) \
                     * np.linalg.norm(exampleData.ptychogram[0], axis=(-1, -2))
hsvplot(np.squeeze(optimizable.probe))
hsvplot(np.squeeze(optimizable.object))

# this will copy any attributes from experimental data that we might care to optimize
# # Set monitor properties
monitor = Monitor()
monitor.figureUpdateFrequency = 1
monitor.objectPlot = 'complex'  # complex abs angle
monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure

# Run the reconstruction
# multiPIE
engine = multiPIE.multiPIE_GPU(optimizable, exampleData, monitor)
# engine = multiPIE.multiPIE(optimizable, exampleData, monitor)

## main parameters
engine.numIterations = 1000
engine.positionOrder = 'random'  # 'sequential' or 'random'
engine.propagator = 'scaledPolychromeASP'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP
engine.betaProbe = 0.25
engine.betaObject = 0.75

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

engine.doReconstruction()


# now save the data
# optimizable.saveResults('reconstruction.hdf5')
