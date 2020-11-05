import matplotlib
matplotlib.use('tkagg')
from fracPy.io import getExampleDataFolder


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

#os.chdir(filePath)

exampleData.loadData(filePath)  # simuRecent  Lenspaper

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
# optimizable.probe = optimizable.probe*np.exp(1.j*2*np.pi/exampleData.wavelength *
#                                              (exampleData.Xp**2+exampleData.Yp**2)/(2*6e-3))
# hsvplot(np.squeeze(optimizable.probe))

# this will copy any attributes from experimental data that we might care to optimize
# # Set monitor properties
monitor = Monitor()
monitor.figureUpdateFrequency = 10
monitor.objectPlot = 'complex'  # complex abs angle
monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure

# exampleData.zo = exampleData.zo
# exampleData.spectralDensity=[exampleData.wavelength]
# exampleData.dxp = exampleData.dxp/1
# Run the reconstruction
# multiPIE
engine = multiPIE.multiPIE_GPU(optimizable, exampleData, monitor)
# engine = multiPIE.multiPIE(optimizable, exampleData, monitor)


## main parameters
engine.numIterations = 100
engine.positionOrder = 'random'  # 'sequential' or 'random'
engine.propagator = 'scaledPolychromeASP'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP
engine.betaProbe = 0.25
engine.betaObject = 0.25

## engine specific parameters:
engine.zPIEgradientStepSize = 100  # gradient step size for axial position correction (typical range [1, 100])

## switches
engine.probePowerCorrectionSwitch = True
engine.modulusEnforcedProbeSwitch = False
engine.comStabilizationSwitch = True
engine.orthogonalizationSwitch = True
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
