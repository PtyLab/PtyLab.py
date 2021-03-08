import matplotlib
matplotlib.use('tkagg')
from fracPy.io import getExampleDataFolder

#matplotlib.use('qt5agg')
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.Optimizable.CalibrationFPM import IlluminationCalibration
from fracPy.engines import ePIE, mPIE, qNewton, mqNewton, aPIE, aPIE_copy, zPIE, e3PIE, pcPIE
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.Params.Params import Params
from fracPy.monitors.Monitor import Monitor as Monitor
import logging
logging.basicConfig(level=logging.INFO)
from fracPy.utils.utils import ifft2c
from fracPy.utils.visualisation import hsvplot
from matplotlib import pyplot as plt
import numpy as np
import h5py



""" 
ptycho data reconstructor 
change data visualization and initialization options manually for now
"""

exampleData = ExperimentalData()

import os
fileName = 'USAF_calibration_700_bin4.hdf5'  # simu.hdf5 or Lenspaper.hdf5
filePath = getExampleDataFolder() / fileName

exampleData.loadData(filePath)

exampleData.operationMode = 'CPM'


probe2 = h5py.File(r'C:\Users\anned\PycharmProjects\fracpy\example_data\USAF_calibration_700_bin4_probe.hdf5', 'r')

# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
optimizable = Optimizable(exampleData)
optimizable.npsm = 2 # Number of probe modes to reconstruct
optimizable.nosm = 1 # Number of object modes to reconstruct
optimizable.nlambda = 1 # len(exampleData.spectralDensity) # Number of wavelength
optimizable.nslice = 1 # Number of object slice
# optimizable.dxp = optimizable.dxd

optimizable.zo=71.30e-3
optimizable.initialProbe = 'circ'
exampleData.entrancePupilDiameter = optimizable.Np / 3 * optimizable.dxp  # initial estimate of beam
optimizable.initialObject = 'ones'
# initialize probe and object and related params
optimizable.prepare_reconstruction()

# customize initial probe quadratic phase
optimizable.probe = probe2['probe'][()]
optimizable.ptychogram=exampleData.ptychogram
optimizable.theta=44
# this will copy any attributes from experimental data that we might care to optimize
# # Set monitor properties
monitor = Monitor()
monitor.figureUpdateFrequency = 1
monitor.objectPlot = 'complex'  # complex abs angle
monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure
monitor.objectPlotZoom = 3.   # control object plot FoV
monitor.probePlotZoom = 3.   # control probe plot FoV

# Run the reconstruction

params = Params()
## main parameters
params.positionOrder = 'random'  # 'sequential' or 'random'
params.propagator = 'Fraunhofer'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP


## switches
params.gpuSwitch = True
params.probePowerCorrectionSwitch = True
params.modulusEnforcedProbeSwitch = False
params.comStabilizationSwitch = True
params.orthogonalizationSwitch = True
params.orthogonalizationFrequency = 10
params.fftshiftSwitch = True
params.intensityConstraint = 'standard'  # standard fluctuation exponential poission
params.absorbingProbeBoundary = False
params.objectContrastSwitch = False
params.absObjectSwitch = False
params.backgroundModeSwitch = True
params.FourierMaskSwitch =True
params.couplingAleph = 1
params.positionCorrectionSwitch = False

# engine_mPIE = mPIE.mPIE(optimizable, exampleData, params, monitor)
# engine_mPIE.numIterations = 50
# engine_mPIE.betaProbe = 0.5
# engine_mPIE.betaObject = 0.5
# engine_mPIE.doReconstruction()


engine = aPIE.aPIE(optimizable, exampleData, params, monitor)
engine.numIterations = 25
engine.betaProbe = 1
engine.betaObject = 1



engine.doReconstruction()
engine.numIterations = 50
engine.betaProbe_m = 0.5
engine.betaObject_m = 0.5
engine.doReconstruction()

engine.doReconstruction()
engine.numIterations = 100
engine.betaProbe_m = 0.5
engine.betaObject_m = 0.5
engine.doReconstruction()


mPIE

engine_mPIE = mPIE.mPIE(optimizable, exampleData, params, monitor)
engine_mPIE.numIterations = 200
engine_mPIE.betaProbe = 0.25
engine_mPIE.betaObject = 0.25
engine_mPIE.doReconstruction()

## choose engine
# ePIE
# engine_ePIE = ePIE.ePIE(optimizable, exampleData, params,monitor)
# engine_ePIE.numIterations = 50
# engine_ePIE.betaProbe = 0.25
# engine_ePIE.betaObject = 0.25
# engine_ePIE.doReconstruction()

# mPIE


# zPIE
# engine_zPIE = zPIE.zPIE(optimizable, exampleData, params,monitor)
# engine_zPIE.numIterations = 50
# engine_zPIE.betaProbe = 0.35
# engine_zPIE.betaObject = 0.35
# engine_zPIE.zPIEgradientStepSize = 1000  # gradient step size for axial position correction (typical range [1, 100])
# engine_zPIE.doReconstruction()

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
