import matplotlib

from fracPy.Engines.BaseEngine import smooth_amplitude

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
# filePath = r'C:\Users\dbs660\PycharmProjects\ptycho_data_analysis\scripts\test.hdf5'
filePath = r'/home/dbs660/PycharmProjects/ptycho_data_analysis/scripts/test.hdf5'
filePath = r'/home/dbs660/Desktop/daniel/31012022_div_1.hdf5'
from fracPy.Monitor.TensorboardMonitor import TensorboardMonitor


experimentalData, reconstruction, params, monitor, ePIE_engine = fracPy.easyInitialize(filePath, operationMode='CPM')
reconstruction.No *= 1.2
reconstruction.No = int(reconstruction.No)

experimentalData.encoder0 -=
# print(reconstruction.No)
# experimentalData.zo = 20e-3
# reconstruction.zo = experimentalData.zo

monitor = TensorboardMonitor()
params.TV_autofocus = True
params.TV_autofocus_stepsize = 10
params.TV_autofocus_intensityonly=False


params.fftshiftFlag = False
params.l2reg = True
params.l2reg_probe_aleph = 1e-1
params.l2reg_object_aleph = 1e-3


params.positionCorrectionSwitch = False
# show_alignment(reconstruction, experimentalData, params, ePIE_engine)
# experimentalData.encoder = -np.fliplr(experimentalData.encoder)
# experimentalData.ptychogram = experimentalData.ptychogram[...,::2,::2]
# experimentalData.xd *= 2
# from fracPy import Reconstruction
# reconstruction: Reconstruction = Reconstruction(experimentalData, params)
## altternative
# experimentalData = ExperimentalData()
# experimentalData.loadData(filePath)
# experimentalData.operationMode = 'CPM'
# experimentalData.showPtychogram()
# almost worked
# experimentalData.zo = 134.7e-3 #90e-3#80.07e-3
# reconstruction.zo = experimentalData.zo


# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
reconstruction.npsm = 4 # Number of probe modes to reconstruct
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
#reconstruction.initialProbe = reconstruction.initialProbe.astype(np.complex64)
# reconstruction.probe *= np.exp(1.j * 2 * np.pi / (reconstruction.wavelength * reconstruction.zo*2 ) *
#                                                     (reconstruction.Xp ** 2 + reconstruction.Yp ** 2))
initial_probe = reconstruction.probe.copy()



reconstruction.describe_reconstruction()
# this will copy any attributes from experimental data that we might care to optimize
# # Set monitor properties
# monitor = Monitor()
monitor.figureUpdateFrequency = 2
monitor.objectPlot = 'complex'  # complex abs angle
monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure
monitor.objectZoom = 0.01  # control object plot FoV
monitor.probeZoom = 0.01#0.5   # control probe plot FoV

# Run the reconstruction

# Params = Params()
## main parameters
params.positionOrder = 'random'  # 'sequential' or 'random'
params.propagatorType = 'Fresnel'# 'Fresnel' #aunhofer'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP
params.fftshiftSwitch = True
params.fftshiftFlag = False
params.positionCorrectionSwitch = False
params.modulusEnforcedProbeSwitch = False

# params.objectSmoothenessSwitch = False
# params.objectSmoothnessAleph = 1e-2
# params.objectSmoothenessWidth = 2

params.probeSmoothenessSwitch = True
params.probeSmoothnessAleph = 1e-2
params.probeSmoothenessWidth = 10



## how do we want to reconstruct?
params.gpuSwitch = True
params.probePowerCorrectionSwitch = True

params.comStabilizationSwitch = True
params.orthogonalizationSwitch = True
params.orthogonalizationFrequency = 10

# params.fftshiftSwitch = True
params.intensityConstraint = 'standard'  # standard fluctuation exponential poission
params.absorbingProbeBoundary = True
params.absorbingProbeBoundaryAleph = 1e-1

params.objectContrastSwitch = False
params.absObjectSwitch = False
params.backgroundModeSwitch = False
params.couplingSwitch = False
params.couplingAleph = 1
params.positionCorrectionSwitch = False



monitor.describe_parameters(params)

## choose mqNewton engine
mqNewton = Engines.mqNewton(reconstruction, experimentalData, params, monitor)

mqNewton.numIterations = 30
mqNewton.betaProbe = .5
mqNewton.betaObject = 1
mqNewton.beta1 = 0.5
mqNewton.beta2 = 0.5
mqNewton.betaProbe_m = 1
mqNewton.betaObject_m = 1
mqNewton.momentum_method = 'NADAM'

# first get some idea
params.TV_autofocus = True
params.comStabilizationSwitch = True
mqNewton.reconstruct()

np.savez('object.npz',  object=reconstruction.object, dxo=reconstruction.dxo,
             wavelength=reconstruction.wavelength)


params.comStabilizationSwitch = True

mqNewton.numIterations = 15
mqNewton.reconstruct()
np.savez('object.npz',  object=reconstruction.object, dxo=reconstruction.dxo,
             wavelength=reconstruction.wavelength)

# now try to focus
params.TV_autofocus = False
mqNewton.numIterations = 100
params.comStabilizationSwitch = True
for i in range(100):
    mqNewton.reconstruct()
    np.savez('object.npz', object=reconstruction.object, dxo=reconstruction.dxo,
             wavelength=reconstruction.wavelength)
#
#
#
# #
# # ## choose ePIE engine
# # ePIE = Engines.ePIE(reconstruction, experimentalData, params, monitor)
# # ePIE.numIterations = 5
# # ePIE.betaProbe = 0.25
# # ePIE.betaObject = 0.25
# # ePIE.reconstruct()
# #
# # # ## choose mPIE engine
# mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
#
# mPIE.numIterations = 100
# mPIE.betaProbe = 0.25
# mPIE.betaObject = 0.25
# # mPIE.reconstruct()
# # #
# # ## choose zPIE engine
# zPIE = Engines.zPIE(reconstruction, experimentalData, params, monitor)
# zPIE.focusObject = True
# zPIE.numIterations = 30
# params.l2reg = False
# zPIE.betaProbe = 0.0
# zPIE.betaObject = 0.25
# zPIE.zPIEgradientStepSize = .5  # gradient step size for axial position correction (typical range [1, 100])
# # zPIE.reconstruct()
#
# params.l2reg = False
# # zPIE.reconstruct()
# params.l2reg = True
#
#
# pcPIE = Engines.pcPIE(reconstruction, experimentalData, params, monitor)
# # mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
#
#
# zPIE.numIterations = 500
# mqNewton.numIterations = 150
# pcPIE.numIterations = 150
#
#
# params.positionCorrectionSwitch = True
#
#
# # pcPIE.reconstruct()
#
# pcPIE.reconstruct()
#
#
# for i in range(5):
#     # reconstruction.probe = reconstruction.initialGuessProbe.copy()
#     if i % 3 == 1:
#         params.comStabilizationSwitch = True
#     else:
#         params.comStabilizationSwitch = False
#
#
#     # zPIE.reconstruct(reconstruction=reconstruction)
#     mqNewton.reconstruct()
#     # params.l2reg = False
#     # zPIE.reconstruct()
#     # params.l2reg = True
#     pcPIE.reconstruct()
#
#
#     # reconstruction.saveResults(f'reconstruction_{i}.hdf5', squeeze=True)
#
# reconstruction.saveResults('final_run_before_PC', squeeze=True)
#
# pcPIE.numIterations = 300
# pcPIE.reconstruct()
# mqNewton.reconstruct(experimentalData)
#
# reconstruction.saveResults('final_run_after_PC', squeeze=True)
#
#
# # now save the data
#
