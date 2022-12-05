from pathlib import Path

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

fileName = 'example:simulation_cpm'   # simu.hdf5 or Lenspaper.hdf5
# filename = r"Preprocessedf7.hdf5"
# filePath = Path(r"C:\Users\dbs2\Documents\projects\ptycho\sobhi") / filename
#r";#getExampleDataFolder() / fileName


# filePath = r'C:\Users\dbs660\PycharmProjects\ptycho_data_analysis\scripts\test.hdf5'
# filePath = r'/home/dbs660/PycharmProjects/ptycho_data_analysis/scripts/test.hdf5'
# filePath = r'/home/dbs660/Desktop/daniel/31012022_div_1.hdf5'
# from fracPy.Monitor.TensorboardMonitor import TensorboardMonitor


experimentalData, reconstruction, params, monitor, ePIE_engine = fracPy.easyInitialize(fileName, operationMode='CPM')
# experimentalData.setOrientation(5)
# reconstruction.No *= 1.2
reconstruction.No = int(reconstruction.No)

# print(experimentalData.encoder.max(), experimentalData.encoder.min())
# random_error_max = 0.00005*0
# reconstruction.positions
# experimentalData.encoder += random_error_max * (np.random.random(experimentalData.encoder.shape)-0.5)
# from fracPy import Reconstruction
# reconstruction = Reconstruction(experimentalData, params)
# reconstruction.copyAttributesFromExperiment(experimentalData)
# casdcsad

# experimentalData.encoder0 -=
# print(reconstruction.No)
# experimentalData.zo = 20e-3
# reconstruction.zo = experimentalData.zo

# monitor = TensorboardMonitor()
params.TV_autofocus = False
params.TV_autofocus_stepsize = 10
params.TV_autofocus_intensityonly=False


params.fftshiftFlag = False
params.l2reg = False
params.l2reg_probe_aleph = 1e-2
params.l2reg_object_aleph = 1e-2


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
# reconstruction.entrancePupilDiameter = reconstruction.Np / 3 * reconstruction.dxp  # initial estimate of beam
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
monitor.objectPlot = 'complex'# 'complex'  # complex abs angle
monitor.verboseLevel = 'low'  # high: plot two figures, low: plot only one figure
monitor.objectZoom = 0.5  # control object plot FoV
monitor.probeZoom = 0.5   # control probe plot FoV

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

params.probeSmoothenessSwitch = False
params.probeSmoothnessAleph = 1e-2
params.probeSmoothenessWidth = 10



## how do we want to reconstruct?
params.gpuSwitch = True
params.probePowerCorrectionSwitch = True
params.comStabilizationSwitch = 3
params.orthogonalizationSwitch = True
params.orthogonalizationFrequency = 10

params.fftshiftSwitch = True
params.intensityConstraint = 'standard'  # standard fluctuation exponential poission
params.absorbingProbeBoundary = False
params.absorbingProbeBoundaryAleph = 1e-1

params.objectContrastSwitch = True
params.absObjectSwitch = False
params.backgroundModeSwitch = False
params.couplingSwitch = True
params.couplingAleph = 1
params.positionCorrectionSwitch = False

params.OPRP = False


monitor.describe_parameters(params)

## choose mqNewton engine
mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)

mPIE.numIterations = 30
mPIE.reconstruct(experimentalData, reconstruction)

params.OPRP = False
mPIE.numIterations = 30
mPIE.reconstruct(experimentalData, reconstruction)
mPIE.comStabilization()
mPIE.reconstruct(experimentalData, reconstruction)

# first get some idea
params.TV_autofocus = False
params.comStabilizationSwitch = False
# mqNewton.reconstruct()

np.savez('object.npz',  object=reconstruction.object, dxo=reconstruction.dxo,
             wavelength=reconstruction.wavelength)


params.comStabilizationSwitch = False

mPIE.numIterations = 15
mPIE.reconstruct()
np.savez('object.npz',  object=reconstruction.object, dxo=reconstruction.dxo,
             wavelength=reconstruction.wavelength)

# now try to focus
params.TV_autofocus = False
mPIE.numIterations = 100
params.comStabilizationSwitch = False
for i in range(100):
    mPIE.reconstruct()
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
