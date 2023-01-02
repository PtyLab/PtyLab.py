from pathlib import Path

import matplotlib

from PtyLab.Engines.BaseEngine import smooth_amplitude

matplotlib.use("tkagg")
import PtyLab
from PtyLab import Reconstruction
from PtyLab.io import getExampleDataFolder
from PtyLab import Engines
import logging

logging.basicConfig(level=logging.INFO)
import numpy as np


""" 
ptycho data reconstructor 
change data visualization and initialization options manually for now
"""
# replace this with the path of the helical beam data. Can be downloaded from
# Loetgering, Lars, et al. "Generation and characterization of focused helical x-ray beams." Science advances 6.7 (2020): eaax8836. 
# https://figshare.com/articles/dataset/PtyLab_helical_beam_data/21671516/1
# place in 'helicalbeam.h5'
fileName = "example:helicalbeam"  # simu.hdf5 or Lenspaper.hdf5

experimentalData, reconstruction, params, monitor, ePIE_engine = PtyLab.easyInitialize(
    fileName, operationMode="CPM"
)
# experimentalData.ptychogram = experimentalData.ptychogram - 3
# experimentalData.ptychogram = np.clip(experimentalData.ptychogram, 0, None)

# I have to truncate on my laptop as I only have 4GB of RAM, but it doesn't seem to affect the reconstruction quality much
experimentalData.ptychogram = experimentalData.ptychogram[::3]
experimentalData.encoder = experimentalData.encoder[::3]


experimentalData.setOrientation(4) #4 # 6
experimentalData._setData()

reconstruction = Reconstruction(experimentalData, params)

monitor.screenshot_directory = './screenshots'

reconstruction.entrancePupilDiameter = reconstruction.Np/3 * reconstruction.dxp
params.objectSmoothenessSwitch = True
params.objectSmoothenessWidth = 2
params.objectSmoothnessAleph = 1e-2

params.probePowerCorrectionSwitch = True
params.comStabilizationSwitch = 1
params.propagatorType = 'Fraunhofer'
params.fftshiftSwitch = True



# optional - use tensorboard monitor instead. To see the results, open tensorboard in the directory ./logs_tensorboard
# from PtyLab.Monitor.TensorboardMonitor import TensorboardMonitor
# monitor = TensorboardMonitor('./logs')

params.l2reg = False


params.positionCorrectionSwitch = False

# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
reconstruction.npsm = 4  # Number of probe modes to reconstruct
reconstruction.nosm = 1  # Number of object modes to reconstruct
reconstruction.nlambda = (
    1  # len(experimentalData.spectralDensity) # Number of wavelength
)
reconstruction.nslice = 1  # Number of object slice


reconstruction.initialProbe = "circ"
reconstruction.initialObject = "ones"
# initialize probe and object and related Params
reconstruction.initializeObjectProbe()


reconstruction.describe_reconstruction()


monitor.figureUpdateFrequency = 1

monitor.objectPlot = "abs"  # 'complex'  # complex abs angle
monitor.verboseLevel = "low"  # high: plot two figures, low: plot only one figure

# Run the reconstruction

## main parameters
params.positionOrder = "random"  # 'sequential' or 'random'
params.propagatorType = "Fraunhofer"#ScaledASP" #Fraunhofer"  # Fresnel'# 'Fresnel' #aunhofer'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP

params.positionCorrectionSwitch = False

params.orthogonalizationSwitch = True
# orthogonalize every ten iterations
params.orthogonalizationFrequency = 10

# differences with lars' implementation
params.absorbingProbeBoundary = False
params.absorbingProbeBoundaryAleph =0.8#1e-1
params.saveMemory = True
monitor.objectZoom = 0.8  # 0.5  # control object plot FoV
monitor.probeZoom = None # 0.5   # control probe plot FoV
params.l2reg_object_aleph = 1e-1



params.objectContrastSwitch = False
params.absObjectSwitch = False
params.backgroundModeSwitch = False
params.couplingSwitch = True
params.couplingAleph = 1
params.positionCorrectionSwitch = False
params.probePowerCorrectionSwitch = True


## computation stuff - how do we want to reconstruct?
params.gpuSwitch = True
# this speeds up some propagators but not all of them are implemented
params.fftshiftSwitch = True


params.intensityConstraint = "standard"#standard"  # standard fluctuation exponential poission

monitor.describe_parameters(params)
#reconstruction.object *= 1
## choose mqNewton engine
mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
# This is according to the paper, as they're not explicitly set in the matlab script I did not set them
# mPIE.numIterations = 25
mPIE.betaProbe = 0.25
mPIE.betaObject = 0.25
# mPIE.frictionM = 0.9
# mPIE.feedbackM = 0.1

# estimate the intensity roughly
# you can now run simple scripts, such as:
mPIE.numIterations = 20
params.l2reg = True
mPIE.reconstruct(experimentalData, reconstruction)

params.l2reg = False
params.comStabilizationSwitch = False
mPIE.reconstruct(experimentalData, reconstruction)

# to check - I don't have enough memory on my laptop
from PtyLab.Engines import OPR_TV
OPR = OPR_TV(reconstruction, experimentalData, params, monitor)
OPR.numIterations = 1000
params.OPR_modes = np.array([0,1])
params.n_subspace = 4


OPR.reconstruct()
