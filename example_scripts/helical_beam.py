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

# upsample??
experimentalData.dxd = experimentalData.dxd / 2
# alternative fourier space upsampling - slow
# padwidth = [[0,0], [padwidth, padwidth], [padwidth,padwidth]]
    # dataset['ptychogram'] = abs(ifft2c(np.pad(fft2c(dataset['ptychogram']), padwidth))).astype(np.float32)
experimentalData.ptychogram = np.repeat(np.repeat(experimentalData.ptychogram, axis=-2, repeats=2), axis=-1, repeats=2)

experimentalData._setData()

reconstruction = Reconstruction(experimentalData, params)

# optional - use tensorboard monitor instead. To see the results, open tensorboard in the directory ./logs_tensorboard
# from PtyLab.Monitor.TensorboardMonitor import TensorboardMonitor
# monitor = TensorboardMonitor('./logs')

# turn these two lines on to see the autofocusing in action
experimentalData.zo = experimentalData.zo - 1e-2
reconstruction.zo = reconstruction.zo - 1e-2

# set this to >1 for larger images
monitor.downsample_everything = 2
monitor.probe_downsampling = 1

# optional - customize orientation (you usually don't need this)
experimentalData.setOrientation(0)

# optional: try to reload the last probe. This can really help convergence, especially for larger probes
reload_probe = False
reload_object = False

# change the number of pixels in the object
reconstruction.No = int(reconstruction.No * 0.9)


params.TV_autofocus = False
params.TV_autofocus_stepsize = 50
params.TV_autofocus_intensityonly = True
params.TV_autofocus_what = "object"
params.TV_autofocus_metric = "TV"
params.TV_autofocus_roi = [[0.3, 0.7], [0.3, 0.7]]  # [[0.45, 0.5], [0.6, 0.65]]
# optional - set the minimum and maximum propagation distance to something reasonable.
# to disable, set to None
params.TV_autofocus_min_z = experimentalData.zo - 2e-2
params.TV_autofocus_max_z = experimentalData.zo + 2e-2


params.l2reg = False
params.l2reg_probe_aleph = 1e-2
params.l2reg_object_aleph = 1e-2


params.positionCorrectionSwitch = False

# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
reconstruction.npsm = 2  # Number of probe modes to reconstruct
reconstruction.nosm = 1  # Number of object modes to reconstruct
reconstruction.nlambda = (
    1  # len(experimentalData.spectralDensity) # Number of wavelength
)
reconstruction.nslice = 1  # Number of object slice


reconstruction.initialProbe = "circ"
print(reconstruction.entrancePupilDiameter)
reconstruction.initialObject = "ones"
# initialize probe and object and related Params
reconstruction.initializeObjectProbe()

# customize initial probe quadratic phase
# reconstruction.initialProbe = reconstruction.initialProbe.astype(np.complex64)
reconstruction.probe *= np.exp(
    1.0j
    * 2
    * np.pi
    / (reconstruction.wavelength * reconstruction.zo * 2)
    * (reconstruction.Xp**2 + reconstruction.Yp**2)
    / 2
)
initial_probe = reconstruction.probe.copy()
reconstruction.object *= 0.3

if reload_probe:
    try:
        print("reloading last probe")
        reconstruction.load_probe("last.hdf5")
    except (FileNotFoundError, RuntimeError):
        print("Cannot re-use last probe")

if reload_object:
    try:
        reconstruction.load_object("last.hdf5")
    except (FileNotFoundError, RuntimeError):
        print("Cannot re-use last probe")


reconstruction.describe_reconstruction()


monitor.figureUpdateFrequency = 1
monitor.objectPlot = "complex"  # 'complex'  # complex abs angle
monitor.verboseLevel = "high"  # high: plot two figures, low: plot only one figure
monitor.objectZoom = None  # 0.5  # control object plot FoV
monitor.probeZoom = None  # 0.5   # control probe plot FoV

# Run the reconstruction

# Params = Params()
## main parameters
params.positionOrder = "random"  # 'sequential' or 'random'
params.propagatorType = "Fresnel"#ScaledASP" #Fraunhofer"  # Fresnel'# 'Fresnel' #aunhofer'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP

params.positionCorrectionSwitch = False
params.modulusEnforcedProbeSwitch = False

params.probeSmoothenessSwitch = True
params.probeSmoothnessAleph = 1e-2
params.probeSmoothenessWidth = 10

params.orthogonalizationSwitch = True
# orthogonalize every ten iterations
params.orthogonalizationFrequency = 10
params.absorbingProbeBoundary = True
params.absorbingProbeBoundaryAleph = 1e-1

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
params.fftshiftSwitch = False


params.intensityConstraint = "standard"  # standard fluctuation exponential poission

monitor.describe_parameters(params)

## choose mqNewton engine
mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
mPIE.numIterations = 25
mPIE.betaProbe = 0.25
mPIE.betaObject = 0.25
mPIE.frictionM = 0.9
mPIE.feedbackM = 0.1
params.comStabilizationSwitch = mPIE.numIterations // 3 + 1

# you can now run simple scripts, such as:
for i in range(8):
    # turn on autofocusing and then l2 regularization, iterate both for about 50 iterations
    params.TV_autofocus = i % 2 == 1
    params.l2reg = i % 2 == 0

    mPIE.reconstruct(experimentalData, reconstruction)
    reconstruction.saveResults("last.hdf5", squeeze=False)
