# matplotlib.use("tkagg")
import logging
from pathlib import Path

import matplotlib.pyplot as plt

import PtyLab
from PtyLab import Engines
from PtyLab.Engines.BaseEngine import smooth_amplitude
from PtyLab.io import getExampleDataFolder

logging.basicConfig(level=logging.INFO)
import argparse

import numpy as np

""" 
ptycho data reconstructor 
change data visualization and initialization options manually for now
"""

# set argparser options
parser = argparse.ArgumentParser(description="Conventional ptychography reconstruction")
parser.add_argument(
    "--file",
    type=str,
    help="Path to the file",
    default=f"{getExampleDataFolder()}/simu.hdf5",
)
parser.add_argument("--gpu", action="store_true", help="GPU switch")
# Parse the arguments from the command line
args = parser.parse_args()
fileName = args.file
gpu_switch = args.gpu

# load the experimental data
experimentalData, reconstruction, params, monitor, ePIE_engine = PtyLab.easyInitialize(
    fileName, operationMode="CPM"
)
# optional - use tensorboard monitor instead. To see the results, open tensorboard in the directory ./logs_tensorboard
from PtyLab.Monitor.TensorboardMonitor import TensorboardMonitor

monitor = TensorboardMonitor()

# turn these two lines on to see the autofocusing in action
# experimentalData.zo = experimentalData.zo + 1e-2
# reconstruction.zo = reconstruction.zo + 1e-2

# set this to >1 for larger images
monitor.downsample_everything = 2
monitor.probe_downsampling = 1

# optional - customize orientation (you usually don't need this)
# experimentalData.setOrientation(3)

# optional: try to reload the last probe. This can really help convergence, especially for larger probes
reload_probe = False
reload_object = False

# change the number of pixels in the object
reconstruction.No = int(reconstruction.No * 0.9)

params.TV_autofocus = False
params.TV_autofocus_stepsize = 50
params.TV_autofocus_intensityonly = False
params.TV_autofocus_what = "object"
params.TV_autofocus_metric = "TV"
params.TV_autofocus_roi = [[0.3, 0.7], [0.3, 0.7]]  # [[0.45, 0.5], [0.6, 0.65]]
# optional - set the minimum and maximum propagation distance to something reasonable.
# to disable, set to None
params.TV_autofocus_min_z = experimentalData.zo - 2e-2
params.TV_autofocus_max_z = experimentalData.zo + 5e-2

params.l2reg = False
params.l2reg_probe_aleph = 1e-2
params.l2reg_object_aleph = 1e-2

params.positionCorrectionSwitch = False

# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
reconstruction.npsm = 1  # Number of probe modes to reconstruct
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
monitor.verboseLevel = "low"  # high: plot two figures, low: plot only one figure
monitor.objectZoom = None  # 0.5  # control object plot FoV
monitor.probeZoom = None  # 0.5   # control probe plot FoV

# Run the reconstruction

# Params = Params()
## main parameters
params.positionOrder = "random"  # 'sequential' or 'random'
params.propagatorType = "ASP"  # Fresnel'# 'Fresnel' #aunhofer'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP

params.positionCorrectionSwitch = False
params.modulusEnforcedProbeSwitch = False

params.probeSmoothenessSwitch = True
params.probeSmoothnessAleph = 1e-2
params.probeSmoothenessWidth = 10
params.comStabilizationSwitch = 10
params.orthogonalizationSwitch = True
# orthogonalize every ten iterations
params.orthogonalizationFrequency = 10
params.absorbingProbeBoundary = False
params.absorbingProbeBoundaryAleph = 1e-1

params.objectContrastSwitch = False
params.absObjectSwitch = False
params.backgroundModeSwitch = False
params.couplingSwitch = True
params.couplingAleph = 1
params.positionCorrectionSwitch = False
params.probePowerCorrectionSwitch = True

## computation stuff - how do we want to reconstruct?
params.gpuSwitch = gpu_switch
# this speeds up some propagators but not all of them are implemented
params.fftshiftSwitch = False

params.intensityConstraint = "standard"  # standard fluctuation exponential poission

monitor.describe_parameters(params)

## choose mqNewton engine
mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
mPIE.numIterations = 50

# you can now run simple scripts, such as:
for i in range(1, 8):
    # turn on autofocusing and then l2 regularization, iterate both for about 50 iterations
    params.TV_autofocus = i % 2 == 1
    params.l2reg = i % 2 == 0

    mPIE.reconstruct(experimentalData, reconstruction)
    reconstruction.saveResults(f"{getExampleDataFolder()}/recon.hdf5", squeeze=False)

plt.show()
