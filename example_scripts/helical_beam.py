from pathlib import Path
import pathlib
import matplotlib

from PtyLab.read_write.readExample import examplePath

try:
    matplotlib.use("tkagg") # this is for people using pycharm pro
except:
    pass
import PtyLab
from PtyLab import Reconstruction
from PtyLab import Engines
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

""" 
ptycho data reconstructor 
change data visualization and initialization options manually for now
"""
# To run this, download the helical beam data from
# https://figshare.com/articles/dataset/PtyLab_helical_beam_data/21671516/1
# and place it in the example_data folder.
# This data is part of the following publication:
# Loetgering, Lars, et al. "Generation and characterization of focused helical x-ray beams." Science advances 6.7 (2020): eaax8836.

fileName = "example:helicalbeam"  # simu.hdf5 or Lenspaper.hdf5
# check if the file exists, download it otherwise
from PtyLab.utils.downloader import download_with_progress

fileName = examplePath(fileName)
if not pathlib.Path(fileName).exists():
    print('Downloading dataset...')
    download_with_progress('https://figshare.com/ndownloader/files/38419391', filename=fileName)

experimentalData, reconstruction, params, monitor, ePIE_engine = PtyLab.easyInitialize(
        fileName, operationMode="CPM"
    )

# This dataset is heavily oversampled. If you are short on memory or want to see a result a bit faster, turn this on.
# experimentalData.ptychogram = experimentalData.ptychogram[::3]
# experimentalData.encoder = experimentalData.encoder[::3]
# experimentalData._setData()

experimentalData.setOrientation(4)  # this corresponds to orientation 1


reconstruction = Reconstruction(experimentalData, params)

monitor.screenshot_directory = "./screenshots"

reconstruction.entrancePupilDiameter = reconstruction.Np / 3 * reconstruction.dxp
params.objectSmoothenessSwitch = True
params.objectSmoothenessWidth = 2
params.objectSmoothnessAleph = 1e-2

params.probePowerCorrectionSwitch = True
params.comStabilizationSwitch = 1
params.propagatorType = "Fresnel"  # ASP'#Fraunhofer'
params.fftshiftSwitch = params.propagatorType in ["Fresnel", "Fraunhofer"]
params.l2reg = False
params.positionCorrectionSwitch = False
params.positionOrder = "weigh_by_normalized_error"  # 'sequential', 'random' or 'weigh_by_error'
params.positionCorrectionSwitch = False
params.orthogonalizationSwitch = True
# orthogonalize every ten iterations
params.orthogonalizationFrequency = 10
params.intensityConstraint = "standard"

# optional - use tensorboard monitor instead. To see the results, open tensorboard in the directory ./logs_tensorboard
from PtyLab.Monitor.TensorboardMonitor import TensorboardMonitor
monitor = TensorboardMonitor('./logs')

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

## Visualization
monitor.figureUpdateFrequency = 1
monitor.objectPlot = "abs"  # 'complex'  # complex abs angle
monitor.verboseLevel = "low"  # high: plot two figures, low: plot only one figure
pathlib.Path("./screenshots").mkdir(exist_ok=True)


# Run the reconstruction

## main parameters


# differences with lars' implementation
params.absorbingProbeBoundary = False
params.absorbingProbeBoundaryAleph = 0.8  # 1e-1
params.saveMemory = True
monitor.objectZoom = 1.5  # 0.5  # control object plot FoV
monitor.probeZoom = None  # 0.5   # control probe plot FoV
params.l2reg_object_aleph = 1e-1


params.objectContrastSwitch = False
params.absObjectSwitch = False
params.backgroundModeSwitch = False
params.couplingSwitch = True
params.couplingAleph = 1
params.positionCorrectionSwitch = False
params.probePowerCorrectionSwitch = True
params.gpuSwitch = True


monitor.describe_parameters(params)
# reconstruction.object *= 1
## choose mqNewton engine
mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
# This is according to the paper, as they're not explicitly set in the matlab script I did not set them
# mPIE.numIterations = 25
mPIE.betaProbe = 0.25
mPIE.betaObject = 0.25
# mPIE.frictionM = 0.9
# mPIE.feedbackM = 0.1


mPIE.numIterations = 5
params.l2reg = True
mPIE.reconstruct(experimentalData, reconstruction)

params.l2reg = False
params.comStabilizationSwitch = False
mPIE.reconstruct(experimentalData, reconstruction)

# # to check - I don't have enough memory on my laptop
from PtyLab.Engines import OPR_TV

OPR = OPR_TV(reconstruction, experimentalData, params, monitor)
OPR.numIterations = 1000
params.OPR_modes = np.array([0, 1])
params.n_subspace = 4
OPR.alpha = 0.25
OPR.betaProbe = 0.25
OPR.betaObject = 0.25
#
#
OPR.reconstruct()
