'''
Please download first OPR_dp_processed.hdf5 from https://figshare.com/search?q=ptylab
This dataset contains diffraction pattern of a filamentous fungus. During the measurement the illumination drifted. In this tutorial we will use a combination of mixed states and orthogonal probe relaxation to still achieve a high image quality.
Please download the file OPR_dp_processed.hdf5 from https://figshare.com/search?q=ptylab
'''

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import PtyLab 
from PtyLab.io import getExampleDataFolder
from PtyLab import Engines
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np

# Location of the preprocessed data file
filePath = '../example_data/OPR_dp_processed.hdf5'
experimentalData, reconstruction, params, monitor, ePIE_engine = PtyLab.easyInitialize(filePath, operationMode='CPM')

# First set the experimental geometry
# The orientation accounts for the direction of the sample postitioners and 
# orientation of the diffraction pattern array
experimentalData.setOrientation(1)
experimentalData.operationMode = 'CPM'
experimentalData.zo = 31.0e-3 # Sample - Detector distance
experimentalData.dxd = 2 * 13.5e-6 # Detector pixel size (2x2 binning)
experimentalData.wavelength = 13.5e-9 # Wavelength
experimentalData._setData()
reconstruction.copyAttributesFromExperiment(experimentalData)
reconstruction.computeParameters()

reconstruction.No = 2000 # Array size of the object array
reconstruction.npsm = 6 # Number of probe modes to reconstruct::
reconstruction.nosm = 1 # Number of object modes to reconstruct
reconstruction.nlambda = 1 # len(exampleData.spectralDensity) # Number of wavelength
reconstruction.nslice = 1 # Number of object slice

# Type of the initial probe
reconstruction.initialProbe = 'circ'
experimentalData.entrancePupilDiameter = 10e-6

# Use simple ones for the initial object
reconstruction.initialObject = 'ones'
reconstruction.initializeObjectProbe()

# Set monitor properties
monitor.figureUpdateFrequency = 5
monitor.objectPlot = 'complex'  # complex abs angle
monitor.verboseLevel = 'low'  # high: plot two figures, low: plot only one figure
monitor.objectZoom = 2   # control object plot FoV
monitor.probeZoom = 0.1   # control probe plot FoV

# params = Reconstruction_parameters()
## main parameters
params.positionOrder = 'random'  # 'sequential' or 'random'
params.propagator = 'Fraunhofer'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP

## how do we want to reconstruct?
params.gpuSwitch = True # Assuming that a GPU is available 
params.probePowerCorrectionSwitch = False 
params.modulusEnforcedProbeSwitch = False
params.comStabilizationSwitch = False 
params.orthogonalizationSwitch = True 
params.orthogonalizationFrequency = 10
params.fftshiftSwitch = False
params.intensityConstraint = 'standard'  # standard fluctuation exponential poission
params.absorbingProbeBoundary = True 
params.objectContrastSwitch = False
params.absObjectSwitch = True 
params.absObjectBeta = 1e-2
params.backgroundModeSwitch = False
params.couplingSwitch = False 
params.couplingAleph = 1
params.positionCorrectionSwitch = False
params.referenceAreaSwitch = False 
params.referenceAreaBeta = 0.1
params.positionCorrectionSwitch = False
params.TV_lam = 5e-6 

# OPR specific parameters
params.OPR_modes = np.array([0, 1, 2]) # Mixed states modes that are used for the OPR
params.n_subspace = 5 # number of singular values that are kept 
params.OPR_tsvd_type = 'randomized' # activates randomized OPR which is faster

# First run a view mPIE iterations
engine_mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
engine_mPIE.numIterations = 25
engine_mPIE.betaProbe = 0.25
engine_mPIE.betaObject = 0.25
engine_mPIE.reconstruct()

# Now start the OPR
engine_OPR = Engines.OPR(reconstruction, experimentalData, params, monitor)
engine_OPR.numIterations = 100 
engine_OPR.betaProbe = 0.99
engine_OPR.betaObject = 0.99
engine_OPR.reconstruct()

# Save test results
reconstruction.saveResults(fileName="recons/test")
# This saves the probe stack (be care full, this array contains a
# probe for each position. Therefore it will require a lot of memory!
reconstruction.saveResults(fileName="recons/test__22", type='probe_stack')
