import matplotlib
matplotlib.use('tkagg')
#matplotlib.use('qt5agg')
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines import ePIE, mPIE, qNewton
from fracPy.monitors.Monitor import Monitor as Monitor
import logging
logging.basicConfig(level=logging.INFO)
from fracPy.utils.utils import ifft2c
from matplotlib import pyplot as plt
import numpy as np
from fracPy.Optimizable.Calibration import IlluminationCalibration

""" 
FPM data reconstructor 
change data visualization and initialization options manually for now
"""

# create an experimentalData object and load a measurement
exampleData = ExperimentalData()
exampleData.loadData('example:simulation_fpm')
exampleData.operationMode = 'FPM'

# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
optimizable = Optimizable(exampleData)
optimizable.npsm = 1 # Number of probe modes to reconstruct
optimizable.nosm = 1  # Number of object modes to reconstruct
optimizable.nlambda = 1  # Number of wavelength
optimizable.prepare_reconstruction()

truePositions = optimizable.positions
trueDiameter = exampleData.entrancePupilDiameter

# translate the positions by 5 pixels
optimizable.positions[:,0] = optimizable.positions[:,0] + 5
optimizable.positions[:,1] = optimizable.positions[:,1] - 3
# change the aperture diameter by 15 pixels (5 x pixel size)
exampleData.entrancePupilDiameter = exampleData.entrancePupilDiameter + 15 * exampleData.dxp
wrongDiameter = exampleData.entrancePupilDiameter


# now see if we can recover that by FPM calibration
calibration = IlluminationCalibration(optimizable, exampleData)
calibration.fit_mode = 'Translation' # position transformation model
calibration.gaussSigma = 3 # sigma used to low-pass filter data, crucial for noisy spectra, usually 2-3
calibration.searchGridSize = 10 # search grid size
calibration.plot = True
calibration.calibrateRadius = True
# update the positions within the optimizable
newPositions, newDiameter, matrixCalib = calibration.runCalibration()


# compare results
print("True diameter", trueDiameter)
print("Wrong diameter", wrongDiameter)
print("Calibrated diameter", newDiameter)
print("Position translation", matrixCalib.translation)
print("Position rotation", matrixCalib.rotation)
# print("Position scale", matrixCalib.scale)
