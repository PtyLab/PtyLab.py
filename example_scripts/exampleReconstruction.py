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

FPM_simulation = True
ptycho_simulation = False


if FPM_simulation:
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

    # calibrate FPM positions
    # calibration = IlluminationCalibration(optimizable, exampleData)
    

    # Set monitor properties
    monitor = Monitor()
    monitor.figureUpdateFrequency = 1
    monitor.objectPlot = 'complex'
    monitor.verboseLevel = 'high'

    # now we want to run an optimizer. First create it.
    # qNewton_engine = qNewton.qNewton(optimizable, exampleData, monitor)
    qNewton_engine = qNewton.qNewton_GPU(optimizable, exampleData, monitor)
    # set any settings involving ePIE in this object.
    qNewton_engine.numIterations = 50
    # now, run the reconstruction
    qNewton_engine.doReconstruction()


""" 
ptycho data reconstructor 
change data visualization and initialization options manually for now
"""
if ptycho_simulation:
    exampleData = ExperimentalData()
    exampleData.loadData('example:simulation_ptycho')
    exampleData.operationMode = 'CPM'

    # now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
    # now create an object to hold everything we're eventually interested in
    optimizable = Optimizable(exampleData)
    optimizable.npsm = 1 # Number of probe modes to reconstruct
    optimizable.nosm = 1 # Number of object modes to reconstruct
    optimizable.nlambda = 1 # Number of wavelength
    optimizable.prepare_reconstruction()
    
    # this will copy any attributes from experimental data that we might care to optimize
    # # Set monitor properties
    monitor = Monitor()
    monitor.figureUpdateFrequency = 2
    monitor.objectPlot = 'complex'
    monitor.verboseLevel = 'high' # high: plot two figures, low: plot only one figure

    # Run the reconstruction
    mPIE_engine = mPIE.mPIE_GPU(optimizable, exampleData, monitor)
    mPIE_engine.numIterations = 50
    mPIE_engine.doReconstruction()
    
    # Compare mPIE to ePIE
    # ePIE_engine = ePIE.ePIE_GPU(optimizable, exampleData, monitor)
    # ePIE_engine.numIterations = 20
    # ePIE_engine.doReconstruction()


    # now save the data
    # optimizable.saveResults('reconstruction.hdf5')
# 