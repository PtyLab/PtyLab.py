import matplotlib
# matplotlib.use('tkagg')
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines import ePIE, mPIE, qNewton
import logging
logging.basicConfig(level=logging.INFO)
from fracPy.utils.utils import ifft2c
from matplotlib import pyplot as plt
import numpy as np

""" 
FPM data reconstructor 
change data visualization and initialization options manually for now
"""
# create an experimentalData object and load a measurement
exampleData = ExperimentalData()
exampleData.loadData('example:simulation_fpm')
# exampleData.loadData('example:simulation_ptycho')
exampleData.operationMode = 'FPM'
# # now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# # now create an object to hold everything we're eventually interested in
optimizable = Optimizable(exampleData)
# # this will copy any attributes from experimental data that we might care to optimize

# now we want to run an optimizer. First create it.
qNewton_engine = qNewton.qNewton(optimizable, exampleData)
# set any settings involving ePIE in this object.
qNewton_engine.numIterations = 10
# now, run the reconstruction
qNewton_engine.doReconstruction()
qNewton_engine.showEndResult()


""" 
ptycho data reconstructor 
change data visualization and initialization options manually for now
"""

exampleData = ExperimentalData()
exampleData.loadData('example:simulation_ptycho')
exampleData.operationMode = 'CPM'

# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
optimizable = Optimizable(exampleData)
# this will copy any attributes from experimental data that we might care to optimize
# now we want to run an optimizer. First create it.
ePIE_engine = ePIE.ePIE(optimizable, exampleData)
# set any settings involving ePIE in this object.
ePIE_engine.numIterations = 100
# now, run the reconstruction
ePIE_engine.doReconstruction()
ePIE_engine.showEndResult()


# now save the data
# optimizable.saveResults('reconstruction.hdf5')
