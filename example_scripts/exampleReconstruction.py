import matplotlib
matplotlib.use('tkagg')
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
#optimizable.initialObject = 'random'
#optimizable.initialObject = 'ones'
# now we want to run an optimizer. First create it.
qNewton_engine = qNewton.qNewton(optimizable, exampleData)
# set any settings involving ePIE in this object.
qNewton_engine.numIterations = 20
# now, run the reconstruction
qNewton_engine.doReconstruction()
qNewton_engine.showEndResult()



# OK, now I want to run a different reconstruction algorithm, what now?
# just create a new reconstruction ePIE_engine
# we initialize it with our data and optimizable parameters so we can forget about them afterwards
ePIE_engine = ePIE.ePIE(optimizable, exampleData)
# set any settings involving ePIE in this object.
ePIE_engine.numIterations = 20
# now, run the reconstruction
ePIE_engine.doReconstruction()

# check FPM recon
initial_guess = ifft2c(optimizable.initialObject[0,:,:])
reconstruction = ifft2c(optimizable.object[0,:,:])
probe = optimizable.probe[0,:,:]
plt.figure(10)
plt.ioff()
plt.subplot(221)
plt.title('initial guess')
plt.imshow(abs(initial_guess))
plt.subplot(222)
plt.title('amplitude')
plt.imshow(abs(reconstruction))
plt.subplot(223)
plt.title('phase')
plt.imshow(np.angle(reconstruction))
plt.subplot(224)
plt.title('probe phase')
plt.imshow(np.abs(probe))
plt.pause(10)

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
ePIE_engine.numIterations = 50
# now, run the reconstruction
ePIE_engine.doReconstruction()

# check ptycho recon
initial_guess = optimizable.initialObject[0,:,:]
reconstruction = optimizable.object[0,:,:]
probe = optimizable.probe[0,:,:]
plt.figure(10)
plt.ioff()
plt.subplot(221)
plt.title('initial guess')
plt.imshow(abs(initial_guess))
plt.subplot(222)
plt.title('amplitude')
plt.imshow(abs(reconstruction))
plt.subplot(223)
plt.title('phase')
plt.imshow(np.angle(reconstruction))
plt.subplot(224)
plt.title('probe phase')
plt.imshow(np.abs(probe))
plt.pause(10)

# mPIE_engine.doReconstruction()

# now save the data
# optimizable.saveResults('reconstruction.hdf5')
