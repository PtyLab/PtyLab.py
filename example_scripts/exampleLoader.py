from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.io import readExample


# This script loads an example dataset.

##############################################################################
# Fourier ptychography - hdf5 data example using fracPy/io
##############################################################################


loaderObject = ExperimentalData('example:fpm_dataset')
#example_data_folder)
#loaderObject.loadExample('fpm_dataset')#loadData

plt.figure(1)
plt.subplot(131)
plt.imshow(loaderObject.ptychogram[0])
plt.subplot(132)
plt.imshow(np.abs(loaderObject.probe))
plt.subplot(133)
plt.imshow(np.angle(loaderObject.probe))
plt.show()

plt.figure(2)
plt.plot(loaderObject.positions[:,0], loaderObject.positions[:,1], '-o')