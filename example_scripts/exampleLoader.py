from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.io import readExample


# This script loads an example dataset.

##############################################################################
# Fourier ptychography - hdf5 data example using fracPy/io
##############################################################################
#example_data_folder = fracpy_directory / 'example_data/fpm_usaft_data.hdf5'

loader_object = ExperimentalData()
#example_data_folder)
loader_object.loadExample('fpm_dataset')#loadData(python_order=False)

plt.figure(1)
plt.subplot(131)
plt.imshow(loader_object.ptychogram[:,:,0])
plt.subplot(132)
plt.imshow(np.abs(loader_object.probe))
plt.subplot(133)
plt.imshow(np.angle(loader_object.probe))
plt.show()

plt.figure(2)
plt.plot(loader_object.positions[:,0], loader_object.positions[:,1], '-o')