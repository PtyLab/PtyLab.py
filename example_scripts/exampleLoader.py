from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.read_write import readExample


# This script loads an example dataset.

##############################################################################
# Fourier ptychography - hdf5 data example using PtyLab/read_write
##############################################################################


loaderObject = ExperimentalData("example:simulation_fpm")
# example_data_folder)
# loaderObject.loadExample('fpm_dataset')#loadData

plt.figure(1)
plt.subplot(131)
plt.imshow(loaderObject.ptychogram[0])
plt.subplot(132)
plt.imshow(np.abs(loaderObject.probe))
plt.subplot(133)
plt.imshow(np.angle(loaderObject.probe))
plt.show()

plt.figure(2)
plt.plot(loaderObject.positions[:, 0], loaderObject.positions[:, 1], "-o")
