from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData

# data directory
fracpy_directory = Path(__file__).parent.parent

##############################################################################
# Ptychography - matlab example using DataLoader
##############################################################################
example_data_folder = fracpy_directory / 'example_data/simulationTiny.mat'

loader_object = ExperimentalData(example_data_folder)
loader_object.load_from_hdf5()

plt.figure(1)
plt.subplot(131)
plt.imshow(loader_object.ptychogram[:,:,0])
plt.subplot(132)
plt.imshow(abs(loader_object.probe))
plt.subplot(133)
plt.imshow(np.angle(loader_object.probe))
plt.show()

##############################################################################
# Fourier ptychography - hdf5 data example using fracPy/io
# TODO(@Tomas) : need to upload it properly
##############################################################################
# example_data_folder = fracpy_directory / 'example_data/USAFT_FPM_data.hdf5'

# loader_object = ExperimentalData(example_data_folder)
# loader_object.load_from_hdf5()

# plt.figure(1)
# plt.subplot(131)
# plt.imshow(loader_object.ptychogram[:,:,0])
# plt.subplot(132)
# plt.imshow(abs(loader_object.probe))
# plt.subplot(133)
# plt.imshow(np.angle(loader_object.probe))
# plt.show()
