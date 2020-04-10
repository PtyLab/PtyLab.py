import tables  # for now
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from fracPy.ptyLab import DataLoader

# data directory
fracpy_directory = Path(__file__).parent.parent
example_data_folder = fracpy_directory / 'example_data/usaft2_441_LED/'

##############################################################################
# 1. Loader with above implementations in DataLoader
##############################################################################
loader_object = DataLoader(example_data_folder)
images = loader_object.load_from_hdf5()
# check data
plt.figure(1)
plt.imshow(np.mean(images[:,:,0:49], axis=2))
plt.show()


##############################################################################
# 2. Loader under the hood
##############################################################################
def load_example(example_folder):
    hdf5_files = example_folder.glob('*.hdf5')
    """ Load an example from the example_data folder. """
    hdf5_file = tables.open_file(next(hdf5_files), mode='r')
    # image_array_crops is name given for the images stored within hdf5
    images = hdf5_file.root.image_array_crops[:,:,:]
    hdf5_file.close()   
    
    """"load k-space values as well"""
    txt_files = example_folder.glob('*.txt')
    positions = np.loadtxt(next(txt_files))
    return images, positions

# load data files
images, positions = load_example(example_data_folder)

# check data
plt.figure(1)
plt.subplot(121)
plt.imshow(np.mean(images[:,:,0:49], axis=2))

plt.subplot(122)
plt.plot(positions[:,0], positions[:,1], '-o')
plt.show()

# for i in range(images.shape[2]):
#     plt.imshow(images[:,:,i])
#     plt.pause(0.1)
