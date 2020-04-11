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
loader_object.load_from_hdf5()



