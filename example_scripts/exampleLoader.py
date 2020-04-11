import tables  # for now
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData

# data directory
fracpy_directory = Path(__file__).parent.parent
example_data_folder = fracpy_directory / 'example_data/usaft2_441_LED/Image_section_0_color_1.hdf5'

##############################################################################
# 1. Loader with above implementations in DataLoader
##############################################################################
loader_object = ExperimentalData(example_data_folder)
loader_object.load_from_hdf5()



