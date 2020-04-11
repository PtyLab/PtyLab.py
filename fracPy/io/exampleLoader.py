import h5py  # for now
from pathlib import Path

fracpy_directory = Path(__file__).parent.parent.parent
example_data_directory = fracpy_directory / 'example_data'


def load_example(example_filename):
    """ Load an example from the example_data folder. """
    archive = h5py.File(example_data_directory / example_filename, 'r')
    #TODO(@tomas_aidukas): Please reimplement this method to actually load our examples using read_hdf5.py
    return archive
