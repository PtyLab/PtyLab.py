
import h5py # for now
from pathlib import Path

fracpy_directory = Path(__file__).parent.parent.parent
example_data_directory = fracpy_directory/'example_data'

def load_example(example_filename):
    """ Load an example from the example_data folder. """
    archive = h5py.File(example_data_directory/example_filename, 'r')
    return archive

if __name__ == '__main__':
    archive = load_example('simulation_tiny.mat')
