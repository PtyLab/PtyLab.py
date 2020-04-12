from readHdf5 import loadInputData
from pathlib import Path

def loadExample(example_data_folder):
    """ Load an example from the example_data folder. """
    archive = loadInputData(example_data_folder)
    return archive

fracpy_directory = Path(__file__).parent.parent.parent
example_data_folder = fracpy_directory / 'example_data/simulationTiny.mat'
data = loadExample(example_data_folder)
