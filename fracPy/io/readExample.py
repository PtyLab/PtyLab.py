from fracPy.io.readHdf5 import loadInputData
from fracPy.io import getExampleDataFolder
from pathlib import Path
import logging
import sys
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
#logger.addHandler(logging.StreamHandler(sys.stderr))

exampleFiles = {
    'fpm_dataset': 'fpm_usaft_data.hdf5',
    'simulationTiny': 'simulationTiny.hdf5'
}

# This is a convenience class to aid in loading a particular example

def listExamples():
    logger.info('Please check the README.md in the example_data folder at %s', getExampleDataFolder())
    logger.info('Currently available datasets: ')
    for key in exampleFiles:
        logger.info('\t %s', key)
    logger.info('To load a specific dataset please run `loadExample(<name>)`.')


def examplePath(key:str):
    """
    Return the full path to a particular example file.
    :param key: which dataset to look for.
    :return:
    """
    try:
        filename = exampleFiles[key]
    except KeyError:
        filename = key
    full_filename = getExampleDataFolder() / filename
    if not full_filename.exists():
        raise FileNotFoundError()
    return full_filename


def loadExample(key:str, *args, **kwargs):
    """ Load an example from the example_data folder. To check which datasets are available, either run
    listExamples() or look in the folder example_data.

    :param key: Key to look for. Will try to look up the filename in readExample.exampleFiles, otherwise it will attempt to
        find the file in example_data.
    :param args: will be passed to loadInputData
    :param kwargs: idem
    :return:
    """
    return loadInputData(examplePath(key), *args, **kwargs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    listExamples()
    loadExample('simulationTiny')