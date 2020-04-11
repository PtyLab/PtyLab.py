from pathlib import Path
import tables
import numpy as np


def loadInputData(filename:Path):
    """
    Load an hdf5 file
    :param filename: the .hdf5 file that has to be loaded. If it's a .mat file it will attempt to load it
    :return:
    """
    filename = Path(filename)
    allowed_extensions = ['.h5', 'hdf5', '.mat']
    if not filename.suffix in allowed_extensions:
        raise NotImplementedError('%s is not a valid extension. Currently, only these extensions are allowed: %s.' %\
                                  (filename.suffix, ['   '.join(allowed_extensions)][0]))
    # TODO(tomas_aidukas): Please implement loading of the data here
    dataset = dict()
    dataset['whateveryouneed'] = np.random.rand()
    return dataset


def checkDataFields(filename):
    """
    Make sure that all the fields in a given .hdf5 file are supported and do some sanity checks.

    This is run before loading the file just to make sure that the file is correctly formatted.
    :param filename: '.hdf5' file with all the necessary attributes.
    :return: None if correct
    :raise: KeyError if one of the attributes is missing.
    """
    # TODO(tomas_aidukas) Please implement this, I think it should go along these lines:
    required_keys = ['I', 'dont', 'know']
    archive = None # load_archive(filename) # This is your part
    # Feel free to change it but this is the gist
    for k in required_keys:
        if k not in archive.keys():
            raise KeyError('hdf5 file misses key %s' % k)
