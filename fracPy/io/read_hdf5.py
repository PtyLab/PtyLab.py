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

loadInputData('invalid')