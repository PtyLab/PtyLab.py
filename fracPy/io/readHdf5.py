from pathlib import Path
import tables
import numpy as np
import logging
from scipy.io import loadmat
import h5py

logger = logging.getLogger('readHdf5')

# this is the list of things that a dataset has to incorporate
required_fields = [
    'ptychogram'
]
# These extensions can be loaded
allowed_extensions = ['.h5', '.hdf5', '.mat']


def pythonize_order(ary):
    """ Change from matlab to python indexing. """
    #TODO (@MaisieD) can you check that this is all we have to do?
    return ary.T

def loadInputData(filename:Path, python_order:bool=True):
    """
    Load an hdf5 file
    :param filename: the .hdf5 file that has to be loaded. If it's a .mat file it will attempt to load it
    :param python_oder:
            Wether to read in the files in a way that is common in python, aka for a list of images the first index is the image and not the pixel.
    :return:
    """
    filename = Path(filename)
    logger.debug('Loading input data: %s', filename)

    if filename.suffix not in allowed_extensions:
        raise NotImplementedError('%s is not a valid extension. Currently, only these extensions are allowed: %s.' %\
                                  (filename.suffix, ['   '.join(allowed_extensions)][0]))
    # TODO(tomas_aidukas): Please implement loading of the data here
    dataset = dict()
    
    # define data order
    if python_order:
        processor = pythonize_order
    else:
        processor = lambda x:x

    # start h5 loading
    try:
        with h5py.File(str(filename), mode='r') as hdf5_file:
            for name in hdf5_file.keys():
                load_to_memory = hdf5_file[str(name)][:]
                dataset[name] = processor(load_to_memory)
                
        # DOESN'T WORK WELL WITH COMPLEX FILES, WILL DELETE (IF LOADED FROM .MAT)
        # with tables.open_file(str(filename), mode='r') as hdf5_file:
        #     # PyTables hierarchy : Table -> Group -> Node
        #     # Table = hdf5_file
        #     # Group = '/' or 'RootGroup' by default (assumed here)
        #     # Loop through Nodes stored in the root group
        #     # 
        #     # This library has problems loading complex arrays from .mat
        #     # it is a problem wheb probe is stored in .mat files
        #     for node in hdf5_file.root._f_walknodes():
        #         try:
        #             dataset[node._v_name] = processor(node.read())
        #         except:
        #             dataset[node._v_name] = None
    except Exception as e:
        logger.error('Error reading hdf5 file!')
        raise e
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
    archive = tables.open_file(filename) # load_archive(filename) # This is your part
    # Feel free to change it but this is the gist
    for k in required_fields:
        if k not in archive.list_nodes():
            raise KeyError('hdf5 file misses key %s' % k)
