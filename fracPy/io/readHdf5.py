from pathlib import Path
import tables
import numpy as np
import logging
from scipy.io import loadmat
import h5py

logger = logging.getLogger('readHdf5')

# TODO (@MaisieD): I used the fields which I found in your .mat file
# this is the list of things that a dataset has to incorporate
required_fields = [
    'ptychogram',       # 3D image stack 
    'probe',            # 2D complex probe
    'wavelength',       # illumination lambda
    'positions',        # diffracted field positions
    'Nd',               # image pixel number
    'xd',               # pixel size
    'zo'                # sample to detector distance
]

# These extensions can be loaded
allowed_extensions = ['.h5', '.hdf5', '.mat']


def pythonizeOrder(ary):
    """ Change from matlab to python indexing. """
    #TODO (@MaisieD) can you check that this is all we have to do?
    return ary.T


def loadInputData(filename:Path, python_order:bool=True):
    """
    Load all values from an hdf5 file into a dictionary, but only with the required fields
    :param filename: the .hdf5 file that has to be loaded. If it's a .mat file it will attempt to load it
    :param python_oder:
            Wether to read in the files in a way that is common in python, aka for a list of images the first index is the image and not the pixel.
    :return:
    """
    filename = Path(filename)
    logger.debug('Loading input data: %s', filename)

    # sanity checks
    if filename.suffix not in allowed_extensions:
        raise NotImplementedError('%s is not a valid extension. Currently, only these extensions are allowed: %s.' %\
                                  (filename.suffix, ['   '.join(allowed_extensions)][0]))
    
    # define data order
    if python_order:
        processor = pythonizeOrder
    else:
        processor = lambda x:x

    # start h5 loading, but check data fields first (defined above)
    dataset = dict()
    try:

        with tables.open_file(str(filename), mode='r') as hdf5_file:
            # PyTables hierarchy : Table -> Group -> Node
            # Go through all nodes hanging from the default
            # 'root' group
            # for node in hdf5_file.root._f_walknodes():
            for node in hdf5_file.walk_nodes("/", "Array"):
                key = node.name
                value = node.read()   
                
                # change image array order
                if key == 'ptychogram':
                    value = processor(value)
                    
                # load only the required fields
                if key in required_fields:                        
                    dataset[key] = value
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
    with tables.open_file(str(filename), mode='r') as hdf5_file:
        # get a list of nodes
        nodes = hdf5_file.list_nodes("/")
        # get the names of each node which will be the field names stored
        # within the hdf5 file
        file_fields = [node.name for node in nodes]
            
    # check if all the required fields are within the file
    for k in required_fields:
        if k not in file_fields:
            raise KeyError('hdf5 file misses key %s' % k)
    
    return None
