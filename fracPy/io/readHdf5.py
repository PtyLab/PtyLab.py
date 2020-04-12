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
    'Nd',               # ?
    'xd',               # ?
    'zo'                # ?
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

    # sanity checks
    if filename.suffix not in allowed_extensions:
        raise NotImplementedError('%s is not a valid extension. Currently, only these extensions are allowed: %s.' %\
                                  (filename.suffix, ['   '.join(allowed_extensions)][0]))
    
    # define data order
    if python_order:
        processor = pythonize_order
    else:
        processor = lambda x:x

    # start h5 loading, but check data fields first (defined above)
    dataset = dict()
    if checkDataFields(filename) == None:
        try:
            with h5py.File(str(filename), mode='r') as hdf5_file:
                for key, val in hdf5_file.items():
                    # complex number stored as a structured numpy arrays
                    # with separate real and imaginary arrays
                    if key in required_fields:     
                        print(key)
                        if key == 'probe':
                            dataset[key] = val[:]['real'] + 1j*val[:]['imag']
                        else:
                            dataset[key] = processor(val[:])

            # DOESN'T WORK WELL WITH COMPLEX FILES, NEED TO DISCUSS
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
    with h5py.File(str(filename), mode='r') as hdf5_file:
        file_fields = set(hdf5_file.keys())
            
    for k in required_fields:
        if k not in file_fields:
            raise KeyError('hdf5 file misses key %s' % k)
    
    return None