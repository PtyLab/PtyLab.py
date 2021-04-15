from pathlib import Path
import tables
import numpy as np
import logging
from scipy.io import loadmat
import h5py

logger = logging.getLogger('readHdf5')


# These are the fields required for ptyLab to work
# ALL VALUES MUST BE IN METERS
required_fields = [
    'ptychogram',       # 3D image stack 
    'wavelength',       # illumination lambda
    'encoder',          # diffracted field positions
    'dxd',              # pixel size
    'zo'                # sample to detector distance
]

# These fields are optional because they will either be computed later
# from the "required_fields" or are required for FPM, but not CPM
# if not provided by the user they will be set as None
optional_fields = [
    'magnification',            # magnification, used for FPM computations of dxp
    'dxp',                      # dxp, can be provided by the user, otherwise will be computed using dxp=dxd/magnification
    'No',                       # number of upsampled pixels
    'Nd',                       # probe/pupil plane size, will be set to Ptychogram size by default
    'entrancePupilDiameter',    # entrance pupil diameter, defined in lens-based microscopes as the aperture diameter, reqquired for FPM
    'spectralDensity',          # CPM parameters: different wavelength
    'theta'                     # CPM parameters: reflection tilt angle
    ]

# These extensions can be loaded
allowed_extensions = ['.h5', '.hdf5', '.mat']


def scalify(l):
    """
    hdf5 file storing (especially when using matlab) can store integers as
    Numpy arrays of size [1,1]. Convert to scalar if that's the case
    """
    # return l if len(l) > 1 else l[0] # <- TODO: doesn't work in all cases!
    l = l.squeeze()
    try:
        return l.item()
    except ValueError:
        return l



def loadInputData(filename:Path):
    """
    Load all values from an hdf5 file into a dictionary, but only with the required fields
    :param filename: the .hdf5 file that has to be loaded. If it's a .mat file it will attempt to load it
    :param python_order:
            Weather to read in the files in a way that is common in python, aka for a list of images the first index
             is the image and not the pixel.
    :return:
    """
    filename = Path(filename)
    logger.debug('Loading input data: %s', filename)

    # sanity checks
    if filename.suffix not in allowed_extensions:
        raise NotImplementedError('%s is not a valid extension. Currently, only these extensions are allowed: %s.' %\
                                  (filename.suffix, ['   '.join(allowed_extensions)][0]))

    # start h5 loading, but check data fields first (defined above)
    dataset = dict()
    try:
        with tables.open_file(str(filename), mode='r') as hdf5_file:
            # PyTables hierarchy : Table -> Group -> Node
            # Go through all nodes hanging from the default
            # for node in hdf5_file.walk_nodes("/", "Array"):
            #     key = node.name
            #     value = node.read()
            #     dataset[key] = scalify(value)
                
            # load the required fields
            for key in required_fields: 
                value = hdf5_file.root[key].read()
                dataset[key] = scalify(value)
           
            # load optional fields, otherwise set to None and compute later
            for key in optional_fields:
                # check if the optional field exists otherwise set to None
                if key in hdf5_file.root:
                    value = hdf5_file.root[key].read()
                    dataset[key] = scalify(value)
                else:
                    dataset[key] = None

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
