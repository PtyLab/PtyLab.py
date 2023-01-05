from pathlib import Path
import tables
import numpy as np
import logging
from scipy.io import loadmat
import h5py

logger = logging.getLogger("readHdf5")

# These extensions can be loaded
allowed_extensions = [".h5", ".hdf5", ".mat"]


def scalify(l):
    """
    hdf5 file storing (especially when using matlab) can store integers as
    Numpy arrays of size [1,1]. Convert to scalar if that's the case
    """
    l = l.squeeze()
    try:
        return l.item()
    except ValueError:
        return l


def loadInputData(filename: Path, requiredFields, optionalFields):
    """
    Load all values from an hdf5 file into a dictionary, but only with the required fields
    :param filename: the .hdf5 file that has to be loaded. If it's a .mat file it will attempt to load it
    :param python_order:
            Weather to read in the files in a way that is common in python, aka for a list of images the first index
             is the image and not the pixel.
    :return:
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f'Could not find file {filename}.')
    logger.debug("Loading input data: %s", filename)

    # sanity checks
    if filename.suffix not in allowed_extensions:
        raise NotImplementedError(
            "%s is not a valid extension. Currently, only these extensions are allowed: %s."
            % (filename.suffix, ["   ".join(allowed_extensions)][0])
        )

    # start h5 loading, but check data fields first (defined above)
    dataset = dict()
    try:
        with tables.open_file(str(filename), mode="r") as hdf5File:

            # load the required fields
            for key in requiredFields:
                value = hdf5File.root[key].read()
                dataset[key] = scalify(value)

            # load optional fields, otherwise set to None and compute later
            for key in optionalFields:
                # check if the optional field exists otherwise set to None
                if key in hdf5File.root:
                    value = hdf5File.root[key].read()
                    dataset[key] = scalify(value)
                else:
                    dataset[key] = None

    except Exception as e:
        logger.error("Error reading hdf5 file!")
        raise e
    if 'encoder' in dataset:
        print(f'Found encoder with shape {dataset["encoder"].shape}')
        if dataset['encoder'].shape[0] < dataset['encoder'].shape[1]:
            dataset['encoder'] = dataset['encoder'].T
            print('Warning: Automatically changing the shape of encoder. ')
            dataset['encoder'] -= dataset['encoder'].mean(axis=0, keepdims=True)
            print(dataset['encoder'].shape, dataset['encoder'].mean(axis=0))
            # dataset['encoder'] *= -1
    # dirty hack for now

    # upsample

    from PtyLab.utils.utils import fft2c, ifft2c
    # dataset['dxd'] = dataset['dxd'] / 2
    # padwidth = dataset['ptychogram'].shape[-1]//2
    # padwidth = [[0,0], [padwidth, padwidth], [padwidth,padwidth]]
    # dataset['ptychogram'] = abs(ifft2c(np.pad(fft2c(dataset['ptychogram']), padwidth))).astype(np.float32)
    # dataset['ptychogram'] = np.repeat(np.repeat(dataset['ptychogram'], axis=-2, repeats=2), axis=-1, repeats=2)

    return dataset


def getOrientation(filename):
    """
    If orientation is given, return it. Otherwise, return None
    """
    orientation = None
    with h5py.File(str(filename), "r") as archive:
        if "orientation" in archive.keys():
            orientation = np.array(archive["orientation"]).ravel()[0].astype(int)
    return int(orientation)


def checkDataFields(filename, requiredFields):
    """
    Make sure that all the fields in a given .hdf5 file are supported and do some sanity checks.

    This is run before loading the file just to make sure that the file is correctly formatted.
    :param filename: '.hdf5' file with all the necessary attributes.
    :return: None if correct
    :raise: KeyError if one of the attributes is missing.
    """
    with tables.open_file(str(filename), mode="r") as hdf5_file:
        # get a list of nodes
        nodes = hdf5_file.list_nodes("/")
        # get the names of each node which will be the field names stored
        # within the hdf5 file
        fileFields = [node.name for node in nodes]

    # check if all the required fields are within the file
    for k in requiredFields:
        if k not in fileFields:
            raise KeyError("hdf5 file misses key %s" % k)

    return None


def getOrientation(filename):
    """
    Get the orientation from the hdf5 file. If not available, set to None
    """
    orientation = None
    try:
        with h5py.File(str(filename), mode="r") as archive:
            orientation = int(np.array(archive["orientation"]).ravel())
    except KeyError as e:
        print(e)
    return orientation
